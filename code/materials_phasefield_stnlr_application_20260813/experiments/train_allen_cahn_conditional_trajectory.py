#!/usr/bin/env python3
"""Train conditional Allen--Cahn trajectory flow models and PEFT adapters."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

torch.backends.cudnn.enabled = False

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import DiT_models  # noqa: E402


def build_model(kind: str):
    common = dict(input_size=64, in_channels=5, num_classes=1, learn_sigma=False)
    if kind == "base":
        return DiT_models["DiT-S/4"](**common)
    if kind == "static":
        return DiT_models["DiT-S/4"](
            **common, use_static_lora=True, static_lora_rank=16
        )
    if kind == "stnlr":
        return DiT_models["DiT-S/4"](
            **common,
            use_nested_tr_mole=True,
            nested_tr_mole_ranks="4,8,16",
            nested_tr_mole_rank_schedule="quantized_t_lora_decay",
            nested_tr_mole_rank_alpha=1.2,
        )
    raise ValueError(kind)


def freeze_adapter_only(model):
    names = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad = "lora_A" in name or "lora_B" in name
        if parameter.requires_grad:
            names.append(name)
    if not names:
        raise RuntimeError("No low-rank adapter parameters found.")
    return names


def condition_input(state, initial, epsilon, reaction, physical_time):
    b, _, h, w = state.shape
    eps_value = ((epsilon - 0.0385) / 0.0165).view(b, 1, 1, 1)
    reaction_value = ((reaction - 1.2) / 0.4).view(b, 1, 1, 1)
    time_value = (physical_time / 0.6).view(b, 1, 1, 1)
    return torch.cat(
        [
            state,
            initial,
            eps_value.expand(b, 1, h, w),
            reaction_value.expand(b, 1, h, w),
            time_value.expand(b, 1, h, w),
        ],
        dim=1,
    )


def flow_reconstruct(model, target, initial, epsilon, reaction, physical_time, t, z):
    tau = ((t.float() + 0.5) / 1000.0).view(-1, 1, 1, 1)
    x_t = (1.0 - tau) * target + tau * z
    y = torch.zeros(target.shape[0], dtype=torch.long, device=target.device)
    prediction = model(
        condition_input(x_t, initial, epsilon, reaction, physical_time), t, y
    )[:, :1]
    target_velocity = z - target
    loss = (prediction - target_velocity).square().mean()
    reconstruction = z - prediction
    return loss, reconstruction


def laplacian_periodic(field, domain_length=1.0):
    h, w = field.shape[-2:]
    ky = 2.0 * math.pi * torch.fft.fftfreq(
        h, d=domain_length / h, device=field.device, dtype=torch.float32
    )
    kx = 2.0 * math.pi * torch.fft.rfftfreq(
        w, d=domain_length / w, device=field.device, dtype=torch.float32
    )
    k2 = ky[:, None].square() + kx[None, :].square()
    transformed = torch.fft.rfft2(field.float(), dim=(-2, -1))
    return torch.fft.irfft2(
        -k2[None, None] * transformed, s=(h, w), dim=(-2, -1)
    )


def pde_residual_loss(
    center,
    minus,
    plus,
    epsilon,
    reaction,
    delta_t,
    mode,
    preconditioner_floor=0.0,
):
    time_derivative = (plus - minus) / (2.0 * delta_t)
    eps = epsilon.view(-1, 1, 1, 1)
    lam = reaction.view(-1, 1, 1, 1)
    rhs = eps.square() * laplacian_periodic(center) + lam * (center - center.pow(3))
    residual = time_derivative - rhs
    scale_field = time_derivative.detach()
    if mode == "preconditioned":
        h, w = residual.shape[-2:]
        ky = 2.0 * math.pi * torch.fft.fftfreq(
            h, d=1.0 / h, device=residual.device, dtype=torch.float32
        )
        kx = 2.0 * math.pi * torch.fft.rfftfreq(
            w, d=1.0 / w, device=residual.device, dtype=torch.float32
        )
        k2 = ky[:, None].square() + kx[None, :].square()
        shift = (2.0 * reaction).view(-1, 1, 1, 1)
        low_pass = shift / (
            shift + epsilon.view(-1, 1, 1, 1).square() * k2[None, None]
        )
        # A positive floor retains high-frequency interface information while
        # preserving the operator-aware spectral reweighting.
        multiplier = preconditioner_floor + (1.0 - preconditioner_floor) * low_pass

        def apply(value):
            transformed = torch.fft.rfft2(value.float(), dim=(-2, -1))
            return torch.fft.irfft2(
                multiplier * transformed, s=(h, w), dim=(-2, -1)
            )

        residual = apply(residual)
        scale_field = apply(scale_field)
    numerator = residual.square().flatten(1).mean(1)
    denominator = scale_field.square().flatten(1).mean(1).detach().clamp_min(1.0e-4)
    return (numerator / denominator).mean()


def free_energy_per_sample(field, epsilon, reaction):
    """Discrete periodic Allen--Cahn free energy for each field in a batch."""
    n = field.shape[-1]
    dx = 1.0 / n
    grad_x = (torch.roll(field, -1, -1) - torch.roll(field, 1, -1)) / (2.0 * dx)
    grad_y = (torch.roll(field, -1, -2) - torch.roll(field, 1, -2)) / (2.0 * dx)
    eps = epsilon.view(-1, 1, 1, 1)
    lam = reaction.view(-1, 1, 1, 1)
    density = (
        0.5 * eps.square() * (grad_x.square() + grad_y.square())
        + 0.25 * lam * (field.square() - 1.0).square()
    )
    return density.flatten(1).mean(1)


def material_structure_loss(
    prediction,
    target,
    epsilon,
    reaction,
    interface_temperature,
    interface_threshold=0.2,
):
    """Return robust energy consistency and a differentiable interface loss.

    The interface term aligns both the spatial soft-interface mask and its area
    fraction.  It therefore targets interface position as well as total extent,
    while leaving the ST-NLR controller and nested factor bank unchanged.
    """
    predicted_energy = free_energy_per_sample(prediction, epsilon, reaction)
    target_energy = free_energy_per_sample(target, epsilon, reaction).detach()
    relative_energy = (predicted_energy - target_energy) / target_energy.clamp_min(1.0e-4)
    energy_loss = F.smooth_l1_loss(
        relative_energy,
        torch.zeros_like(relative_energy),
        beta=0.1,
    )

    predicted_mask = torch.sigmoid(
        (interface_threshold - prediction.abs()) / interface_temperature
    )
    target_mask = torch.sigmoid(
        (interface_threshold - target.detach().abs()) / interface_temperature
    )
    mask_loss = F.smooth_l1_loss(predicted_mask, target_mask, beta=0.05)
    predicted_fraction = predicted_mask.flatten(1).mean(1)
    target_fraction = target_mask.flatten(1).mean(1)
    fraction_loss = F.smooth_l1_loss(
        (predicted_fraction - target_fraction) / target_fraction.clamp_min(0.02),
        torch.zeros_like(target_fraction),
        beta=0.1,
    )
    interface_loss = mask_loss + 0.5 * fraction_loss
    return energy_loss, interface_loss


def sample_batch(data, indices, batch_size, device, generator):
    chosen = torch.randint(
        0, len(indices), (batch_size,), generator=generator, device=device
    )
    trajectory_ids = indices[chosen]
    time_ids = torch.randint(1, data["fields"].shape[1] - 1, (batch_size,), generator=generator, device=device)
    cpu_ids = trajectory_ids.cpu().numpy()
    cpu_t = time_ids.cpu().numpy()
    fields = data["fields"]

    def take(offset):
        array = fields[cpu_ids, cpu_t + offset]
        return torch.from_numpy(array).to(device=device).unsqueeze(1)

    initial = torch.from_numpy(data["initial"][cpu_ids]).to(device=device).unsqueeze(1)
    epsilon = torch.from_numpy(data["epsilon"][cpu_ids]).to(device=device)
    reaction = torch.from_numpy(data["reaction"][cpu_ids]).to(device=device)
    times = torch.from_numpy(data["times"][cpu_t]).to(device=device)
    delta_t = float(data["times"][1] - data["times"][0])
    return initial, take(-1), take(0), take(1), epsilon, reaction, times, delta_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--kind", choices=["base", "static", "stnlr"], required=True)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--residual", choices=["none", "raw", "preconditioned"], default="none")
    parser.add_argument("--residual-weight", type=float, default=0.05)
    parser.add_argument("--preconditioner-floor", type=float, default=0.0)
    parser.add_argument("--energy-weight", type=float, default=0.0)
    parser.add_argument("--interface-weight", type=float, default=0.0)
    parser.add_argument("--interface-temperature", type=float, default=0.04)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-count", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--save-adapter-only",
        action="store_true",
        help="Store only LoRA tensors for adapter models; the frozen base is reloaded separately.",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    payload = np.load(args.data)
    data = {key: payload[key] for key in payload.files}
    count = data["fields"].shape[0]
    train_count = args.train_count if args.train_count > 0 else count
    indices = torch.arange(train_count, device=device)
    model = build_model(args.kind).to(device)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu")
        state = checkpoint.get("model", checkpoint)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"loaded={args.init_checkpoint} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if args.kind != "base":
        names = freeze_adapter_only(model)
        print(f"strict_adapter_only tensors={len(names)}", flush=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(
        f"kind={args.kind} residual={args.residual} total={sum(p.numel() for p in model.parameters())} "
        f"trainable={sum(p.numel() for p in trainable)}",
        flush=True,
    )
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    generator = torch.Generator(device=device).manual_seed(args.seed + 913)
    model.train()
    start = time.time()
    running_fm = running_pde = running_energy = running_interface = running_total = 0.0
    for step in range(1, args.steps + 1):
        initial, minus, center, plus, epsilon, reaction, physical_time, delta_t = sample_batch(
            data, indices, args.batch_size, device, generator
        )
        t = torch.randint(0, 1000, (args.batch_size,), generator=generator, device=device)
        t = t[:1].repeat(args.batch_size)
        z = torch.randn(center.shape, generator=generator, device=device)
        if args.residual == "none" and args.kind == "base":
            fm_loss, _ = flow_reconstruct(
                model, center, initial, epsilon, reaction, physical_time, t, z
            )
            pde_loss = center.new_tensor(0.0)
            energy_loss = center.new_tensor(0.0)
            interface_loss = center.new_tensor(0.0)
        else:
            fm_minus, reconstructed_minus = flow_reconstruct(
                model, minus, initial, epsilon, reaction, physical_time - delta_t, t, z
            )
            fm_center, reconstructed_center = flow_reconstruct(
                model, center, initial, epsilon, reaction, physical_time, t, z
            )
            fm_plus, reconstructed_plus = flow_reconstruct(
                model, plus, initial, epsilon, reaction, physical_time + delta_t, t, z
            )
            fm_loss = (fm_minus + fm_center + fm_plus) / 3.0
            if args.residual == "none":
                pde_loss = center.new_tensor(0.0)
            else:
                pde_loss = pde_residual_loss(
                    reconstructed_center,
                    reconstructed_minus,
                    reconstructed_plus,
                    epsilon,
                    reaction,
                    delta_t,
                    args.residual,
                    args.preconditioner_floor,
                )
            material_terms = [
                material_structure_loss(
                    prediction,
                    target,
                    epsilon,
                    reaction,
                    args.interface_temperature,
                )
                for prediction, target in (
                    (reconstructed_minus, minus),
                    (reconstructed_center, center),
                    (reconstructed_plus, plus),
                )
            ]
            energy_loss = torch.stack([term[0] for term in material_terms]).mean()
            interface_loss = torch.stack([term[1] for term in material_terms]).mean()
        loss = (
            fm_loss
            + (args.residual_weight * pde_loss if args.residual != "none" else 0.0)
            + args.energy_weight * energy_loss
            + args.interface_weight * interface_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()
        running_fm += float(fm_loss.detach())
        running_pde += float(pde_loss.detach())
        running_energy += float(energy_loss.detach())
        running_interface += float(interface_loss.detach())
        running_total += float(loss.detach())
        if step % args.log_every == 0:
            elapsed = time.time() - start
            denom = args.log_every
            print(
                f"step={step:05d} total={running_total/denom:.6f} fm={running_fm/denom:.6f} "
                f"pde={running_pde/denom:.6f} energy={running_energy/denom:.6f} "
                f"interface={running_interface/denom:.6f} steps_per_sec={step/elapsed:.2f}",
                flush=True,
            )
            running_fm = running_pde = running_energy = running_interface = running_total = 0.0
    model_state = model.state_dict()
    adapter_only = bool(args.save_adapter_only and args.kind != "base")
    if adapter_only:
        model_state = {
            name: value
            for name, value in model_state.items()
            if "lora_A" in name or "lora_B" in name
        }
    checkpoint = {
        "model": model_state,
        "args": vars(args),
        "trainable_parameters": sum(p.numel() for p in trainable),
        "adapter_only": adapter_only,
        "base_checkpoint": str(args.init_checkpoint) if adapter_only else None,
    }
    torch.save(checkpoint, args.out_dir / "final.pt")
    (args.out_dir / "config.json").write_text(
        json.dumps({**vars(args), "data": str(args.data), "out_dir": str(args.out_dir), "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None}, indent=2)
    )
    print(f"saved={args.out_dir / 'final.pt'} elapsed={time.time()-start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
