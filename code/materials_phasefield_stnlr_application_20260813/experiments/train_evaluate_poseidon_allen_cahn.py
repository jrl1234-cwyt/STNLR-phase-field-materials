from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import NestedTRMoLE_Linear, StaticLoRA_Linear  # noqa: E402
from materials_phasefield_stnlr_application_20260813.experiments.evaluate_allen_cahn_conditional_trajectory import summarize  # noqa: E402
from materials_phasefield_stnlr_application_20260813.experiments.train_allen_cahn_conditional_trajectory import material_structure_loss  # noqa: E402


class PoseidonNestedLinear(NestedTRMoLE_Linear):
    def __init__(self, source: nn.Linear, schedule: str, u_mid_rank: int):
        if schedule == "early_high":
            rank_schedule = "quantized_t_lora_decay"
            fixed_edge_rank = None
            fixed_mid_rank = None
        elif schedule == "u_shaped":
            rank_schedule = "fixed_bins"
            fixed_edge_rank = 16
            fixed_mid_rank = u_mid_rank
        else:
            raise ValueError(f"Unsupported nested schedule: {schedule}")
        super().__init__(
            source.in_features,
            source.out_features,
            bias=source.bias is not None,
            nested_ranks=(4, 8, 16),
            rank_schedule=rank_schedule,
            rank_alpha=1.2,
            fixed_edge_rank=fixed_edge_rank,
            fixed_mid_rank=fixed_mid_rank,
            lora_alpha=1.0,
            base_trainable=False,
            num_timesteps=2,
        )
        self.copy_base_from_linear(source)
        self.current_time = None
        self.forced_rank = None

    def rank_from_timestep(self, timestep):
        if self.forced_rank is not None:
            return int(self.forced_rank)
        return super().rank_from_timestep(timestep)

    def forward(self, x):
        return super().forward(x, self.current_time)


def is_adapter_target(name: str, module: nn.Module) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    return any(
        marker in name
        for marker in (
            ".attention.self.query",
            ".attention.self.key",
            ".attention.self.value",
            ".attention.output.dense",
            ".intermediate.dense",
            ".output.dense",
        )
    )


def replace_linears(
    model: nn.Module,
    kind: str,
    nested_schedule: str,
    u_mid_rank: int,
    static_rank: int = 16,
) -> list[str]:
    names = [name for name, module in model.named_modules() if is_adapter_target(name, module)]
    for name in names:
        parent_name, child_name = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        source = getattr(parent, child_name)
        if kind == "static":
            replacement = StaticLoRA_Linear(
                source.in_features,
                source.out_features,
                bias=source.bias is not None,
                rank=static_rank,
                lora_alpha=1.0,
                base_trainable=False,
            )
            replacement.copy_base_from_linear(source)
        elif kind == "stnlr":
            replacement = PoseidonNestedLinear(source, nested_schedule, u_mid_rank)
        else:
            raise ValueError(kind)
        setattr(parent, child_name, replacement)
    return names


def set_nested_time(
    model: nn.Module,
    lead_time: torch.Tensor,
    direction: str = "decay",
) -> None:
    controller_time = lead_time if direction == "decay" else 1.0 - lead_time
    for module in model.modules():
        if isinstance(module, PoseidonNestedLinear):
            module.current_time = controller_time


@torch.no_grad()
def initialize_scalar_heads_from_pretrained(model: nn.Module, checkpoint_dir: Path) -> None:
    """Project the four-channel pretrained heads to one scalar phase field.

    Channel averaging gives static and nested adapters the same deterministic
    task-head initialization and avoids coupling the comparison to a random
    cross-physics input/recovery head.
    """
    state = torch.load(checkpoint_dir / "pytorch_model.bin", map_location="cpu")
    input_weight = state["embeddings.patch_embeddings.projection.weight"].mean(
        dim=1, keepdim=True
    )
    model.embeddings.patch_embeddings.projection.weight.copy_(input_weight)
    model.embeddings.patch_embeddings.projection.bias.copy_(
        state["embeddings.patch_embeddings.projection.bias"]
    )
    recovery_weight = state["patch_recovery.projection.weight"].mean(
        dim=1, keepdim=True
    )
    model.patch_recovery.projection.weight.copy_(recovery_weight)
    model.patch_recovery.projection.bias.copy_(
        state["patch_recovery.projection.bias"].mean().reshape(1)
    )
    mixup_weight = state["patch_recovery.mixup.weight"].mean(
        dim=(0, 1), keepdim=True
    )
    model.patch_recovery.mixup.weight.copy_(mixup_weight)


@torch.no_grad()
def initialize_multichannel_heads_from_pretrained(
    model: nn.Module,
    checkpoint_dir: Path,
    channels: int,
) -> None:
    """Deterministically initialize a smaller task head from Poseidon's channels."""
    state = torch.load(checkpoint_dir / "pytorch_model.bin", map_location="cpu")
    model.embeddings.patch_embeddings.projection.weight.copy_(
        state["embeddings.patch_embeddings.projection.weight"][:, :channels]
    )
    model.embeddings.patch_embeddings.projection.bias.copy_(
        state["embeddings.patch_embeddings.projection.bias"]
    )
    model.patch_recovery.projection.weight.copy_(
        state["patch_recovery.projection.weight"][:, :channels]
    )
    model.patch_recovery.projection.bias.copy_(
        state["patch_recovery.projection.bias"][:channels]
    )
    model.patch_recovery.mixup.weight.copy_(
        state["patch_recovery.mixup.weight"][:channels, :channels]
    )


def build_poseidon(args, device):
    sys.path.insert(0, str(args.poseidon_code))
    from scOT.model import ScOT, ScOTConfig

    config = ScOTConfig.from_pretrained(args.poseidon_checkpoint)
    channels = int(getattr(args, "num_channels", 1))
    config.num_channels = channels
    config.num_out_channels = channels
    config.channel_slice_list_normalized_loss = None
    model = ScOT.from_pretrained(
        args.poseidon_checkpoint,
        config=config,
        ignore_mismatched_sizes=True,
    )
    if channels == 1:
        initialize_scalar_heads_from_pretrained(model, args.poseidon_checkpoint)
    else:
        initialize_multichannel_heads_from_pretrained(
            model, args.poseidon_checkpoint, channels
        )
    replaced = []
    if args.kind in {"static", "stnlr"}:
        replaced = replace_linears(
            model,
            args.kind,
            args.nested_schedule,
            args.u_mid_rank,
            getattr(args, "static_rank", 16),
        )
        for parameter in model.parameters():
            parameter.requires_grad = False
        for name, parameter in model.named_parameters():
            if (
                "lora_A" in name
                or "lora_B" in name
                or name.startswith("embeddings.patch_embeddings.")
                or name.startswith("patch_recovery.")
            ):
                parameter.requires_grad = True
    elif args.kind == "full":
        for parameter in model.parameters():
            parameter.requires_grad = True
    else:
        raise ValueError(args.kind)
    return model.to(device), replaced


def load_data(path: Path):
    payload = np.load(path)
    return {key: payload[key] for key in payload.files}


def training_batch(data, count, batch_size, device, generator):
    trajectory_ids = torch.randint(0, count, (batch_size,), generator=generator, device=device)
    time_id = int(
        torch.randint(
            1,
            data["fields"].shape[1],
            (1,),
            generator=generator,
            device=device,
        ).item()
    )
    ids = trajectory_ids.cpu().numpy()
    initial = torch.from_numpy(data["initial"][ids]).to(device).unsqueeze(1)
    target = torch.from_numpy(data["fields"][ids, time_id]).to(device).unsqueeze(1)
    lead = float(data["times"][time_id] / data["times"][-1])
    lead_time = torch.full((batch_size,), lead, device=device)
    epsilon = torch.from_numpy(data["epsilon"][ids]).to(device)
    reaction = torch.from_numpy(data["reaction"][ids]).to(device)
    return initial, target, lead_time, epsilon, reaction


@torch.no_grad()
def evaluate(model, kind, data, start, count, batch_size, device, nested_time_direction):
    fields = torch.from_numpy(data["fields"][start : start + count]).to(device).unsqueeze(2)
    initials = torch.from_numpy(data["initial"][start : start + count]).to(device).unsqueeze(1)
    epsilon = torch.from_numpy(data["epsilon"][start : start + count]).to(device)
    reaction = torch.from_numpy(data["reaction"][start : start + count]).to(device)
    times = torch.from_numpy(data["times"]).to(device)
    predictions = []
    model.eval()
    for begin in range(0, count, batch_size):
        end = min(begin + batch_size, count)
        batch_predictions = []
        for physical_time in times:
            lead = (physical_time / times[-1]).expand(end - begin)
            if kind == "stnlr":
                set_nested_time(model, lead, nested_time_direction)
            output = model(pixel_values=initials[begin:end], time=lead).output
            batch_predictions.append(output)
        predictions.append(torch.stack(batch_predictions, dim=1))
    predicted = torch.cat(predictions, dim=0)
    return summarize(
        predicted,
        fields,
        epsilon,
        reaction,
        float(times[1] - times[0]),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--poseidon-code", type=Path, default=Path("/tmp/poseidon-stnlr"))
    parser.add_argument("--poseidon-checkpoint", type=Path, default=Path("/tmp/poseidon_model"))
    parser.add_argument("--kind", choices=["static", "stnlr", "full"], required=True)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5.0e-4)
    parser.add_argument("--energy-weight", type=float, default=0.0)
    parser.add_argument("--interface-weight", type=float, default=0.0)
    parser.add_argument("--interface-temperature", type=float, default=0.04)
    parser.add_argument(
        "--nested-time-direction",
        choices=["decay", "growth"],
        default="decay",
        help="Map physical lead time to decreasing or increasing active prefix rank.",
    )
    parser.add_argument(
        "--nested-schedule",
        choices=["early_high", "u_shaped"],
        default="early_high",
        help="Physical-time rank profile for the shared nested factor bank.",
    )
    parser.add_argument(
        "--u-mid-rank",
        type=int,
        choices=[4, 8],
        default=8,
        help="Middle-third prefix rank for the U-shaped material schedule.",
    )
    parser.add_argument(
        "--skip-checkpoint",
        action="store_true",
        help="Write metrics without a model checkpoint (useful for full-FT upper bounds).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-count", type=int, default=100)
    parser.add_argument("--eval-start", type=int, default=120)
    parser.add_argument("--eval-count", type=int, default=40)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data = load_data(args.data)
    model, replaced = build_poseidon(args, device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    print(
        f"kind={args.kind} total={sum(p.numel() for p in model.parameters())} "
        f"trainable={sum(p.numel() for p in trainable)} replaced={len(replaced)}",
        flush=True,
    )
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    generator = torch.Generator(device=device).manual_seed(args.seed + 1701)
    model.train()
    start_time = time.time()
    running = running_field = running_energy = running_interface = 0.0
    for step in range(1, args.steps + 1):
        initial, target, lead, epsilon, reaction = training_batch(
            data, args.train_count, args.batch_size, device, generator
        )
        if args.kind == "stnlr":
            set_nested_time(model, lead, args.nested_time_direction)
        prediction = model(pixel_values=initial, time=lead).output
        field_loss = (prediction - target).square().mean()
        energy_loss, interface_loss = material_structure_loss(
            prediction,
            target,
            epsilon,
            reaction,
            args.interface_temperature,
        )
        loss = (
            field_loss
            + args.energy_weight * energy_loss
            + args.interface_weight * interface_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()
        running += float(loss.detach())
        running_field += float(field_loss.detach())
        running_energy += float(energy_loss.detach())
        running_interface += float(interface_loss.detach())
        if step % 100 == 0:
            print(
                f"step={step:05d} loss={running/100:.6f} field={running_field/100:.6f} "
                f"energy={running_energy/100:.6f} interface={running_interface/100:.6f} "
                f"steps_per_sec={step/(time.time()-start_time):.2f}",
                flush=True,
            )
            running = running_field = running_energy = running_interface = 0.0

    metrics = evaluate(
        model,
        args.kind,
        data,
        args.eval_start,
        args.eval_count,
        args.batch_size,
        device,
        args.nested_time_direction,
    )
    trainable_state = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if any(
            marker in name
            for marker in (
                "lora_A",
                "lora_B",
                "embeddings.patch_embeddings.",
                "patch_recovery.",
            )
        )
    }
    if args.kind == "full":
        trainable_state = model.state_dict()
    if not args.skip_checkpoint:
        torch.save(
            {
                "model": trainable_state,
                "kind": args.kind,
                "args": vars(args),
                "trainable_parameters": sum(p.numel() for p in trainable),
            },
            args.out_dir / "final.pt",
        )
    result = {
        "kind": args.kind,
        "metrics": metrics,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in trainable),
        "replaced_linear_layers": len(replaced),
        "nested_schedule": args.nested_schedule if args.kind == "stnlr" else None,
        "u_mid_rank": args.u_mid_rank if args.kind == "stnlr" and args.nested_schedule == "u_shaped" else None,
        "elapsed_seconds": time.time() - start_time,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
