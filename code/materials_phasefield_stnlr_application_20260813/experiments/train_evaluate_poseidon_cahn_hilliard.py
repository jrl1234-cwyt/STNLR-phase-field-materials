#!/usr/bin/env python3
"""Matched static and nested-prefix Poseidon transfer on Cahn--Hilliard."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from train_evaluate_poseidon_allen_cahn import (
    PoseidonNestedLinear,
    build_poseidon,
    load_data,
    set_nested_time,
)
from train_poseidon_physics_calibrated_nested import spectral_distillation


def force_rank(model, rank):
    for module in model.modules():
        if isinstance(module, PoseidonNestedLinear):
            module.forced_rank = rank


def laplacian(field):
    n = field.shape[-1]
    return (
        torch.roll(field, 1, -1) + torch.roll(field, -1, -1)
        + torch.roll(field, 1, -2) + torch.roll(field, -1, -2) - 4.0 * field
    ) * float(n * n)


def free_energy(field, epsilon, reaction):
    n = field.shape[-1]
    gx = 0.5 * n * (torch.roll(field, -1, -1) - torch.roll(field, 1, -1))
    gy = 0.5 * n * (torch.roll(field, -1, -2) - torch.roll(field, 1, -2))
    eps = epsilon.reshape(-1, 1, 1, 1)
    lam = reaction.reshape(-1, 1, 1, 1)
    density = 0.5 * eps.square() * (gx.square() + gy.square())
    density = density + 0.25 * lam * (field.square() - 1.0).square()
    return density.flatten(1).mean(1)


def spectral_centroid(field):
    n = field.shape[-1]
    freq = torch.fft.fftfreq(n, d=1.0 / n, device=field.device)
    kx, ky = torch.meshgrid(freq, freq, indexing="ij")
    radius = torch.sqrt(kx.square() + ky.square()).reshape(1, 1, n, n)
    centered = field - field.mean(dim=(-2, -1), keepdim=True)
    power = torch.fft.fft2(centered.float(), dim=(-2, -1)).abs().square()
    return (power * radius).sum(dim=(-3, -2, -1)) / power.sum(dim=(-3, -2, -1)).clamp_min(1e-8)


def physics_losses(prediction, target, epsilon, reaction):
    pred_energy = free_energy(prediction, epsilon, reaction)
    true_energy = free_energy(target, epsilon, reaction).detach()
    energy = ((pred_energy - true_energy) / true_energy.abs().clamp_min(1e-4)).square().mean()
    pred_mass = prediction.mean(dim=(-3, -2, -1))
    true_mass = target.mean(dim=(-3, -2, -1))
    mass = (pred_mass - true_mass).square().mean()
    pred_scale = spectral_centroid(prediction)
    true_scale = spectral_centroid(target).detach()
    structure = ((pred_scale - true_scale) / true_scale.clamp_min(1e-4)).square().mean()
    return energy, mass, structure


def training_batch(data, count, batch_size, device, generator):
    trajectory_ids = torch.randint(0, count, (batch_size,), generator=generator, device=device)
    time_id = int(torch.randint(1, data["fields"].shape[1], (1,), generator=generator, device=device).item())
    ids = trajectory_ids.cpu().numpy()
    initial = torch.from_numpy(data["initial"][ids]).to(device).unsqueeze(1)
    target = torch.from_numpy(data["fields"][ids, time_id]).to(device).unsqueeze(1)
    lead_value = float(data["times"][time_id] / data["times"][-1])
    lead = torch.full((batch_size,), lead_value, device=device)
    epsilon = torch.from_numpy(data["epsilon"][ids]).to(device)
    reaction = torch.from_numpy(data["reaction"][ids]).to(device)
    return initial, target, lead, epsilon, reaction


@torch.no_grad()
def predict(model, data, start, count, batch_size, rank=None):
    device = next(model.parameters()).device
    initials = torch.from_numpy(data["initial"][start:start + count]).to(device).unsqueeze(1)
    times = torch.from_numpy(data["times"]).to(device)
    if rank is not None:
        force_rank(model, rank)
    rows = []
    model.eval()
    for begin in range(0, count, batch_size):
        end = min(count, begin + batch_size)
        time_rows = []
        for physical_time in times:
            lead = (physical_time / times[-1]).expand(end - begin)
            set_nested_time(model, lead, "decay")
            time_rows.append(model(pixel_values=initials[begin:end], time=lead).output)
        rows.append(torch.stack(time_rows, dim=1))
    return torch.cat(rows, dim=0)


def summarize(predicted, target, initial, epsilon, reaction, mobility, delta_t):
    eps_floor = 1e-8
    relative = torch.linalg.vector_norm((predicted - target).flatten(2), dim=2) / torch.linalg.vector_norm(target.flatten(2), dim=2).clamp_min(eps_floor)
    pred_mass = predicted.mean(dim=(-3, -2, -1))
    initial_mass = initial.mean(dim=(-3, -2, -1)).unsqueeze(1)
    mass_drift = (pred_mass - initial_mass).abs()
    pred_energy = free_energy(predicted[:, -1], epsilon, reaction)
    true_energy = free_energy(target[:, -1], epsilon, reaction)
    energy_error = (pred_energy - true_energy).abs() / true_energy.abs().clamp_min(eps_floor)
    pred_centroid = spectral_centroid(predicted[:, -1])
    true_centroid = spectral_centroid(target[:, -1])
    centroid_error = (pred_centroid - true_centroid).abs() / true_centroid.clamp_min(eps_floor)

    center = predicted[:, 1:-1].reshape(-1, 1, predicted.shape[-2], predicted.shape[-1])
    minus = predicted[:, :-2].reshape_as(center)
    plus = predicted[:, 2:].reshape_as(center)
    repeats = predicted.shape[1] - 2
    eps_r = epsilon[:, None].expand(-1, repeats).reshape(-1, 1, 1, 1)
    reaction_r = reaction[:, None].expand(-1, repeats).reshape(-1, 1, 1, 1)
    mobility_r = mobility[:, None].expand(-1, repeats).reshape(-1, 1, 1, 1)
    time_derivative = (plus - minus) / (2.0 * delta_t)
    chemical_potential = reaction_r * (center.pow(3) - center) - eps_r.square() * laplacian(center)
    rhs = mobility_r * laplacian(chemical_potential)
    residual = time_derivative - rhs
    target_center = target[:, 1:-1].reshape_as(center)
    target_minus = target[:, :-2].reshape_as(center)
    target_plus = target[:, 2:].reshape_as(center)
    target_time_derivative = (target_plus - target_minus) / (2.0 * delta_t)
    target_chemical_potential = reaction_r * (target_center.pow(3) - target_center) - eps_r.square() * laplacian(target_center)
    target_rhs = mobility_r * laplacian(target_chemical_potential)
    target_residual = target_time_derivative - target_rhs
    normalizer = target_time_derivative.flatten(1).square().mean(1).sqrt().clamp_min(1e-4)
    residual_relative = residual.flatten(1).square().mean(1).sqrt() / normalizer
    target_residual_relative = target_residual.flatten(1).square().mean(1).sqrt() / normalizer
    return {
        "trajectory_relative_l2_mean": float(relative.mean()),
        "terminal_relative_l2_mean": float(relative[:, -1].mean()),
        "trajectory_mass_drift_mae": float(mass_drift.mean()),
        "maximum_mass_drift_mean": float(mass_drift.max(dim=1).values.mean()),
        "terminal_free_energy_relative_error_mean": float(energy_error.mean()),
        "terminal_structure_factor_centroid_relative_error_mean": float(centroid_error.mean()),
        "trajectory_pde_residual_relative_rms": float(residual_relative.mean()),
        "reference_discretization_residual_relative_rms": float(target_residual_relative.mean()),
        "trajectory_relative_l2_per_time": [float(x) for x in relative.mean(0)],
        "mass_drift_per_time": [float(x) for x in mass_drift.mean(0)],
    }


def evaluate(model, data, start, count, batch_size, rank=None):
    device = next(model.parameters()).device
    predicted = predict(model, data, start, count, batch_size, rank)
    target = torch.from_numpy(data["fields"][start:start + count]).to(device).unsqueeze(2)
    initial = torch.from_numpy(data["initial"][start:start + count]).to(device).unsqueeze(1)
    epsilon = torch.from_numpy(data["epsilon"][start:start + count]).to(device)
    reaction = torch.from_numpy(data["reaction"][start:start + count]).to(device)
    mobility = torch.from_numpy(data["mobility"][start:start + count]).to(device)
    return predicted, summarize(predicted, target, initial, epsilon, reaction, mobility, float(data["times"][1] - data["times"][0]))


def build(kind, device):
    args = SimpleNamespace(
        poseidon_code=Path("/tmp/poseidon-stnlr"),
        poseidon_checkpoint=Path("/tmp/poseidon_model"),
        kind=kind,
        nested_schedule="u_shaped",
        nested_time_direction="decay",
        u_mid_rank=8,
    )
    return build_poseidon(args, device)


def load_static_checkpoint(model, path):
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model"], strict=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["static", "nested"], required=True)
    parser.add_argument("--static-checkpoint", type=Path)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--energy-weight", type=float, default=0.02)
    parser.add_argument("--mass-weight", type=float, default=2.0)
    parser.add_argument("--structure-weight", type=float, default=0.01)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument(
        "--distill-target",
        choices=["frozen_teacher", "current_rank16"],
        default="frozen_teacher",
    )
    parser.add_argument("--spectral-weight", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-count", type=int, default=100)
    parser.add_argument("--val-start", type=int, default=100)
    parser.add_argument("--val-count", type=int, default=20)
    parser.add_argument("--test-start", type=int, default=120)
    parser.add_argument("--test-count", type=int, default=40)
    args = parser.parse_args()
    if args.mode == "nested" and args.static_checkpoint is None:
        parser.error("--static-checkpoint is required for nested mode")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data = load_data(args.data)

    teacher = None
    if args.mode == "static":
        model, replaced = build("static", device)
        if args.static_checkpoint is not None:
            load_static_checkpoint(model, args.static_checkpoint)
    else:
        model, replaced = build("stnlr", device)
        load_static_checkpoint(model, args.static_checkpoint)
        if args.distill_target == "frozen_teacher":
            teacher, _ = build("static", device)
            load_static_checkpoint(teacher, args.static_checkpoint)
            teacher.eval()
            for parameter in teacher.parameters():
                parameter.requires_grad = False

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    # ``steps=0`` is used by the strict-trace re-evaluation utility to load a
    # trained nested checkpoint and recompute validation/test metrics without
    # changing weights.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.steps))
    generator = torch.Generator(device=device).manual_seed(args.seed + 2381)
    rank_weights = {16: 1.0, 8: 0.5, 4: 0.25}
    start_time = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        initial, target, lead, epsilon, reaction = training_batch(data, args.train_count, args.batch_size, device, generator)
        teacher_prediction = None
        if teacher is not None:
            with torch.no_grad():
                teacher_prediction = teacher(pixel_values=initial, time=lead).output
        optimizer.zero_grad(set_to_none=True)
        ranks = (16, 8, 4) if args.mode == "nested" else (16,)
        logged = {}
        current_rank16_prediction = None
        for rank in ranks:
            if args.mode == "nested":
                force_rank(model, rank)
                set_nested_time(model, lead, "decay")
            prediction = model(pixel_values=initial, time=lead).output
            field = (prediction - target).square().mean()
            energy, mass, structure = physics_losses(prediction, target, epsilon, reaction)
            loss = field + args.energy_weight * energy + args.mass_weight * mass + args.structure_weight * structure
            distill_target = teacher_prediction
            if args.mode == "nested" and args.distill_target == "current_rank16":
                if rank == 16:
                    current_rank16_prediction = prediction.detach()
                    distill_target = None
                else:
                    distill_target = current_rank16_prediction
            if distill_target is not None:
                loss = loss + args.distill_weight * (prediction - distill_target).square().mean()
                loss = loss + args.spectral_weight * spectral_distillation(prediction, distill_target)
            (rank_weights[rank] * loss).backward()
            logged[rank] = float(loss.detach())
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()
        if step % 100 == 0:
            print(f"step={step:04d} " + " ".join(f"rank{rank}={value:.6f}" for rank, value in logged.items()) + f" steps_per_sec={step/(time.time()-start_time):.2f}", flush=True)

    if args.mode == "static":
        _, test_metrics = evaluate(model, data, args.test_start, args.test_count, args.batch_size)
        result = {"mode": args.mode, "test_metrics": test_metrics}
        rank_trace = None
    else:
        validation_predictions = {}
        validation_metrics = {}
        validation_target = torch.from_numpy(data["fields"][args.val_start:args.val_start + args.val_count]).to(device).unsqueeze(2)
        validation_initial = torch.from_numpy(data["initial"][args.val_start:args.val_start + args.val_count]).to(device).unsqueeze(1)
        eps_val = torch.from_numpy(data["epsilon"][args.val_start:args.val_start + args.val_count]).to(device)
        reaction_val = torch.from_numpy(data["reaction"][args.val_start:args.val_start + args.val_count]).to(device)
        mobility_val = torch.from_numpy(data["mobility"][args.val_start:args.val_start + args.val_count]).to(device)
        for rank in (4, 8, 16):
            validation_predictions[rank] = predict(model, data, args.val_start, args.val_count, args.batch_size, rank)
            validation_metrics[rank] = summarize(validation_predictions[rank], validation_target, validation_initial, eps_val, reaction_val, mobility_val, float(data["times"][1]-data["times"][0]))
        rank_trace = []
        reference_rel = torch.linalg.vector_norm((validation_predictions[16] - validation_target).flatten(2), dim=2).mean(0)
        for time_id in range(validation_target.shape[1]):
            chosen = 16
            target_energy = free_energy(validation_target[:, time_id], eps_val, reaction_val)
            target_centroid = spectral_centroid(validation_target[:, time_id])
            reference_energy = (
                (free_energy(validation_predictions[16][:, time_id], eps_val, reaction_val) - target_energy).abs()
                / target_energy.abs().clamp_min(1e-8)
            ).mean()
            reference_centroid = (
                (spectral_centroid(validation_predictions[16][:, time_id]) - target_centroid).abs()
                / target_centroid.clamp_min(1e-8)
            ).mean()
            for rank in (4, 8, 16):
                candidate_rel = torch.linalg.vector_norm((validation_predictions[rank][:, time_id] - validation_target[:, time_id]).flatten(1), dim=1).mean()
                candidate_mass = (validation_predictions[rank][:, time_id].mean((-3,-2,-1)) - validation_initial.mean((-3,-2,-1))).abs().mean()
                ref_mass = (validation_predictions[16][:, time_id].mean((-3,-2,-1)) - validation_initial.mean((-3,-2,-1))).abs().mean()
                candidate_energy = (
                    (free_energy(validation_predictions[rank][:, time_id], eps_val, reaction_val) - target_energy).abs()
                    / target_energy.abs().clamp_min(1e-8)
                ).mean()
                candidate_centroid = (
                    (spectral_centroid(validation_predictions[rank][:, time_id]) - target_centroid).abs()
                    / target_centroid.clamp_min(1e-8)
                ).mean()
                if (
                    candidate_rel <= 1.02 * reference_rel[time_id] + 1e-4
                    and candidate_mass <= 1.05 * ref_mass + 1e-5
                    and candidate_energy <= 1.05 * reference_energy + 5e-4
                    and candidate_centroid <= 1.05 * reference_centroid + 5e-4
                ):
                    chosen = rank
                    break
            rank_trace.append(chosen)
        test_by_rank = {rank: predict(model, data, args.test_start, args.test_count, args.batch_size, rank) for rank in (4, 8, 16)}
        calibrated = torch.stack([test_by_rank[rank][:, time_id] for time_id, rank in enumerate(rank_trace)], dim=1)
        target = torch.from_numpy(data["fields"][args.test_start:args.test_start + args.test_count]).to(device).unsqueeze(2)
        initial = torch.from_numpy(data["initial"][args.test_start:args.test_start + args.test_count]).to(device).unsqueeze(1)
        eps_test = torch.from_numpy(data["epsilon"][args.test_start:args.test_start + args.test_count]).to(device)
        reaction_test = torch.from_numpy(data["reaction"][args.test_start:args.test_start + args.test_count]).to(device)
        mobility_test = torch.from_numpy(data["mobility"][args.test_start:args.test_start + args.test_count]).to(device)
        delta_t = float(data["times"][1] - data["times"][0])
        result = {
            "mode": args.mode,
            "validation_selected_rank_trace": rank_trace,
            "mean_active_rank": float(np.mean(rank_trace)),
            "validation_per_rank": validation_metrics,
            "test_per_rank": {
                str(rank): summarize(
                    test_by_rank[rank], target, initial, eps_test, reaction_test,
                    mobility_test, delta_t,
                )
                for rank in (4, 8, 16)
            },
            "test_rank16_metrics": summarize(test_by_rank[16], target, initial, eps_test, reaction_test, mobility_test, delta_t),
            "test_calibrated_metrics": summarize(calibrated, target, initial, eps_test, reaction_test, mobility_test, delta_t),
        }

    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    state = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name in trainable_names
    }
    torch.save({"model": state, "kind": args.mode, "rank_trace": rank_trace, "args": vars(args)}, args.out_dir / "final.pt")
    result.update({
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "replaced_linear_layers": len(replaced),
        "elapsed_seconds": time.time() - start_time,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    })
    (args.out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
