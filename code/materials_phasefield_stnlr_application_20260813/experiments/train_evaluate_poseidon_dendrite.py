#!/usr/bin/env python3
"""Matched Poseidon-T adapter pilot for PFHub-3-type dendrite trajectories.

The frozen multi-PDE backbone receives the initial phase/temperature pair and
the normalized physical lead time.  Static continuation and the nested bank
start from the same trained rank-16 checkpoint and receive the same additional
optimization budget.  Validation data select the smallest admissible prefix at
each output time; test data are used once for the final comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from train_evaluate_poseidon_allen_cahn import (  # noqa: E402
    PoseidonNestedLinear,
    build_poseidon,
    load_data,
    set_nested_time,
)


def model_args(kind: str) -> SimpleNamespace:
    return SimpleNamespace(
        poseidon_code=Path("/tmp/poseidon-stnlr"),
        poseidon_checkpoint=Path("/tmp/poseidon_model"),
        kind=kind,
        nested_schedule="u_shaped",
        u_mid_rank=8,
        num_channels=2,
    )


def force_rank(model, rank: int | None) -> None:
    for module in model.modules():
        if isinstance(module, PoseidonNestedLinear):
            module.forced_rank = rank


def adapter_state(model) -> dict[str, torch.Tensor]:
    markers = ("lora_A", "lora_B", "embeddings.patch_embeddings.", "patch_recovery.")
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if any(marker in name for marker in markers)
    }


def load_from_checkpoint(kind: str, checkpoint: Path, device: torch.device):
    model, replaced = build_poseidon(model_args(kind), device)
    payload = torch.load(checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    relevant_missing = [name for name in missing if "lora_" in name]
    if relevant_missing or unexpected:
        raise RuntimeError(
            f"adapter checkpoint mismatch: missing={relevant_missing}, unexpected={unexpected}"
        )
    return model, replaced


def batch(
    data,
    train_count,
    batch_size,
    device,
    generator,
    terminal_sample_probability: float = 0.0,
):
    ids = torch.randint(0, train_count, (batch_size,), generator=generator, device=device)
    draw = float(torch.rand((), generator=generator, device=device))
    if draw < terminal_sample_probability:
        time_id = data["fields"].shape[1] - 1
    else:
        time_id = int(
            torch.randint(
                1,
                data["fields"].shape[1],
                (1,),
                generator=generator,
                device=device,
            )
        )
    host_ids = ids.cpu().numpy()
    initial = torch.from_numpy(data["initial"][host_ids]).to(device)
    target = torch.from_numpy(data["fields"][host_ids, time_id]).to(device)
    lead_value = float(data["times"][time_id] / data["times"][-1])
    lead = torch.full((batch_size,), lead_value, device=device)
    return initial, target, lead


def spatial_gradients(field: torch.Tensor, dx: float):
    # field [B, H, W]
    padded = F.pad(field[:, None], (1, 1, 1, 1), mode="replicate")[:, 0]
    gx = (padded[:, 1:-1, 2:] - padded[:, 1:-1, :-2]) / (2.0 * dx)
    gy = (padded[:, 2:, 1:-1] - padded[:, :-2, 1:-1]) / (2.0 * dx)
    return gx, gy


def free_energy(state: torch.Tensor, diffusion: float, epsilon4: float, dx: float):
    # state [B, 2, H, W]
    phi, temperature = state[:, 0], state[:, 1]
    gx, gy = spatial_gradients(phi, dx)
    theta = torch.atan2(gy, gx)
    width = 1.0 + epsilon4 * torch.cos(4.0 * theta)
    phi2 = phi.square()
    coupling = diffusion / 0.6267
    chemical = (
        -0.5 * phi2
        + 0.25 * phi2.square()
        + coupling
        * temperature
        * phi
        * (1.0 - (2.0 / 3.0) * phi2 + 0.2 * phi2.square())
    )
    return (0.5 * width.square() * (gx.square() + gy.square()) + chemical).mean((-2, -1))


def material_losses(
    prediction,
    target,
    diffusion: float,
    epsilon4: float,
    dx: float,
    lead: torch.Tensor,
    args,
):
    pred = prediction
    truth = target
    phi_loss = (pred[:, 0] - truth[:, 0]).square().mean()
    temperature_loss = ((pred[:, 1] - truth[:, 1]) / 0.3).square().mean()
    pred_fraction = ((pred[:, 0] + 1.0) * 0.5).mean((-2, -1))
    true_fraction = ((truth[:, 0] + 1.0) * 0.5).mean((-2, -1))
    fraction_error = pred_fraction - true_fraction
    if args.normalize_solid_loss:
        fraction_error = fraction_error / true_fraction.abs().clamp_min(
            args.solid_scale_floor
        )
    fraction_loss = fraction_error.square().mean()
    pred_axis_extent = 0.5 * (
        torch.sigmoid(pred[:, 0, 0, :] / 0.05).mean(-1)
        + torch.sigmoid(pred[:, 0, :, 0] / 0.05).mean(-1)
    )
    true_axis_extent = 0.5 * (
        torch.sigmoid(truth[:, 0, 0, :] / 0.05).mean(-1)
        + torch.sigmoid(truth[:, 0, :, 0] / 0.05).mean(-1)
    )
    tip_extent_loss = (pred_axis_extent - true_axis_extent).square().mean()
    pred_energy = free_energy(prediction, diffusion, epsilon4, dx)
    true_energy = free_energy(target, diffusion, epsilon4, dx)
    energy_loss = (
        (pred_energy - true_energy) / true_energy.abs().clamp_min(1.0e-3)
    ).square().mean()
    pred_gx, pred_gy = spatial_gradients(pred[:, 0], dx)
    true_gx, true_gy = spatial_gradients(truth[:, 0], dx)
    interface_weight = torch.exp(-truth[:, 0].square() / 0.08)
    pred_soft_solid = torch.sigmoid(pred[:, 0] / args.interface_level_temperature)
    true_soft_solid = torch.sigmoid(truth[:, 0] / args.interface_level_temperature)
    interface_level_loss = (
        (1.0 + 4.0 * interface_weight)
        * (pred_soft_solid - true_soft_solid).square()
    ).mean()
    gradient_loss = (
        interface_weight
        * (
            (torch.sqrt(pred_gx.square() + pred_gy.square() + 1.0e-8)
             - torch.sqrt(true_gx.square() + true_gy.square() + 1.0e-8)).square()
        )
    ).mean()
    material_multiplier = 1.0 + args.late_material_weight * float(lead.mean()) ** 2
    total = (
        phi_loss
        + args.temperature_weight * temperature_loss
        + material_multiplier
        * (
            args.solid_weight * fraction_loss
            + args.tip_weight * tip_extent_loss
            + args.energy_weight * energy_loss
        )
        + args.interface_level_weight * interface_level_loss
        + args.interface_weight * gradient_loss
    )
    return total, {
        "phi": phi_loss,
        "temperature": temperature_loss,
        "solid_fraction": fraction_loss,
        "tip_extent": tip_extent_loss,
        "energy": energy_loss,
        "interface_level": interface_level_loss,
        "interface_gradient": gradient_loss,
    }


def tip_position(phi: torch.Tensor) -> torch.Tensor:
    # phi [B,T,H,W], position normalized by stored grid size.
    solid_x = phi[:, :, 0, :] >= 0.5
    solid_y = phi[:, :, :, 0] >= 0.5
    coordinates = torch.arange(phi.shape[-1], device=phi.device, dtype=phi.dtype)
    x_tip = (solid_x * coordinates).amax(-1)
    y_tip = (solid_y * coordinates).amax(-1)
    return 0.5 * (x_tip + y_tip) / max(1, phi.shape[-1] - 1)


def terminal_interface_chamfer(pred_phi: torch.Tensor, true_phi: torch.Tensor) -> float:
    """Symmetric terminal zero-contour distance, normalized by image width."""
    values = []
    scale = float(max(pred_phi.shape[-2:]) - 1)
    for pred_row, true_row in zip(pred_phi, true_phi):
        pred_points = torch.nonzero(pred_row.abs() <= 0.15, as_tuple=False).float()
        true_points = torch.nonzero(true_row.abs() <= 0.15, as_tuple=False).float()
        if pred_points.numel() == 0 or true_points.numel() == 0:
            values.append(torch.tensor(1.0, device=pred_phi.device))
            continue
        # Bound the pairwise distance matrix without introducing randomness.
        if pred_points.shape[0] > 2048:
            ids = torch.linspace(0, pred_points.shape[0] - 1, 2048, device=pred_phi.device).long()
            pred_points = pred_points[ids]
        if true_points.shape[0] > 2048:
            ids = torch.linspace(0, true_points.shape[0] - 1, 2048, device=true_phi.device).long()
            true_points = true_points[ids]
        distances = torch.cdist(pred_points, true_points) / max(scale, 1.0)
        values.append(0.5 * (distances.min(1).values.mean() + distances.min(0).values.mean()))
    return float(torch.stack(values).mean())


def summarize(predicted, target, diffusion: float, epsilon4: float, dx: float):
    # Both tensors [B,T,2,H,W].
    pred = predicted
    truth = target
    phase_rel = torch.linalg.vector_norm((pred[:, :, 0] - truth[:, :, 0]).flatten(2), dim=2) / torch.linalg.vector_norm(truth[:, :, 0].flatten(2), dim=2).clamp_min(1e-8)
    temp_rel = torch.linalg.vector_norm((pred[:, :, 1] - truth[:, :, 1]).flatten(2), dim=2) / torch.linalg.vector_norm(truth[:, :, 1].flatten(2), dim=2).clamp_min(1e-8)
    solid_pred = ((pred[:, :, 0] + 1.0) * 0.5).mean((-2, -1))
    solid_true = ((truth[:, :, 0] + 1.0) * 0.5).mean((-2, -1))
    energy_errors = []
    for time_id in range(pred.shape[1]):
        pred_e = free_energy(pred[:, time_id], diffusion, epsilon4, dx)
        true_e = free_energy(truth[:, time_id], diffusion, epsilon4, dx)
        energy_errors.append((pred_e - true_e).abs() / true_e.abs().clamp_min(1e-3))
    energy_error = torch.stack(energy_errors, dim=1)
    pred_tip = tip_position(pred[:, :, 0])
    true_tip = tip_position(truth[:, :, 0])
    tip_error = (pred_tip - true_tip).abs()
    return {
        "phase_trajectory_relative_l2": float(phase_rel.mean()),
        "phase_terminal_relative_l2": float(phase_rel[:, -1].mean()),
        "temperature_trajectory_relative_l2": float(temp_rel.mean()),
        "temperature_terminal_relative_l2": float(temp_rel[:, -1].mean()),
        "solid_fraction_mae": float((solid_pred - solid_true).abs().mean()),
        "free_energy_relative_error": float(energy_error.mean()),
        "tip_position_mae": float(tip_error.mean()),
        "tip_terminal_mae": float(tip_error[:, -1].mean()),
        "terminal_interface_chamfer": terminal_interface_chamfer(
            pred[:, -1, 0], truth[:, -1, 0]
        ),
    }


def pfhub_observable_curves(
    predicted: torch.Tensor,
    target: torch.Tensor,
    times: np.ndarray,
    diffusion: float,
    epsilon4: float,
    dx: float,
) -> dict:
    """Export PFHub scalar curves and the terminal contour for the first test row."""
    pred = predicted[:1]
    truth = target[:1]
    pred_solid = ((pred[:, :, 0] + 1.0) * 0.5).mean((-2, -1))[0]
    true_solid = ((truth[:, :, 0] + 1.0) * 0.5).mean((-2, -1))[0]
    pred_energy, true_energy = [], []
    for time_id in range(pred.shape[1]):
        pred_energy.append(free_energy(pred[:, time_id], diffusion, epsilon4, dx)[0])
        true_energy.append(free_energy(truth[:, time_id], diffusion, epsilon4, dx)[0])
    pred_tip = tip_position(pred[:, :, 0])[0] * (pred.shape[-1] - 1) * dx
    true_tip = tip_position(truth[:, :, 0])[0] * (truth.shape[-1] - 1) * dx

    def contour(field: torch.Tensor) -> list[list[float]]:
        points = torch.nonzero(field.abs() <= 0.15, as_tuple=False).float()
        if points.shape[0] > 4096:
            ids = torch.linspace(0, points.shape[0] - 1, 4096, device=points.device).long()
            points = points[ids]
        return torch.stack((points[:, 1], points[:, 0]), dim=1).mul(dx).cpu().tolist()

    return {
        "time": [float(value) for value in times],
        "solid_fraction": {
            "prediction": pred_solid.cpu().tolist(),
            "reference": true_solid.cpu().tolist(),
        },
        "free_energy_density_mean": {
            "prediction": torch.stack(pred_energy).cpu().tolist(),
            "reference": torch.stack(true_energy).cpu().tolist(),
        },
        "tip_position": {
            "prediction": pred_tip.cpu().tolist(),
            "reference": true_tip.cpu().tolist(),
        },
        "phase_field_1500": {
            "prediction_xy": contour(pred[0, -1, 0]),
            "reference_xy": contour(truth[0, -1, 0]),
        },
    }


@torch.no_grad()
def predict(model, kind, data, start, count, batch_size, device, rank=None):
    initials = torch.from_numpy(data["initial"][start:start + count]).to(device)
    times = torch.from_numpy(data["times"]).to(device)
    if kind == "stnlr":
        force_rank(model, rank)
    outputs = []
    model.eval()
    for begin in range(0, count, batch_size):
        end = min(count, begin + batch_size)
        rows = []
        for physical_time in times:
            lead = (physical_time / times[-1]).expand(end - begin)
            if kind == "stnlr":
                set_nested_time(model, lead, "decay")
            rows.append(model(pixel_values=initials[begin:end], time=lead).output)
        outputs.append(torch.stack(rows, dim=1))
    return torch.cat(outputs, dim=0)


def per_sample_time_metrics(predicted, target, diffusion: float, epsilon4: float, dx: float):
    pred = predicted
    truth = target
    phase = torch.linalg.vector_norm((pred[:, :, 0] - truth[:, :, 0]).flatten(2), dim=2) / torch.linalg.vector_norm(truth[:, :, 0].flatten(2), dim=2).clamp_min(1e-8)
    temperature = torch.linalg.vector_norm((pred[:, :, 1] - truth[:, :, 1]).flatten(2), dim=2) / torch.linalg.vector_norm(truth[:, :, 1].flatten(2), dim=2).clamp_min(1e-8)
    solid = (((pred[:, :, 0] + 1.0) * 0.5).mean((-2, -1)) - ((truth[:, :, 0] + 1.0) * 0.5).mean((-2, -1))).abs()
    tips = (tip_position(pred[:, :, 0]) - tip_position(truth[:, :, 0])).abs()
    energies = []
    for time_id in range(pred.shape[1]):
        pred_e = free_energy(pred[:, time_id], diffusion, epsilon4, dx)
        true_e = free_energy(truth[:, time_id], diffusion, epsilon4, dx)
        energies.append((pred_e - true_e).abs() / true_e.abs().clamp_min(1e-3))
    return {
        "phase": phase,
        "temperature": temperature,
        "solid_fraction": solid,
        "free_energy": torch.stack(energies, 1),
        "tip_position": tips,
    }


def per_time_metrics(predicted, target, diffusion: float, epsilon4: float, dx: float):
    return {
        metric: values.mean(0)
        for metric, values in per_sample_time_metrics(
            predicted, target, diffusion, epsilon4, dx
        ).items()
    }


def global_budget_rank_trace(
    diagnostics: dict[int, dict[str, torch.Tensor]],
    field_tolerance: float,
    material_tolerance: float,
) -> tuple[list[int], dict[str, float], dict[str, float]]:
    """Select the lowest-cost validation-feasible nested-prefix trace.

    The search starts from the all-rank-16 reference and greedily accepts the
    least damaging one-step downgrade (16->8 or 8->4) while the *trajectory-
    averaged* validation errors remain inside fixed field/material budgets.
    Test predictions are not inspected by this routine.
    """
    metric_names = tuple(diagnostics[16])
    field_metrics = {"phase", "temperature"}
    reference = {
        metric: float(diagnostics[16][metric].mean()) for metric in metric_names
    }
    tolerances = {
        metric: field_tolerance if metric in field_metrics else material_tolerance
        for metric in metric_names
    }
    trace = [16] * len(diagnostics[16][metric_names[0]])

    def aggregate(candidate: list[int]) -> dict[str, float]:
        return {
            metric: float(
                torch.stack(
                    [diagnostics[rank][metric][time_id] for time_id, rank in enumerate(candidate)]
                ).mean()
            )
            for metric in metric_names
        }

    def feasible(values: dict[str, float]) -> bool:
        return all(
            values[metric] <= (1.0 + tolerances[metric]) * reference[metric]
            for metric in metric_names
        )

    while True:
        candidates = []
        for time_id, rank in enumerate(trace):
            lower_rank = {16: 8, 8: 4}.get(rank)
            if lower_rank is None:
                continue
            candidate = list(trace)
            candidate[time_id] = lower_rank
            values = aggregate(candidate)
            if not feasible(values):
                continue
            utilization = [
                values[metric]
                / max((1.0 + tolerances[metric]) * reference[metric], 1e-12)
                for metric in metric_names
            ]
            candidates.append(
                (max(utilization), sum(utilization), time_id, lower_rank, candidate, values)
            )
        if not candidates:
            break
        candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        trace = candidates[0][4]

    values = aggregate(trace)
    ratios = {
        metric: values[metric] / max(reference[metric], 1e-12)
        for metric in metric_names
    }
    return trace, values, ratios


def robust_trajectory_budget_rank_trace(
    diagnostics: dict[int, dict[str, torch.Tensor]],
    field_tolerance: float,
    energy_solid_tolerance: float,
    tip_tolerance: float,
) -> tuple[list[int], dict[str, list[float]], dict[str, list[float]]]:
    """Choose a trace that satisfies budgets for every validation trajectory.

    Each diagnostic tensor has shape [validation trajectory, physical time].
    The all-rank-16 trajectory is the reference. A downgrade is accepted only
    when every validation trajectory remains within every prescribed budget.
    """
    metric_names = tuple(diagnostics[16])
    tolerances = {
        "phase": field_tolerance,
        "temperature": field_tolerance,
        "solid_fraction": energy_solid_tolerance,
        "free_energy": energy_solid_tolerance,
        "tip_position": tip_tolerance,
    }
    reference = {
        metric: diagnostics[16][metric].mean(1) for metric in metric_names
    }
    trace = [16] * diagnostics[16][metric_names[0]].shape[1]

    def aggregate(candidate: list[int]) -> dict[str, torch.Tensor]:
        return {
            metric: torch.stack(
                [diagnostics[rank][metric][:, time_id] for time_id, rank in enumerate(candidate)],
                dim=1,
            ).mean(1)
            for metric in metric_names
        }

    def feasible(values: dict[str, torch.Tensor]) -> bool:
        return all(
            bool(torch.all(values[metric] <= (1.0 + tolerances[metric]) * reference[metric]))
            for metric in metric_names
        )

    while True:
        candidates = []
        for time_id, rank in enumerate(trace):
            lower_rank = {16: 8, 8: 4}.get(rank)
            if lower_rank is None:
                continue
            candidate = list(trace)
            candidate[time_id] = lower_rank
            values = aggregate(candidate)
            if not feasible(values):
                continue
            utilization = torch.cat(
                [
                    values[metric]
                    / ((1.0 + tolerances[metric]) * reference[metric]).clamp_min(1e-12)
                    for metric in metric_names
                ]
            )
            candidates.append(
                (
                    float(utilization.max()),
                    float(utilization.mean()),
                    time_id,
                    lower_rank,
                    candidate,
                    values,
                )
            )
        if not candidates:
            break
        candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        trace = candidates[0][4]

    values = aggregate(trace)
    value_lists = {
        metric: [float(value) for value in values[metric]] for metric in metric_names
    }
    ratio_lists = {
        metric: [
            float(value) for value in (values[metric] / reference[metric].clamp_min(1e-12))
        ]
        for metric in metric_names
    }
    return trace, value_lists, ratio_lists


def train(model, kind, data, args, device):
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if args.steps <= 0:
        return 0.0, sum(parameter.numel() for parameter in trainable)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    generator = torch.Generator(device=device).manual_seed(args.seed + 8831)
    weights = {
        16: 1.0,
        8: args.rank8_loss_weight,
        4: args.rank4_loss_weight,
    }
    started = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        initial, target, lead = batch(
            data,
            args.train_count,
            args.batch_size,
            device,
            generator,
            args.terminal_sample_probability,
        )
        optimizer.zero_grad(set_to_none=True)
        ranks = (16, 8, 4) if kind == "stnlr" else (16,)
        logged = {}
        rank16 = None
        for rank in ranks:
            if kind == "stnlr":
                force_rank(model, rank)
                set_nested_time(model, lead, "decay")
            prediction = model(pixel_values=initial, time=lead).output
            loss, parts = material_losses(
                prediction,
                target,
                args.diffusion,
                args.epsilon4,
                args.effective_dx,
                lead,
                args,
            )
            if kind == "stnlr" and rank < 16:
                loss = loss + args.distill_weight * (prediction - rank16).square().mean()
            if rank == 16:
                rank16 = prediction.detach()
            (weights[rank] * loss).backward()
            logged[rank] = float(loss.detach())
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()
        if step % args.log_every == 0:
            speed = step / (time.time() - started)
            details = " ".join(f"r{rank}={value:.5f}" for rank, value in logged.items())
            print(f"step={step}/{args.steps} {details} steps_per_sec={speed:.2f}", flush=True)
    return time.time() - started, sum(parameter.numel() for parameter in trainable)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--test-data",
        type=Path,
        help="Optional independent audit set; validation and training remain on --data.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--kind", choices=("static_initial", "static_continue", "stnlr"), required=True)
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--distill-weight", type=float, default=0.5)
    parser.add_argument("--temperature-weight", type=float, default=1.0)
    parser.add_argument("--terminal-sample-probability", type=float, default=0.0)
    parser.add_argument("--rank8-loss-weight", type=float, default=0.5)
    parser.add_argument("--rank4-loss-weight", type=float, default=0.25)
    parser.add_argument("--solid-weight", type=float, default=5.0)
    parser.add_argument("--normalize-solid-loss", action="store_true")
    parser.add_argument("--solid-scale-floor", type=float, default=0.005)
    parser.add_argument("--late-material-weight", type=float, default=0.0)
    parser.add_argument("--tip-weight", type=float, default=0.0)
    parser.add_argument("--energy-weight", type=float, default=0.03)
    parser.add_argument("--interface-weight", type=float, default=0.2)
    parser.add_argument("--interface-level-weight", type=float, default=0.0)
    parser.add_argument("--interface-level-temperature", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-count", type=int, default=20)
    parser.add_argument("--val-start", type=int, default=20)
    parser.add_argument("--val-count", type=int, default=4)
    parser.add_argument("--test-start", type=int, default=24)
    parser.add_argument("--test-count", type=int, default=8)
    parser.add_argument("--diffusion", type=float, required=True)
    parser.add_argument("--epsilon4", type=float, required=True)
    parser.add_argument("--effective-dx", type=float, default=1.0)
    parser.add_argument("--selection-tolerance", type=float, default=0.02)
    parser.add_argument(
        "--selection-policy",
        choices=("pointwise", "global_budget", "robust_trajectory_budget"),
        default="pointwise",
    )
    parser.add_argument("--selection-field-tolerance", type=float, default=0.02)
    parser.add_argument("--selection-material-tolerance", type=float, default=0.05)
    parser.add_argument("--selection-energy-solid-tolerance", type=float, default=0.03)
    parser.add_argument("--selection-tip-tolerance", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data = load_data(args.data)
    test_data = load_data(args.test_data) if args.test_data is not None else data
    if args.kind == "static_initial":
        model, replaced = build_poseidon(model_args("static"), device)
        model_kind = "static"
    else:
        if args.source_checkpoint is None:
            raise ValueError("--source-checkpoint is required for continuation/calibration")
        model_kind = "stnlr" if args.kind == "stnlr" else "static"
        model, replaced = load_from_checkpoint(model_kind, args.source_checkpoint, device)

    elapsed, trainable_count = train(model, model_kind, data, args, device)
    target_val = torch.from_numpy(data["fields"][args.val_start:args.val_start + args.val_count]).to(device)
    target_test = torch.from_numpy(
        test_data["fields"][args.test_start:args.test_start + args.test_count]
    ).to(device)
    result = {
        "kind": args.kind,
        "trainable_parameters": trainable_count,
        "replaced_linear_layers": len(replaced),
        "elapsed_seconds": elapsed,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    if model_kind == "static":
        test_prediction = predict(
            model, model_kind, test_data, args.test_start, args.test_count, args.batch_size, device
        )
        result["test_metrics"] = summarize(test_prediction, target_test, args.diffusion, args.epsilon4, args.effective_dx)
        result["pfhub_first_test_observables"] = pfhub_observable_curves(
            test_prediction, target_test, test_data["times"], args.diffusion,
            args.epsilon4, args.effective_dx
        )
    else:
        val_predictions = {}
        test_predictions = {}
        diagnostics = {}
        sample_diagnostics = {}
        for rank in (4, 8, 16):
            val_predictions[rank] = predict(model, model_kind, data, args.val_start, args.val_count, args.batch_size, device, rank)
            test_predictions[rank] = predict(
                model, model_kind, test_data, args.test_start, args.test_count,
                args.batch_size, device, rank
            )
            diagnostics[rank] = per_time_metrics(val_predictions[rank], target_val, args.diffusion, args.epsilon4, args.effective_dx)
            sample_diagnostics[rank] = per_sample_time_metrics(
                val_predictions[rank], target_val, args.diffusion,
                args.epsilon4, args.effective_dx
            )
        selection_summary = {}
        if args.selection_policy == "robust_trajectory_budget":
            trace, validation_values, validation_ratios = robust_trajectory_budget_rank_trace(
                sample_diagnostics,
                args.selection_field_tolerance,
                args.selection_energy_solid_tolerance,
                args.selection_tip_tolerance,
            )
            selection_summary = {
                "policy": "robust_trajectory_budget",
                "field_tolerance": args.selection_field_tolerance,
                "energy_solid_tolerance": args.selection_energy_solid_tolerance,
                "tip_tolerance": args.selection_tip_tolerance,
                "validation_per_trajectory_selected_metrics": validation_values,
                "validation_per_trajectory_selected_to_rank16_ratios": validation_ratios,
            }
        elif args.selection_policy == "global_budget":
            trace, validation_values, validation_ratios = global_budget_rank_trace(
                diagnostics,
                args.selection_field_tolerance,
                args.selection_material_tolerance,
            )
            selection_summary = {
                "policy": "global_budget",
                "field_tolerance": args.selection_field_tolerance,
                "material_tolerance": args.selection_material_tolerance,
                "validation_selected_metrics": validation_values,
                "validation_selected_to_rank16_ratios": validation_ratios,
            }
        else:
            trace = []
            reference = diagnostics[16]
            for time_id in range(target_val.shape[1]):
                selected = 16
                for rank in (4, 8, 16):
                    admissible = all(
                        diagnostics[rank][metric][time_id]
                        <= (1.0 + args.selection_tolerance) * reference[metric][time_id] + 1e-4
                        for metric in diagnostics[rank]
                    )
                    if admissible:
                        selected = rank
                        break
                trace.append(selected)
            selection_summary = {
                "policy": "pointwise",
                "tolerance": args.selection_tolerance,
            }
        calibrated = torch.stack(
            [test_predictions[rank][:, time_id] for time_id, rank in enumerate(trace)],
            dim=1,
        )
        result["validation_selected_rank_trace"] = trace
        result["mean_active_rank"] = float(np.mean(trace))
        result["selection_summary"] = selection_summary
        result["test_calibrated_metrics"] = summarize(calibrated, target_test, args.diffusion, args.epsilon4, args.effective_dx)
        result["pfhub_first_test_observables"] = pfhub_observable_curves(
            calibrated, target_test, test_data["times"], args.diffusion,
            args.epsilon4, args.effective_dx
        )
        result["test_fixed_prefix_metrics"] = {
            str(rank): summarize(test_predictions[rank], target_test, args.diffusion, args.epsilon4, args.effective_dx)
            for rank in (4, 8, 16)
        }
        result["validation_per_rank_per_time"] = {
            str(rank): {key: [float(value) for value in values] for key, values in diagnostics[rank].items()}
            for rank in (4, 8, 16)
        }

    torch.save(
        {
            "model": adapter_state(model),
            "kind": args.kind,
            "trainable_parameters": trainable_count,
            "args": vars(args),
        },
        args.out_dir / "final.pt",
    )
    (args.out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != "validation_per_rank_per_time"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
