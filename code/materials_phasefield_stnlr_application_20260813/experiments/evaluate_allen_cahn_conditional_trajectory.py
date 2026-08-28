#!/usr/bin/env python3
"""Evaluate conditional Allen--Cahn trajectory generators."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

torch.backends.cudnn.enabled = False

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_allen_cahn_conditional_trajectory import (  # noqa: E402
    build_model,
    condition_input,
    laplacian_periodic,
)


@torch.no_grad()
def generate(model, initial, epsilon, reaction, physical_time, noise, steps):
    x = noise.clone()
    y = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    h = -1.0 / steps
    for node in range(steps):
        s = 1.0 - node / steps
        s_next = max(0.0, s + h)
        t = torch.full((x.shape[0],), int(round(s * 999)), dtype=torch.long, device=x.device)
        t_next = torch.full((x.shape[0],), int(round(s_next * 999)), dtype=torch.long, device=x.device)
        v1 = model(condition_input(x, initial, epsilon, reaction, physical_time), t, y)[:, :1]
        euler = x + h * v1
        v2 = model(condition_input(euler, initial, epsilon, reaction, physical_time), t_next, y)[:, :1]
        x = x + 0.5 * h * (v1 + v2)
    return x


def free_energy(field, epsilon, reaction):
    n = field.shape[-1]
    dx = 1.0 / n
    gx = (torch.roll(field, -1, -1) - torch.roll(field, 1, -1)) / (2.0 * dx)
    gy = (torch.roll(field, -1, -2) - torch.roll(field, 1, -2)) / (2.0 * dx)
    eps = epsilon.view(-1, 1, 1, 1)
    lam = reaction.view(-1, 1, 1, 1)
    density = 0.5 * eps.square() * (gx.square() + gy.square()) + 0.25 * lam * (field.square() - 1.0).square()
    return density.flatten(1).mean(1)


def summarize(predicted, target, epsilon, reaction, delta_t):
    eps = 1.0e-8
    error = (predicted - target).flatten(2)
    target_flat = target.flatten(2)
    relative = torch.linalg.vector_norm(error, dim=2) / torch.linalg.vector_norm(target_flat, dim=2).clamp_min(eps)
    terminal_relative = relative[:, -1]
    interface_pred = (predicted.abs() < 0.2).float().mean(dim=(-3, -2, -1))
    interface_target = (target.abs() < 0.2).float().mean(dim=(-3, -2, -1))
    terminal_interface = (interface_pred[:, -1] - interface_target[:, -1]).abs()
    terminal_energy_pred = free_energy(predicted[:, -1], epsilon, reaction)
    terminal_energy_target = free_energy(target[:, -1], epsilon, reaction)
    terminal_energy_error = (terminal_energy_pred - terminal_energy_target).abs() / terminal_energy_target.abs().clamp_min(eps)

    center = predicted[:, 1:-1].reshape(-1, 1, 64, 64)
    minus = predicted[:, :-2].reshape(-1, 1, 64, 64)
    plus = predicted[:, 2:].reshape(-1, 1, 64, 64)
    repeated_epsilon = epsilon[:, None].expand(-1, predicted.shape[1] - 2).reshape(-1)
    repeated_reaction = reaction[:, None].expand(-1, predicted.shape[1] - 2).reshape(-1)
    time_derivative = (plus - minus) / (2.0 * delta_t)
    rhs = repeated_epsilon.view(-1, 1, 1, 1).square() * laplacian_periodic(center) + repeated_reaction.view(-1, 1, 1, 1) * (center - center.pow(3))
    residual = time_derivative - rhs
    residual_relative = residual.flatten(1).square().mean(1).sqrt() / time_derivative.flatten(1).square().mean(1).sqrt().clamp_min(1.0e-4)
    return {
        "trajectory_relative_l2_mean": float(relative.mean()),
        "terminal_relative_l2_mean": float(terminal_relative.mean()),
        "terminal_interface_fraction_mae": float(terminal_interface.mean()),
        "terminal_free_energy_relative_error_mean": float(terminal_energy_error.mean()),
        "trajectory_pde_residual_relative_rms": float(residual_relative.mean()),
        "trajectory_relative_l2_per_time": [float(x) for x in relative.mean(0)],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--kind", choices=["base", "static", "stnlr"], required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", type=int, default=120)
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Do not store full predicted and target trajectories.",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    payload = np.load(args.data)
    selection = slice(args.start, args.start + args.count)
    target = torch.from_numpy(payload["fields"][selection]).to(device).unsqueeze(2)
    initial = torch.from_numpy(payload["initial"][selection]).to(device).unsqueeze(1)
    epsilon = torch.from_numpy(payload["epsilon"][selection]).to(device)
    reaction = torch.from_numpy(payload["reaction"][selection]).to(device)
    times = torch.from_numpy(payload["times"]).to(device)
    model = build_model(args.kind).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = checkpoint["model"]
    if checkpoint.get("adapter_only", False):
        base_path = checkpoint.get("base_checkpoint")
        if not base_path:
            raise RuntimeError("Adapter-only checkpoint does not identify its frozen base checkpoint.")
        base_state = torch.load(base_path, map_location="cpu")["model"]
        _, base_unexpected = model.load_state_dict(base_state, strict=False)
        if base_unexpected:
            raise RuntimeError(f"base checkpoint mismatch unexpected={base_unexpected[:5]}")
        expected_adapter_names = {
            name
            for name in model.state_dict()
            if "lora_A" in name or "lora_B" in name
        }
        missing_adapter = sorted(expected_adapter_names.difference(state))
        _, unexpected = model.load_state_dict(state, strict=False)
        if missing_adapter or unexpected:
            raise RuntimeError(
                f"adapter checkpoint mismatch missing={missing_adapter[:5]} "
                f"unexpected={unexpected[:5]}"
            )
    else:
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"checkpoint mismatch missing={missing[:5]} unexpected={unexpected[:5]}"
            )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    predictions = []
    for begin in range(0, args.count, args.batch_size):
        end = min(begin + args.batch_size, args.count)
        init_batch = initial[begin:end]
        eps_batch = epsilon[begin:end]
        reaction_batch = reaction[begin:end]
        shared_noise = torch.randn(
            (end - begin, 1, 64, 64), generator=generator, device=device
        )
        trajectory = []
        for physical_time in times:
            time_batch = physical_time.expand(end - begin)
            trajectory.append(
                generate(
                    model, init_batch, eps_batch, reaction_batch, time_batch,
                    shared_noise, args.steps
                )
            )
        predictions.append(torch.stack(trajectory, dim=1))
    predicted = torch.cat(predictions, dim=0)
    metrics = summarize(
        predicted,
        target,
        epsilon,
        reaction,
        float(times[1] - times[0]),
    )
    result = {"name": args.name, "metrics": metrics, "count": args.count, "steps": args.steps}
    (args.out / "metrics.json").write_text(json.dumps(result, indent=2))
    if not args.metrics_only:
        np.savez_compressed(
            args.out / "trajectories.npz",
            predicted=predicted.cpu().numpy(),
            target=target.cpu().numpy(),
            times=times.cpu().numpy(),
        )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
