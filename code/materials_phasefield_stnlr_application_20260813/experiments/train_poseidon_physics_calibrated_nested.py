#!/usr/bin/env python3
"""Physics-calibrated nested-prefix pilot with static-teacher distillation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from evaluate_allen_cahn_conditional_trajectory import free_energy, summarize
from train_allen_cahn_conditional_trajectory import material_structure_loss
from train_evaluate_poseidon_allen_cahn import (
    PoseidonNestedLinear,
    build_poseidon,
    load_data,
    set_nested_time,
    training_batch,
)


def force_rank(model, rank):
    for module in model.modules():
        if isinstance(module, PoseidonNestedLinear):
            module.forced_rank = rank


def load_teacher_and_student(checkpoint, device):
    payload = torch.load(checkpoint, map_location="cpu")
    common = dict(
        poseidon_code=Path("/tmp/poseidon-stnlr"),
        poseidon_checkpoint=Path("/tmp/poseidon_model"),
        u_mid_rank=8,
    )
    teacher_args = SimpleNamespace(kind="static", nested_schedule="early_high", **common)
    student_args = SimpleNamespace(kind="stnlr", nested_schedule="u_shaped", **common)
    teacher, _ = build_poseidon(teacher_args, device)
    student, _ = build_poseidon(student_args, device)
    teacher.load_state_dict(payload["model"], strict=False)
    # Static and nested rank-16 adapters share factor shapes, so this provides
    # an exact rank-16 initialization before low-prefix calibration.
    student.load_state_dict(payload["model"], strict=False)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    return teacher, student


def spectral_distillation(student, teacher, floor=0.02):
    delta = torch.fft.rfft2((student - teacher).float(), dim=(-2, -1))
    reference = torch.fft.rfft2(teacher.float(), dim=(-2, -1)).abs().square()
    scale = reference + floor * reference.mean(dim=(-2, -1), keepdim=True)
    return (delta.abs().square() / scale.clamp_min(1e-8)).mean()


@torch.no_grad()
def predict_at_rank(model, data, start, count, batch_size, rank):
    device = next(model.parameters()).device
    initials = torch.from_numpy(data["initial"][start:start + count]).to(device).unsqueeze(1)
    times = torch.from_numpy(data["times"]).to(device)
    force_rank(model, rank)
    predictions = []
    model.eval()
    for begin in range(0, count, batch_size):
        end = min(count, begin + batch_size)
        rows = []
        for physical_time in times:
            lead = (physical_time / times[-1]).expand(end - begin)
            set_nested_time(model, lead, "decay")
            rows.append(model(pixel_values=initials[begin:end], time=lead).output)
        predictions.append(torch.stack(rows, dim=1))
    return torch.cat(predictions), times


def per_time_metrics(predicted, target, epsilon, reaction):
    relative = torch.linalg.vector_norm((predicted - target).flatten(2), dim=2) / torch.linalg.vector_norm(target.flatten(2), dim=2).clamp_min(1e-8)
    interface = ((predicted.abs() < 0.2).float().mean(dim=(-3, -2, -1)) - (target.abs() < 0.2).float().mean(dim=(-3, -2, -1))).abs()
    energy = []
    for time_id in range(predicted.shape[1]):
        pred_e = free_energy(predicted[:, time_id], epsilon, reaction)
        true_e = free_energy(target[:, time_id], epsilon, reaction)
        energy.append(((pred_e - true_e).abs() / true_e.abs().clamp_min(1e-8)).mean())
    return {
        "relative_l2": relative.mean(0),
        "interface_mae": interface.mean(0),
        "energy_relative_error": torch.stack(energy),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--static-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument(
        "--distill-target",
        choices=("frozen_teacher", "current_rank16"),
        default="frozen_teacher",
    )
    parser.add_argument("--spectral-weight", type=float, default=1e-3)
    parser.add_argument("--energy-weight", type=float, default=0.04)
    parser.add_argument("--interface-weight", type=float, default=0.02)
    parser.add_argument("--interface-temperature", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-count", type=int, default=100)
    parser.add_argument("--eval-start", type=int, default=100)
    parser.add_argument("--eval-count", type=int, default=20)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data = load_data(args.data)
    teacher, student = load_teacher_and_student(args.static_checkpoint, device)
    trainable = [parameter for parameter in student.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    generator = torch.Generator(device=device).manual_seed(args.seed + 9713)
    student.train()
    weights = {16: 1.0, 8: 0.5, 4: 0.25}
    torch.cuda.synchronize(device)
    training_started = time.perf_counter()
    for step in range(1, args.steps + 1):
        initial, target, lead, epsilon, reaction = training_batch(
            data, args.train_count, args.batch_size, device, generator
        )
        teacher_output = None
        if args.distill_target == "frozen_teacher":
            with torch.no_grad():
                teacher_output = teacher(pixel_values=initial, time=lead).output
        optimizer.zero_grad(set_to_none=True)
        logged = {}
        current_rank16_prediction = None
        for rank in (16, 8, 4):
            force_rank(student, rank)
            prediction = student(pixel_values=initial, time=lead).output
            field = (prediction - target).square().mean()
            energy, interface = material_structure_loss(
                prediction, target, epsilon, reaction, args.interface_temperature
            )
            distill_target = teacher_output
            if args.distill_target == "current_rank16":
                if rank == 16:
                    current_rank16_prediction = prediction.detach()
                    distill_target = None
                else:
                    distill_target = current_rank16_prediction
            if distill_target is None:
                distill = prediction.new_zeros(())
                spectral = prediction.new_zeros(())
            else:
                distill = (prediction - distill_target).square().mean()
                spectral = spectral_distillation(prediction, distill_target)
            loss = (
                field + args.energy_weight * energy + args.interface_weight * interface
                + args.distill_weight * distill + args.spectral_weight * spectral
            )
            (weights[rank] * loss).backward()
            logged[rank] = float(loss.detach())
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()
        if step % 100 == 0:
            print(f"step={step} rank16={logged[16]:.6f} rank8={logged[8]:.6f} rank4={logged[4]:.6f}", flush=True)
    torch.cuda.synchronize(device)
    training_elapsed_seconds = time.perf_counter() - training_started

    start, count = args.eval_start, args.eval_count
    fields = torch.from_numpy(data["fields"][start:start + count]).to(device).unsqueeze(2)
    epsilon = torch.from_numpy(data["epsilon"][start:start + count]).to(device)
    reaction = torch.from_numpy(data["reaction"][start:start + count]).to(device)
    predictions, diagnostics = {}, {}
    for rank in (4, 8, 16):
        predictions[rank], times = predict_at_rank(student, data, start, count, args.batch_size, rank)
        diagnostics[rank] = per_time_metrics(predictions[rank], fields, epsilon, reaction)
    teacher_prediction, _ = predict_at_rank(teacher, data, start, count, args.batch_size, 16)
    reference = diagnostics[16]
    trace = []
    for time_id in range(len(times)):
        selected = 16
        for rank in (4, 8, 16):
            row = diagnostics[rank]
            if (
                row["relative_l2"][time_id] <= 1.02 * reference["relative_l2"][time_id] + 1e-4
                and row["interface_mae"][time_id] <= 1.02 * reference["interface_mae"][time_id] + 1e-4
                and row["energy_relative_error"][time_id] <= 1.02 * reference["energy_relative_error"][time_id] + 2e-4
            ):
                selected = rank
                break
        trace.append(selected)
    calibrated = torch.stack([predictions[rank][:, time_id] for time_id, rank in enumerate(trace)], dim=1)
    delta_t = float(times[1] - times[0])
    result = {
        "trainable_parameters": sum(p.numel() for p in trainable),
        "validation_selected_rank_trace": trace,
        "mean_active_rank": float(np.mean(trace)),
        "training_elapsed_seconds": training_elapsed_seconds,
        "training_seconds_per_step": training_elapsed_seconds / max(1, args.steps),
        "static_teacher_metrics": summarize(teacher_prediction, fields, epsilon, reaction, delta_t),
        "rank16_metrics": summarize(predictions[16], fields, epsilon, reaction, delta_t),
        "calibrated_metrics": summarize(calibrated, fields, epsilon, reaction, delta_t),
        "per_rank_per_time": {
            str(rank): {key: [float(x) for x in values] for key, values in diagnostics[rank].items()}
            for rank in (4, 8, 16)
        },
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    torch.save({
        "model": {name: value.detach().cpu() for name, value in student.state_dict().items() if "lora_" in name or name.startswith("embeddings.patch_embeddings.") or name.startswith("patch_recovery.")},
        "rank_trace": trace,
        "args": vars(args),
    }, args.out_dir / "final.pt")
    print(json.dumps({key: value for key, value in result.items() if key != "per_rank_per_time"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
