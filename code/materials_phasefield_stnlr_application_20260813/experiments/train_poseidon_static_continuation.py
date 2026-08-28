#!/usr/bin/env python3
"""Continue a static rank-16 Poseidon adapter under the matched calibration budget."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from train_allen_cahn_conditional_trajectory import material_structure_loss
from train_evaluate_poseidon_allen_cahn import (
    build_poseidon,
    evaluate,
    load_data,
    training_batch,
)
from train_poseidon_physics_calibrated_nested import spectral_distillation


def load_static_pair(checkpoint: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu")
    common = dict(
        poseidon_code=Path("/tmp/poseidon-stnlr"),
        poseidon_checkpoint=Path("/tmp/poseidon_model"),
        kind="static",
        nested_schedule="early_high",
        nested_time_direction="decay",
        u_mid_rank=8,
    )
    teacher, replaced = build_poseidon(SimpleNamespace(**common), device)
    student, _ = build_poseidon(SimpleNamespace(**common), device)
    teacher.load_state_dict(payload["model"], strict=False)
    student.load_state_dict(payload["model"], strict=False)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    return teacher, student, replaced


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--static-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--spectral-weight", type=float, default=1.0e-3)
    parser.add_argument("--energy-weight", type=float, default=0.04)
    parser.add_argument("--interface-weight", type=float, default=0.02)
    parser.add_argument("--interface-temperature", type=float, default=0.04)
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
    teacher, student, replaced = load_static_pair(args.static_checkpoint, device)
    trainable = [parameter for parameter in student.parameters() if parameter.requires_grad]
    before = evaluate(
        teacher, "static", data, args.eval_start, args.eval_count,
        args.batch_size, device, "decay",
    )

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    generator = torch.Generator(device=device).manual_seed(args.seed + 9713)
    student.train()
    torch.cuda.synchronize(device)
    start_time = time.perf_counter()
    running = {"total": 0.0, "field": 0.0, "energy": 0.0, "interface": 0.0,
               "distill": 0.0, "spectral": 0.0}
    use_fixed_teacher = args.distill_weight > 0.0 or args.spectral_weight > 0.0

    for step in range(1, args.steps + 1):
        initial, target, lead, epsilon, reaction = training_batch(
            data, args.train_count, args.batch_size, device, generator
        )
        prediction = student(pixel_values=initial, time=lead).output
        field = (prediction - target).square().mean()
        energy, interface = material_structure_loss(
            prediction, target, epsilon, reaction, args.interface_temperature
        )
        if use_fixed_teacher:
            with torch.no_grad():
                teacher_output = teacher(pixel_values=initial, time=lead).output
            distill = (prediction - teacher_output).square().mean()
            spectral = spectral_distillation(prediction, teacher_output)
        else:
            distill = prediction.new_zeros(())
            spectral = prediction.new_zeros(())
        loss = (
            field
            + args.energy_weight * energy
            + args.interface_weight * interface
            + args.distill_weight * distill
            + args.spectral_weight * spectral
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()
        for key, value in {
            "total": loss, "field": field, "energy": energy,
            "interface": interface, "distill": distill, "spectral": spectral,
        }.items():
            running[key] += float(value.detach())
        if step % 100 == 0:
            summary = " ".join(f"{key}={value / 100:.6f}" for key, value in running.items())
            print(f"step={step:04d} {summary} steps_per_sec={step / (time.perf_counter() - start_time):.2f}", flush=True)
            running = {key: 0.0 for key in running}

    torch.cuda.synchronize(device)
    training_elapsed_seconds = time.perf_counter() - start_time

    after = evaluate(
        student, "static", data, args.eval_start, args.eval_count,
        args.batch_size, device, "decay",
    )
    trainable_state = {
        name: value.detach().cpu()
        for name, value in student.state_dict().items()
        if any(marker in name for marker in (
            "lora_A", "lora_B", "embeddings.patch_embeddings.", "patch_recovery."
        ))
    }
    torch.save(
        {
            "model": trainable_state,
            "kind": "static_continued_rank16",
            "source_checkpoint": str(args.static_checkpoint),
            "args": vars(args),
            "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        },
        args.out_dir / "final.pt",
    )
    result = {
        "kind": "static_continued_rank16",
        "source_checkpoint": str(args.static_checkpoint),
        "before_metrics": before,
        "after_metrics": after,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "replaced_linear_layers": len(replaced),
        "training_elapsed_seconds": training_elapsed_seconds,
        "training_seconds_per_step": training_elapsed_seconds / max(1, args.steps),
        "elapsed_seconds": training_elapsed_seconds,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
