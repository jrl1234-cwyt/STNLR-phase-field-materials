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


def model_args(static_rank: int) -> SimpleNamespace:
    return SimpleNamespace(
        poseidon_code=Path("/tmp/poseidon-stnlr"),
        poseidon_checkpoint=Path("/tmp/poseidon_model"),
        kind="static",
        static_rank=static_rank,
        nested_schedule="early_high",
        nested_time_direction="decay",
        u_mid_rank=8,
    )


def load_prefix_state(model, source_state: dict[str, torch.Tensor]) -> None:
    target_state = model.state_dict()
    converted: dict[str, torch.Tensor] = {}
    for name, source in source_state.items():
        if name not in target_state:
            continue
        target = target_state[name]
        if source.shape == target.shape:
            converted[name] = source
        elif name.endswith("lora_A") and source.ndim == 2:
            converted[name] = source[: target.shape[0], : target.shape[1]]
        elif name.endswith("lora_B") and source.ndim == 2:
            converted[name] = source[: target.shape[0], : target.shape[1]]
        else:
            raise RuntimeError(
                f"unsupported checkpoint conversion for {name}: "
                f"source={tuple(source.shape)} target={tuple(target.shape)}"
            )
    missing, unexpected = model.load_state_dict(converted, strict=False)
    relevant_missing = [
        name for name in missing
        if "lora_" in name
        or name.startswith("embeddings.patch_embeddings.")
        or name.startswith("patch_recovery.")
    ]
    if relevant_missing or unexpected:
        raise RuntimeError(
            f"rank-control checkpoint mismatch: missing={relevant_missing}, "
            f"unexpected={unexpected}"
        )


def load_pair(checkpoint: Path, static_rank: int, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu")
    teacher, _ = build_poseidon(model_args(16), device)
    student, replaced = build_poseidon(model_args(static_rank), device)
    teacher.load_state_dict(payload["model"], strict=False)
    load_prefix_state(student, payload["model"])
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    return teacher, student, replaced


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--static-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--static-rank", type=int, choices=(4, 7, 8), required=True)
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
    teacher, student, replaced = load_pair(
        args.static_checkpoint, args.static_rank, device
    )
    trainable = [p for p in student.parameters() if p.requires_grad]
    before = evaluate(
        student, "static", data, args.eval_start, args.eval_count,
        args.batch_size, device, "decay",
    )

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps
    )
    generator = torch.Generator(device=device).manual_seed(args.seed + 9713)
    student.train()
    started = time.time()
    running = {
        key: 0.0 for key in
        ("total", "field", "energy", "interface", "distill", "spectral")
    }
    for step in range(1, args.steps + 1):
        initial, target, lead, epsilon, reaction = training_batch(
            data, args.train_count, args.batch_size, device, generator
        )
        with torch.no_grad():
            teacher_output = teacher(pixel_values=initial, time=lead).output
        prediction = student(pixel_values=initial, time=lead).output
        field = (prediction - target).square().mean()
        energy, interface = material_structure_loss(
            prediction, target, epsilon, reaction, args.interface_temperature
        )
        distill = (prediction - teacher_output).square().mean()
        spectral = spectral_distillation(prediction, teacher_output)
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
        for name, value in {
            "total": loss,
            "field": field,
            "energy": energy,
            "interface": interface,
            "distill": distill,
            "spectral": spectral,
        }.items():
            running[name] += float(value.detach())
        if step % 100 == 0:
            summary = " ".join(
                f"{name}={value / 100:.6f}" for name, value in running.items()
            )
            print(
                f"rank={args.static_rank} step={step:04d} {summary} "
                f"steps_per_sec={step / (time.time() - started):.2f}",
                flush=True,
            )
            running = {name: 0.0 for name in running}

    after = evaluate(
        student, "static", data, args.eval_start, args.eval_count,
        args.batch_size, device, "decay",
    )
    trainable_state = {
        name: value.detach().cpu()
        for name, value in student.state_dict().items()
        if any(marker in name for marker in (
            "lora_A", "lora_B", "embeddings.patch_embeddings.",
            "patch_recovery.",
        ))
    }
    torch.save(
        {
            "model": trainable_state,
            "kind": f"static_rank{args.static_rank}_continued",
            "source_checkpoint": str(args.static_checkpoint),
            "args": vars(args),
            "trainable_parameters": sum(p.numel() for p in trainable),
        },
        args.out_dir / "final.pt",
    )
    result = {
        "kind": f"static_rank{args.static_rank}_continued",
        "source_checkpoint": str(args.static_checkpoint),
        "initial_prefix_metrics": before,
        "after_metrics": after,
        "trainable_parameters": sum(p.numel() for p in trainable),
        "replaced_linear_layers": len(replaced),
        "elapsed_seconds": time.time() - started,
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
