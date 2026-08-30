#!/usr/bin/env python3
"""Strict rank-flexible LoRA controls for the Poseidon Allen--Cahn task.

The controls preserve the defining training rules of DyLoRA and
MatryoshkaLoRA while matching the ST-NLR backbone, static rank-16 starting
point, material objectives, calibration updates, data split, and validation
tolerances.  DyLoRA samples ranks 1--16 and freezes the preceding prefix.
MatryoshkaLoRA uses the diagonal prefix aggregation in Algorithm 1 over
training ranks {4, 8, 16}.  Both controls may select rank 4, 8, or 16 at each
validation time before the rank trace is frozen for testing.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from materials_phasefield_stnlr_application_20260813.experiments.evaluate_allen_cahn_conditional_trajectory import (  # noqa: E402
    free_energy,
    summarize,
)
from materials_phasefield_stnlr_application_20260813.experiments.train_allen_cahn_conditional_trajectory import (  # noqa: E402
    material_structure_loss,
)
from materials_phasefield_stnlr_application_20260813.experiments.train_evaluate_poseidon_allen_cahn import (  # noqa: E402
    build_poseidon,
    initialize_scalar_heads_from_pretrained,
    is_adapter_target,
    load_data,
    training_batch,
)


class RankFlexibleLinear(nn.Module):
    """Frozen linear map plus a maximum rank-16 ordered LoRA bank."""

    def __init__(self, source: nn.Linear, rank: int = 16, alpha: float = 1.0):
        super().__init__()
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.weight = nn.Parameter(source.weight.detach().clone(), requires_grad=False)
        self.bias = (
            nn.Parameter(source.bias.detach().clone(), requires_grad=False)
            if source.bias is not None
            else None
        )
        self.lora_A = nn.Parameter(torch.empty(self.rank, source.in_features))
        self.lora_B = nn.Parameter(torch.zeros(source.out_features, self.rank))
        nn.init.normal_(self.lora_A, std=0.02)
        self.inference_rank = self.rank

    def prefix_update(
        self,
        x: torch.Tensor,
        rank: int,
        freeze_preceding: bool = False,
    ) -> torch.Tensor:
        rank = max(1, min(int(rank), self.rank))
        if freeze_preceding and rank > 1:
            a = torch.cat((self.lora_A[: rank - 1].detach(), self.lora_A[rank - 1 : rank]), dim=0)
            b = torch.cat((self.lora_B[:, : rank - 1].detach(), self.lora_B[:, rank - 1 : rank]), dim=1)
        else:
            a = self.lora_A[:rank]
            b = self.lora_B[:, :rank]
        return (self.alpha / rank) * F.linear(F.linear(x, a), b)


class StrictDyLoRALinear(RankFlexibleLinear):
    """Original DyLoRA full range with its frozen-prefix update rule."""

    def __init__(self, source: nn.Linear, rank: int = 16, alpha: float = 1.0):
        super().__init__(source, rank, alpha)
        self.training_rank = rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        if self.training:
            return base + self.prefix_update(x, self.training_rank, freeze_preceding=True)
        return base + self.prefix_update(x, self.inference_rank)


class MatryoshkaLoRALinear(RankFlexibleLinear):
    """Algorithm-1 diagonal aggregation for the prefixes {4, 8, 16}."""

    def __init__(
        self,
        source: nn.Linear,
        rank: int = 16,
        alpha: float = 1.0,
        train_ranks: tuple[int, ...] = (4, 8, 16),
    ):
        super().__init__(source, rank, alpha)
        self.train_ranks = tuple(sorted({int(r) for r in train_ranks}))
        if not self.train_ranks or self.train_ranks[-1] != self.rank:
            raise ValueError("Matryoshka training ranks must end at the maximum rank")
        # Independent implementation of P_i = sum_{r in S: r >= i} alpha/r.
        diagonal = [
            sum(self.alpha / r for r in self.train_ranks if r >= component + 1)
            for component in range(self.rank)
        ]
        self.register_buffer("prefix_diagonal", torch.tensor(diagonal), persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        if self.training:
            a = self.lora_A * self.prefix_diagonal.to(dtype=x.dtype)[:, None]
            return base + F.linear(F.linear(x, a), self.lora_B)
        return base + self.prefix_update(x, self.inference_rank)


def replace_linears(model: nn.Module, method: str) -> list[str]:
    names = [name for name, module in model.named_modules() if is_adapter_target(name, module)]
    cls = StrictDyLoRALinear if method == "dylora" else MatryoshkaLoRALinear
    for name in names:
        parent_name, child_name = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        setattr(parent, child_name, cls(getattr(parent, child_name)))
    return names


def build_student(args, device: torch.device):
    sys.path.insert(0, str(args.poseidon_code))
    from scOT.model import ScOT, ScOTConfig

    config = ScOTConfig.from_pretrained(args.poseidon_checkpoint)
    config.num_channels = 1
    config.num_out_channels = 1
    config.channel_slice_list_normalized_loss = None
    model = ScOT.from_pretrained(
        args.poseidon_checkpoint,
        config=config,
        ignore_mismatched_sizes=True,
    )
    initialize_scalar_heads_from_pretrained(model, args.poseidon_checkpoint)
    replaced = replace_linears(model, args.method)
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
    return model.to(device), replaced


def set_training_rank(model: nn.Module, rank: int) -> None:
    for module in model.modules():
        if isinstance(module, StrictDyLoRALinear):
            module.training_rank = int(rank)


def set_inference_rank(model: nn.Module, rank: int) -> None:
    for module in model.modules():
        if isinstance(module, RankFlexibleLinear):
            module.inference_rank = int(rank)


def spectral_distillation(student: torch.Tensor, teacher: torch.Tensor, floor: float = 0.02):
    delta = torch.fft.rfft2((student - teacher).float(), dim=(-2, -1))
    reference = torch.fft.rfft2(teacher.float(), dim=(-2, -1)).abs().square()
    scale = reference + floor * reference.mean(dim=(-2, -1), keepdim=True)
    return (delta.abs().square() / scale.clamp_min(1.0e-8)).mean()


@torch.no_grad()
def predict_at_rank(model, data, start: int, count: int, batch_size: int, rank: int):
    device = next(model.parameters()).device
    initial = torch.from_numpy(data["initial"][start : start + count]).to(device).unsqueeze(1)
    times = torch.from_numpy(data["times"]).to(device)
    set_inference_rank(model, rank)
    model.eval()
    predictions = []
    for begin in range(0, count, batch_size):
        end = min(count, begin + batch_size)
        rows = []
        for physical_time in times:
            lead = (physical_time / times[-1]).expand(end - begin)
            rows.append(model(pixel_values=initial[begin:end], time=lead).output)
        predictions.append(torch.stack(rows, dim=1))
    return torch.cat(predictions), times


def per_time_metrics(predicted, target, epsilon, reaction):
    relative = torch.linalg.vector_norm((predicted - target).flatten(2), dim=2)
    relative = relative / torch.linalg.vector_norm(target.flatten(2), dim=2).clamp_min(1.0e-8)
    interface = (
        (predicted.abs() < 0.2).float().mean(dim=(-3, -2, -1))
        - (target.abs() < 0.2).float().mean(dim=(-3, -2, -1))
    ).abs()
    energy = []
    for time_id in range(predicted.shape[1]):
        predicted_energy = free_energy(predicted[:, time_id], epsilon, reaction)
        target_energy = free_energy(target[:, time_id], epsilon, reaction)
        energy.append(((predicted_energy - target_energy).abs() / target_energy.abs().clamp_min(1.0e-8)).mean())
    return {
        "relative_l2": relative.mean(0),
        "interface_mae": interface.mean(0),
        "energy_relative_error": torch.stack(energy),
    }


def select_rank_trace(diagnostics: dict[int, dict[str, torch.Tensor]]) -> list[int]:
    reference = diagnostics[16]
    trace = []
    time_count = len(reference["relative_l2"])
    for time_id in range(time_count):
        selected = 16
        for rank in (4, 8, 16):
            row = diagnostics[rank]
            if (
                row["relative_l2"][time_id] <= 1.02 * reference["relative_l2"][time_id] + 1.0e-4
                and row["interface_mae"][time_id] <= 1.02 * reference["interface_mae"][time_id] + 1.0e-4
                and row["energy_relative_error"][time_id] <= 1.02 * reference["energy_relative_error"][time_id] + 2.0e-4
            ):
                selected = rank
                break
        trace.append(selected)
    return trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--static-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--method", choices=("dylora", "matryoshkalora"), required=True)
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
    parser.add_argument("--val-start", type=int, default=100)
    parser.add_argument("--val-count", type=int, default=20)
    parser.add_argument("--test-start", type=int, default=120)
    parser.add_argument("--test-count", type=int, default=40)
    parser.add_argument("--poseidon-code", type=Path, default=Path("/tmp/poseidon-stnlr"))
    parser.add_argument("--poseidon-checkpoint", type=Path, default=Path("/tmp/poseidon_model"))
    parser.add_argument("--skip-checkpoint", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data = load_data(args.data)
    static_payload = torch.load(args.static_checkpoint, map_location="cpu")

    teacher_args = SimpleNamespace(
        kind="static",
        nested_schedule="early_high",
        u_mid_rank=8,
        poseidon_code=args.poseidon_code,
        poseidon_checkpoint=args.poseidon_checkpoint,
    )
    teacher, _ = build_poseidon(teacher_args, device)
    teacher.load_state_dict(static_payload["model"], strict=False)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    student, replaced = build_student(args, device)
    student.load_state_dict(static_payload["model"], strict=False)
    trainable = [parameter for parameter in student.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    generator = torch.Generator(device=device).manual_seed(args.seed + 9713)
    rank_generator = torch.Generator(device="cpu").manual_seed(args.seed + 29017)

    student.train()
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        initial, target, lead, epsilon, reaction = training_batch(
            data, args.train_count, args.batch_size, device, generator
        )
        if args.method == "dylora":
            sampled_rank = int(torch.randint(1, 17, (1,), generator=rank_generator).item())
            set_training_rank(student, sampled_rank)
        else:
            sampled_rank = 0
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
        if step % 100 == 0 or step == args.steps:
            speed = step / (time.perf_counter() - started)
            print(
                f"step={step} method={args.method} sampled_rank={sampled_rank} "
                f"loss={float(loss):.6f} field={float(field):.6f} steps_per_sec={speed:.3f}",
                flush=True,
            )
    torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started

    validation_target = torch.from_numpy(
        data["fields"][args.val_start : args.val_start + args.val_count]
    ).to(device).unsqueeze(2)
    val_epsilon = torch.from_numpy(
        data["epsilon"][args.val_start : args.val_start + args.val_count]
    ).to(device)
    val_reaction = torch.from_numpy(
        data["reaction"][args.val_start : args.val_start + args.val_count]
    ).to(device)
    validation_predictions = {}
    diagnostics = {}
    for rank in (4, 8, 16):
        validation_predictions[rank], val_times = predict_at_rank(
            student, data, args.val_start, args.val_count, args.batch_size, rank
        )
        diagnostics[rank] = per_time_metrics(
            validation_predictions[rank], validation_target, val_epsilon, val_reaction
        )
    trace = select_rank_trace(diagnostics)

    test_target = torch.from_numpy(
        data["fields"][args.test_start : args.test_start + args.test_count]
    ).to(device).unsqueeze(2)
    test_epsilon = torch.from_numpy(
        data["epsilon"][args.test_start : args.test_start + args.test_count]
    ).to(device)
    test_reaction = torch.from_numpy(
        data["reaction"][args.test_start : args.test_start + args.test_count]
    ).to(device)
    test_predictions = {}
    test_metrics_by_rank = {}
    for rank in (4, 8, 16):
        test_predictions[rank], test_times = predict_at_rank(
            student, data, args.test_start, args.test_count, args.batch_size, rank
        )
        test_metrics_by_rank[rank] = summarize(
            test_predictions[rank],
            test_target,
            test_epsilon,
            test_reaction,
            float(test_times[1] - test_times[0]),
        )
    calibrated = torch.stack(
        [test_predictions[rank][:, time_id] for time_id, rank in enumerate(trace)],
        dim=1,
    )
    calibrated_metrics = summarize(
        calibrated,
        test_target,
        test_epsilon,
        test_reaction,
        float(test_times[1] - test_times[0]),
    )

    result = {
        "method": args.method,
        "training_rule": (
            "uniform ranks 1-16 with preceding prefix detached"
            if args.method == "dylora"
            else "Algorithm-1 diagonal aggregation over ranks 4,8,16"
        ),
        "validation_selected_rank_trace": trace,
        "mean_active_rank": float(np.mean(trace)),
        "calibrated_test_metrics": calibrated_metrics,
        "rank16_test_metrics": test_metrics_by_rank[16],
        "test_metrics_by_rank": {str(rank): value for rank, value in test_metrics_by_rank.items()},
        "validation_per_rank_per_time": {
            str(rank): {
                metric: [float(value) for value in values]
                for metric, values in diagnostics[rank].items()
            }
            for rank in (4, 8, 16)
        },
        "stored_adapter_rank_equivalent": 16,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "replaced_linear_layers": len(replaced),
        "training_seconds": training_seconds,
        "training_seconds_per_step": training_seconds / max(1, args.steps),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    if not args.skip_checkpoint:
        state = {
            name: value.detach().cpu()
            for name, value in student.state_dict().items()
            if "lora_" in name
            or "prefix_diagonal" in name
            or name.startswith("embeddings.patch_embeddings.")
            or name.startswith("patch_recovery.")
        }
        torch.save({"model": state, "result": result}, args.out_dir / "final.pt")
    print(
        json.dumps(
            {key: value for key, value in result.items() if "per_rank" not in key},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
