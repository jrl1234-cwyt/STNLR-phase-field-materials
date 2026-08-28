#!/usr/bin/env python3
"""Matched-budget dynamic-adapter baselines for the Poseidon Allen--Cahn task.

Two alternatives isolate the value of a shared nested prefix bank:

* ``adalora`` progressively allocates a global rank budget by first-order
  parameter sensitivity.  Every layer stores rank 16, while the final average
  active rank is matched to the calibrated ST-NLR trace.
* ``timestep_expert`` stores four independent rank-4 experts (rank-16 total
  storage) and hard-routes physical lead-time quartiles to one expert.

Both baselines start from the same pretrained Poseidon-T backbone, use the same
task heads and static rank-16 teacher, and optimize the same material losses.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from materials_phasefield_stnlr_application_20260813.experiments.evaluate_allen_cahn_conditional_trajectory import summarize
from materials_phasefield_stnlr_application_20260813.experiments.train_allen_cahn_conditional_trajectory import material_structure_loss
from materials_phasefield_stnlr_application_20260813.experiments.train_evaluate_poseidon_allen_cahn import (
    build_poseidon,
    initialize_scalar_heads_from_pretrained,
    is_adapter_target,
    load_data,
    training_batch,
)


class AdaLoraBudgetLinear(nn.Module):
    """Rank-16 LoRA with a globally allocated component mask."""

    def __init__(self, source: nn.Linear, rank: int = 16, alpha: float = 1.0):
        super().__init__()
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.rank = rank
        self.alpha = alpha
        self.weight = nn.Parameter(source.weight.detach().clone(), requires_grad=False)
        self.bias = (
            nn.Parameter(source.bias.detach().clone(), requires_grad=False)
            if source.bias is not None else None
        )
        self.lora_A = nn.Parameter(torch.empty(rank, source.in_features))
        self.lora_B = nn.Parameter(torch.zeros(source.out_features, rank))
        nn.init.normal_(self.lora_A, std=0.02)
        self.register_buffer("rank_mask", torch.ones(rank), persistent=True)
        self.register_buffer("importance", torch.zeros(rank), persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = self.rank_mask.to(dtype=x.dtype)
        a = self.lora_A * mask[:, None]
        b = self.lora_B * mask[None, :]
        return F.linear(x, self.weight, self.bias) + (self.alpha / self.rank) * F.linear(F.linear(x, a), b)

    @torch.no_grad()
    def update_importance(self, beta: float = 0.9) -> None:
        if self.lora_A.grad is None or self.lora_B.grad is None:
            return
        sensitivity_a = (self.lora_A * self.lora_A.grad).abs().mean(dim=1)
        sensitivity_b = (self.lora_B * self.lora_B.grad).abs().mean(dim=0)
        magnitude = self.lora_A.norm(dim=1) * self.lora_B.norm(dim=0)
        score = sensitivity_a + sensitivity_b + 1.0e-3 * magnitude
        self.importance.mul_(beta).add_(score, alpha=1.0 - beta)


class TimestepExpertLinear(nn.Module):
    """Four independent rank-4 LoRA experts with deterministic time routing."""

    def __init__(self, source: nn.Linear, experts: int = 4, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.num_experts = experts
        self.rank = rank
        self.alpha = alpha
        self.weight = nn.Parameter(source.weight.detach().clone(), requires_grad=False)
        self.bias = (
            nn.Parameter(source.bias.detach().clone(), requires_grad=False)
            if source.bias is not None else None
        )
        self.experts_A = nn.Parameter(torch.empty(experts, rank, source.in_features))
        self.experts_B = nn.Parameter(torch.zeros(experts, source.out_features, rank))
        nn.init.normal_(self.experts_A, std=0.02)
        self.current_time: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.current_time is None:
            raise RuntimeError("physical lead time was not set for timestep expert")
        # The experiment deliberately draws one physical time for the complete
        # batch.  ScOT later folds spatial windows into its leading dimension,
        # so routing once here also keeps all windows of one field consistent.
        expert_id = min(int(float(self.current_time.flatten()[0]) * self.num_experts), self.num_experts - 1)
        base = F.linear(x, self.weight, self.bias)
        low = F.linear(x, self.experts_A[expert_id])
        update = (self.alpha / self.rank) * F.linear(low, self.experts_B[expert_id])
        return base + update


def replace_baseline_linears(model: nn.Module, method: str) -> list[str]:
    names = [name for name, module in model.named_modules() if is_adapter_target(name, module)]
    for name in names:
        parent_name, child_name = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        source = getattr(parent, child_name)
        replacement = AdaLoraBudgetLinear(source) if method == "adalora" else TimestepExpertLinear(source)
        setattr(parent, child_name, replacement)
    return names


def build_student(args, device: torch.device):
    sys.path.insert(0, str(args.poseidon_code))
    from scOT.model import ScOT, ScOTConfig

    config = ScOTConfig.from_pretrained(args.poseidon_checkpoint)
    config.num_channels = 1
    config.num_out_channels = 1
    config.channel_slice_list_normalized_loss = None
    model = ScOT.from_pretrained(
        args.poseidon_checkpoint, config=config, ignore_mismatched_sizes=True
    )
    initialize_scalar_heads_from_pretrained(model, args.poseidon_checkpoint)
    replaced = replace_baseline_linears(model, args.method)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for name, parameter in model.named_parameters():
        if (
            "lora_A" in name or "lora_B" in name or "experts_A" in name or "experts_B" in name
            or name.startswith("embeddings.patch_embeddings.") or name.startswith("patch_recovery.")
        ):
            parameter.requires_grad = True
    return model.to(device), replaced


def set_physical_time(model: nn.Module, lead: torch.Tensor) -> None:
    for module in model.modules():
        if isinstance(module, TimestepExpertLinear):
            module.current_time = lead


@torch.no_grad()
def initialize_from_static(student: nn.Module, static_state: dict[str, torch.Tensor], method: str) -> None:
    # Task heads share names and shapes with the static checkpoint.
    student.load_state_dict(static_state, strict=False)
    if method == "adalora":
        return
    # Each independent time expert starts from the best rank-4 approximation
    # of the static rank-16 update.  The small SVD is evaluated in its 16-D core.
    for name, module in student.named_modules():
        if not isinstance(module, TimestepExpertLinear):
            continue
        a = static_state[f"{name}.lora_A"].float()
        b = static_state[f"{name}.lora_B"].float()
        qb, rb = torch.linalg.qr(b, mode="reduced")
        qa, ra = torch.linalg.qr(a.T, mode="reduced")
        u_core, singular, vh_core = torch.linalg.svd(rb @ ra.T, full_matrices=False)
        u = qb @ u_core[:, : module.rank]
        vh = vh_core[: module.rank] @ qa.T
        root = torch.sqrt(singular[: module.rank] / 4.0).clamp_min(0.0)
        b4 = u * root[None, :]
        a4 = root[:, None] * vh
        for expert in range(module.num_experts):
            module.experts_A[expert].copy_(a4.to(module.experts_A))
            module.experts_B[expert].copy_(b4.to(module.experts_B))


def spectral_distillation(student: torch.Tensor, teacher: torch.Tensor, floor: float = 0.02):
    delta = torch.fft.rfft2((student - teacher).float(), dim=(-2, -1))
    reference = torch.fft.rfft2(teacher.float(), dim=(-2, -1)).abs().square()
    scale = reference + floor * reference.mean(dim=(-2, -1), keepdim=True)
    return (delta.abs().square() / scale.clamp_min(1.0e-8)).mean()


@torch.no_grad()
def allocate_adalora_budget(model: nn.Module, average_rank: float) -> float:
    layers = [module for module in model.modules() if isinstance(module, AdaLoraBudgetLinear)]
    minimum = 4
    total_components = int(round(average_rank * len(layers)))
    extra = max(0, total_components - minimum * len(layers))
    candidates = []
    for layer_id, layer in enumerate(layers):
        # Preserve the four strongest directions in every layer, then allocate
        # the remaining global budget by importance.
        order = torch.argsort(layer.importance, descending=True)
        layer.rank_mask.zero_()
        layer.rank_mask[order[:minimum]] = 1.0
        for component in order[minimum:]:
            candidates.append((float(layer.importance[component]), layer_id, int(component)))
    candidates.sort(key=lambda row: row[0], reverse=True)
    for _, layer_id, component in candidates[:extra]:
        layers[layer_id].rank_mask[component] = 1.0
    return float(np.mean([float(layer.rank_mask.sum()) for layer in layers]))


def current_adalora_rank(model: nn.Module) -> float:
    ranks = [float(module.rank_mask.sum()) for module in model.modules() if isinstance(module, AdaLoraBudgetLinear)]
    return float(np.mean(ranks))


@torch.no_grad()
def evaluate(model, method, data, start, count, batch_size, device):
    fields = torch.from_numpy(data["fields"][start:start + count]).to(device).unsqueeze(2)
    initials = torch.from_numpy(data["initial"][start:start + count]).to(device).unsqueeze(1)
    epsilon = torch.from_numpy(data["epsilon"][start:start + count]).to(device)
    reaction = torch.from_numpy(data["reaction"][start:start + count]).to(device)
    times = torch.from_numpy(data["times"]).to(device)
    predictions = []
    model.eval()
    for begin in range(0, count, batch_size):
        end = min(count, begin + batch_size)
        rows = []
        for physical_time in times:
            lead = (physical_time / times[-1]).expand(end - begin)
            if method == "timestep_expert":
                set_physical_time(model, lead)
            rows.append(model(pixel_values=initials[begin:end], time=lead).output)
        predictions.append(torch.stack(rows, dim=1))
    predicted = torch.cat(predictions)
    return summarize(predicted, fields, epsilon, reaction, float(times[1] - times[0]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--static-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--method", choices=("adalora", "timestep_expert"), required=True)
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--target-average-rank", type=float, default=4.36)
    parser.add_argument("--budget-warmup", type=int, default=150)
    parser.add_argument("--budget-final-step", type=int, default=1200)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--spectral-weight", type=float, default=1.0e-3)
    parser.add_argument("--energy-weight", type=float, default=0.04)
    parser.add_argument("--interface-weight", type=float, default=0.02)
    parser.add_argument("--interface-temperature", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-count", type=int, default=100)
    parser.add_argument("--eval-start", type=int, default=120)
    parser.add_argument("--eval-count", type=int, default=40)
    parser.add_argument("--poseidon-code", type=Path, default=Path("/tmp/poseidon-stnlr"))
    parser.add_argument("--poseidon-checkpoint", type=Path, default=Path("/tmp/poseidon_model"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data = load_data(args.data)
    static_payload = torch.load(args.static_checkpoint, map_location="cpu")
    teacher_args = SimpleNamespace(
        kind="static", nested_schedule="early_high", u_mid_rank=8,
        poseidon_code=args.poseidon_code, poseidon_checkpoint=args.poseidon_checkpoint,
    )
    teacher, _ = build_poseidon(teacher_args, device)
    teacher.load_state_dict(static_payload["model"], strict=False)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    student, replaced = build_student(args, device)
    initialize_from_static(student, static_payload["model"], args.method)
    trainable = [parameter for parameter in student.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    generator = torch.Generator(device=device).manual_seed(args.seed + 260821)
    started = time.time()
    student.train()
    for step in range(1, args.steps + 1):
        initial, target, lead, epsilon, reaction = training_batch(
            data, args.train_count, args.batch_size, device, generator
        )
        if args.method == "timestep_expert":
            set_physical_time(student, lead)
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
            field + args.energy_weight * energy + args.interface_weight * interface
            + args.distill_weight * distill + args.spectral_weight * spectral
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.method == "adalora":
            for module in student.modules():
                if isinstance(module, AdaLoraBudgetLinear):
                    module.update_importance()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()
        if args.method == "adalora" and step >= args.budget_warmup:
            progress = min(1.0, (step - args.budget_warmup) / max(1, args.budget_final_step - args.budget_warmup))
            target = 16.0 + progress * (args.target_average_rank - 16.0)
            active_rank = allocate_adalora_budget(student, target)
        else:
            active_rank = 4.0 if args.method == "timestep_expert" else current_adalora_rank(student)
        if step % 100 == 0:
            speed = step / (time.time() - started)
            print(
                f"step={step} loss={float(loss):.6f} field={float(field):.6f} "
                f"active_rank={active_rank:.3f} steps_per_sec={speed:.3f}", flush=True,
            )
    metrics = evaluate(student, args.method, data, args.eval_start, args.eval_count, args.batch_size, device)
    active_rank = current_adalora_rank(student) if args.method == "adalora" else 4.0
    result = {
        "method": args.method,
        "metrics": metrics,
        "stored_adapter_rank_equivalent": 16,
        "mean_active_rank": active_rank,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "replaced_linear_layers": len(replaced),
        "training_seconds": time.time() - started,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    state = {
        name: value.detach().cpu() for name, value in student.state_dict().items()
        if any(token in name for token in ("lora_", "experts_", "rank_mask", "importance"))
        or name.startswith("embeddings.patch_embeddings.") or name.startswith("patch_recovery.")
    }
    torch.save({"model": state, "result": result}, args.out_dir / "final.pt")
    print(json.dumps({key: value for key, value in result.items() if key != "args"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
