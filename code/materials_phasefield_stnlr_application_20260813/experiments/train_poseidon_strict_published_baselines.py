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


HEAD_PREFIXES = ("embeddings.patch_embeddings.", "patch_recovery.")


class AdaLoraSVDLinear(nn.Module):
    """Frozen linear map plus the P diag(lambda) Q AdaLoRA increment."""

    def __init__(self, source: nn.Linear, rank: int = 16, alpha: float = 16.0):
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
        # Names follow the authors' implementation: B=P, E=lambda, A=Q.
        self.lora_A = nn.Parameter(torch.empty(self.rank, self.in_features))
        self.lora_E = nn.Parameter(torch.zeros(self.rank))
        self.lora_B = nn.Parameter(torch.empty(self.out_features, self.rank))
        nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
        nn.init.normal_(self.lora_B, mean=0.0, std=0.02)
        self.register_buffer("ranknum", torch.tensor(float(self.rank)), persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update_a = self.lora_A * self.lora_E[:, None]
        update = F.linear(F.linear(x, update_a), self.lora_B)
        return F.linear(x, self.weight, self.bias) + self.alpha * update / self.ranknum.clamp_min(1.0e-5)


class TimeStepMasterLinear(nn.Module):
    """Eight rank-4 core experts and one gated global context expert."""

    def __init__(
        self,
        source: nn.Linear,
        fine_experts: int = 8,
        rank: int = 4,
        alpha: float = 4.0,
        num_times: int = 11,
    ):
        super().__init__()
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.num_fine_experts = int(fine_experts)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.num_times = int(num_times)
        self.weight = nn.Parameter(source.weight.detach().clone(), requires_grad=False)
        self.bias = (
            nn.Parameter(source.bias.detach().clone(), requires_grad=False)
            if source.bias is not None
            else None
        )
        self.core_A = nn.Parameter(torch.empty(self.num_fine_experts, self.rank, self.in_features))
        self.core_B = nn.Parameter(torch.zeros(self.num_fine_experts, self.out_features, self.rank))
        self.context_A = nn.Parameter(torch.empty(self.rank, self.in_features))
        self.context_B = nn.Parameter(torch.zeros(self.out_features, self.rank))
        nn.init.normal_(self.core_A, mean=0.0, std=0.02)
        nn.init.normal_(self.context_A, mean=0.0, std=0.02)

        # Equation (6): G(z_t,t)=F(z_t)+E(t).  Zero initialization makes the
        # beginning of assembling exactly equal to the fostered core path.
        self.router_feature = nn.Linear(self.in_features, 1, bias=False)
        self.router_time = nn.Embedding(self.num_times, 1)
        nn.init.zeros_(self.router_feature.weight)
        nn.init.zeros_(self.router_time.weight)
        self.current_time: torch.Tensor | None = None
        self.stage_mode = "assembled"
        self.last_gate = torch.tensor(0.0)

    def _route(self) -> tuple[int, int]:
        if self.current_time is None:
            raise RuntimeError("physical time must be set before a TSM forward pass")
        lead = float(self.current_time.detach().flatten()[0].clamp(0.0, 1.0))
        core_id = min(int(lead * self.num_fine_experts), self.num_fine_experts - 1)
        time_id = min(int(round(lead * (self.num_times - 1))), self.num_times - 1)
        return core_id, time_id

    def _update(self, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (self.alpha / self.rank) * F.linear(F.linear(x, a), b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        core_id, time_id = self._route()
        base = F.linear(x, self.weight, self.bias)
        core = self._update(x, self.core_A[core_id], self.core_B[core_id])
        context = self._update(x, self.context_A, self.context_B)
        if self.stage_mode == "core":
            return base + core
        if self.stage_mode == "context":
            return base + context
        if self.stage_mode != "assembled":
            raise ValueError(f"unknown TSM stage mode: {self.stage_mode}")
        feature = x.reshape(-1, x.shape[-1]).mean(dim=0, keepdim=True)
        time_index = torch.tensor([time_id], device=x.device, dtype=torch.long)
        gate = self.router_feature(feature).reshape(()) + self.router_time(time_index).reshape(())
        self.last_gate = gate.detach()
        return base + core + gate * context


def replace_linears(model: nn.Module, method: str, num_times: int) -> list[str]:
    names = [name for name, module in model.named_modules() if is_adapter_target(name, module)]
    for name in names:
        parent_name, child_name = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        source = getattr(parent, child_name)
        replacement = (
            AdaLoraSVDLinear(source)
            if method == "adalora"
            else TimeStepMasterLinear(source, num_times=num_times)
        )
        setattr(parent, child_name, replacement)
    return names


def build_student(args, device: torch.device, num_times: int):
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
    replaced = replace_linears(model, args.method, num_times)
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model.to(device), replaced


@torch.no_grad()
def load_paired_start(
    student: nn.Module,
    static_state: dict[str, torch.Tensor],
    method: str,
) -> dict[str, float]:
    # Base matrices and scalar target heads have matching names.  Adapter
    # factors are converted explicitly below instead of being copied by name.
    compatible = {
        name: value
        for name, value in static_state.items()
        if ".lora_A" not in name and ".lora_B" not in name
    }
    student.load_state_dict(compatible, strict=False)
    relative_factor_errors = []
    for name, module in student.named_modules():
        if not isinstance(module, (AdaLoraSVDLinear, TimeStepMasterLinear)):
            continue
        static_a = static_state[f"{name}.lora_A"].float()
        static_b = static_state[f"{name}.lora_B"].float()
        # The paired static adapter uses alpha/r = 1/16.
        static_delta = (static_b @ static_a) / 16.0
        u, singular, vh = torch.linalg.svd(static_delta, full_matrices=False)
        if isinstance(module, AdaLoraSVDLinear):
            # AdaLoRA alpha/ranknum=16/16=1 at initialization.
            module.lora_B.copy_(u[:, : module.rank].to(module.lora_B))
            module.lora_E.copy_(singular[: module.rank].to(module.lora_E))
            module.lora_A.copy_(vh[: module.rank].to(module.lora_A))
            reconstructed = module.lora_B @ (module.lora_A * module.lora_E[:, None])
        else:
            root = torch.sqrt(singular[: module.rank].clamp_min(0.0))
            b4 = u[:, : module.rank] * root[None, :]
            a4 = root[:, None] * vh[: module.rank]
            module.context_A.copy_(a4.to(module.context_A))
            module.context_B.copy_(b4.to(module.context_B))
            for expert in range(module.num_fine_experts):
                module.core_A[expert].copy_(a4.to(module.core_A))
                module.core_B[expert].copy_(b4.to(module.core_B))
            reconstructed = b4 @ a4
        comparison_delta = static_delta.to(reconstructed)
        denominator = torch.linalg.norm(comparison_delta).clamp_min(1.0e-12)
        relative_factor_errors.append(
            float(torch.linalg.norm(reconstructed.float() - comparison_delta) / denominator)
        )
    return {
        "mean_relative_factorization_error": float(np.mean(relative_factor_errors)),
        "max_relative_factorization_error": float(np.max(relative_factor_errors)),
    }


def set_tsm_state(model: nn.Module, lead: torch.Tensor, mode: str) -> None:
    for module in model.modules():
        if isinstance(module, TimeStepMasterLinear):
            module.current_time = lead
            module.stage_mode = mode


def set_trainable(model: nn.Module, method: str, stage: str) -> list[nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for name, parameter in model.named_parameters():
        if method == "adalora":
            enabled = any(token in name for token in ("lora_A", "lora_B", "lora_E")) or name.startswith(HEAD_PREFIXES)
        elif stage == "fostering":
            enabled = any(token in name for token in ("core_A", "core_B", "context_A", "context_B")) or name.startswith(HEAD_PREFIXES)
        elif stage == "assembling":
            enabled = "router_feature" in name or "router_time" in name
        else:
            raise ValueError(stage)
        parameter.requires_grad = enabled
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def spectral_distillation(student: torch.Tensor, teacher: torch.Tensor, floor: float = 0.02):
    delta = torch.fft.rfft2((student - teacher).float(), dim=(-2, -1))
    reference = torch.fft.rfft2(teacher.float(), dim=(-2, -1)).abs().square()
    scale = reference + floor * reference.mean(dim=(-2, -1), keepdim=True)
    return (delta.abs().square() / scale.clamp_min(1.0e-8)).mean()


def objective(prediction, target, teacher_output, epsilon, reaction, args):
    field = (prediction - target).square().mean()
    energy, interface = material_structure_loss(
        prediction,
        target,
        epsilon,
        reaction,
        args.interface_temperature,
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
    return loss, {
        "field": float(field.detach()),
        "energy": float(energy.detach()),
        "interface": float(interface.detach()),
        "distill": float(distill.detach()),
        "spectral": float(spectral.detach()),
    }


def orthogonality_regularizer(model: nn.Module, weight: float) -> torch.Tensor:
    terms = []
    for module in model.modules():
        if not isinstance(module, AdaLoraSVDLinear):
            continue
        identity = torch.eye(module.rank, device=module.lora_A.device, dtype=module.lora_A.dtype)
        terms.append(torch.linalg.matrix_norm(module.lora_A @ module.lora_A.T - identity, ord="fro"))
        terms.append(torch.linalg.matrix_norm(module.lora_B.T @ module.lora_B - identity, ord="fro"))
    if not terms:
        return next(model.parameters()).new_zeros(())
    return float(weight) * torch.stack(terms).mean()


class AdaRankAllocator:
    """Algorithm 1 from AdaLoRA, adapted to the local module names."""

    def __init__(
        self,
        model: nn.Module,
        total_steps: int,
        target_total_rank: int,
        initial_warmup: int,
        final_warmup: int,
        mask_interval: int,
        beta1: float,
        beta2: float,
    ):
        self.modules = [module for module in model.modules() if isinstance(module, AdaLoraSVDLinear)]
        self.total_steps = int(total_steps)
        self.target_total_rank = int(target_total_rank)
        self.initial_warmup = int(initial_warmup)
        self.final_warmup = int(final_warmup)
        self.mask_interval = int(mask_interval)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.initial_total_rank = sum(module.rank for module in self.modules)
        self.statistics: dict[int, dict[str, list[torch.Tensor]]] = {}
        for module in self.modules:
            rows = []
            for parameter in (module.lora_A, module.lora_E, module.lora_B):
                rows.append([torch.zeros_like(parameter), torch.zeros_like(parameter)])
            self.statistics[id(module)] = {"rows": rows}
        self.selected_masks = [torch.ones(module.rank, dtype=torch.bool, device=module.lora_E.device) for module in self.modules]

    @torch.no_grad()
    def update_importance(self) -> None:
        for module in self.modules:
            rows = self.statistics[id(module)]["rows"]
            for parameter, (smooth, uncertainty) in zip((module.lora_A, module.lora_E, module.lora_B), rows):
                if parameter.grad is None:
                    continue
                instant = (parameter * parameter.grad).abs()
                smooth.mul_(self.beta1).add_(instant, alpha=1.0 - self.beta1)
                uncertainty.mul_(self.beta2).add_((instant - smooth).abs(), alpha=1.0 - self.beta2)

    def scheduled_rank(self, step: int) -> tuple[int, bool]:
        if step <= self.initial_warmup:
            return self.initial_total_rank, False
        if step > self.total_steps - self.final_warmup:
            return self.target_total_rank, True
        remaining = 1.0 - (step - self.initial_warmup) / (
            self.total_steps - self.final_warmup - self.initial_warmup
        )
        budget = self.target_total_rank + (self.initial_total_rank - self.target_total_rank) * remaining**3
        return int(budget), step % self.mask_interval == 0

    @torch.no_grad()
    def mask(self, step: int) -> tuple[int, float | None]:
        current_rank, should_mask = self.scheduled_rank(step)
        if not should_mask:
            return current_rank, None
        scores = []
        per_module = []
        for module in self.modules:
            rows = self.statistics[id(module)]["rows"]
            score_a = rows[0][0] * rows[0][1]
            score_e = rows[1][0] * rows[1][1]
            score_b = rows[2][0] * rows[2][1]
            triplet = score_e.reshape(-1) + score_a.mean(dim=1) + score_b.mean(dim=0)
            per_module.append(triplet)
            scores.append(triplet)
        all_scores = torch.cat(scores)
        keep = min(max(int(current_rank), 1), all_scores.numel())
        selected = torch.topk(all_scores, k=keep, largest=True, sorted=False).indices
        global_mask = torch.zeros_like(all_scores, dtype=torch.bool)
        global_mask[selected] = True
        threshold = float(all_scores[selected].min())
        offset = 0
        for module_id, (module, triplet) in enumerate(zip(self.modules, per_module)):
            local = global_mask[offset : offset + triplet.numel()]
            module.lora_E.masked_fill_(~local, 0.0)
            self.selected_masks[module_id] = local.clone()
            offset += triplet.numel()
        return keep, threshold

    def mean_active_rank(self) -> float:
        return float(sum(int(mask.sum()) for mask in self.selected_masks) / len(self.selected_masks))

    def rank_distribution(self) -> list[int]:
        return [int(mask.sum()) for mask in self.selected_masks]


@torch.no_grad()
def evaluate(model, method, data, start, count, batch_size, device):
    fields = torch.from_numpy(data["fields"][start : start + count]).to(device).unsqueeze(2)
    initials = torch.from_numpy(data["initial"][start : start + count]).to(device).unsqueeze(1)
    epsilon = torch.from_numpy(data["epsilon"][start : start + count]).to(device)
    reaction = torch.from_numpy(data["reaction"][start : start + count]).to(device)
    times = torch.from_numpy(data["times"]).to(device)
    predictions = []
    model.eval()
    for begin in range(0, count, batch_size):
        end = min(count, begin + batch_size)
        rows = []
        for physical_time in times:
            lead = (physical_time / times[-1]).expand(end - begin)
            if method == "timestep_master":
                set_tsm_state(model, lead, "assembled")
            rows.append(model(pixel_values=initials[begin:end], time=lead).output)
        predictions.append(torch.stack(rows, dim=1))
    predicted = torch.cat(predictions)
    summary = summarize(predicted, fields, epsilon, reaction, float(times[1] - times[0]))
    error = (predicted - fields).flatten(2)
    target_flat = fields.flatten(2)
    relative = torch.linalg.vector_norm(error, dim=2) / torch.linalg.vector_norm(
        target_flat, dim=2
    ).clamp_min(1.0e-8)
    interface_pred = (predicted.abs() < 0.2).float().mean(dim=(-3, -2, -1))
    interface_target = (fields.abs() < 0.2).float().mean(dim=(-3, -2, -1))
    energy_pred = free_energy(predicted[:, -1], epsilon, reaction)
    energy_target = free_energy(fields[:, -1], epsilon, reaction)
    per_sample = {
        "trajectory_relative_l2": relative.mean(1).cpu().tolist(),
        "terminal_relative_l2": relative[:, -1].cpu().tolist(),
        "terminal_interface_fraction_mae": (
            interface_pred[:, -1] - interface_target[:, -1]
        ).abs().cpu().tolist(),
        "terminal_free_energy_relative_error": (
            (energy_pred - energy_target).abs()
            / energy_target.abs().clamp_min(1.0e-8)
        ).cpu().tolist(),
    }
    return summary, per_sample


def make_optimizer(parameters, lr: float, steps: int):
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    return optimizer, scheduler


def train_adalora(student, teacher, data, args, device):
    trainable = set_trainable(student, "adalora", "adalora")
    optimizer, scheduler = make_optimizer(trainable, args.lr, args.steps)
    generator = torch.Generator(device=device).manual_seed(args.seed + 9713)
    layer_count = sum(isinstance(module, AdaLoraSVDLinear) for module in student.modules())
    target_total_rank = int(round(args.target_average_rank * layer_count))
    allocator = AdaRankAllocator(
        student,
        args.steps,
        target_total_rank,
        args.initial_warmup,
        args.final_warmup,
        args.mask_interval,
        args.beta1,
        args.beta2,
    )
    started = time.perf_counter()
    student.train()
    for step in range(1, args.steps + 1):
        initial, target, lead, epsilon, reaction = training_batch(
            data, args.train_count, args.batch_size, device, generator
        )
        with torch.no_grad():
            teacher_output = teacher(pixel_values=initial, time=lead).output
        prediction = student(pixel_values=initial, time=lead).output
        base_loss, parts = objective(prediction, target, teacher_output, epsilon, reaction, args)
        orthogonal = orthogonality_regularizer(student, args.orthogonal_weight)
        loss = base_loss + orthogonal
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        allocator.update_importance()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        active_total, threshold = allocator.mask(step)
        scheduler.step()
        if step % args.log_interval == 0 or step == args.steps:
            speed = step / (time.perf_counter() - started)
            print(
                f"step={step} stage=adalora loss={float(loss):.6f} field={parts['field']:.6f} "
                f"orthogonal={float(orthogonal):.6f} active_rank={active_total/layer_count:.4f} "
                f"threshold={threshold} steps_per_sec={speed:.3f}",
                flush=True,
            )
    return {
        "training_seconds": time.perf_counter() - started,
        "mean_active_rank": allocator.mean_active_rank(),
        "rank_distribution": allocator.rank_distribution(),
        "target_total_rank": target_total_rank,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
    }


def train_timestep_master(student, teacher, data, args, device):
    generator = torch.Generator(device=device).manual_seed(args.seed + 9713)
    trainable = set_trainable(student, "timestep_master", "fostering")
    optimizer, scheduler = make_optimizer(trainable, args.lr, args.fostering_steps)
    started = time.perf_counter()
    student.train()
    for step in range(1, args.fostering_steps + 1):
        initial, target, lead, epsilon, reaction = training_batch(
            data, args.train_count, args.batch_size, device, generator
        )
        with torch.no_grad():
            teacher_output = teacher(pixel_values=initial, time=lead).output
        optimizer.zero_grad(set_to_none=True)
        set_tsm_state(student, lead, "core")
        prediction = student(pixel_values=initial, time=lead).output
        core_loss, core_parts = objective(prediction, target, teacher_output, epsilon, reaction, args)
        (0.5 * core_loss).backward()
        set_tsm_state(student, lead, "context")
        prediction = student(pixel_values=initial, time=lead).output
        context_loss, _ = objective(prediction, target, teacher_output, epsilon, reaction, args)
        (0.5 * context_loss).backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()
        if step % args.log_interval == 0 or step == args.fostering_steps:
            speed = step / (time.perf_counter() - started)
            print(
                f"step={step} stage=fostering loss={float(0.5*(core_loss+context_loss)):.6f} "
                f"field={core_parts['field']:.6f} steps_per_sec={speed:.3f}",
                flush=True,
            )

    fostering_seconds = time.perf_counter() - started
    trainable_router = set_trainable(student, "timestep_master", "assembling")
    optimizer, scheduler = make_optimizer(trainable_router, args.router_lr, args.assembling_steps)
    assembling_started = time.perf_counter()
    student.train()
    for step in range(1, args.assembling_steps + 1):
        initial, target, lead, epsilon, reaction = training_batch(
            data, args.train_count, args.batch_size, device, generator
        )
        with torch.no_grad():
            teacher_output = teacher(pixel_values=initial, time=lead).output
        set_tsm_state(student, lead, "assembled")
        prediction = student(pixel_values=initial, time=lead).output
        loss, parts = objective(prediction, target, teacher_output, epsilon, reaction, args)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_router, 1.0)
        optimizer.step()
        scheduler.step()
        if step % args.log_interval == 0 or step == args.assembling_steps:
            gates = [float(module.last_gate) for module in student.modules() if isinstance(module, TimeStepMasterLinear)]
            speed = step / (time.perf_counter() - assembling_started)
            print(
                f"step={step} stage=assembling loss={float(loss):.6f} field={parts['field']:.6f} "
                f"mean_gate={float(np.mean(gates)):.6f} steps_per_sec={speed:.3f}",
                flush=True,
            )
    assembling_seconds = time.perf_counter() - assembling_started
    adapter_parameters = sum(
        parameter.numel()
        for name, parameter in student.named_parameters()
        if any(token in name for token in ("core_A", "core_B", "context_A", "context_B"))
    )
    router_parameters = sum(
        parameter.numel()
        for name, parameter in student.named_parameters()
        if "router_feature" in name or "router_time" in name
    )
    return {
        "training_seconds": fostering_seconds + assembling_seconds,
        "fostering_seconds": fostering_seconds,
        "assembling_seconds": assembling_seconds,
        "stored_adapter_rank_equivalent": 36,
        "active_adapter_rank_equivalent": 8,
        "expert_parameters": adapter_parameters,
        "router_parameters": router_parameters,
        "fostering_trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "assembling_trainable_parameters": sum(parameter.numel() for parameter in trainable_router),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--static-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--method", choices=("adalora", "timestep_master"), required=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--fostering-steps", type=int, default=600)
    parser.add_argument("--assembling-steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--router-lr", type=float, default=3.0e-4)
    parser.add_argument("--target-average-rank", type=float, default=4.359375)
    parser.add_argument("--initial-warmup", type=int, default=100)
    parser.add_argument("--final-warmup", type=int, default=100)
    parser.add_argument("--mask-interval", type=int, default=10)
    parser.add_argument("--beta1", type=float, default=0.85)
    parser.add_argument("--beta2", type=float, default=0.85)
    parser.add_argument("--orthogonal-weight", type=float, default=0.1)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--spectral-weight", type=float, default=1.0e-3)
    parser.add_argument("--energy-weight", type=float, default=0.04)
    parser.add_argument("--interface-weight", type=float, default=0.02)
    parser.add_argument("--interface-temperature", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-count", type=int, default=100)
    parser.add_argument("--eval-start", type=int, default=120)
    parser.add_argument("--eval-count", type=int, default=40)
    parser.add_argument("--log-interval", type=int, default=100)
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

    student, replaced = build_student(args, device, len(data["times"]))
    factorization = load_paired_start(student, static_payload["model"], args.method)
    if args.method == "adalora":
        training = train_adalora(student, teacher, data, args, device)
    else:
        training = train_timestep_master(student, teacher, data, args, device)
    metrics, per_sample_records = evaluate(
        student,
        args.method,
        data,
        args.eval_start,
        args.eval_count,
        args.batch_size,
        device,
    )
    result = {
        "method": args.method,
        "paper": (
            "AdaLoRA, ICLR 2023"
            if args.method == "adalora"
            else "TimeStep Master, ICML 2025"
        ),
        "metrics": metrics,
        "per_sample_records": per_sample_records,
        "factorization_audit": factorization,
        "replaced_linear_layers": len(replaced),
        **training,
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not args.skip_checkpoint:
        tokens = (
            ("lora_A", "lora_B", "lora_E", "ranknum")
            if args.method == "adalora"
            else ("core_A", "core_B", "context_A", "context_B", "router_feature", "router_time")
        )
        state = {
            name: value.detach().cpu()
            for name, value in student.state_dict().items()
            if any(token in name for token in tokens) or name.startswith(HEAD_PREFIXES)
        }
        torch.save({"model": state, "result": result}, args.out_dir / "final.pt")
    print(json.dumps({key: value for key, value in result.items() if key != "args"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
