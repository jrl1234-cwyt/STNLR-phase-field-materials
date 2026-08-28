# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TR_MoLE_Linear(nn.Module):
    """
    Time-routed mixture of low-rank experts around a frozen/full-rank base matrix.
    """
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        expert_ranks=(4, 8, 16, 32),
        time_emb_dim=None,
        router_hidden_dim=64,
        lora_alpha=1.0,
        base_trainable=False,
        router_temperature=1.0,
        top_k=0,
        top_k_mode="sample",
    ):
        super().__init__()
        if len(expert_ranks) == 0:
            raise ValueError("expert_ranks must contain at least one rank.")
        self.in_features = in_features
        self.out_features = out_features
        self.expert_ranks = tuple(int(rank) for rank in expert_ranks)
        if any(rank <= 0 for rank in self.expert_ranks):
            raise ValueError(f"All expert ranks must be positive, got {self.expert_ranks}.")

        time_emb_dim = time_emb_dim or in_features
        self.num_experts = len(self.expert_ranks)
        self.max_expert_rank = max(self.expert_ranks)
        self.lora_alpha = float(lora_alpha)
        self.router_temperature = float(router_temperature)
        if self.router_temperature <= 0:
            raise ValueError(f"router_temperature must be positive, got {self.router_temperature}.")
        self.top_k = int(top_k)
        if self.top_k < 0:
            raise ValueError(f"top_k must be non-negative, got {self.top_k}.")
        if top_k_mode not in {"sample", "batch"}:
            raise ValueError(f"top_k_mode must be 'sample' or 'batch', got {top_k_mode}.")
        self.top_k_mode = top_k_mode
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=base_trainable)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features), requires_grad=base_trainable)
        else:
            self.register_parameter("bias", None)

        self.experts_A = nn.Parameter(torch.empty(self.num_experts, self.max_expert_rank, in_features))
        self.experts_B = nn.Parameter(torch.empty(self.num_experts, out_features, self.max_expert_rank))
        self.router = nn.Sequential(
            nn.Linear(time_emb_dim, router_hidden_dim),
            nn.SiLU(),
            nn.Linear(router_hidden_dim, self.num_experts),
        )
        rank_mask = torch.zeros(self.num_experts, self.max_expert_rank, dtype=torch.float32)
        for i, rank in enumerate(self.expert_ranks):
            rank_mask[i, :rank] = 1
        self.register_buffer("rank_mask", rank_mask, persistent=False)
        self.register_buffer(
            "expert_scales",
            torch.tensor([self.lora_alpha / rank for rank in self.expert_ranks], dtype=torch.float32),
            persistent=False,
        )
        self.current_routing_weights = None
        self.last_routing_weights = None
        self.last_router_entropy = None
        self.router_entropy = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        nn.init.normal_(self.experts_A, std=0.02)
        nn.init.constant_(self.experts_B, 0)
        with torch.no_grad():
            mask_A = self.rank_mask[:, :, None].to(dtype=self.experts_A.dtype)
            mask_B = self.rank_mask[:, None, :].to(dtype=self.experts_B.dtype)
            self.experts_A.mul_(mask_A)
            self.experts_B.mul_(mask_B)

    @torch.no_grad()
    def copy_base_from_linear(self, linear):
        self.weight.copy_(linear.weight)
        if self.bias is not None and linear.bias is not None:
            self.bias.copy_(linear.bias)

    def _padded_expert_weights(self):
        mask_A = self.rank_mask[:, :, None].to(device=self.experts_A.device, dtype=self.experts_A.dtype)
        mask_B = self.rank_mask[:, None, :].to(device=self.experts_B.device, dtype=self.experts_B.dtype)
        return self.experts_A * mask_A, self.experts_B * mask_B

    def _selected_expert_out(self, x, expert_indices, expert_weights):
        padded_A, padded_B = self._padded_expert_weights()
        batch, top_k = expert_indices.shape
        expert_out = None
        for j in range(top_k):
            indices = expert_indices[:, j]
            selected_A = padded_A.index_select(0, indices)
            selected_B = padded_B.index_select(0, indices)
            lx = torch.einsum("bti,bri->btr", x, selected_A)
            lx = torch.einsum("btr,bor->bto", lx, selected_B)
            scale = self.expert_scales.to(device=x.device, dtype=x.dtype).index_select(0, indices).view(batch, 1, 1)
            weighted_lx = expert_weights[:, j].view(batch, 1, 1) * scale * lx
            expert_out = weighted_lx if expert_out is None else expert_out + weighted_lx
        return expert_out

    def forward(self, x, t_emb):
        """
        x: [batch, tokens, in_features]
        t_emb: [batch, time_emb_dim], usually DiT's conditioning vector c.
        """
        base_out = F.linear(x, self.weight, self.bias)
        routing_logits = self.router(t_emb) / self.router_temperature
        routing_weights = F.softmax(routing_logits, dim=-1)
        router_entropy = -(routing_weights * routing_weights.clamp_min(1e-12).log()).sum(dim=-1).mean()
        self.router_entropy = router_entropy
        self.last_router_entropy = router_entropy.detach()
        self.current_routing_weights = routing_weights
        self.last_routing_weights = routing_weights.detach()

        expert_out = torch.zeros_like(base_out)
        expert_weight_shape = [routing_weights.shape[0]] + [1] * (x.dim() - 1)
        if self.top_k == 0 or self.top_k >= self.num_experts:
            for i in range(self.num_experts):
                rank = self.expert_ranks[i]
                lx = F.linear(x, self.experts_A[i, :rank])
                lx = F.linear(lx, self.experts_B[i, :, :rank])
                scale = self.expert_scales[i].to(device=lx.device, dtype=lx.dtype)
                expert_weight = routing_weights[:, i].view(*expert_weight_shape)
                expert_out = expert_out + expert_weight * scale * lx
        elif self.top_k_mode == "batch":
            selected_experts = routing_weights.mean(dim=0).topk(self.top_k).indices
            selected_weights = routing_weights[:, selected_experts]
            selected_weights = selected_weights / selected_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            expert_indices = selected_experts.unsqueeze(0).expand(routing_weights.shape[0], -1)
            expert_out = self._selected_expert_out(x, expert_indices, selected_weights)
        else:
            top_values, top_indices = routing_weights.topk(self.top_k, dim=-1)
            top_values = top_values / top_values.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            expert_out = self._selected_expert_out(x, top_indices, top_values)
        return base_out + expert_out


class StaticLoRA_Linear(nn.Module):
    """
    Fixed-rank LoRA layer around the same base matrix used by DiT Linear layers.
    When dynamic rank sampling is enabled, the same parameter pool can emulate
    a DyLoRA-style search-free baseline by sampling prefix ranks during training.
    """
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        rank=32,
        lora_alpha=1.0,
        base_trainable=False,
        use_dora=False,
        dora_eps=1e-6,
        dynamic_rank_candidates=None,
        dynamic_rank_sampling=False,
        eval_rank=None,
    ):
        super().__init__()
        rank = int(rank)
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}.")
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.lora_alpha = float(lora_alpha)
        self.use_dora = bool(use_dora)
        self.dora_eps = float(dora_eps)
        if self.dora_eps <= 0:
            raise ValueError(f"dora_eps must be positive, got {self.dora_eps}.")
        candidates = dynamic_rank_candidates
        if candidates is None:
            candidates = [rank]
        else:
            candidates = sorted({int(r) for r in candidates if int(r) > 0})
            candidates = [r for r in candidates if r <= rank]
            if not candidates:
                raise ValueError("dynamic_rank_candidates must contain at least one positive rank <= rank.")
        self.dynamic_rank_candidates = tuple(candidates)
        self.dynamic_rank_sampling = bool(dynamic_rank_sampling and len(self.dynamic_rank_candidates) > 1)
        self.eval_rank = int(eval_rank) if eval_rank is not None else int(self.dynamic_rank_candidates[-1])
        self.eval_rank = self._closest_available_rank(self.eval_rank)
        self.last_active_rank = None
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=base_trainable)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features), requires_grad=base_trainable)
        else:
            self.register_parameter("bias", None)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))
        if self.use_dora:
            self.dora_magnitude = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("dora_magnitude", None)
        self.scale = self.lora_alpha / rank
        self.reset_parameters()

    def _closest_available_rank(self, rank_value):
        rank_value = int(rank_value)
        candidates = [r for r in self.dynamic_rank_candidates if r <= rank_value]
        if candidates:
            return candidates[-1]
        return self.dynamic_rank_candidates[0]

    def set_eval_rank(self, rank):
        self.eval_rank = self._closest_available_rank(rank)

    def _active_rank(self):
        if self.dynamic_rank_sampling and self.training:
            return int(random.choice(self.dynamic_rank_candidates))
        return int(self.eval_rank)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        nn.init.normal_(self.lora_A, std=0.02)
        nn.init.constant_(self.lora_B, 0)
        if self.dora_magnitude is not None:
            with torch.no_grad():
                self.dora_magnitude.copy_(self.weight.norm(dim=1).clamp_min(self.dora_eps))

    @torch.no_grad()
    def copy_base_from_linear(self, linear):
        self.weight.copy_(linear.weight)
        if self.bias is not None and linear.bias is not None:
            self.bias.copy_(linear.bias)
        if self.dora_magnitude is not None:
            self.dora_magnitude.copy_(self.weight.norm(dim=1).clamp_min(self.dora_eps))

    def forward(self, x):
        active_rank = self._active_rank()
        self.last_active_rank = torch.tensor(float(active_rank), device=x.device)
        lora_A = self.lora_A[:active_rank]
        lora_B = self.lora_B[:, :active_rank]
        scale = self.lora_alpha / active_rank
        if not self.use_dora:
            base_out = F.linear(x, self.weight, self.bias)
            lora_out = F.linear(F.linear(x, lora_A), lora_B)
            return base_out + scale * lora_out

        # DoRA-style decomposition: learn magnitude separately from update direction.
        delta_weight = scale * (lora_B @ lora_A)
        merged_weight = self.weight + delta_weight
        direction = merged_weight / merged_weight.norm(dim=1, keepdim=True).clamp_min(self.dora_eps)
        dora_weight = direction * self.dora_magnitude[:, None]
        return F.linear(x, dora_weight, self.bias)


class NestedTRMoLE_Linear(nn.Module):
    """
    Time-budgeted nested low-rank expert segments.

    This is the compute-saving TR-MoLE path: instead of evaluating several
    independent LoRA experts and mixing them, it keeps one max-rank LoRA bank
    and slices a timestep-dependent prefix rank. The slice turns into a smaller
    real matmul, which is the key difference from soft mixture routing.
    """
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        nested_ranks=(4, 8, 16, 32),
        rank_schedule="fixed_bins",
        rank_alpha=1.0,
        fixed_edge_rank=None,
        fixed_mid_rank=None,
        lora_alpha=1.0,
        base_trainable=False,
        num_timesteps=1000,
        use_dora=False,
        dora_eps=1e-6,
        orth_init=False,
        fast_infer_mode="off",
        fast_infer_base_rank=0,
        fast_infer_keep_base_until=0.0,
        fast_infer_trainable=False,
    ):
        super().__init__()
        ranks = parse_tr_mole_ranks(nested_ranks)
        ranks = tuple(sorted(set(int(rank) for rank in ranks)))
        if any(rank <= 0 for rank in ranks):
            raise ValueError(f"All nested ranks must be positive, got {ranks}.")
        if ranks[-1] > min(in_features, out_features):
            raise ValueError(
                f"Max nested rank {ranks[-1]} cannot exceed min(in_features, out_features)="
                f"{min(in_features, out_features)}."
            )
        if rank_schedule not in {
            "fixed_bins", "mid_high", "linear_high_noise", "linear_low_noise",
            "t_lora_decay", "quantized_t_lora_decay",
            "reverse_t_lora_decay", "shuffled_t_lora_decay",
        }:
            raise ValueError(f"Unsupported nested rank schedule: {rank_schedule}.")

        self.in_features = in_features
        self.out_features = out_features
        self.nested_ranks = ranks
        self.max_rank = ranks[-1]
        self.rank_schedule = rank_schedule
        self.rank_alpha = float(rank_alpha)
        if self.rank_alpha <= 0:
            raise ValueError(f"rank_alpha must be positive, got {self.rank_alpha}.")
        self.fixed_edge_rank = int(fixed_edge_rank) if fixed_edge_rank is not None else None
        self.fixed_mid_rank = int(fixed_mid_rank) if fixed_mid_rank is not None else None
        if self.fixed_edge_rank is not None and self.fixed_edge_rank not in self.nested_ranks:
            raise ValueError(
                f"fixed_edge_rank={self.fixed_edge_rank} must be one of nested_ranks={self.nested_ranks}."
            )
        if self.fixed_mid_rank is not None and self.fixed_mid_rank not in self.nested_ranks:
            raise ValueError(
                f"fixed_mid_rank={self.fixed_mid_rank} must be one of nested_ranks={self.nested_ranks}."
            )
        self.lora_alpha = float(lora_alpha)
        self.use_dora = bool(use_dora)
        self.dora_eps = float(dora_eps)
        self.orth_init = bool(orth_init)
        if fast_infer_mode not in {"off", "svd", "qr", "drop"}:
            raise ValueError(f"Unsupported fast_infer_mode: {fast_infer_mode}.")
        self.fast_infer_mode = str(fast_infer_mode)
        self.fast_infer_base_rank = int(fast_infer_base_rank)
        self.fast_infer_keep_base_until = float(max(0.0, min(1.0, fast_infer_keep_base_until)))
        self.fast_infer_trainable = bool(fast_infer_trainable)
        self._base_svd_rank_cached = 0
        self._base_fast_mode_cached = "off"
        if self.dora_eps <= 0:
            raise ValueError(f"dora_eps must be positive, got {self.dora_eps}.")
        self.num_timesteps = int(num_timesteps)
        self.rank_warmup_progress = 1.0
        self.rank_shift_steps = 0
        self.lora_gate = 1.0
        self.lora_skip_low_t = -1.0
        self.lora_skip_high_t = 2.0
        self.lora_skip_epsilon = 0.0
        self.lora_skip_transition_width = 0.0
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=base_trainable)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features), requires_grad=base_trainable)
        else:
            self.register_parameter("bias", None)
        self.lora_A = nn.Parameter(torch.empty(self.max_rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.max_rank))
        self.register_buffer("_base_svd_A", torch.empty(0), persistent=False)  # [r, in]
        self.register_buffer("_base_svd_B", torch.empty(0), persistent=False)  # [out, r]
        self.register_parameter("fast_svd_A", None)  # [r, in], optional trainable fast-SVD factor
        self.register_parameter("fast_svd_B", None)  # [out, r], optional trainable fast-SVD factor
        if self.use_dora:
            self.dora_magnitude = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("dora_magnitude", None)
        self.last_active_rank = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if self.orth_init:
            with torch.no_grad():
                # T-LoRA-style orthogonal basis initialization for LoRA factors.
                base = torch.randn(
                    self.in_features,
                    self.max_rank,
                    device=self.lora_A.device,
                    dtype=self.lora_A.dtype,
                )
                q, _ = torch.linalg.qr(base, mode="reduced")
                self.lora_A.copy_(q.transpose(0, 1).contiguous())
        else:
            nn.init.normal_(self.lora_A, std=0.02)
        nn.init.constant_(self.lora_B, 0)
        if self.dora_magnitude is not None:
            with torch.no_grad():
                self.dora_magnitude.copy_(self.weight.norm(dim=1).clamp_min(self.dora_eps))
        self._maybe_refresh_base_svd_cache()

    @torch.no_grad()
    def copy_base_from_linear(self, linear):
        self.weight.copy_(linear.weight)
        if self.bias is not None and linear.bias is not None:
            self.bias.copy_(linear.bias)
        if self.dora_magnitude is not None:
            self.dora_magnitude.copy_(self.weight.norm(dim=1).clamp_min(self.dora_eps))
        self._maybe_refresh_base_svd_cache()

    def set_fast_infer(self, mode="off", base_rank=None, keep_base_until=None):
        if mode not in {"off", "svd", "qr", "drop"}:
            raise ValueError(f"Unsupported fast infer mode: {mode}.")
        self.fast_infer_mode = str(mode)
        if base_rank is not None:
            self.fast_infer_base_rank = int(base_rank)
        if keep_base_until is not None:
            self.fast_infer_keep_base_until = float(max(0.0, min(1.0, keep_base_until)))
        self._maybe_refresh_base_svd_cache()

    def set_fast_infer_trainable(self, enabled=True, reinit_from_cache=True):
        """
        Enable training on cached fast low-rank factors only.
        The same factor slots are reused for SVD or QR fast infer modes.
        """
        self.fast_infer_trainable = bool(enabled)
        self._maybe_refresh_base_svd_cache()
        if not self.fast_infer_trainable:
            if self.fast_svd_A is not None:
                self.fast_svd_A.requires_grad = False
            if self.fast_svd_B is not None:
                self.fast_svd_B.requires_grad = False
            return

        if self._base_svd_A.numel() == 0 or self._base_svd_B.numel() == 0:
            self._maybe_refresh_base_svd_cache()

        if self.fast_svd_A is None or self.fast_svd_A.shape != self._base_svd_A.shape:
            self.fast_svd_A = nn.Parameter(self._base_svd_A.detach().clone(), requires_grad=True)
        elif reinit_from_cache:
            with torch.no_grad():
                self.fast_svd_A.copy_(self._base_svd_A)
            self.fast_svd_A.requires_grad = True

        if self.fast_svd_B is None or self.fast_svd_B.shape != self._base_svd_B.shape:
            self.fast_svd_B = nn.Parameter(self._base_svd_B.detach().clone(), requires_grad=True)
        elif reinit_from_cache:
            with torch.no_grad():
                self.fast_svd_B.copy_(self._base_svd_B)
            self.fast_svd_B.requires_grad = True

    @torch.no_grad()
    def _factorize_weight_for_fast_infer(self, weight_fp32, rank):
        if self.fast_infer_mode == "svd":
            try:
                u, s, vh = torch.linalg.svd(weight_fp32, full_matrices=False)
            except RuntimeError:
                # Fallback to CPU SVD for environments with limited GPU SVD support.
                u, s, vh = torch.linalg.svd(weight_fp32.cpu(), full_matrices=False)
                u = u.to(device=self.weight.device)
                s = s.to(device=self.weight.device)
                vh = vh.to(device=self.weight.device)
            b = u[:, :rank] * s[:rank].unsqueeze(0)   # [out, r]
            a = vh[:rank, :]                           # [r, in]
            return a, b

        if self.fast_infer_mode == "qr":
            from scipy import linalg as scipy_linalg

            weight_np = weight_fp32.detach().cpu().numpy()
            q, r, piv = scipy_linalg.qr(weight_np, mode="economic", pivoting=True)
            q = torch.from_numpy(q[:, :rank]).to(device=self.weight.device, dtype=torch.float32)
            r = torch.from_numpy(r[:rank, :]).to(device=self.weight.device, dtype=torch.float32)
            a = torch.zeros(rank, self.in_features, device=self.weight.device, dtype=torch.float32)
            a[:, piv] = r
            b = q
            return a, b

        raise ValueError(f"Unsupported fast factorization mode: {self.fast_infer_mode}")

    @torch.no_grad()
    def _maybe_refresh_base_svd_cache(self):
        if self.fast_infer_mode not in {"svd", "qr"}:
            return
        rank = int(self.fast_infer_base_rank)
        if rank <= 0:
            rank = self.max_rank
        rank = max(1, min(rank, min(self.out_features, self.in_features)))
        if (
            self._base_svd_rank_cached == rank
            and self._base_fast_mode_cached == self.fast_infer_mode
            and self._base_svd_A.numel() != 0
            and self._base_svd_B.numel() != 0
        ):
            return
        weight_fp32 = self.weight.detach().to(dtype=torch.float32)
        a, b = self._factorize_weight_for_fast_infer(weight_fp32, rank)
        self._base_svd_A = a.to(device=self.weight.device, dtype=self.weight.dtype)
        self._base_svd_B = b.to(device=self.weight.device, dtype=self.weight.dtype)
        self._base_svd_rank_cached = rank
        self._base_fast_mode_cached = self.fast_infer_mode
        if self.fast_infer_trainable:
            # Keep trainable factors shape in sync; re-init only on shape mismatch.
            if self.fast_svd_A is None or self.fast_svd_A.shape != self._base_svd_A.shape:
                self.fast_svd_A = nn.Parameter(self._base_svd_A.detach().clone(), requires_grad=True)
            if self.fast_svd_B is None or self.fast_svd_B.shape != self._base_svd_B.shape:
                self.fast_svd_B = nn.Parameter(self._base_svd_B.detach().clone(), requires_grad=True)

    def _rank_index_from_score(self, score):
        score = float(max(0.0, min(1.0, score)))
        rank_idx = min(int(score * len(self.nested_ranks)), len(self.nested_ranks) - 1)
        return rank_idx

    def _ceil_available_rank(self, rank_value):
        rank_value = float(rank_value)
        for rank in self.nested_ranks:
            if rank >= rank_value:
                return rank
        return self.max_rank

    def _floor_available_rank(self, rank_value):
        rank_value = float(rank_value)
        for rank in reversed(self.nested_ranks):
            if rank <= rank_value:
                return rank
        return self.nested_ranks[0]

    def set_rank_warmup_progress(self, progress):
        self.rank_warmup_progress = float(max(0.0, min(1.0, progress)))

    def set_rank_shift_steps(self, shift_steps):
        max_shift = len(self.nested_ranks) - 1
        shift_steps = int(shift_steps)
        self.rank_shift_steps = max(-max_shift, min(max_shift, shift_steps))

    def set_lora_gate(self, gate):
        self.lora_gate = float(max(0.0, min(1.0, gate)))

    def set_lora_timestep_skip(self, low_t=None, high_t=None, epsilon=None, transition_width=None):
        self.lora_skip_low_t = -1.0 if low_t is None else float(max(0.0, min(1.0, low_t)))
        self.lora_skip_high_t = 2.0 if high_t is None else float(max(0.0, min(1.0, high_t)))
        if epsilon is not None:
            self.lora_skip_epsilon = float(max(0.0, min(1.0, epsilon)))
        if transition_width is not None:
            self.lora_skip_transition_width = float(max(0.0, transition_width))

    def _timestep_to_norm(self, timestep):
        if timestep is None:
            return None
        timestep_value = timestep.detach().float().reshape(-1)[0].item()
        t_norm = timestep_value / max(self.num_timesteps - 1, 1)
        return float(max(0.0, min(1.0, t_norm)))

    def _apply_rank_shift(self, rank):
        try:
            idx = self.nested_ranks.index(int(rank))
        except ValueError:
            return rank
        idx = max(0, min(len(self.nested_ranks) - 1, idx + self.rank_shift_steps))
        return self.nested_ranks[idx]

    def rank_from_timestep(self, timestep):
        if timestep is None:
            return self.max_rank
        # Follow T-LoRA's per-step masking spirit: use one representative timestep
        # instead of averaging a mixed batch (which would collapse to mid-rank).
        t_norm = self._timestep_to_norm(timestep)
        if self.rank_schedule == "fixed_bins":
            default_edge_rank = self.nested_ranks[min(1, len(self.nested_ranks) - 1)]
            edge_rank = self.fixed_edge_rank if self.fixed_edge_rank is not None else default_edge_rank
            mid_rank = self.fixed_mid_rank if self.fixed_mid_rank is not None else self.max_rank
            base_rank = mid_rank if (1.0 / 3.0 <= t_norm < 2.0 / 3.0) else edge_rank
        elif self.rank_schedule == "mid_high":
            mid_score = 1.0 - abs(2.0 * t_norm - 1.0)
            base_rank = self.nested_ranks[self._rank_index_from_score(mid_score)]
        elif self.rank_schedule == "linear_high_noise":
            base_rank = self.nested_ranks[self._rank_index_from_score(t_norm)]
        elif self.rank_schedule == "linear_low_noise":
            base_rank = self.nested_ranks[self._rank_index_from_score(1.0 - t_norm)]
        elif self.rank_schedule == "t_lora_decay":
            # T-LoRA style: lower noise (small t) gets larger rank, high noise gets smaller rank.
            min_rank = self.nested_ranks[0]
            score = (1.0 - t_norm) ** self.rank_alpha
            target_rank = min_rank + score * (self.max_rank - min_rank)
            base_rank = self._floor_available_rank(target_rank)
        elif self.rank_schedule == "quantized_t_lora_decay":
            # Monotone low-noise schedule with a nonzero interval assigned to
            # every stored prefix, including the maximum rank.  Quantizing the
            # normalized score avoids activating the last prefix only at t=0.
            score = (1.0 - t_norm) ** self.rank_alpha
            base_rank = self.nested_ranks[self._rank_index_from_score(score)]
        elif self.rank_schedule == "reverse_t_lora_decay":
            min_rank = self.nested_ranks[0]
            score = t_norm ** self.rank_alpha
            target_rank = min_rank + score * (self.max_rank - min_rank)
            base_rank = self._floor_available_rank(target_rank)
        elif self.rank_schedule == "shuffled_t_lora_decay":
            permutation = (3, 0, 6, 1, 7, 4, 2, 5)
            scaled = min(t_norm * len(permutation), len(permutation) - 1e-8)
            bin_index = int(scaled)
            within_bin = scaled - bin_index
            shuffled_t = (permutation[bin_index] + within_bin) / len(permutation)
            min_rank = self.nested_ranks[0]
            score = (1.0 - shuffled_t) ** self.rank_alpha
            target_rank = min_rank + score * (self.max_rank - min_rank)
            base_rank = self._floor_available_rank(target_rank)
        else:
            base_rank = self.max_rank

        base_rank = self._apply_rank_shift(base_rank)
        progress = self.rank_warmup_progress
        if progress < 1.0:
            warmup_rank = base_rank + (1.0 - progress) * (self.max_rank - base_rank)
            return self._ceil_available_rank(warmup_rank)
        return base_rank

    def forward(self, x, timestep=None):
        rank = self.rank_from_timestep(timestep)
        self.last_active_rank = torch.tensor(float(rank), device=x.device)
        gate = float(self.lora_gate)
        # Keep training behavior unchanged unless fast_infer is explicitly enabled.
        t_norm = self._timestep_to_norm(timestep)
        keep_full_base = (
            self.fast_infer_mode == "off"
            or self.use_dora
            or (
                self.fast_infer_keep_base_until > 0.0
                and t_norm is not None
                and t_norm <= self.fast_infer_keep_base_until
            )
        )
        if keep_full_base:
            base_out = F.linear(x, self.weight, self.bias)
        elif self.fast_infer_mode == "drop":
            base_out = x.new_zeros(*x.shape[:-1], self.out_features)
        elif self.fast_infer_mode in {"svd", "qr"}:
            if self.fast_infer_trainable and self.fast_svd_A is not None and self.fast_svd_B is not None:
                a = self.fast_svd_A.to(device=x.device, dtype=x.dtype)
                b = self.fast_svd_B.to(device=x.device, dtype=x.dtype)
                base_out = F.linear(F.linear(x, a), b, self.bias)
            else:
                if self._base_svd_A.numel() == 0 or self._base_svd_B.numel() == 0:
                    self._maybe_refresh_base_svd_cache()
                a = self._base_svd_A.to(device=x.device, dtype=x.dtype)
                b = self._base_svd_B.to(device=x.device, dtype=x.dtype)
                base_out = F.linear(F.linear(x, a), b, self.bias)
        else:
            base_out = F.linear(x, self.weight, self.bias)
        if gate <= 0.0:
            return base_out
        residual_gate = gate
        if t_norm is not None and self.lora_skip_transition_width > 0.0 and self.lora_skip_epsilon > 0.0:
            width = max(self.lora_skip_transition_width, 1e-12)
            residual_scale = 1.0
            if self.lora_skip_low_t >= 0.0:
                residual_scale *= 1.0 / (1.0 + math.exp(-(t_norm - self.lora_skip_low_t) / width))
            if self.lora_skip_high_t <= 1.0:
                residual_scale *= 1.0 / (1.0 + math.exp(-(self.lora_skip_high_t - t_norm) / width))
            residual_gate = gate * max(float(residual_scale), float(self.lora_skip_epsilon))
        elif t_norm is not None and (t_norm < self.lora_skip_low_t or t_norm > self.lora_skip_high_t):
            if self.lora_skip_epsilon <= 0.0:
                return base_out
            residual_gate = gate * self.lora_skip_epsilon
        lora_A = self.lora_A[:rank]
        lora_B = self.lora_B[:, :rank]
        scale = self.lora_alpha / rank
        if not self.use_dora:
            lora_out = F.linear(F.linear(x, lora_A), lora_B)
            return base_out + residual_gate * scale * lora_out

        delta_weight = residual_gate * scale * (lora_B @ lora_A)
        merged_weight = self.weight + delta_weight
        direction = merged_weight / merged_weight.norm(dim=1, keepdim=True).clamp_min(self.dora_eps)
        dora_weight = direction * self.dora_magnitude[:, None]
        return F.linear(x, dora_weight, self.bias)


class StaticLoRAAttention(nn.Module):
    """
    DiT attention with fixed-rank LoRA projections for qkv and output projection.
    """
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.,
        proj_drop=0.,
        norm_layer=nn.LayerNorm,
        **lora_kwargs,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = StaticLoRA_Linear(dim, dim * 3, bias=qkv_bias, **lora_kwargs)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = StaticLoRA_Linear(dim, dim, bias=True, **lora_kwargs)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = self.q_norm(q)
        k = self.k_norm(k)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(batch, tokens, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class StaticLoRAMlp(nn.Module):
    """
    DiT MLP with fixed-rank LoRA linear projections.
    """
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.,
        norm_layer=None,
        **lora_kwargs,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = StaticLoRA_Linear(in_features, hidden_features, bias=True, **lora_kwargs)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = StaticLoRA_Linear(hidden_features, out_features, bias=True, **lora_kwargs)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class NestedTRMoLEAttention(nn.Module):
    """
    DiT attention with timestep-selected nested low-rank prefixes.
    """
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.,
        proj_drop=0.,
        norm_layer=nn.LayerNorm,
        **nested_kwargs,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = NestedTRMoLE_Linear(dim, dim * 3, bias=qkv_bias, **nested_kwargs)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = NestedTRMoLE_Linear(dim, dim, bias=True, **nested_kwargs)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, timestep):
        batch, tokens, channels = x.shape
        qkv = self.qkv(x, timestep).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = self.q_norm(q)
        k = self.k_norm(k)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(batch, tokens, channels)
        x = self.proj(x, timestep)
        x = self.proj_drop(x)
        return x


class NestedTRMoLEMlp(nn.Module):
    """
    DiT MLP with timestep-selected nested low-rank prefixes.
    """
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.,
        norm_layer=None,
        **nested_kwargs,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = NestedTRMoLE_Linear(in_features, hidden_features, bias=True, **nested_kwargs)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = NestedTRMoLE_Linear(hidden_features, out_features, bias=True, **nested_kwargs)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x, timestep):
        x = self.fc1(x, timestep)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x, timestep)
        x = self.drop2(x)
        return x


class TimeRoutedAttention(nn.Module):
    """
    DiT attention with TR-MoLE projections for qkv and output projection.
    """
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.,
        proj_drop=0.,
        norm_layer=nn.LayerNorm,
        **tr_mole_kwargs,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = TR_MoLE_Linear(dim, dim * 3, bias=qkv_bias, **tr_mole_kwargs)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = TR_MoLE_Linear(dim, dim, bias=True, **tr_mole_kwargs)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, c):
        batch, tokens, channels = x.shape
        qkv = self.qkv(x, c).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = self.q_norm(q)
        k = self.k_norm(k)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(batch, tokens, channels)
        x = self.proj(x, c)
        x = self.proj_drop(x)
        return x


class TimeRoutedMlp(nn.Module):
    """
    DiT MLP with TR-MoLE linear projections.
    """
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.,
        norm_layer=None,
        **tr_mole_kwargs,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = TR_MoLE_Linear(in_features, hidden_features, bias=True, **tr_mole_kwargs)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = TR_MoLE_Linear(hidden_features, out_features, bias=True, **tr_mole_kwargs)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x, c):
        x = self.fc1(x, c)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x, c)
        x = self.drop2(x)
        return x


def parse_tr_mole_ranks(ranks):
    if isinstance(ranks, str):
        ranks = tuple(int(rank.strip()) for rank in ranks.split(",") if rank.strip())
    else:
        ranks = tuple(int(rank) for rank in ranks)
    if len(ranks) == 0:
        raise ValueError("TR-MoLE ranks cannot be empty.")
    return ranks


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        use_tr_mole=False,
        use_static_lora=False,
        use_dylora=False,
        use_nested_tr_mole=False,
        tr_mole_ranks=(4, 8, 16, 32),
        tr_mole_router_hidden_dim=64,
        tr_mole_alpha=1.0,
        tr_mole_train_base=False,
        tr_mole_router_temperature=1.0,
        tr_mole_top_k=0,
        tr_mole_top_k_mode="sample",
        nested_tr_mole_ranks=(4, 8, 16, 32),
        nested_tr_mole_rank_schedule="fixed_bins",
        nested_tr_mole_rank_alpha=1.0,
        nested_tr_mole_fixed_edge_rank=None,
        nested_tr_mole_fixed_mid_rank=None,
        nested_tr_mole_alpha=1.0,
        nested_tr_mole_train_base=False,
        nested_tr_mole_num_timesteps=1000,
        nested_tr_mole_use_dora=False,
        nested_tr_mole_orth_init=False,
        static_lora_rank=32,
        static_lora_alpha=1.0,
        static_lora_train_base=False,
        static_lora_use_dora=False,
        dylora_ranks=(4, 8, 16, 32),
        dylora_alpha=1.0,
        dylora_train_base=False,
        dylora_use_dora=False,
        dylora_eval_rank=None,
        dora_eps=1e-6,
        **block_kwargs,
    ):
        super().__init__()
        enabled_modes = int(use_tr_mole) + int(use_static_lora) + int(use_dylora) + int(use_nested_tr_mole)
        if enabled_modes > 1:
            raise ValueError("use_tr_mole, use_static_lora, use_dylora and use_nested_tr_mole are mutually exclusive.")
        self.use_tr_mole = use_tr_mole
        self.use_static_lora = use_static_lora
        self.use_dylora = use_dylora
        self.use_nested_tr_mole = use_nested_tr_mole
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        tr_mole_kwargs = dict(
            expert_ranks=parse_tr_mole_ranks(tr_mole_ranks),
            time_emb_dim=hidden_size,
            router_hidden_dim=tr_mole_router_hidden_dim,
            lora_alpha=tr_mole_alpha,
            base_trainable=tr_mole_train_base,
            router_temperature=tr_mole_router_temperature,
            top_k=tr_mole_top_k,
            top_k_mode=tr_mole_top_k_mode,
        )
        nested_kwargs = dict(
            nested_ranks=parse_tr_mole_ranks(nested_tr_mole_ranks),
            rank_schedule=nested_tr_mole_rank_schedule,
            rank_alpha=nested_tr_mole_rank_alpha,
            fixed_edge_rank=nested_tr_mole_fixed_edge_rank,
            fixed_mid_rank=nested_tr_mole_fixed_mid_rank,
            lora_alpha=nested_tr_mole_alpha,
            base_trainable=nested_tr_mole_train_base,
            num_timesteps=nested_tr_mole_num_timesteps,
            use_dora=nested_tr_mole_use_dora,
            dora_eps=dora_eps,
            orth_init=nested_tr_mole_orth_init,
        )
        static_lora_kwargs = dict(
            rank=static_lora_rank,
            lora_alpha=static_lora_alpha,
            base_trainable=static_lora_train_base,
            use_dora=static_lora_use_dora,
            dora_eps=dora_eps,
        )
        dylora_rank_candidates = parse_tr_mole_ranks(dylora_ranks)
        dylora_kwargs = dict(
            rank=dylora_rank_candidates[-1],
            lora_alpha=dylora_alpha,
            base_trainable=dylora_train_base,
            use_dora=dylora_use_dora,
            dora_eps=dora_eps,
            dynamic_rank_candidates=dylora_rank_candidates,
            dynamic_rank_sampling=True,
            eval_rank=dylora_eval_rank if dylora_eval_rank is not None else dylora_rank_candidates[-1],
        )
        if use_tr_mole:
            self.attn = TimeRoutedAttention(
                hidden_size,
                num_heads=num_heads,
                qkv_bias=True,
                **tr_mole_kwargs,
                **block_kwargs,
            )
        elif use_static_lora:
            self.attn = StaticLoRAAttention(
                hidden_size,
                num_heads=num_heads,
                qkv_bias=True,
                **static_lora_kwargs,
                **block_kwargs,
            )
        elif use_dylora:
            self.attn = StaticLoRAAttention(
                hidden_size,
                num_heads=num_heads,
                qkv_bias=True,
                **dylora_kwargs,
                **block_kwargs,
            )
        elif use_nested_tr_mole:
            self.attn = NestedTRMoLEAttention(
                hidden_size,
                num_heads=num_heads,
                qkv_bias=True,
                **nested_kwargs,
                **block_kwargs,
            )
        else:
            self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        if use_tr_mole:
            self.mlp = TimeRoutedMlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0,
                **tr_mole_kwargs,
            )
        elif use_static_lora:
            self.mlp = StaticLoRAMlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0,
                **static_lora_kwargs,
            )
        elif use_dylora:
            self.mlp = StaticLoRAMlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0,
                **dylora_kwargs,
            )
        elif use_nested_tr_mole:
            self.mlp = NestedTRMoLEMlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0,
                **nested_kwargs,
            )
        else:
            self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, t_emb=None, timestep=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        attn_in = modulate(self.norm1(x), shift_msa, scale_msa)
        mlp_in = modulate(self.norm2(x), shift_mlp, scale_mlp)
        if self.use_tr_mole:
            router_emb = t_emb if t_emb is not None else c
            x = x + gate_msa.unsqueeze(1) * self.attn(attn_in, router_emb)
            x = x + gate_mlp.unsqueeze(1) * self.mlp(mlp_in, router_emb)
        elif self.use_nested_tr_mole:
            x = x + gate_msa.unsqueeze(1) * self.attn(attn_in, timestep)
            x = x + gate_mlp.unsqueeze(1) * self.mlp(mlp_in, timestep)
        elif self.use_static_lora or self.use_dylora:
            x = x + gate_msa.unsqueeze(1) * self.attn(attn_in)
            x = x + gate_mlp.unsqueeze(1) * self.mlp(mlp_in)
        else:
            x = x + gate_msa.unsqueeze(1) * self.attn(attn_in)
            x = x + gate_mlp.unsqueeze(1) * self.mlp(mlp_in)
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        use_tr_mole=False,
        use_static_lora=False,
        use_dylora=False,
        use_nested_tr_mole=False,
        tr_mole_ranks=(4, 8, 16, 32),
        tr_mole_router_hidden_dim=64,
        tr_mole_alpha=1.0,
        tr_mole_train_base=False,
        tr_mole_router_temperature=1.0,
        tr_mole_top_k=0,
        tr_mole_top_k_mode="sample",
        nested_tr_mole_ranks=(4, 8, 16, 32),
        nested_tr_mole_rank_schedule="fixed_bins",
        nested_tr_mole_rank_alpha=1.0,
        nested_tr_mole_fixed_edge_rank=None,
        nested_tr_mole_fixed_mid_rank=None,
        nested_tr_mole_alpha=1.0,
        nested_tr_mole_train_base=False,
        nested_tr_mole_num_timesteps=1000,
        nested_tr_mole_use_dora=False,
        nested_tr_mole_orth_init=False,
        static_lora_rank=32,
        static_lora_alpha=1.0,
        static_lora_train_base=False,
        static_lora_use_dora=False,
        dylora_ranks=(4, 8, 16, 32),
        dylora_alpha=1.0,
        dylora_train_base=False,
        dylora_use_dora=False,
        dylora_eval_rank=None,
        dora_eps=1e-6,
    ):
        super().__init__()
        enabled_modes = int(use_tr_mole) + int(use_static_lora) + int(use_dylora) + int(use_nested_tr_mole)
        if enabled_modes > 1:
            raise ValueError("use_tr_mole, use_static_lora, use_dylora and use_nested_tr_mole are mutually exclusive.")
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.use_tr_mole = use_tr_mole
        self.use_static_lora = use_static_lora
        self.use_dylora = use_dylora
        self.use_nested_tr_mole = use_nested_tr_mole
        self.tr_mole_ranks = parse_tr_mole_ranks(tr_mole_ranks)
        self.nested_tr_mole_ranks = parse_tr_mole_ranks(nested_tr_mole_ranks)
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_size,
                num_heads,
                mlp_ratio=mlp_ratio,
                use_tr_mole=use_tr_mole,
                use_static_lora=use_static_lora,
                use_dylora=use_dylora,
                use_nested_tr_mole=use_nested_tr_mole,
                tr_mole_ranks=self.tr_mole_ranks,
                tr_mole_router_hidden_dim=tr_mole_router_hidden_dim,
                tr_mole_alpha=tr_mole_alpha,
                tr_mole_train_base=tr_mole_train_base,
                tr_mole_router_temperature=tr_mole_router_temperature,
                tr_mole_top_k=tr_mole_top_k,
                tr_mole_top_k_mode=tr_mole_top_k_mode,
                nested_tr_mole_ranks=self.nested_tr_mole_ranks,
                nested_tr_mole_rank_schedule=nested_tr_mole_rank_schedule,
                nested_tr_mole_rank_alpha=nested_tr_mole_rank_alpha,
                nested_tr_mole_fixed_edge_rank=nested_tr_mole_fixed_edge_rank,
                nested_tr_mole_fixed_mid_rank=nested_tr_mole_fixed_mid_rank,
                nested_tr_mole_alpha=nested_tr_mole_alpha,
                nested_tr_mole_train_base=nested_tr_mole_train_base,
                nested_tr_mole_num_timesteps=nested_tr_mole_num_timesteps,
                nested_tr_mole_use_dora=nested_tr_mole_use_dora,
                nested_tr_mole_orth_init=nested_tr_mole_orth_init,
                static_lora_rank=static_lora_rank,
                static_lora_alpha=static_lora_alpha,
                static_lora_train_base=static_lora_train_base,
                static_lora_use_dora=static_lora_use_dora,
                dylora_ranks=dylora_ranks,
                dylora_alpha=dylora_alpha,
                dylora_train_base=dylora_train_base,
                dylora_use_dora=dylora_use_dora,
                dylora_eval_rank=dylora_eval_rank,
                dora_eps=dora_eps,
            ) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        t_raw = t
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        t_emb = t                                # Pure timestep embedding for TR-MoLE routers.
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)
        for block in self.blocks:
            x = block(x, c, t_emb, t_raw)        # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x

    @torch.no_grad()
    def get_tr_mole_routing_summary(self):
        """
        Returns mean routing weights from the most recent forward pass.
        Keys identify block/module/projection, values are [num_experts] CPU tensors.
        """
        summary = {}
        if not self.use_tr_mole:
            return summary
        for block_idx, block in enumerate(self.blocks):
            modules = {
                "attn.qkv": block.attn.qkv,
                "attn.proj": block.attn.proj,
                "mlp.fc1": block.mlp.fc1,
                "mlp.fc2": block.mlp.fc2,
            }
            for name, module in modules.items():
                if module.last_routing_weights is not None:
                    summary[f"blocks.{block_idx}.{name}"] = module.last_routing_weights.mean(dim=0).cpu()
        return summary

    @torch.no_grad()
    def get_nested_active_rank_summary(self):
        summary = {}
        if not self.use_nested_tr_mole:
            return summary
        for block_idx, block in enumerate(self.blocks):
            modules = {
                "attn.qkv": block.attn.qkv,
                "attn.proj": block.attn.proj,
                "mlp.fc1": block.mlp.fc1,
                "mlp.fc2": block.mlp.fc2,
            }
            for name, module in modules.items():
                rank_value = getattr(module, "last_active_rank", None)
                if rank_value is not None:
                    summary[f"blocks.{block_idx}.{name}"] = float(rank_value.detach().cpu().item())
        return summary

    def iter_nested_tr_mole_modules(self):
        if not self.use_nested_tr_mole:
            return []
        modules = []
        for block_idx, block in enumerate(self.blocks):
            module_map = {
                "attn.qkv": block.attn.qkv,
                "attn.proj": block.attn.proj,
                "mlp.fc1": block.mlp.fc1,
                "mlp.fc2": block.mlp.fc2,
            }
            for sub_name, module in module_map.items():
                modules.append((f"blocks.{block_idx}.{sub_name}", module))
        return modules

    def set_nested_rank_shift_map(self, shift_map):
        if not self.use_nested_tr_mole:
            return
        shift_map = shift_map or {}
        for name, module in self.iter_nested_tr_mole_modules():
            if hasattr(module, "set_rank_shift_steps"):
                module.set_rank_shift_steps(int(shift_map.get(name, 0)))

    def set_nested_lora_gate_map(self, gate_map):
        if not self.use_nested_tr_mole:
            return
        gate_map = gate_map or {}
        for name, module in self.iter_nested_tr_mole_modules():
            if hasattr(module, "set_lora_gate"):
                module.set_lora_gate(float(gate_map.get(name, 1.0)))

    def set_nested_lora_timestep_skip(self, low_t=None, high_t=None, epsilon=0.0, transition_width=0.0):
        if not self.use_nested_tr_mole:
            return
        for _, module in self.iter_nested_tr_mole_modules():
            if hasattr(module, "set_lora_timestep_skip"):
                module.set_lora_timestep_skip(
                    low_t=low_t,
                    high_t=high_t,
                    epsilon=epsilon,
                    transition_width=transition_width,
                )

    @torch.no_grad()
    def get_nested_lora_gate_summary(self):
        summary = {}
        if not self.use_nested_tr_mole:
            return summary
        for name, module in self.iter_nested_tr_mole_modules():
            summary[name] = float(getattr(module, "lora_gate", 1.0))
        return summary

    @torch.no_grad()
    def get_nested_rank_shift_summary(self):
        summary = {}
        if not self.use_nested_tr_mole:
            return summary
        for name, module in self.iter_nested_tr_mole_modules():
            summary[name] = int(getattr(module, "rank_shift_steps", 0))
        return summary

    def set_nested_rank_warmup_progress(self, progress):
        if not self.use_nested_tr_mole:
            return
        progress = float(max(0.0, min(1.0, progress)))
        for _, module in self.iter_nested_tr_mole_modules():
            if hasattr(module, "set_rank_warmup_progress"):
                module.set_rank_warmup_progress(progress)

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}
