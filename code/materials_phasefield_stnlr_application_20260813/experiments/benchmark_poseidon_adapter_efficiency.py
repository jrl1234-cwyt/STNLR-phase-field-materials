#!/usr/bin/env python3
"""Measure Poseidon-T adapter rank, compute, latency, and inference memory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import StaticLoRA_Linear
from train_evaluate_poseidon_allen_cahn import (
    PoseidonNestedLinear,
    build_poseidon,
    load_data,
    set_nested_time,
)


def load_model(checkpoint: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu")
    saved = payload["args"]
    checkpoint_kind = payload.get("kind")
    if checkpoint_kind in {"static", "static_continued_rank16"}:
        kind = "static"
    elif checkpoint_kind in {None, "stnlr", "nested"}:
        kind = "stnlr"
    else:
        raise ValueError(f"Unsupported checkpoint kind: {checkpoint_kind}")
    saved = dict(saved)
    saved["kind"] = kind
    if "rank_trace" in payload:
        saved["_fixed_rank_trace"] = list(payload["rank_trace"])
    config = SimpleNamespace(
        kind=kind,
        nested_schedule=saved.get("nested_schedule", "early_high"),
        u_mid_rank=int(saved.get("u_mid_rank", 8)),
        poseidon_code=Path("/tmp/poseidon-stnlr"),
        poseidon_checkpoint=Path("/tmp/poseidon_model"),
    )
    model, _ = build_poseidon(config, device)
    model.load_state_dict(payload["model"], strict=False)
    return model.eval(), saved


@torch.no_grad()
def trajectory_forward(model, kind, initial, times, direction):
    outputs = []
    fixed_trace = getattr(model, "_benchmark_fixed_rank_trace", None)
    for time_id, physical_time in enumerate(times):
        lead = (physical_time / times[-1]).expand(initial.shape[0])
        if kind == "stnlr":
            set_nested_time(model, lead, direction)
            if fixed_trace is not None:
                for module in model.modules():
                    if isinstance(module, PoseidonNestedLinear):
                        module.forced_rank = int(fixed_trace[time_id])
        outputs.append(model(pixel_values=initial, time=lead).output)
    return outputs


def benchmark_one(
    checkpoint: Path,
    data_path: Path,
    repeats: int,
    timing_rounds: int,
    trace_metrics: Path | None = None,
):
    device = torch.device("cuda")
    model, saved = load_model(checkpoint, device)
    if trace_metrics is not None:
        saved["_fixed_rank_trace"] = json.loads(trace_metrics.read_text())["rank_trace"]
    data = load_data(data_path)
    initial = torch.from_numpy(data["initial"][:1]).to(device).unsqueeze(1)
    times = torch.from_numpy(data["times"]).to(device)
    kind = saved["kind"]
    if "_fixed_rank_trace" in saved:
        model._benchmark_fixed_rank_trace = saved["_fixed_rank_trace"]
    direction = saved.get("nested_time_direction", "decay")

    adapter_macs = 0.0
    ranks = []
    active_factor_parameters = []

    def adapter_hook(module, inputs, _output):
        nonlocal adapter_macs
        vectors = inputs[0].numel() // module.in_features
        rank = int(float(module.last_active_rank))
        adapter_macs += vectors * rank * (module.in_features + module.out_features)
        ranks.append(rank)
        active_factor_parameters.append(rank * (module.in_features + module.out_features))

    handles = []
    for module in model.modules():
        if isinstance(module, (PoseidonNestedLinear, StaticLoRA_Linear)):
            handles.append(module.register_forward_hook(adapter_hook))

    for _ in range(3):
        trajectory_forward(model, kind, initial, times, direction)
    torch.cuda.synchronize()

    adapter_macs = 0.0
    ranks.clear()
    active_factor_parameters.clear()
    torch.cuda.reset_peak_memory_stats()
    trajectory_forward(model, kind, initial, times, direction)
    torch.cuda.synchronize()
    peak_bytes = torch.cuda.max_memory_allocated()
    adapter_macs_per_trajectory = adapter_macs
    rank_trace = []
    fixed_trace = saved.get("_fixed_rank_trace")
    for time_id, physical_time in enumerate(times):
        lead = (physical_time / times[-1]).expand(1)
        if kind == "stnlr":
            set_nested_time(model, lead, direction)
            if fixed_trace is not None:
                for module in model.modules():
                    if isinstance(module, PoseidonNestedLinear):
                        module.forced_rank = int(fixed_trace[time_id])
        model(pixel_values=initial, time=lead).output
        module_ranks = [int(float(module.last_active_rank)) for module in model.modules() if isinstance(module, (PoseidonNestedLinear, StaticLoRA_Linear))]
        rank_trace.append(float(np.mean(module_ranks)))

    latency_rounds_ms = []
    for _ in range(timing_rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            trajectory_forward(model, kind, initial, times, direction)
        end.record()
        torch.cuda.synchronize()
        latency_rounds_ms.append(start.elapsed_time(end) / repeats)
    latency_ms = float(np.mean(latency_rounds_ms))

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_flops=True) as prof:
        trajectory_forward(model, kind, initial, times, direction)
        torch.cuda.synchronize()
    profiled_flops = float(sum(event.flops for event in prof.key_averages()))

    for handle in handles:
        handle.remove()
    stored_adapter_parameters = sum(
        module.lora_A.numel() + module.lora_B.numel()
        for module in model.modules()
        if isinstance(module, (PoseidonNestedLinear, StaticLoRA_Linear))
    )
    head_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" not in name
    )
    mean_active_factor_parameters = float(np.mean([
        sum(
            int(rank) * (module.in_features + module.out_features)
            for module in model.modules()
            if isinstance(module, (PoseidonNestedLinear, StaticLoRA_Linear))
        )
        for rank in rank_trace
    ]))
    result = {
        "kind": kind,
        "nested_schedule": saved.get("nested_schedule") if kind == "stnlr" else None,
        "u_mid_rank": saved.get("u_mid_rank") if saved.get("nested_schedule") == "u_shaped" else None,
        "physical_time_nodes": len(times),
        "rank_trace": rank_trace,
        "mean_active_rank": float(np.mean(rank_trace)),
        "stored_adapter_parameters": int(stored_adapter_parameters),
        "always_active_task_head_parameters": int(head_parameters),
        "mean_active_adapter_factor_parameters": mean_active_factor_parameters,
        "adapter_gmac_per_11_time_trajectory": adapter_macs_per_trajectory / 1e9,
        "profiled_gflop_per_11_time_trajectory": profiled_flops / 1e9,
        "latency_ms_per_11_time_trajectory_batch1": latency_ms,
        "latency_median_ms_per_11_time_trajectory_batch1": float(np.median(latency_rounds_ms)),
        "latency_std_ms_per_11_time_trajectory_batch1": float(np.std(latency_rounds_ms)),
        "latency_min_ms_per_11_time_trajectory_batch1": float(np.min(latency_rounds_ms)),
        "latency_max_ms_per_11_time_trajectory_batch1": float(np.max(latency_rounds_ms)),
        "latency_rounds_ms_per_11_time_trajectory_batch1": latency_rounds_ms,
        "latency_ms_per_field_batch1": latency_ms / len(times),
        "peak_inference_memory_mib": peak_bytes / 2**20,
        "repeats": repeats,
        "timing_rounds": timing_rounds,
    }
    del model
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--name", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--trace-metrics", type=Path, action="append",
        help="Optional evaluation metrics JSON whose rank_trace overrides the checkpoint trace.",
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--timing-rounds", type=int, default=5)
    args = parser.parse_args()
    if len(args.checkpoint) != len(args.name):
        raise ValueError("--checkpoint and --name counts must match")
    if args.trace_metrics is not None and len(args.trace_metrics) != len(args.name):
        raise ValueError("--trace-metrics must be omitted or supplied once per --name")
    traces = args.trace_metrics or [None] * len(args.name)
    result = {
        name: benchmark_one(checkpoint, args.data, args.repeats, args.timing_rounds, trace)
        for name, checkpoint, trace in zip(args.name, args.checkpoint, traces)
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
