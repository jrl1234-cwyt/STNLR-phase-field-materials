#!/usr/bin/env python3
"""Re-evaluate a frozen TimeStep Master checkpoint on paired test trajectories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from materials_phasefield_stnlr_application_20260813.experiments.train_evaluate_poseidon_allen_cahn import load_data
from materials_phasefield_stnlr_application_20260813.experiments.train_poseidon_strict_published_baselines import (
    build_student,
    evaluate,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-start", type=int, default=120)
    parser.add_argument("--eval-count", type=int, default=40)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--poseidon-code", type=Path, default=Path("/tmp/poseidon-stnlr"))
    parser.add_argument("--poseidon-checkpoint", type=Path, default=Path("/tmp/poseidon_model"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    data = load_data(args.data)
    build_args = SimpleNamespace(
        method="timestep_master",
        poseidon_code=args.poseidon_code,
        poseidon_checkpoint=args.poseidon_checkpoint,
    )
    model, replaced = build_student(build_args, device, len(data["times"]))
    payload = torch.load(args.checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected checkpoint keys: {unexpected[:8]}")
    model.eval()
    summary, per_sample = evaluate(
        model,
        "timestep_master",
        data,
        args.eval_start,
        args.eval_count,
        args.batch_size,
        device,
    )
    result = {
        "method": "timestep_master",
        "seed": args.seed,
        "test_start": args.eval_start,
        "test_count": args.eval_count,
        "metrics": summary,
        "per_sample_records": per_sample,
        "checkpoint": str(args.checkpoint),
        "replaced_linear_layers": len(replaced),
        "missing_frozen_or_reconstructed_keys": list(missing),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"metrics": summary, "out": str(args.out)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
