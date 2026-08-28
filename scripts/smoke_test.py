#!/usr/bin/env python3
"""Lightweight release smoke test without loading the external backbone."""

from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

required = [
    ROOT / "data/full/allen_cahn/eps0022_lam14.npz",
    ROOT / "data/full/cahn_hilliard/eps0020_lam10.npz",
    ROOT / "data/full/pfhub3/standard.npz",
    ROOT / "results/allen_cahn_main.json",
    ROOT / "checkpoints/allen_cahn/stnlr_eps0022_lam14_seed0.pt",
]
for path in required:
    if not path.is_file():
        raise FileNotFoundError(path)

for path in required[:3]:
    with np.load(path) as data:
        if not data.files:
            raise RuntimeError(f"empty dataset: {path}")
        print(path.relative_to(ROOT), {k: data[k].shape for k in data.files})

json.loads((ROOT / "results/allen_cahn_main.json").read_text())
print("release smoke test: PASS")
