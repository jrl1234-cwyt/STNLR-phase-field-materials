#!/usr/bin/env python3
"""Create machine-readable metadata for all supplied NumPy datasets."""

from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
records = []
for path in sorted((ROOT / "data/full").rglob("*.npz")):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path) as data:
        arrays = {
            key: {"shape": list(data[key].shape), "dtype": str(data[key].dtype)}
            for key in data.files
        }
    records.append(
        {
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": digest,
            "arrays": arrays,
        }
    )
(ROOT / "data/MANIFEST.json").write_text(json.dumps({"datasets": records}, indent=2) + "\n")
print(f"indexed {len(records)} datasets")
