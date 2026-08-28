#!/usr/bin/env python3
"""Print a compact inventory of the machine-readable result records."""

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
for path in sorted((ROOT / "results").glob("*.json")):
    payload = json.loads(path.read_text())
    keys = ", ".join(list(payload)[:8]) if isinstance(payload, dict) else type(payload).__name__
    print(f"{path.name}: {keys}")
