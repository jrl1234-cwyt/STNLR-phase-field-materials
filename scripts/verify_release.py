#!/usr/bin/env python3
"""Verify the SHA-256 release manifest."""

from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[1]
manifest = ROOT / "MANIFEST.sha256"
failures = []
checked = 0
for line in manifest.read_text().splitlines():
    if not line.strip():
        continue
    expected, rel = line.split("  ", 1)
    path = ROOT / rel
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "MISSING"
    if digest != expected:
        failures.append(rel)
    checked += 1
if failures:
    raise SystemExit("checksum failure: " + ", ".join(failures))
print(f"verified {checked} files: PASS")
