#!/usr/bin/env python3
"""Rebuild the deterministic SHA-256 manifest for the complete release."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "MANIFEST.sha256"


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return (
        path.is_file()
        and relative.as_posix() != "MANIFEST.sha256"
        and ".git" not in relative.parts
        and "__pycache__" not in relative.parts
        and path.suffix != ".pyc"
    )


rows = []
for path in sorted((item for item in ROOT.rglob("*") if included(item)), key=lambda p: p.relative_to(ROOT).as_posix()):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    rows.append(f"{digest}  {path.relative_to(ROOT).as_posix()}")
OUTPUT.write_text("\n".join(rows) + "\n", encoding="utf-8")
print(f"wrote {OUTPUT} with {len(rows)} files")
