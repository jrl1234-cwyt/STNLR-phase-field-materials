#!/usr/bin/env python3
"""Validate and normalize the user-supplied Figure 1 SVG for the manuscript."""

from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET


REPLACEMENTS: dict[str, str] = {}


REQUIRED_TEXT = (
    "Physical-time-adaptive nested low-rank transfer",
    "E[u; θ]",
    "∂q/∂τ = H(q; θ)",
    "Nested low-rank",
    "r = {4, 8, 16}",
    "τ ↦ r(τ)",
    "shared bank in attention / MLP maps",
    "q̂τ",
    "structure factor",
    "spectral distillation",
    "D",
    "min",
    "rⱼ⋆ = min Fⱼ",
)

STYLE_REPLACEMENTS: dict[str, str] = {}


def normalize(source: Path, output: Path) -> None:
    text = source.read_text(encoding="utf-8")
    for old, new in REPLACEMENTS.items():
        if old not in text:
            raise RuntimeError(f"Expected SVG fragment was not found: {old}")
        text = text.replace(old, new, 1)
    for old, new in STYLE_REPLACEMENTS.items():
        if old not in text:
            raise RuntimeError(f"Expected SVG style was not found: {old}")
        text = text.replace(old, new)
    ET.fromstring(text)
    for fragment in REQUIRED_TEXT:
        if fragment not in text:
            raise RuntimeError(f"Required manuscript term is missing: {fragment}")
    if 'width="1600" height="880" viewBox="0 0 1600 880"' not in text:
        raise RuntimeError("Unexpected Figure 1 canvas or viewBox")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(f"validated and wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    normalize(args.source, args.output)


if __name__ == "__main__":
    main()
