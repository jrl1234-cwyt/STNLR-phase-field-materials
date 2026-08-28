#!/usr/bin/env bash
set -euo pipefail

release_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
external_root="${1:-$release_root/external}"
mkdir -p "$external_root"

if [ ! -d "$external_root/poseidon/.git" ]; then
  git clone https://github.com/camlab-ethz/poseidon.git "$external_root/poseidon"
fi
git -C "$external_root/poseidon" fetch --all --tags
git -C "$external_root/poseidon" checkout b8fa28f59bd7f7673323f28d11a12c6f3a215c61

hf download camlab-ethz/Poseidon-T --local-dir "$external_root/Poseidon-T"
weight_file="$(find "$external_root/Poseidon-T" -maxdepth 1 -type f \( -name '*.safetensors' -o -name 'pytorch_model.bin' \) | head -n 1)"
if [ -n "$weight_file" ]; then
  echo "a363f7317fbc3a900a318fc63cc53197705d95fce0e0ce28dd3c8844a89112e2  $weight_file" | sha256sum --check
fi
