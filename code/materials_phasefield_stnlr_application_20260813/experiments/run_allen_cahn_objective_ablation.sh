#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 GPU_ID SEED_LIST RELEASE_ROOT RESULT_ROOT" >&2
  exit 2
fi

gpu_id="$1"
seed_list="$2"
release_root="$3"
result_root="$4"
target="eps0022_lam14"
trainer="$release_root/code/materials_phasefield_stnlr_application_20260813/experiments/train_poseidon_physics_calibrated_nested.py"
data="$release_root/data/full/allen_cahn/${target}.npz"

IFS=',' read -r -a seeds <<< "$seed_list"
for seed in "${seeds[@]}"; do
  static_checkpoint="$release_root/checkpoints/allen_cahn/static_rank16_${target}_seed${seed}.pt"
  if [[ ! -s "$static_checkpoint" ]]; then
    echo "missing paired static checkpoint: $static_checkpoint" >&2
    echo "The release supplies seed 0 as the representative checkpoint; generate the other seeds with the documented static-rank-16 protocol before a full three-seed rerun." >&2
    exit 3
  fi
  for variant in full_stnlr no_field_distillation no_spectral_distillation no_material_objectives; do
    out="$result_root/objective_ablation/$target/seed${seed}/$variant"
    mkdir -p "$out"
    if [[ -s "$out/metrics.json" && -s "$out/final.pt" ]]; then
      echo "skip complete $target seed=$seed variant=$variant"
      continue
    fi
    extra=(--distill-weight 1.0 --spectral-weight 0.001 --energy-weight 0.04 --interface-weight 0.02)
    if [[ "$variant" == "no_field_distillation" ]]; then
      extra=(--distill-weight 0.0 --spectral-weight 0.001 --energy-weight 0.04 --interface-weight 0.02)
    elif [[ "$variant" == "no_spectral_distillation" ]]; then
      extra=(--distill-weight 1.0 --spectral-weight 0.0 --energy-weight 0.04 --interface-weight 0.02)
    elif [[ "$variant" == "no_material_objectives" ]]; then
      extra=(--distill-weight 1.0 --spectral-weight 0.001 --energy-weight 0.0 --interface-weight 0.0)
    fi
    echo "START $(date --iso-8601=seconds) gpu=$gpu_id target=$target seed=$seed variant=$variant"
    CUDA_VISIBLE_DEVICES="$gpu_id" python -u "$trainer" \
      --data "$data" --static-checkpoint "$static_checkpoint" --out-dir "$out" \
      --steps 600 --batch-size 4 --lr 0.0003 --seed "$seed" \
      --train-count 100 --eval-start 100 --eval-count 20 \
      --distill-target current_rank16 "${extra[@]}" \
      > "$out/train.log" 2>&1
    echo "DONE $(date --iso-8601=seconds) gpu=$gpu_id target=$target seed=$seed variant=$variant"
  done
done
