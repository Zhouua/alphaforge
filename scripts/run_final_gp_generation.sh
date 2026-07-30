#!/usr/bin/env bash
set -euo pipefail

: "${ALPHAFORGE_QLIB_PATH:?Set ALPHAFORGE_QLIB_PATH to Qlib cn_data}"

SEEDS="${GP_SEEDS:-0}"
DEVICE="${GP_DEVICE:-cuda:0}"
POPULATION="${GP_POPULATION_SIZE:-1000}"
GENERATIONS="${GP_GENERATIONS:-40}"
LIBRARY_SIZE="${GP_LIBRARY_SIZE:-100}"
MIN_FACTORS="${GP_MIN_FACTORS:-50}"

if (( MIN_FACTORS < 50 )); then
  echo "GP_MIN_FACTORS must be at least 50." >&2
  exit 2
fi
if (( LIBRARY_SIZE < MIN_FACTORS )); then
  echo "GP_LIBRARY_SIZE must be >= GP_MIN_FACTORS." >&2
  exit 2
fi

mkdir -p logs

for seed in ${SEEDS}; do
  PYTHONUNBUFFERED=1 python train_GP.py \
    --instrument=csi300 \
    --train_start=2010-01-01 \
    --train_end=2019-11-30 \
    --valid_start=2020-01-01 \
    --valid_end=2021-11-30 \
    --qlib_path="${ALPHAFORGE_QLIB_PATH}" \
    --seed="[${seed}]" \
    --population_size="${POPULATION}" \
    --generations="${GENERATIONS}" \
    --library_size="${LIBRARY_SIZE}" \
    --min_factors="${MIN_FACTORS}" \
    --device="${DEVICE}" \
    2>&1 | tee "logs/gp_seed${seed}.log"
done
