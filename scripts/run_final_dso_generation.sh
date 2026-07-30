#!/usr/bin/env bash
set -euo pipefail

: "${ALPHAFORGE_QLIB_PATH:?Set ALPHAFORGE_QLIB_PATH to Qlib cn_data}"

SEEDS="${DSO_SEEDS:-0}"
DEVICE="${DSO_DEVICE:-cuda:0}"
N_SAMPLES="${DSO_N_SAMPLES:-20000}"
BATCH_SIZE="${DSO_BATCH_SIZE:-128}"
LIBRARY_SIZE="${DSO_LIBRARY_SIZE:-100}"
MIN_FACTORS="${DSO_MIN_FACTORS:-50}"

if (( MIN_FACTORS < 50 )); then
  echo "DSO_MIN_FACTORS must be at least 50." >&2
  exit 2
fi
if (( LIBRARY_SIZE < MIN_FACTORS )); then
  echo "DSO_LIBRARY_SIZE must be >= DSO_MIN_FACTORS." >&2
  exit 2
fi

mkdir -p logs

for seed in ${SEEDS}; do
  PYTHONUNBUFFERED=1 python train_DSO.py \
    --instrument=csi300 \
    --train_start=2010-01-01 \
    --train_end=2019-11-30 \
    --valid_start=2020-01-01 \
    --valid_end=2021-11-30 \
    --qlib_path="${ALPHAFORGE_QLIB_PATH}" \
    --seeds="[${seed}]" \
    --n_samples="${N_SAMPLES}" \
    --batch_size="${BATCH_SIZE}" \
    --library_size="${LIBRARY_SIZE}" \
    --min_factors="${MIN_FACTORS}" \
    --device="${DEVICE}" \
    2>&1 | tee "logs/dso_seed${seed}.log"
done
