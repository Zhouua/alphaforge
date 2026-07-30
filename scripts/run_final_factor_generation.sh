#!/usr/bin/env bash
set -euo pipefail

: "${ALPHAFORGE_QLIB_PATH:?Set ALPHAFORGE_QLIB_PATH to Qlib cn_data}"

SEEDS="${ALPHAFORGE_SEEDS:-0}"
DEVICE="${ALPHAFORGE_DEVICE:-cuda:0}"
RUN_PREFIX="${ALPHAFORGE_RUN_PREFIX:-alphamining_aligned}"
ZOO_SIZE="${ALPHAFORGE_ZOO_SIZE:-100}"
MIN_FACTORS="${ALPHAFORGE_MIN_FACTORS:-50}"
FORCE="${ALPHAFORGE_FORCE:-0}"

if (( MIN_FACTORS < 50 )); then
  echo "ALPHAFORGE_MIN_FACTORS must be at least 50." >&2
  exit 2
fi
if (( ZOO_SIZE < MIN_FACTORS )); then
  echo "ALPHAFORGE_ZOO_SIZE must be >= ALPHAFORGE_MIN_FACTORS." >&2
  exit 2
fi

mkdir -p logs

for seed in ${SEEDS}; do
  run_dir="out/${RUN_PREFIX}_csi300_2010-01-01_2019-11-30_2020-01-01_2021-11-30_${seed}"
  library_path="${run_dir}/factor_library.json"

  if [[ -f "${library_path}" && "${FORCE}" != "1" ]]; then
    echo "Reusing existing factor library: ${library_path}"
  else
    echo "Generating AlphaForge factor library for seed ${seed}"
    PYTHONUNBUFFERED=1 python train_AFF.py \
      --instruments=csi300 \
      --train_start=2010-01-01 \
      --train_end=2019-11-30 \
      --valid_start=2020-01-01 \
      --valid_end=2021-11-30 \
      --qlib_path="${ALPHAFORGE_QLIB_PATH}" \
      --seeds="[${seed}]" \
      --save_name="${RUN_PREFIX}" \
      --zoo_size="${ZOO_SIZE}" \
      --initial_candidates=10000 \
      --candidates_per_round=1000 \
      --max_rounds=15 \
      --corr_thresh=0.7 \
      --ic_thresh=0.01 \
      --icir_thresh=0.1 \
      --device="${DEVICE}" \
      2>&1 | tee "logs/${RUN_PREFIX}_seed${seed}.log"
  fi

  python - "${run_dir}" "${MIN_FACTORS}" <<'PY'
import csv
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
minimum = int(sys.argv[2])
library_path = run_dir / "factor_library.json"
metadata_path = run_dir / "run_metadata.json"

library = json.loads(library_path.read_text(encoding="utf-8"))
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
expressions = library["exprs"]
scores = library["train_scores"]

if library.get("selection_data") != "train":
    raise SystemExit("Refusing library not selected on train.")
if metadata.get("test_data_loaded") is not False:
    raise SystemExit("Refusing library whose run loaded test data.")
if len(expressions) != len(scores):
    raise SystemExit("Expressions and train scores are misaligned.")
if len(expressions) < minimum:
    raise SystemExit(
        f"Only {len(expressions)} factors were generated; require >= {minimum}."
    )

csv_path = run_dir / "factor_library_for_db.csv"
with csv_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "factor_id",
            "method",
            "seed",
            "expression",
            "train_ic",
            "universe",
            "target",
            "train_start",
            "train_end",
            "protocol_version",
        ],
    )
    writer.writeheader()
    for index, (expression, score) in enumerate(
        zip(expressions, scores), start=1
    ):
        writer.writerow(
            {
                "factor_id": (
                    f"alphaforge_o2o10_seed{library['seed']}_{index:03d}"
                ),
                "method": "AlphaForge",
                "seed": library["seed"],
                "expression": expression,
                "train_ic": score,
                "universe": "csi300",
                "target": "Ref($open, -11) / Ref($open, -1) - 1",
                "train_start": "2010-01-01",
                "train_end": "2019-11-30",
                "protocol_version": library["protocol_version"],
            }
        )

print(f"PASS: {len(expressions)} factors")
print(f"JSON: {library_path}")
print(f"DB CSV: {csv_path}")
PY
done
