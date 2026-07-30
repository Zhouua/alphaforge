"""Common factor-library output helpers for all search methods."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable

from experiment_protocol import PROTOCOL_VERSION, QLIB_TARGET, protocol_dict


def select_train_factors(
    scored_expressions: Iterable[tuple[str, float]],
    *,
    library_size: int,
    min_factors: int,
) -> list[tuple[str, float]]:
    """Deduplicate, validate, and rank a train-scored expression collection."""
    if min_factors < 50:
        raise ValueError("min_factors must be at least 50.")
    if library_size < min_factors:
        raise ValueError("library_size must be >= min_factors.")

    best_by_expression: dict[str, float] = {}
    for expression, score in scored_expressions:
        expression = str(expression).strip()
        score = float(score)
        if not expression or not math.isfinite(score) or score <= 0:
            continue
        previous = best_by_expression.get(expression)
        if previous is None or score > previous:
            best_by_expression[expression] = score

    selected = sorted(
        best_by_expression.items(),
        key=lambda item: (-item[1], item[0]),
    )[:library_size]
    if len(selected) < min_factors:
        raise RuntimeError(
            f"Only {len(selected)} valid unique factors were generated; "
            f"require at least {min_factors}. Increase the search budget."
        )
    return selected


def write_factor_library(
    scored_expressions: Iterable[tuple[str, float]],
    output_dir: str | Path,
    *,
    method: str,
    method_id: str,
    seed: int,
    run_name: str,
    library_size: int = 100,
    min_factors: int = 50,
    metadata: dict | None = None,
) -> Path:
    """Write common JSON, DB CSV, and leakage-audit metadata."""
    selected = select_train_factors(
        scored_expressions,
        library_size=library_size,
        min_factors=min_factors,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    expressions = [expression for expression, _ in selected]
    scores = [score for _, score in selected]
    payload = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "method": method,
        "selection_data": "train",
        "exprs": expressions,
        "train_scores": scores,
        "seed": seed,
        "run_name": run_name,
        "protocol": protocol_dict(include_test=False),
    }
    library_path = output / "factor_library.json"
    temporary = library_path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary.replace(library_path)

    csv_path = output / "factor_library_for_db.csv"
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
        for index, (expression, score) in enumerate(selected, start=1):
            writer.writerow(
                {
                    "factor_id": (
                        f"{method_id}_o2o10_seed{seed}_{index:03d}"
                    ),
                    "method": method,
                    "seed": seed,
                    "expression": expression,
                    "train_ic": score,
                    "universe": "csi300",
                    "target": QLIB_TARGET,
                    "train_start": "2010-01-01",
                    "train_end": "2019-11-30",
                    "protocol_version": PROTOCOL_VERSION,
                }
            )

    audit = {
        "method": method,
        "seed": seed,
        "run_name": run_name,
        "library_size": len(selected),
        "factor_library": library_path.name,
        "db_csv": csv_path.name,
        "selection_data": "train",
        "test_data_loaded": False,
    }
    if metadata:
        audit.update(metadata)
    metadata_path = output / "run_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(audit, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return library_path
