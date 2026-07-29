"""Export an AlphaForge Builders pickle as a portable factor library."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

from experiment_protocol import PROTOCOL_VERSION, protocol_dict


def export_builders(
    builders,
    output_path: str | Path,
    *,
    seed: int | None = None,
    run_name: str | None = None,
) -> Path:
    pairs = [
        (str(expr), float(score))
        for expr, score in zip(builders.exprs, builders.scores)
    ]
    if len(pairs) != len(builders.exprs) or len(pairs) != len(builders.scores):
        raise ValueError("AlphaForge expressions and train scores are misaligned.")
    pairs.sort(key=lambda item: item[1], reverse=True)
    expressions = [expression for expression, _ in pairs]
    scores = [score for _, score in pairs]
    payload = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "method": "AlphaForge",
        "selection_data": "train",
        "exprs": expressions,
        "train_scores": scores,
        "seed": seed,
        "run_name": run_name,
        "protocol": protocol_dict(include_test=False),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary.replace(output)
    return output


def main(
    input_path: str,
    output_path: str = "factor_library.json",
    seed: int | None = None,
    run_name: str | None = None,
):
    with open(input_path, "rb") as file:
        builders = pickle.load(file)
    result = export_builders(
        builders,
        output_path,
        seed=seed,
        run_name=run_name,
    )
    print(result)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
