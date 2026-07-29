"""Compare AFF, GP, DSO, and LLM factor predictions on identical Qlib splits."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr
from alphagen_generic.features import target
from gan.utils.data import get_data_by_dates


def _load_prediction(path, expected_shape, device):
    value = torch.load(path, map_location=device)
    if isinstance(value, dict):
        if "prediction" not in value:
            raise ValueError(
                f"{path} is a dictionary but has no 'prediction' entry."
            )
        value = value["prediction"]
    value = torch.as_tensor(value, dtype=torch.float, device=device).squeeze()
    if tuple(value.shape) != tuple(expected_shape):
        raise ValueError(
            f"{path} has shape {tuple(value.shape)}, expected "
            f"{tuple(expected_shape)} (days, CSI300 instruments)."
        )
    value[~torch.isfinite(value)] = torch.nan
    return value


def _rank_ic_by_chunks(prediction, label, chunk_size=64):
    chunks = []
    for start in range(0, len(prediction), chunk_size):
        chunks.append(
            batch_spearmanr(
                prediction[start:start + chunk_size],
                label[start:start + chunk_size],
            )
        )
    return torch.cat(chunks)


def _summarize(prediction, label):
    ic = batch_pearsonr(prediction, label)
    rank_ic = _rank_ic_by_chunks(prediction, label)
    label_count = torch.isfinite(label).sum().item()
    joint_count = (torch.isfinite(prediction) & torch.isfinite(label)).sum().item()

    def stats(values, prefix):
        values = values[torch.isfinite(values)]
        mean = values.mean().item()
        std = values.std(unbiased=True).item()
        return {
            prefix: mean,
            f"{prefix}_std": std,
            f"{prefix}ir": mean / std if std > 0 else np.nan,
        }

    result = {}
    result.update(stats(ic, "ic"))
    result.update(stats(rank_ic, "rank_ic"))
    result["coverage"] = joint_count / label_count if label_count else 0.
    result["n_days"] = len(prediction)
    return result


def main(
    manifest: str,
    qlib_path: str,
    output_dir: str = "comparison_results",
    instruments: str = "csi300",
    train_start: str = "2010-01-01",
    train_end: str = "2019-11-30",
    valid_start: str = "2020-01-01",
    valid_end: str = "2021-11-30",
    test_start: str = "2022-01-01",
    test_end: str = "2025-12-31",
):
    """Evaluate prediction tensors listed in a JSON manifest.

    Manifest format:
    ``{"AFF": {"valid": "...pt", "test": "...pt"}, "LLM": {...}}``.
    Each tensor must be ``[days, instruments]`` in AlphaForge/Qlib ordering.
    Factor sign is selected once on validation IC and then frozen for test.
    """
    with open(manifest, encoding="utf-8") as file:
        methods = json.load(file)
    if not methods:
        raise ValueError("The prediction manifest is empty.")

    returned = get_data_by_dates(
        train_start=train_start,
        train_end=train_end,
        valid_start=valid_start,
        valid_end=valid_end,
        test_start=test_start,
        test_end=test_end,
        instruments=instruments,
        target=target,
        freq="day",
        qlib_path=qlib_path,
    )
    _, _, valid_data, _, test_data, _, _ = returned
    valid_label = target.evaluate(valid_data)
    test_label = target.evaluate(test_data)
    device = valid_label.device

    results = {}
    for method, paths in methods.items():
        if not {"valid", "test"} <= set(paths):
            raise ValueError(f"{method} must define both 'valid' and 'test' paths.")
        valid_prediction = _load_prediction(
            paths["valid"], valid_label.shape, device
        )
        test_prediction = _load_prediction(
            paths["test"], test_label.shape, device
        )

        raw_valid_ic = batch_pearsonr(
            valid_prediction, valid_label
        ).mean().item()
        orientation = 1. if raw_valid_ic >= 0 else -1.
        valid_metrics = _summarize(valid_prediction * orientation, valid_label)
        test_metrics = _summarize(test_prediction * orientation, test_label)
        results[method] = {
            "orientation_selected_on_valid": orientation,
            "valid": valid_metrics,
            "test": test_metrics,
        }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    rows = []
    for method, method_result in results.items():
        for split in ("valid", "test"):
            rows.append(
                {
                    "method": method,
                    "split": split,
                    **method_result[split],
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(output_path / "metrics.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    import fire

    fire.Fire(main)
