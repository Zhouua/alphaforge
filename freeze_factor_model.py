"""Fit and freeze the common Qlib Ridge model without loading test data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from aligned_factor_data import (
    load_factor_library,
    load_stock_data,
    make_qlib_frame,
    prediction_metrics,
)
from experiment_protocol import (
    INSTRUMENTS,
    MODEL,
    PROTOCOL_VERSION,
    SPLITS,
    protocol_dict,
)


def _device(name: str) -> str:
    if name == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {name}")
    return name


def main(
    factor_library: str,
    qlib_path: str,
    output_dir: str = "frozen_factor_model",
    instruments: str = INSTRUMENTS,
    train_start: str = SPLITS.train_start,
    train_end: str = SPLITS.train_end,
    valid_start: str = SPLITS.valid_start,
    valid_end: str = SPLITS.valid_end,
    max_factors: int = 100,
    device: str = "auto",
):
    """Fit Ridge on train, report validation, and freeze all model choices."""
    library = load_factor_library(factor_library)
    expressions = library["exprs"][:max_factors]
    if not expressions:
        raise ValueError("No factors remain after max_factors.")
    resolved_device = _device(device)

    train_data = load_stock_data(
        qlib_path=qlib_path,
        instruments=instruments,
        start=train_start,
        end=train_end,
        device=resolved_device,
    )
    valid_data = load_stock_data(
        qlib_path=qlib_path,
        instruments=instruments,
        start=valid_start,
        end=valid_end,
        device=resolved_device,
    )
    train_frame = make_qlib_frame(train_data, expressions)
    valid_frame = make_qlib_frame(valid_data, expressions)
    combined = pd.concat([train_frame, valid_frame]).sort_index()

    from qlib.contrib.model.linear import LinearModel
    from qlib.data.dataset import DataHandlerLP, DatasetH

    handler = DataHandlerLP.from_df(combined)
    dataset = DatasetH(
        handler=handler,
        segments={
            "train": (train_start, train_end),
            "valid": (valid_start, valid_end),
        },
    )
    model = LinearModel(
        estimator=MODEL.estimator,
        alpha=MODEL.alpha,
        fit_intercept=MODEL.fit_intercept,
        include_valid=MODEL.include_valid,
    )
    model.fit(dataset)
    valid_prediction = model.predict(dataset, segment="valid")
    valid_label = valid_frame["label"]["LABEL0"]
    metrics, daily = prediction_metrics(valid_prediction, valid_label)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    library_sha256 = hashlib.sha256(
        Path(factor_library).read_bytes()
    ).hexdigest()
    frozen = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "created_without_test_data": True,
        "factor_library_sha256": library_sha256,
        "expressions": expressions,
        "feature_names": [
            f"factor_{index:03d}" for index in range(len(expressions))
        ],
        "model": {
            "class": "qlib.contrib.model.linear.LinearModel",
            "estimator": MODEL.estimator,
            "alpha": MODEL.alpha,
            "fit_intercept": MODEL.fit_intercept,
            "include_valid": MODEL.include_valid,
            "coefficients": np.asarray(model.coef_).tolist(),
            "intercept": float(model.intercept_),
        },
        "data": {
            "instruments": instruments,
            "train": [train_start, train_end],
            "valid": [valid_start, valid_end],
        },
        "validation_metrics": metrics,
        "protocol": protocol_dict(include_test=False),
    }
    with open(output / "frozen_model.json", "w", encoding="utf-8") as file:
        json.dump(frozen, file, ensure_ascii=False, indent=2)
        file.write("\n")
    valid_prediction.rename("score").to_csv(
        output / "validation_predictions.csv.gz",
        compression="gzip",
    )
    daily.to_csv(output / "validation_daily_ic.csv")
    with open(output / "validation_metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(output / "frozen_model.json")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
