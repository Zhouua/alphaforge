"""Utilities shared by leakage-safe validation and public test evaluation."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from alphagen.data.expression import *  # noqa: F403 - expression namespace
from alphagen.utils.pytorch_utils import normalize_by_day
from alphagen_generic.features import *  # noqa: F403 - expression namespace
from alphagen_qlib.stock_data import FeatureType, StockData
from experiment_protocol import PROTOCOL_VERSION, TARGET_SELL_SHIFT


def load_factor_library(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as file:
        payload = json.load(file)
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(
            "Factor library protocol mismatch: "
            f"expected {PROTOCOL_VERSION!r}, got "
            f"{payload.get('protocol_version')!r}."
        )
    if payload.get("selection_data") != "train":
        raise ValueError(
            "The factor library must declare selection_data='train'."
        )
    expressions = payload.get("exprs")
    if not isinstance(expressions, list) or not expressions:
        raise ValueError("The factor library has no expressions.")
    if len(set(expressions)) != len(expressions):
        raise ValueError("The factor library contains duplicate expressions.")
    return payload


def parse_expression(text: str) -> Expression:
    text = re.sub(r"\bopen\b", "open_", text)
    namespace = {
        name: value
        for name, value in globals().items()
        if isinstance(value, (Expression, type))
    }
    namespace["__builtins__"] = {}
    value = eval(text, namespace, {})
    if not isinstance(value, Expression):
        raise ValueError(f"Not an AlphaForge expression: {text}")
    return value


def load_stock_data(
    *,
    qlib_path: str,
    instruments: str,
    start: str,
    end: str,
    device: str,
) -> StockData:
    return StockData(
        instruments,
        start,
        end,
        max_backtrack_days=100,
        max_future_days=TARGET_SELL_SHIFT,
        raw=False,
        qlib_path={"day": str(Path(qlib_path).expanduser().resolve())},
        freq="day",
        device=torch.device(device),
    )


def evaluate_factor_frame(
    data: StockData,
    expression_texts: list[str],
) -> pd.DataFrame:
    """Evaluate and cross-sectionally z-score factors on each trading day."""
    values = []
    for text in expression_texts:
        value = parse_expression(text).evaluate(data)
        value = normalize_by_day(value)
        values.append(value)
    tensor = torch.stack(values, dim=-1)
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    columns = [f"factor_{index:03d}" for index in range(len(values))]
    frame = data.make_dataframe(tensor, columns=columns)
    frame.index.names = ["datetime", "instrument"]
    if data.max_future_days == 0:
        raw_open = data.data[data.max_backtrack_days:, FeatureType.OPEN]
    else:
        raw_open = data.data[
            data.max_backtrack_days:-data.max_future_days,
            FeatureType.OPEN,
        ]
    available_mask = (
        torch.isfinite(raw_open).detach().cpu().numpy().reshape(-1)
    )
    return frame.loc[available_mask]


def evaluate_label_frame(data: StockData) -> pd.DataFrame:
    label = target.evaluate(data)  # noqa: F405
    frame = data.make_dataframe(label, columns=["LABEL0"])
    frame.index.names = ["datetime", "instrument"]
    return frame


def make_qlib_frame(
    data: StockData,
    expression_texts: list[str],
) -> pd.DataFrame:
    features = evaluate_factor_frame(data, expression_texts)
    labels = evaluate_label_frame(data).reindex(features.index)
    frame = pd.concat({"feature": features, "label": labels}, axis=1)
    return frame.sort_index()


def prediction_metrics(
    prediction: pd.Series,
    label: pd.Series,
) -> tuple[dict, pd.DataFrame]:
    aligned = pd.concat(
        [prediction.rename("score"), label.rename("label")],
        axis=1,
    ).dropna()
    if aligned.empty:
        raise ValueError("Prediction and label have no finite overlapping rows.")

    def daily_metric(day: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "ic": day["score"].corr(day["label"], method="pearson"),
                "rank_ic": day["score"].corr(
                    day["label"], method="spearman"
                ),
            }
        )

    daily = aligned.groupby(level="datetime", sort=True).apply(daily_metric)
    result = {}
    for name in ("ic", "rank_ic"):
        values = daily[name].dropna()
        mean = float(values.mean())
        std = float(values.std(ddof=1))
        result[name] = mean
        result[f"{name}_std"] = std
        result[f"{name}ir"] = mean / std if std > 0 else np.nan
    result["n_dates"] = int(len(daily))
    result["n_samples"] = int(len(aligned))
    return result, daily


def predict_from_frozen(
    features: pd.DataFrame,
    coefficients: list[float],
    intercept: float,
) -> pd.Series:
    coefficient_array = np.asarray(coefficients, dtype=np.float64)
    if features.shape[1] != len(coefficient_array):
        raise ValueError(
            f"Expected {len(coefficient_array)} factors, got "
            f"{features.shape[1]}."
        )
    score = features.to_numpy(dtype=np.float64) @ coefficient_array + intercept
    return pd.Series(score, index=features.index, name="score")
