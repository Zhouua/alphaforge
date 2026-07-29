"""Public test-only evaluator for a model frozen on train/validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from aligned_factor_data import (
    evaluate_factor_frame,
    evaluate_label_frame,
    load_stock_data,
    predict_from_frozen,
    prediction_metrics,
)
from experiment_protocol import (
    BACKTEST,
    PROTOCOL_VERSION,
    SPLITS,
)


def _device(name: str) -> str:
    if name == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {name}")
    return name


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def _risk_dict(report: pd.DataFrame) -> dict:
    from qlib.contrib.evaluate import risk_analysis

    analyses = {
        "benchmark": risk_analysis(report["bench"], freq="1day"),
        "excess_return_without_cost": risk_analysis(
            report["return"] - report["bench"],
            freq="1day",
        ),
        "excess_return_with_cost": risk_analysis(
            report["return"] - report["bench"] - report["cost"],
            freq="1day",
        ),
    }
    return {
        name: _jsonable(frame["risk"].to_dict())
        for name, frame in analyses.items()
    }


def main(
    frozen_model: str,
    qlib_path: str,
    output_dir: str = "public_test_results",
    test_start: str = SPLITS.test_start,
    test_end: str = SPLITS.test_end,
    device: str = "auto",
):
    """Evaluate a frozen artifact; no model or factor selection is performed."""
    with open(frozen_model, encoding="utf-8") as file:
        frozen = json.load(file)
    if frozen.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Frozen model protocol does not match this evaluator.")
    if frozen.get("created_without_test_data") is not True:
        raise ValueError(
            "Refusing a model that does not attest test-data isolation."
        )
    model = frozen["model"]
    expected_model = {
        "estimator": "ridge",
        "alpha": 10.0,
        "fit_intercept": False,
        "include_valid": False,
    }
    for key, expected in expected_model.items():
        if model.get(key) != expected:
            raise ValueError(
                f"Frozen model has {key}={model.get(key)!r}; "
                f"expected {expected!r}."
            )

    instruments = frozen["data"]["instruments"]
    resolved_device = _device(device)
    test_data = load_stock_data(
        qlib_path=qlib_path,
        instruments=instruments,
        start=test_start,
        end=test_end,
        device=resolved_device,
    )
    features = evaluate_factor_frame(test_data, frozen["expressions"])
    if features.columns.tolist() != frozen["feature_names"]:
        raise ValueError("Frozen feature order does not match evaluated factors.")
    prediction = predict_from_frozen(
        features,
        model["coefficients"],
        model["intercept"],
    )
    labels = evaluate_label_frame(test_data)["LABEL0"]

    from qlib.data import D

    expected_dates = pd.DatetimeIndex(
        D.calendar(start_time=test_start, end_time=test_end, freq="day")
    )
    actual_dates = prediction.index.unique("datetime")
    if len(expected_dates) == 0 or not actual_dates.equals(expected_dates):
        raise ValueError(
            "The Qlib provider cannot evaluate the complete public test "
            f"calendar. Expected {len(expected_dates)} dates ending "
            f"{expected_dates[-1] if len(expected_dates) else 'N/A'}, got "
            f"{len(actual_dates)} ending "
            f"{actual_dates[-1] if len(actual_dates) else 'N/A'}. The provider "
            "must contain at least 11 trading sessions after test_end for the "
            "10-day Open-to-Open label."
        )

    signal_metrics, daily_ic = prediction_metrics(prediction, labels)

    from qlib.backtest import backtest, executor
    from qlib.contrib.strategy import TopkDropoutStrategy

    strategy = TopkDropoutStrategy(
        signal=prediction,
        topk=BACKTEST.topk,
        n_drop=BACKTEST.n_drop,
    )
    executor_obj = executor.SimulatorExecutor(
        time_per_step="day",
        generate_portfolio_metrics=True,
    )
    portfolio_metrics, _ = backtest(
        start_time=test_start,
        end_time=test_end,
        account=BACKTEST.account,
        benchmark=BACKTEST.benchmark,
        exchange_kwargs={
            "freq": "day",
            "limit_threshold": None,
            "deal_price": BACKTEST.deal_price,
            "open_cost": BACKTEST.open_cost,
            "close_cost": BACKTEST.close_cost,
            "min_cost": BACKTEST.min_cost,
        },
        executor=executor_obj,
        strategy=strategy,
    )
    if "1day" in portfolio_metrics:
        report, positions = portfolio_metrics["1day"]
    else:
        report, positions = next(iter(portfolio_metrics.values()))
    risk = _risk_dict(report)
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "frozen_model": str(Path(frozen_model).resolve()),
        "test": [test_start, test_end],
        "signal_metrics": signal_metrics,
        "portfolio_risk": risk,
        "backtest": {
            "benchmark": BACKTEST.benchmark,
            "topk": BACKTEST.topk,
            "n_drop": BACKTEST.n_drop,
            "account": BACKTEST.account,
            "deal_price": BACKTEST.deal_price,
            "open_cost": BACKTEST.open_cost,
            "close_cost": BACKTEST.close_cost,
            "min_cost": BACKTEST.min_cost,
            "limit_threshold": None,
        },
    }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(
        output / "test_predictions.csv.gz",
        compression="gzip",
    )
    daily_ic.to_csv(output / "test_daily_ic.csv")
    report.to_csv(output / "portfolio_report.csv")
    pd.to_pickle(positions, output / "positions.pkl")
    with open(output / "summary.json", "w", encoding="utf-8") as file:
        json.dump(_jsonable(summary), file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import fire

    fire.Fire(main)
