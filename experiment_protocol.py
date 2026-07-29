"""Single source of truth for the AlphaForge/AlphaMining experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass


PROTOCOL_VERSION = "alphamining-aligned-csi300-o2o10-v1"


@dataclass(frozen=True)
class SplitConfig:
    train_start: str = "2010-01-01"
    train_end: str = "2019-11-30"
    valid_start: str = "2020-01-01"
    valid_end: str = "2021-11-30"
    test_start: str = "2022-01-01"
    test_end: str = "2025-12-31"


@dataclass(frozen=True)
class ModelConfig:
    estimator: str = "ridge"
    alpha: float = 10.0
    fit_intercept: bool = False
    include_valid: bool = False


@dataclass(frozen=True)
class BacktestConfig:
    benchmark: str = "SH000300"
    topk: int = 50
    n_drop: int = 5
    account: int = 100_000_000
    deal_price: str = "open"
    open_cost: float = 0.0005
    close_cost: float = 0.0015
    min_cost: float = 5.0


SPLITS = SplitConfig()
MODEL = ModelConfig()
BACKTEST = BacktestConfig()

INSTRUMENTS = "csi300"
FEATURES = ("open", "high", "low", "close", "volume", "vwap")

# Qlib Ref uses a negative offset for future observations.  A signal produced
# on day t is traded at open[t+1] and exits at open[t+11], i.e. ten sessions.
TARGET_BUY_SHIFT = 1
TARGET_SELL_SHIFT = 11
TARGET_HORIZON = TARGET_SELL_SHIFT - TARGET_BUY_SHIFT
QLIB_TARGET = "Ref($open, -11) / Ref($open, -1) - 1"


def protocol_dict(*, include_test: bool = True) -> dict:
    """Return JSON-serializable protocol metadata."""
    splits = asdict(SPLITS)
    if not include_test:
        splits.pop("test_start")
        splits.pop("test_end")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": "Qlib cn_data",
        "features": list(FEATURES),
        "feature_semantics": "provider-native Qlib fields (raw=False)",
        "instruments": INSTRUMENTS,
        "splits": splits,
        "target": {
            "name": "Open-to-Open 10D",
            "qlib_expression": QLIB_TARGET,
            "buy_shift": TARGET_BUY_SHIFT,
            "sell_shift": TARGET_SELL_SHIFT,
            "holding_trading_days": TARGET_HORIZON,
        },
        "model": asdict(MODEL),
        "strategy": asdict(BACKTEST),
    }
