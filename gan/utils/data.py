import os
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd
import torch

from alphagen.data.expression import *
from alphagen_generic.features import *


DEFAULT_QLIB_PATH = "~/.qlib/qlib_data/cn_data"


def _resolve_qlib_path(qlib_path: Optional[str]) -> str:
    """Resolve the Qlib provider path without baking a machine-specific path in code."""
    raw_path = qlib_path or os.environ.get("ALPHAFORGE_QLIB_PATH", DEFAULT_QLIB_PATH)
    resolved = str(Path(raw_path).expanduser().resolve())
    if not Path(resolved).is_dir():
        raise FileNotFoundError(
            f"Qlib data directory does not exist: {resolved}. "
            "Pass --qlib_path or set ALPHAFORGE_QLIB_PATH."
        )
    return resolved


def _validate_splits(
    train_start: str,
    train_end: str,
    valid_start: str,
    valid_end: str,
    test_start: str,
    test_end: str,
) -> None:
    values = [
        pd.Timestamp(train_start),
        pd.Timestamp(train_end),
        pd.Timestamp(valid_start),
        pd.Timestamp(valid_end),
        pd.Timestamp(test_start),
        pd.Timestamp(test_end),
    ]
    if not (
        values[0] <= values[1]
        < values[2] <= values[3]
        < values[4] <= values[5]
    ):
        raise ValueError(
            "Expected non-overlapping chronological splits: "
            "train_start <= train_end < valid_start <= valid_end "
            "< test_start <= test_end."
        )


def _cache_name(
    instruments: str,
    target,
    freq: str,
    train_start: str,
    train_end: str,
    valid_start: str,
    valid_end: str,
    test_start: str,
    test_end: str,
    qlib_path: str,
) -> str:
    target_name = str(target).replace("/", "_").replace(" ", "")
    provider_name = Path(qlib_path).name
    return (
        f"{instruments}_pkl_{target_name}_{freq}_{provider_name}_"
        f"{train_start}_{train_end}_{valid_start}_{valid_end}_{test_start}_{test_end}"
    )


def _search_cache_name(
    instruments: str,
    target,
    freq: str,
    train_start: str,
    train_end: str,
    valid_start: str,
    valid_end: str,
    qlib_path: str,
) -> str:
    target_name = str(target).replace("/", "_").replace(" ", "")
    provider_name = Path(qlib_path).name
    return (
        f"{instruments}_search_pkl_{target_name}_{freq}_{provider_name}_"
        f"{train_start}_{train_end}_{valid_start}_{valid_end}"
    )


def get_search_data_by_dates(
    train_start: str,
    train_end: str,
    valid_start: str,
    valid_end: str,
    instruments: str,
    target,
    freq: str = "day",
    qlib_path: Optional[str] = None,
    device: str = "cuda:0",
):
    """Load only train/validation data for factor discovery.

    This deliberately has no test arguments and creates no cache spanning the
    test period.  It is the stage-1 leakage boundary.
    """
    train_start_ts = pd.Timestamp(train_start)
    train_end_ts = pd.Timestamp(train_end)
    valid_start_ts = pd.Timestamp(valid_start)
    valid_end_ts = pd.Timestamp(valid_end)
    if not (
        train_start_ts <= train_end_ts < valid_start_ts <= valid_end_ts
    ):
        raise ValueError(
            "Expected train_start <= train_end < valid_start <= valid_end."
        )

    resolved_qlib_path = _resolve_qlib_path(qlib_path)
    provider_uri = {freq: resolved_qlib_path}
    from gan.utils import load_pickle, save_pickle

    name = _search_cache_name(
        instruments,
        target,
        freq,
        train_start,
        train_end,
        valid_start,
        valid_end,
        resolved_qlib_path,
    )
    cache_dir = Path("pkl") / name
    common = dict(
        raw=False,
        qlib_path=provider_uri,
        freq=freq,
        device=torch.device("cpu"),
    )
    try:
        data = load_pickle(cache_dir / "data.pkl")
        data_valid = load_pickle(cache_dir / "data_valid.pkl")
    except (FileNotFoundError, EOFError, AttributeError):
        print(f"Search cache not found; loading from Qlib: {resolved_qlib_path}")
        data = StockData(instruments, train_start, train_end, **common)
        data_valid = StockData(instruments, valid_start, valid_end, **common)
        cache_dir.mkdir(parents=True, exist_ok=True)
        save_pickle(data, cache_dir / "data.pkl")
        save_pickle(data_valid, cache_dir / "data_valid.pkl")

    target_device = torch.device(device)
    for stock_data in (data, data_valid):
        stock_data.data = stock_data.data.to(target_device)
        stock_data.device = target_device
    return data, data_valid, name


def get_data_by_dates(
    train_start: str,
    train_end: str,
    valid_start: str,
    valid_end: str,
    test_start: str,
    test_end: str,
    instruments: str,
    target,
    freq: str = "day",
    qlib_path: Optional[str] = None,
    device: str = "cuda:0",
):
    """Load AlphaForge data with explicit, reproducible date boundaries.

    The returned objects keep AlphaForge's historical tuple layout so existing
    callers can migrate from ``get_data_by_year`` without downstream changes.
    """
    _validate_splits(
        train_start,
        train_end,
        valid_start,
        valid_end,
        test_start,
        test_end,
    )
    resolved_qlib_path = _resolve_qlib_path(qlib_path)
    provider_uri = {freq: resolved_qlib_path}

    from gan.utils import load_pickle, save_pickle

    get_data_my = StockData
    valid_head_start = str(pd.Timestamp(valid_start) - pd.DateOffset(years=2))[:10]
    test_head_start = str(pd.Timestamp(test_start) - pd.DateOffset(years=2))[:10]
    name = _cache_name(
        instruments,
        target,
        freq,
        train_start,
        train_end,
        valid_start,
        valid_end,
        test_start,
        test_end,
        resolved_qlib_path,
    )
    cache_dir = Path("pkl") / name

    try:
        data = load_pickle(cache_dir / "data.pkl")
        data_valid = load_pickle(cache_dir / "data_valid.pkl")
        data_valid_withhead = load_pickle(cache_dir / "data_valid_withhead.pkl")
        data_test = load_pickle(cache_dir / "data_test.pkl")
        data_test_withhead = load_pickle(cache_dir / "data_test_withhead.pkl")
    except (FileNotFoundError, EOFError, AttributeError):
        print(f"Data cache not found; loading from Qlib: {resolved_qlib_path}")
        common = dict(
            raw=False,
            qlib_path=provider_uri,
            freq=freq,
            device=torch.device("cpu"),
        )
        data = get_data_my(instruments, train_start, train_end, **common)
        data_valid = get_data_my(instruments, valid_start, valid_end, **common)
        data_valid_withhead = get_data_my(
            instruments, valid_head_start, valid_end, **common
        )
        data_test = get_data_my(instruments, test_start, test_end, **common)
        data_test_withhead = get_data_my(
            instruments, test_head_start, test_end, **common
        )

        cache_dir.mkdir(parents=True, exist_ok=True)
        save_pickle(data, cache_dir / "data.pkl")
        save_pickle(data_valid, cache_dir / "data_valid.pkl")
        save_pickle(data_valid_withhead, cache_dir / "data_valid_withhead.pkl")
        save_pickle(data_test, cache_dir / "data_test.pkl")
        save_pickle(data_test_withhead, cache_dir / "data_test_withhead.pkl")

    try:
        data_all = load_pickle(cache_dir / "data_all.pkl")
    except (FileNotFoundError, EOFError, AttributeError):
        data_all = get_data_my(
            instruments,
            train_start,
            test_end,
            raw=False,
            qlib_path=provider_uri,
            freq=freq,
            device=torch.device("cpu"),
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        save_pickle(data_all, cache_dir / "data_all.pkl")

    target_device = torch.device(device)
    for stock_data in (
        data_all,
        data,
        data_valid,
        data_valid_withhead,
        data_test,
        data_test_withhead,
    ):
        stock_data.data = stock_data.data.to(target_device)
        stock_data.device = target_device

    if data_test.n_days <= 0:
        raise ValueError(
            "The test split contains no evaluable days. The target needs future "
            "observations; extend the Qlib calendar/data beyond test_end."
        )
    if data_test._dates[-1] <= pd.Timestamp(test_end):
        warnings.warn(
            "Qlib does not extend beyond test_end. Because the default target "
            "uses a future price, the last test dates cannot be scored. Extend the "
            "provider beyond test_end to evaluate the complete requested range.",
            RuntimeWarning,
        )

    return (
        data_all,
        data,
        data_valid,
        data_valid_withhead,
        data_test,
        data_test_withhead,
        name,
    )


def get_data_by_year(
    train_start=2010,
    train_end=2019,
    valid_year=2020,
    test_year=2021,
    instruments=None,
    target=None,
    freq=None,
    qlib_path: Optional[str] = None,
    device: str = "cuda:0",
):
    """Backward-compatible whole-year wrapper around ``get_data_by_dates``."""
    return get_data_by_dates(
        train_start=f"{train_start}-01-01",
        train_end=f"{train_end}-12-31",
        valid_start=f"{valid_year}-01-01",
        valid_end=f"{valid_year}-12-31",
        test_start=f"{test_year}-01-01",
        test_end=f"{test_year}-12-31",
        instruments=instruments,
        target=target,
        freq=freq,
        qlib_path=qlib_path,
        device=device,
    )
