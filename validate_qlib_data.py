"""Read-only preflight for the train/validation portion of Qlib cn_data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_protocol import FEATURES, INSTRUMENTS, SPLITS


def main(
    qlib_path: str,
    instruments: str = INSTRUMENTS,
):
    import qlib
    from qlib.config import REG_CN
    from qlib.data import D

    provider = str(Path(qlib_path).expanduser().resolve())
    if not Path(provider).is_dir():
        raise FileNotFoundError(provider)
    qlib.init(provider_uri=provider, region=REG_CN)

    calendar = pd.DatetimeIndex(
        D.calendar(
            start_time=SPLITS.train_start,
            end_time=SPLITS.valid_end,
            freq="day",
        )
    )
    if calendar.empty:
        raise ValueError("No train/validation daily calendar is available.")
    sample_end = calendar[min(4, len(calendar) - 1)]
    fields = [f"${name}" for name in FEATURES]
    sample = D.features(
        D.instruments(instruments),
        fields,
        start_time=calendar[0],
        end_time=sample_end,
        freq="day",
    )
    non_null = sample.notna().sum()
    missing = [
        field for field, count in non_null.items() if int(count) == 0
    ]
    if missing:
        raise ValueError(f"Qlib fields have no sample data: {missing}")
    benchmark = D.features(
        ["SH000300"],
        ["$open"],
        start_time=calendar[0],
        end_time=sample_end,
        freq="day",
    )
    if benchmark["$open"].notna().sum() == 0:
        raise ValueError("SH000300 benchmark open data is unavailable.")

    result = {
        "provider": provider,
        "instruments": instruments,
        "features": list(FEATURES),
        "train_valid_calendar_start": str(calendar[0].date()),
        "train_valid_calendar_end": str(calendar[-1].date()),
        "train_valid_trading_days": int(len(calendar)),
        "sample_rows": int(len(sample)),
        "sample_instruments": int(
            sample.index.get_level_values("instrument").nunique()
        ),
        "finite_values_by_field": {
            str(field): int(np.isfinite(sample[field]).sum())
            for field in sample.columns
        },
        "test_data_read": False,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import fire

    fire.Fire(main)
