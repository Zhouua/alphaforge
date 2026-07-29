"""Create one compact AlphaForge/AlphaMining public-test comparison table."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def main(
    manifest: str,
    output: str = "public_test_comparison.csv",
):
    with open(manifest, encoding="utf-8") as file:
        paths = json.load(file)
    rows = []
    for method, path in paths.items():
        with open(path, encoding="utf-8") as file:
            result = json.load(file)
        signal = result["signal_metrics"]
        risk = result["portfolio_risk"]["excess_return_with_cost"]
        rows.append(
            {
                "method": method,
                "test_ic": signal["ic"],
                "test_icir": signal["icir"],
                "test_rank_ic": signal["rank_ic"],
                "test_rank_icir": signal["rank_icir"],
                "annualized_excess_return_with_cost": risk.get(
                    "annualized_return"
                ),
                "information_ratio_with_cost": risk.get(
                    "information_ratio"
                ),
                "max_drawdown_with_cost": risk.get("max_drawdown"),
            }
        )
    frame = pd.DataFrame(rows).sort_values("method")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    import fire

    fire.Fire(main)
