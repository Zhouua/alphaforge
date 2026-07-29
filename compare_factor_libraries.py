"""Fairly compare candidate expression libraries from AFF, GP, DSO, and LLM."""

import json
import pickle
import re
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from alphagen.data.expression import *
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr
from alphagen.utils.pytorch_utils import normalize_by_day
from alphagen_generic.features import *
from gan.utils.data import get_data_by_dates


def _extract_json_expressions(raw):
    if isinstance(raw, list):
        return raw
    if "exprs" in raw:
        return raw["exprs"]
    if "result" in raw and "candidate_library" in raw["result"]:
        return raw["result"]["candidate_library"]["exprs"]
    if "result" in raw and "pool" in raw["result"]:
        return raw["result"]["pool"]["exprs"]
    if "pool" in raw and isinstance(raw["pool"], dict):
        return raw["pool"]["exprs"]
    if "cache" in raw:
        return list(raw["cache"])
    raise ValueError("JSON has no supported expression list or GP cache.")


def _load_expressions(path):
    path = Path(path)
    if path.suffix == ".json":
        with open(path, encoding="utf-8") as file:
            expressions = _extract_json_expressions(json.load(file))
    elif path.suffix == ".csv":
        frame = pd.read_csv(path)
        column = "exprs" if "exprs" in frame else "exprs_str"
        expressions = frame[column].tolist()
    elif path.suffix in {".pkl", ".pickle"}:
        with open(path, "rb") as file:
            raw = pickle.load(file)
        if not hasattr(raw, "exprs"):
            raise ValueError(f"{path} does not contain an AlphaForge Builders object.")
        expressions = raw.exprs
    else:
        raise ValueError(f"Unsupported factor-library format: {path.suffix}")

    result = []
    seen = set()
    for expression in expressions:
        if expression is None:
            continue
        text = str(expression)
        if text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _parse_expression(text):
    # Serialized AlphaForge expressions call the OPEN feature "open", whereas
    # the Python variable is deliberately named open_ to avoid shadowing.
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


def _expression_length(text):
    text = re.sub(r"\bopen\b", "open_", text)
    root = ast.parse(text, mode="eval").body

    def count(node):
        if isinstance(node, ast.Call):
            return 1 + sum(count(arg) for arg in node.args)
        if isinstance(node, ast.BinOp):
            return 1 + count(node.left) + count(node.right)
        if isinstance(node, ast.UnaryOp):
            return 1 + count(node.operand)
        if isinstance(node, (ast.Name, ast.Constant)):
            return 1
        return 1 + sum(count(child) for child in ast.iter_child_nodes(node))

    return count(root)


def _rank_ic(prediction, label, chunk_size=32):
    values = []
    for start in range(0, len(prediction), chunk_size):
        values.append(
            batch_spearmanr(
                prediction[start:start + chunk_size],
                label[start:start + chunk_size],
            )
        )
    return torch.cat(values)


def _metrics(prediction, label):
    ic = batch_pearsonr(prediction, label)
    rank_ic = _rank_ic(prediction, label)
    ic_mean = ic.mean().item()
    ic_std = ic.std(unbiased=True).item()
    rank_ic_mean = rank_ic.mean().item()
    rank_ic_std = rank_ic.std(unbiased=True).item()
    return {
        "ic": ic_mean,
        "ic_std": ic_std,
        "icir": ic_mean / ic_std if ic_std > 0 else np.nan,
        "rank_ic": rank_ic_mean,
        "rank_ic_std": rank_ic_std,
        "rank_icir": (
            rank_ic_mean / rank_ic_std if rank_ic_std > 0 else np.nan
        ),
    }


def _evaluate_expression(expression, data):
    value = expression.evaluate(data).clone()
    value[~torch.isfinite(value)] = torch.nan
    return normalize_by_day(value)


def main(
    manifest: str,
    qlib_path: str,
    top_k: int = 10,
    max_library_size: int = 100,
    max_expression_length: int = 20,
    output_dir: str = "library_comparison_results",
    instruments: str = "csi300",
    train_start: str = "2010-01-01",
    train_end: str = "2019-11-30",
    valid_start: str = "2020-01-01",
    valid_end: str = "2021-11-30",
    test_start: str = "2022-01-01",
    test_end: str = "2025-12-31",
):
    """Compare libraries using one common validation selector and combiner.

    The manifest is a JSON mapping method names to a JSON/CSV/PKL library path.
    For each method, expressions are ranked by absolute validation IC. Their
    signs are chosen on validation, frozen, and the top-k normalized factors
    are equally weighted. Test data never participates in selection.
    """
    with open(manifest, encoding="utf-8") as file:
        libraries = json.load(file)
    if not libraries:
        raise ValueError("The factor-library manifest is empty.")

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

    factor_rows = []
    summaries = []
    yearly_rows = []
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for method, library_path in libraries.items():
        candidates = _load_expressions(library_path)
        if len(candidates) > max_library_size:
            raise ValueError(
                f"{method} contains {len(candidates)} candidates, exceeding "
                f"max_library_size={max_library_size}. Preselect the library "
                "using train only so every method gets the same capacity."
            )
        valid_candidates = []
        for text in candidates:
            try:
                expression_length = _expression_length(text)
                if expression_length > max_expression_length:
                    continue
                expression = _parse_expression(text)
                raw_prediction = expression.evaluate(valid_data).clone()
                label_mask = torch.isfinite(valid_label)
                coverage = (
                    (torch.isfinite(raw_prediction) & label_mask).sum().item()
                    / label_mask.sum().item()
                )
                raw_prediction[~torch.isfinite(raw_prediction)] = torch.nan
                prediction = normalize_by_day(raw_prediction)
                valid_ic = batch_pearsonr(
                    prediction, valid_label
                ).mean().item()
            except Exception as error:
                factor_rows.append(
                    {
                        "method": method,
                        "expression": text,
                        "status": f"invalid: {type(error).__name__}",
                    }
                )
                continue
            if not np.isfinite(valid_ic) or coverage < 0.8:
                continue
            valid_candidates.append(
                {
                    "text": text,
                    "expression": expression,
                    "orientation": 1. if valid_ic >= 0 else -1.,
                    "valid_ic": valid_ic,
                    "coverage": coverage,
                }
            )

        valid_candidates.sort(
            key=lambda item: abs(item["valid_ic"]),
            reverse=True,
        )
        selected = valid_candidates[:top_k]
        if not selected:
            raise ValueError(f"{method} has no valid expressions after filtering.")

        valid_values = []
        test_values = []
        for rank, item in enumerate(selected, start=1):
            orientation = item["orientation"]
            valid_prediction = (
                _evaluate_expression(item["expression"], valid_data)
                * orientation
            )
            test_prediction = (
                _evaluate_expression(item["expression"], test_data)
                * orientation
            )
            valid_metrics = _metrics(valid_prediction, valid_label)
            test_metrics = _metrics(test_prediction, test_label)
            valid_values.append(valid_prediction)
            test_values.append(test_prediction)
            factor_rows.append(
                {
                    "method": method,
                    "rank": rank,
                    "expression": item["text"],
                    "orientation": orientation,
                    "status": "selected",
                    **{f"valid_{key}": value for key, value in valid_metrics.items()},
                    **{f"test_{key}": value for key, value in test_metrics.items()},
                }
            )

        valid_ensemble = normalize_by_day(torch.stack(valid_values).mean(dim=0))
        test_ensemble = normalize_by_day(torch.stack(test_values).mean(dim=0))
        valid_ensemble_metrics = _metrics(valid_ensemble, valid_label)
        test_ensemble_metrics = _metrics(test_ensemble, test_label)
        if test_data.max_future_days == 0:
            test_dates = test_data._dates[test_data.max_backtrack_days:]
        else:
            test_dates = test_data._dates[
                test_data.max_backtrack_days:-test_data.max_future_days
            ]
        for year in sorted(set(test_dates.year)):
            year_mask = torch.as_tensor(
                np.asarray(test_dates.year == year),
                dtype=torch.bool,
                device=test_ensemble.device,
            )
            yearly_rows.append(
                {
                    "method": method,
                    "year": int(year),
                    **_metrics(
                        test_ensemble[year_mask],
                        test_label[year_mask],
                    ),
                }
            )
        summaries.append(
            {
                "method": method,
                "candidates_loaded": len(candidates),
                "candidates_valid": len(valid_candidates),
                "selected": len(selected),
                **{
                    f"valid_{key}": value
                    for key, value in valid_ensemble_metrics.items()
                },
                **{
                    f"test_{key}": value
                    for key, value in test_ensemble_metrics.items()
                },
            }
        )
        torch.save(
            valid_ensemble.detach().cpu(),
            output_path / f"{method}_pred_valid.pt",
        )
        torch.save(
            test_ensemble.detach().cpu(),
            output_path / f"{method}_pred_test.pt",
        )

    factor_frame = pd.DataFrame(factor_rows)
    summary_frame = pd.DataFrame(summaries)
    yearly_frame = pd.DataFrame(yearly_rows)
    factor_frame.to_csv(output_path / "selected_factors.csv", index=False)
    summary_frame.to_csv(output_path / "summary.csv", index=False)
    yearly_frame.to_csv(output_path / "test_by_year.csv", index=False)
    with open(output_path / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summaries, file, ensure_ascii=False, indent=2)
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    import fire

    fire.Fire(main)
