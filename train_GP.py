"""Leakage-safe GP factor generation for the aligned AlphaMining protocol."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
from pathlib import Path

import numpy as np
import torch

from alphagen.data.expression import *
from alphagen.utils.correlation import batch_pearsonr
from alphagen.utils.pytorch_utils import normalize_by_day
from alphagen.utils.random import reseed_everything
from alphagen_generic.features import (
    close,
    high,
    low,
    open_,
    target,
    volume,
    vwap,
)
from alphagen_generic.operators import funcs as generic_funcs
from experiment_protocol import INSTRUMENTS, SPLITS
from factor_library_io import write_factor_library
from gan.utils.data import get_search_data_by_dates
from gplearn.fitness import make_fitness
from gplearn.functions import make_function
from gplearn.genetic import SymbolicRegressor
from symbolic_search_config import (
    MAX_EXPRESSION_LENGTH,
    required_backtrack_days,
)


EXPRESSION_NAMESPACE = {
    name: value
    for name, value in globals().items()
    if isinstance(value, (Expression, type))
}
EXPRESSION_NAMESPACE["__builtins__"] = {}


def parse_expression(text: str) -> Expression:
    text = re.sub(r"\bopen\b", "open_", text)
    expression = eval(text, EXPRESSION_NAMESPACE, {})
    if not isinstance(expression, Expression):
        raise ValueError(f"Not an AlphaForge expression: {text}")
    return expression


def expression_length(text: str) -> int:
    root = ast.parse(re.sub(r"\bopen\b", "open_", text), mode="eval").body

    def count(node) -> int:
        if isinstance(node, ast.Call):
            return 1 + sum(count(argument) for argument in node.args)
        if isinstance(node, ast.BinOp):
            return 1 + count(node.left) + count(node.right)
        if isinstance(node, ast.UnaryOp):
            return 1 + count(node.operand)
        if isinstance(node, (ast.Name, ast.Constant)):
            return 1
        return 1 + sum(count(child) for child in ast.iter_child_nodes(node))

    return count(root)


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instrument", default=INSTRUMENTS)
    parser.add_argument("--seed", default="[0]")
    parser.add_argument("--freq", default="day")
    parser.add_argument("--cuda", default="0")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train_start", default=SPLITS.train_start)
    parser.add_argument("--train_end", default=SPLITS.train_end)
    parser.add_argument("--valid_start", default=SPLITS.valid_start)
    parser.add_argument("--valid_end", default=SPLITS.valid_end)
    parser.add_argument("--test_start", default=None)
    parser.add_argument("--test_end", default=None)
    parser.add_argument("--qlib_path", default=None)
    parser.add_argument("--population_size", type=int, default=1000)
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument(
        "--max_expression_length",
        type=int,
        default=MAX_EXPRESSION_LENGTH,
    )
    parser.add_argument("--library_size", type=int, default=100)
    parser.add_argument("--min_factors", type=int, default=50)
    parser.add_argument("--output_root", default="out_gp")
    args = parser.parse_args()
    if args.test_start is not None or args.test_end is not None:
        parser.error(
            "GP generation must not receive test dates. Test belongs only to "
            "the public evaluator."
        )
    try:
        args.seed = ast.literal_eval(args.seed)
    except (SyntaxError, ValueError) as error:
        parser.error(f"--seed must be a Python-style integer list: {error}")
    if isinstance(args.seed, int):
        args.seed = [args.seed]
    if not isinstance(args.seed, (list, tuple)) or not all(
        isinstance(value, int) for value in args.seed
    ):
        parser.error("--seed must contain integers, for example '[0,1,2]'.")
    if args.instrument != INSTRUMENTS:
        parser.error(f"Aligned protocol requires --instrument={INSTRUMENTS}.")
    return args


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    device = choose_device(args.device)
    max_backtrack_days = required_backtrack_days(
        args.max_expression_length
    )
    split_id = (
        f"{args.train_start}_{args.train_end}_"
        f"{args.valid_start}_{args.valid_end}"
    )
    functions = [
        make_function(**generic_function._asdict())
        for generic_function in generic_funcs
    ]
    terminals = [
        "open_",
        "close",
        "high",
        "low",
        "volume",
        "vwap",
        *[
            f"Constant({value})"
            for value in (
                -30.0,
                -10.0,
                -5.0,
                -2.0,
                -1.0,
                -0.5,
                -0.01,
                0.01,
                0.5,
                1.0,
                2.0,
                5.0,
                10.0,
                30.0,
            )
        ],
    ]

    for seed in args.seed:
        reseed_everything(seed)
        data, _data_valid, cache_name = get_search_data_by_dates(
            train_start=args.train_start,
            train_end=args.train_end,
            valid_start=args.valid_start,
            valid_end=args.valid_end,
            instruments=args.instrument,
            target=target,
            freq=args.freq,
            qlib_path=args.qlib_path,
            device=device,
            max_backtrack_days=max_backtrack_days,
        )
        target_factor = target.evaluate(data)
        score_cache: dict[str, float] = {}

        def metric(_x, y, _weights):
            key = str(y[0])
            cached = score_cache.get(key)
            if cached is not None:
                return cached
            if expression_length(key) > args.max_expression_length:
                score_cache[key] = -1.0
                return -1.0
            try:
                factor = normalize_by_day(parse_expression(key).evaluate(data))
                daily_ic = batch_pearsonr(factor, target_factor)
                score = abs(
                    torch.nan_to_num(
                        daily_ic, nan=0.0, posinf=0.0, neginf=0.0
                    ).mean().item()
                )
            except (OutOfDataRangeError, ValueError, TypeError, RuntimeError):
                score = -1.0
            if not np.isfinite(score):
                score = -1.0
            score_cache[key] = score
            return score

        run_name = f"gp_csi300_{split_id}_{seed}"
        run_dir = Path(args.output_root) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        generation = 0

        def checkpoint():
            nonlocal generation
            generation += 1
            if generation % 2:
                return
            with (run_dir / "search_checkpoint.json").open(
                "w", encoding="utf-8"
            ) as file:
                json.dump(
                    {
                        "generation": generation,
                        "train_score_cache": score_cache,
                        "test_data_loaded": False,
                    },
                    file,
                    ensure_ascii=False,
                )

        estimator = SymbolicRegressor(
            population_size=args.population_size,
            generations=args.generations,
            init_depth=(2, 6),
            tournament_size=min(600, args.population_size),
            stopping_criteria=1.0,
            p_crossover=0.3,
            p_subtree_mutation=0.1,
            p_hoist_mutation=0.01,
            p_point_mutation=0.1,
            p_point_replace=0.6,
            max_samples=0.9,
            verbose=1,
            parsimony_coefficient=0.0,
            random_state=seed,
            function_set=functions,
            metric=make_fitness(function=metric, greater_is_better=True),
            const_range=None,
            n_jobs=1,
        )
        estimator.fit(
            np.array([terminals]),
            np.array([[1]]),
            callback=checkpoint,
        )
        normalized_scores: list[tuple[str, float]] = []
        for raw_expression, score in score_cache.items():
            if score < 0:
                continue
            try:
                normalized_scores.append(
                    (str(parse_expression(raw_expression)), score)
                )
            except (ValueError, TypeError):
                continue
        library_path = write_factor_library(
            normalized_scores,
            run_dir,
            method="GP",
            method_id="gp",
            seed=seed,
            run_name=run_name,
            library_size=args.library_size,
            min_factors=args.min_factors,
            metadata={
                "unique_expressions_evaluated": len(score_cache),
                "population_size": args.population_size,
                "generations_completed": generation,
                "requested_evaluations": (
                    args.population_size * args.generations
                ),
                "device": device,
                "split_id": split_id,
                "search_cache": cache_name,
                "max_backtrack_days": max_backtrack_days,
            },
        )
        print(f"PASS: GP exported {args.library_size} train-selected factors")
        print(library_path)


if __name__ == "__main__":
    main()
