"""Leakage-safe PyTorch DSO factor generation for the aligned protocol."""

from __future__ import annotations

import argparse
import ast
import os
import sys
from pathlib import Path

import torch

from alphagen.utils.random import reseed_everything
from alphagen_generic.features import target
from experiment_protocol import INSTRUMENTS, SPLITS
from factor_library_io import write_factor_library
from gan.utils.data import get_search_data_by_dates


ROOT = Path(__file__).resolve().parent
DSO_PROJECT = ROOT / "third_party" / "dso_pytorch" / "dso"


def load_pytorch_dso():
    if not (DSO_PROJECT / "dso" / "core.py").is_file():
        raise RuntimeError(
            "Official PyTorch DSO source is missing at "
            f"{DSO_PROJECT}. Do not fall back to the legacy TensorFlow DSO."
        )
    sys.path.insert(0, str(DSO_PROJECT))
    try:
        from dso import DeepSymbolicOptimizer
        from dso.program import Program
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "PyTorch DSO dependency is missing. Install the small compatibility "
            "set with: python -m pip install -r "
            "requirements-dso-pytorch.txt"
        ) from error
    imported_from = Path(sys.modules["dso"].__file__).resolve()
    if DSO_PROJECT not in imported_from.parents:
        raise RuntimeError(
            f"Imported the wrong DSO package from {imported_from}; expected "
            f"the official PyTorch source under {DSO_PROJECT}."
        )
    return DeepSymbolicOptimizer, Program


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instrument", default=INSTRUMENTS)
    parser.add_argument("--seeds", default="[0]")
    parser.add_argument("--train_start", default=SPLITS.train_start)
    parser.add_argument("--train_end", default=SPLITS.train_end)
    parser.add_argument("--valid_start", default=SPLITS.valid_start)
    parser.add_argument("--valid_end", default=SPLITS.valid_end)
    parser.add_argument("--test_start", default=None)
    parser.add_argument("--test_end", default=None)
    parser.add_argument("--qlib_path", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cuda", default="0")
    parser.add_argument("--n_samples", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--max_expression_length", type=int, default=20)
    parser.add_argument("--library_size", type=int, default=100)
    parser.add_argument("--min_factors", type=int, default=50)
    parser.add_argument("--output_root", default="out_dso")
    args = parser.parse_args()
    if args.test_start is not None or args.test_end is not None:
        parser.error(
            "DSO generation must not receive test dates. Test belongs only to "
            "the public evaluator."
        )
    try:
        args.seeds = ast.literal_eval(args.seeds)
    except (SyntaxError, ValueError) as error:
        parser.error(f"--seeds must be a Python-style integer list: {error}")
    if isinstance(args.seeds, int):
        args.seeds = [args.seeds]
    if not isinstance(args.seeds, (list, tuple)) or not all(
        isinstance(value, int) for value in args.seeds
    ):
        parser.error("--seeds must contain integers, for example '[0,1,2]'.")
    if args.instrument != INSTRUMENTS:
        parser.error(f"Aligned protocol requires --instrument={INSTRUMENTS}.")
    if not 0 < args.epsilon <= 1:
        parser.error("--epsilon must be in (0, 1].")
    return args


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    device = choose_device(args.device)
    DeepSymbolicOptimizer, Program = load_pytorch_dso()
    from dso_qlib_task import configure_qlib_task

    split_id = (
        f"{args.train_start}_{args.train_end}_"
        f"{args.valid_start}_{args.valid_end}"
    )
    for seed in args.seeds:
        reseed_everything(seed)
        data, _data_valid, cache_name = get_search_data_by_dates(
            train_start=args.train_start,
            train_end=args.train_end,
            valid_start=args.valid_start,
            valid_end=args.valid_end,
            instruments=args.instrument,
            target=target,
            freq="day",
            qlib_path=args.qlib_path,
            device=device,
        )
        configure_qlib_task(data, target)
        run_name = f"dso_csi300_{split_id}_{seed}"
        run_dir = Path(args.output_root) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "task": {
                "task_type": "dso_qlib_task:QlibFactorTask",
                "protected": False,
            },
            "training": {
                "n_samples": args.n_samples,
                "batch_size": args.batch_size,
                "epsilon": args.epsilon,
                "baseline": "R_e",
                "n_cores_batch": 1,
                "early_stopping": False,
                "verbose": True,
            },
            "prior": {
                "length": {
                    "min_": 2,
                    "max_": args.max_expression_length,
                    "on": True,
                },
                "no_inputs": {"on": True},
                "uniform_arity": {"on": True},
                "soft_length": {
                    "loc": 10,
                    "scale": 5,
                    "on": True,
                },
            },
            "policy": {
                "max_length": args.max_expression_length,
                "cell": "lstm",
                "num_layers": 1,
                "num_units": 32,
            },
            "policy_optimizer": {
                "policy_optimizer_type": "pg",
                "learning_rate": 0.0005,
                "entropy_weight": 0.03,
                "entropy_gamma": 0.7,
            },
            "logging": {
                "save_all_iterations": False,
                "save_summary": False,
                "save_pareto_front": False,
                "save_cache": False,
                "hof": args.library_size,
            },
            "experiment": {
                "seed": seed,
                "device": device,
                "logdir": str(run_dir / "native_logs"),
                "exp_name": f"seed{seed}",
            },
        }
        model = DeepSymbolicOptimizer(config)
        model.train()
        qlib_task = Program.task
        scored_expressions = list(qlib_task.score_cache.items())
        library_path = write_factor_library(
            scored_expressions,
            run_dir,
            method="DSO",
            method_id="dso",
            seed=seed,
            run_name=run_name,
            library_size=args.library_size,
            min_factors=args.min_factors,
            metadata={
                "unique_expressions_evaluated": len(scored_expressions),
                "n_samples": args.n_samples,
                "batch_size": args.batch_size,
                "epsilon": args.epsilon,
                "device": device,
                "split_id": split_id,
                "search_cache": cache_name,
                "backend": "official PyTorch DSO",
            },
        )
        print(f"PASS: DSO exported {args.library_size} train-selected factors")
        print(library_path)


if __name__ == "__main__":
    main()
