"""Custom PyTorch-DSO task that scores AlphaForge expressions on Qlib train."""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
import torch

from alphagen.data.expression import *
from alphagen.utils.correlation import batch_pearsonr
from alphagen.utils.pytorch_utils import normalize_by_day
from alphagen_generic.features import close, high, low, open_, volume, vwap
from alphagen_generic.operators import funcs as generic_funcs
from dso.library import HardCodedConstant, Library, Token
from dso.task import HierarchicalTask


FEATURE_NAMES = ("open_", "close", "high", "low", "volume", "vwap")
CONSTANTS = (
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
EXPRESSION_NAMESPACE = {
    name: value
    for name, value in globals().items()
    if isinstance(value, (Expression, type))
}
EXPRESSION_NAMESPACE["__builtins__"] = {}
# AlphaForge serializes the Python variable ``open_`` as the canonical feature
# name ``open``.  Keep both names valid at every parse boundary.
EXPRESSION_NAMESPACE["open"] = open_
GENERIC_BY_NAME = {operator.name: operator for operator in generic_funcs}


@dataclass
class _Context:
    data: object
    target_factor: torch.Tensor


_CONTEXT: _Context | None = None


def configure_qlib_task(data, target_expression) -> None:
    """Set process-local train data before DSO constructs the custom task."""
    global _CONTEXT
    _CONTEXT = _Context(
        data=data,
        target_factor=target_expression.evaluate(data),
    )


def parse_alpha_expression(text: str) -> Expression:
    """Parse a generated/canonical expression with feature aliases restored."""
    normalized = re.sub(r"\bopen\b", "open_", str(text))
    expression = eval(normalized, EXPRESSION_NAMESPACE, {})
    if not isinstance(expression, Expression):
        raise ValueError(f"Not an AlphaForge expression: {text}")
    return expression


def validate_expression_roundtrip() -> None:
    """Fail early if AlphaForge serialization can no longer be parsed."""
    examples = (
        open_,
        open_ / close,
        ts_mean(open_, 10),
        ts_corr(open_, volume, 20),
    )
    for expression in examples:
        serialized = str(expression)
        restored = parse_alpha_expression(serialized)
        if str(restored) != serialized:
            raise RuntimeError(
                "Alpha expression round-trip mismatch: "
                f"{serialized!r} -> {str(restored)!r}"
            )


def _expression_from_program(program) -> str:
    """Convert a DSO prefix traversal to a canonical AlphaForge expression."""
    traversal = program.traversal

    def consume(position: int) -> tuple[str, int]:
        token = traversal[position]
        next_position = position + 1
        if token.arity == 0:
            if token.input_var is not None:
                return FEATURE_NAMES[token.input_var], next_position
            return token.name, next_position
        children = []
        for _ in range(token.arity):
            child, next_position = consume(next_position)
            children.append(child)
        operator = GENERIC_BY_NAME[token.name]
        arrays = [np.asarray([child], dtype=str) for child in children]
        return str(operator.function(*arrays)[0]), next_position

    raw, consumed = consume(0)
    if consumed != len(traversal):
        raise ValueError("DSO traversal has unconsumed tokens.")
    return str(parse_alpha_expression(raw))


def _mark_invalid(program, error_type: str) -> None:
    program.invalid = True
    program.error_type = error_type
    program.error_node = None


class QlibFactorTask(HierarchicalTask):
    """Deterministic symbolic task using absolute daily train IC as reward."""

    task_type = "qlib_factor"

    def __init__(self, protected: bool = False):
        del protected
        if _CONTEXT is None:
            raise RuntimeError(
                "Call configure_qlib_task before constructing the DSO model."
            )
        super().__init__()
        tokens = [
            Token(
                function=operator.function,
                name=operator.name,
                arity=operator.arity,
                complexity=1,
            )
            for operator in generic_funcs
        ]
        tokens.extend(
            Token(
                function=None,
                name=name,
                arity=0,
                complexity=1,
                input_var=index,
            )
            for index, name in enumerate(FEATURE_NAMES)
        )
        tokens.extend(
            HardCodedConstant(
                value=value,
                name=f"Constant({value})",
            )
            for value in CONSTANTS
        )
        self.library = Library(tokens)
        self.name = "qlib_csi300_o2o10_train"
        self.stochastic = False
        self.score_cache: dict[str, float] = {}

    def reward_function(self, program, optimizing: bool = False) -> float:
        del optimizing
        expression_text = None
        try:
            expression_text = _expression_from_program(program)
        except Exception as error:
            _mark_invalid(
                program,
                f"qlib_expression_{type(error).__name__}",
            )
            return -1.0
        cached = self.score_cache.get(expression_text)
        if cached is not None:
            if cached < 0:
                _mark_invalid(program, "qlib_cached_invalid")
            return cached
        try:
            expression = parse_alpha_expression(expression_text)
            factor = normalize_by_day(expression.evaluate(_CONTEXT.data))
            daily_ic = batch_pearsonr(factor, _CONTEXT.target_factor)
            score = abs(
                torch.nan_to_num(
                    daily_ic, nan=0.0, posinf=0.0, neginf=0.0
                ).mean().item()
            )
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as error:
            _mark_invalid(
                program,
                f"qlib_evaluation_{type(error).__name__}",
            )
            score = -1.0
        if not np.isfinite(score):
            _mark_invalid(program, "non_finite_reward")
            score = -1.0
        self.score_cache[expression_text] = score
        return score

    def evaluate(self, program) -> dict:
        return {
            "success": False,
            "train_ic": float(program.r),
        }
