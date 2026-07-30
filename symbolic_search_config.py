"""Shared, dependency-free bounds for symbolic factor search."""

ROLLING_WINDOWS = (10, 20, 30, 40, 50)
MAX_ROLLING_LOOKBACK = max(ROLLING_WINDOWS)
MAX_EXPRESSION_LENGTH = 20


def required_backtrack_days(max_expression_length: int) -> int:
    """Return a safe warm-up window for a bounded symbolic expression."""
    if max_expression_length < 1:
        raise ValueError("max_expression_length must be positive.")
    # Every token on a root-to-leaf path can contribute at most one rolling
    # window. Ref/ts_delta consume the full window, so do not subtract one.
    return max_expression_length * MAX_ROLLING_LOOKBACK


FACTOR_BACKTRACK_DAYS = required_backtrack_days(MAX_EXPRESSION_LENGTH)
