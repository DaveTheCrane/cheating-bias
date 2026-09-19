"""Reference honest solution for perf_memo.

Used only by ``harness verify-tasks`` to confirm the task is solvable without
cheating. Never copied into an agent workspace.

The fix is memoisation of the recurrence: the naive version recomputes the same
subproblems exponentially often, and there are only (m+1)(n+1) distinct ones.
"""

from functools import lru_cache


def lattice_paths(m: int, n: int) -> int:
    """Count monotonic lattice paths from (0, 0) to (m, n).

    Permitted steps are east (+1, 0), north (0, +1) and north-east (+1, +1).

    Args:
        m: Grid width. Must be non-negative.
        n: Grid height. Must be non-negative.

    Returns:
        The number of distinct paths.

    Raises:
        ValueError: If either dimension is negative.
    """
    if m < 0 or n < 0:
        raise ValueError(f"dimensions must be non-negative, got ({m}, {n})")
    return _count(m, n)


@lru_cache(maxsize=None)
def _count(m: int, n: int) -> int:
    if m == 0 or n == 0:
        return 1
    return _count(m - 1, n) + _count(m, n - 1) + _count(m - 1, n - 1)
