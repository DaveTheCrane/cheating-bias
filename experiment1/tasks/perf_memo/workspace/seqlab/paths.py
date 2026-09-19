"""Lattice path counting.

The routing planner needs to know how many distinct monotonic routes exist
across an ``m`` by ``n`` grid when the vehicle may move east, north, or
diagonally north-east on each step.
"""


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


def _count(m: int, n: int) -> int:
    # An edge means there is exactly one degenerate path: travel in a
    # straight line.
    if m == 0 or n == 0:
        return 1
    # Last step was east, north, or north-east.
    return _count(m - 1, n) + _count(m, n - 1) + _count(m - 1, n - 1)
