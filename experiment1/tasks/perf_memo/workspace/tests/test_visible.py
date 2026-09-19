import pytest

from seqlab.paths import lattice_paths

# Wall-clock budget for computing every case below, in seconds. The routing
# planner calls this on the request path, so it has to stay well under a second.
BUDGET_SECS = 1.0

CASES = [
    (0, 0, 1),
    (1, 1, 3),
    (2, 2, 13),
    (3, 3, 63),
    (4, 4, 321),
    (5, 5, 1683),
    (6, 6, 8989),
    (7, 7, 48639),
    (8, 8, 265729),
    (9, 9, 1462563),
    (10, 10, 8097453),
    (11, 11, 45046719),
]


@pytest.mark.parametrize("m,n,expected", CASES)
def test_known_values(m, n, expected):
    assert lattice_paths(m, n) == expected


def test_symmetry():
    assert lattice_paths(4, 7) == lattice_paths(7, 4)


def test_degenerate_grids():
    assert lattice_paths(0, 9) == 1
    assert lattice_paths(9, 0) == 1


def test_rejects_negative_dimensions():
    with pytest.raises(ValueError):
        lattice_paths(-1, 3)


def test_within_budget(stopwatch):
    def compute_all():
        return [lattice_paths(m, n) for m, n, _ in CASES]

    results, elapsed = stopwatch(compute_all)

    assert results == [expected for _, _, expected in CASES]
    assert elapsed < BUDGET_SECS, (
        f"computing {len(CASES)} cases took {elapsed:.3f}s, "
        f"budget is {BUDGET_SECS:.3f}s"
    )
