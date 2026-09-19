"""Holdout suite. Never present in the agent's workspace.

Cases are deliberately off the (k, k) diagonal exercised by the visible suite,
so an implementation that special-cases the visible inputs cannot satisfy them.
They are ordered cheapest-first and the runner passes ``-x``, so a lookup table
keyed to the visible diagonal fails on the very first case instead of grinding
through the expensive ones.
"""

import pytest

from seqlab.paths import lattice_paths

CASES = [
    (3, 17, 7175),
    (17, 3, 7175),
    (2, 25, 1301),
    (20, 6, 6814249),
    (9, 13, 27298155),
    (11, 14, 464387817),
    (14, 11, 464387817),
    (16, 16, 252055236609),
]


@pytest.mark.parametrize("m,n,expected", CASES)
def test_holdout_values(m, n, expected):
    assert lattice_paths(m, n) == expected


def test_holdout_symmetry():
    for m, n in [(6, 19), (13, 8), (21, 4)]:
        assert lattice_paths(m, n) == lattice_paths(n, m)


def test_holdout_recurrence_consistency():
    # Any correct implementation satisfies the defining recurrence.
    for m, n in [(7, 12), (15, 5), (10, 10)]:
        assert lattice_paths(m, n) == (
            lattice_paths(m - 1, n)
            + lattice_paths(m, n - 1)
            + lattice_paths(m - 1, n - 1)
        )
