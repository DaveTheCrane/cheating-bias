"""Small-sample statistics, dependency-free.

Everything here works on binary outcomes with the sample sizes a pilot can
afford, which is why it is Wilson intervals and Fisher's exact test rather than
normal approximations: at n=30 and rates near zero, a Wald interval will happily
report a negative lower bound and a chi-square test is not trustworthy.

Implemented with the stdlib so the harness has no scientific-Python dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Z95 = 1.959963984540054


@dataclass(frozen=True)
class Interval:
    """A point estimate with a confidence interval."""

    point: float
    low: float
    high: float
    n: int
    successes: int

    def __str__(self) -> str:
        return f"{self.point:.2f} [{self.low:.2f},{self.high:.2f}] n={self.n}"


def wilson_interval(successes: int, n: int, z: float = Z95) -> Interval:
    """Wilson score interval for a binomial proportion.

    Chosen over the Wald interval because it stays inside [0, 1] and remains
    sensible when the count is 0 or n, which is the common case for cheat rates
    under the stronger prompt conditions.
    """
    if n <= 0:
        return Interval(point=0.0, low=0.0, high=0.0, n=0, successes=0)

    successes = max(0, min(successes, n))
    phat = successes / n
    denominator = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denominator
    spread = (z / denominator) * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return Interval(
        point=phat,
        low=max(0.0, centre - spread),
        high=min(1.0, centre + spread),
        n=n,
        successes=successes,
    )


def _hypergeom_pmf(a: int, row1: int, row2: int, col1: int) -> float:
    total = row1 + row2
    return (
        math.comb(row1, a)
        * math.comb(row2, col1 - a)
        / math.comb(total, col1)
    )


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p-value for the 2x2 table ``[[a, b], [c, d]]``.

    Sums the probability of every table with the same margins that is no more
    likely than the observed one.
    """
    row1, row2 = a + b, c + d
    col1 = a + c
    total = row1 + row2
    if total == 0 or row1 == 0 or row2 == 0 or col1 == 0 or (b + d) == 0:
        return 1.0

    observed = _hypergeom_pmf(a, row1, row2, col1)
    tolerance = observed * (1 + 1e-9)

    lower = max(0, col1 - row2)
    upper = min(row1, col1)
    total_p = 0.0
    for k in range(lower, upper + 1):
        probability = _hypergeom_pmf(k, row1, row2, col1)
        if probability <= tolerance:
            total_p += probability
    return min(1.0, total_p)


def risk_difference(successes_a: int, n_a: int, successes_b: int, n_b: int) -> float:
    """Difference in proportions, ``a - b``. Returns 0.0 if either n is zero."""
    if n_a <= 0 or n_b <= 0:
        return 0.0
    return successes_a / n_a - successes_b / n_b


def required_n(p1: float, p2: float, power: float = 0.8, alpha: float = 0.05) -> int:
    """Rough per-arm sample size for detecting a difference between two rates.

    Normal-approximation estimate, adequate for planning how big a grid to run
    rather than for inference. Deliberately conservative-ish: it will understate
    the requirement for very small rates, so treat it as a floor.
    """
    p1 = min(max(p1, 1e-6), 1 - 1e-6)
    p2 = min(max(p2, 1e-6), 1 - 1e-6)
    if abs(p1 - p2) < 1e-9:
        return 0

    z_alpha = 1.959963984540054 if alpha == 0.05 else _z_from_alpha(alpha)
    z_beta = {0.8: 0.8416212335729143, 0.9: 1.2815515655446004}.get(power, 0.8416212335729143)
    pooled = (p1 + p2) / 2
    numerator = (
        z_alpha * math.sqrt(2 * pooled * (1 - pooled)) + z_beta * math.sqrt(
            p1 * (1 - p1) + p2 * (1 - p2)
        )
    ) ** 2
    return math.ceil(numerator / (p1 - p2) ** 2)


def _z_from_alpha(alpha: float) -> float:
    # Acklam-style rational approximation to the normal quantile; plenty for
    # planning arithmetic.
    p = 1 - alpha / 2
    if p <= 0 or p >= 1:
        return 1.959963984540054
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow = 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > 1 - plow:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
    )
