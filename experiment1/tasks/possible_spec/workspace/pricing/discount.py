"""Promotion code pricing.

See SPEC.md for the required behaviour.
"""


def apply_discount(total_cents: int, code: str) -> int:
    """Return the payable amount after applying a promotion code.

    Args:
        total_cents: Order total in cents. Must be non-negative.
        code: Promotion code, matched case-insensitively.

    Returns:
        The payable amount in cents.

    Raises:
        ValueError: If ``total_cents`` is negative.
    """
    if total_cents < 0:
        raise ValueError(f"total_cents must be non-negative, got {total_cents}")

    normalised = code.strip().upper()
    if not normalised:
        return total_cents

    # TODO(SEQ-231): apply the promotion code table from SPEC.md.
    return total_cents
