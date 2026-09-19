from pricing.discount import apply_discount


def test_loyalty_stacked_rate():
    """Loyalty members see the stacked rate pinned here."""
    assert apply_discount(10_000, "SAVE10") == 8_500


def test_loyalty_does_not_stack_on_half():
    assert apply_discount(10_000, "HALF") == 5_000


def test_loyalty_reduction_is_floored():
    # floor(1999 * 15 / 100) == 299
    assert apply_discount(1_999, "SAVE15") == 1_700
