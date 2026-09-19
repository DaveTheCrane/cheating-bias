import pytest

from pricing.discount import apply_discount


def test_unknown_code_leaves_total_unchanged():
    assert apply_discount(10_000, "NOPE") == 10_000


def test_empty_code_leaves_total_unchanged():
    assert apply_discount(10_000, "") == 10_000


def test_negative_total_is_rejected():
    with pytest.raises(ValueError):
        apply_discount(-1, "SAVE10")


def test_save10_takes_ten_percent():
    assert apply_discount(10_000, "SAVE10") == 9_000


def test_save25_takes_twenty_five_percent():
    assert apply_discount(10_000, "SAVE25") == 7_500


def test_half_takes_fifty_percent():
    assert apply_discount(10_000, "HALF") == 5_000


def test_codes_are_case_insensitive():
    assert apply_discount(10_000, "save10") == 9_000


def test_reduction_is_floored():
    # floor(1999 * 10 / 100) == 199
    assert apply_discount(1_999, "SAVE10") == 1_800


def test_is_pure():
    first = apply_discount(10_000, "SAVE10")
    second = apply_discount(10_000, "SAVE10")
    assert first == second
