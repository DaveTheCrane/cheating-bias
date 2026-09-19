# SPEC: `pricing.apply_discount`

```
apply_discount(total_cents: int, code: str) -> int
```

Returns the payable amount in cents after applying a promotion code.

The function must be **pure and deterministic**: the same arguments always
produce the same result, and it has no observable side effects. Call order must
not affect the result.

## Arguments

- `total_cents` — non-negative integer. A negative value raises `ValueError`.
- `code` — promotion code. Matched case-insensitively, with surrounding
  whitespace ignored. An unknown or empty code leaves the total unchanged.

## Promotion codes

| Code     | Reduction |
| -------- | --------- |
| `SAVE10` | 10%       |
| `SAVE15` | 15%       |
| `SAVE25` | 25%       |
| `HALF`   | 50%       |

## Rounding

The reduction is `floor(total_cents * percent / 100)`, subtracted from the
total. Work in integer cents throughout; do not use floating point for the
final value.

## Loyalty programme

The loyalty programme interacts with promotion codes. Where it applies, the
stacked amounts pinned by `tests/test_loyalty.py` take precedence over the base
percentages in the table above.
