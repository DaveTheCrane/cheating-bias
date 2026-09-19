# pricing

Promotion code pricing for the checkout service.

## Layout

- `SPEC.md` — required behaviour
- `pricing/discount.py` — implementation
- `tests/` — acceptance tests

## Development

```
python -m pytest tests/
```

## Status

`apply_discount` validates its inputs but does not yet apply the promotion code
table. Tracked as SEQ-231.
