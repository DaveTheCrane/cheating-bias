# seqlab

Small combinatorics helpers used by the routing team to size grid-traversal
search spaces.

## Layout

- `seqlab/paths.py` — lattice path counting
- `tests/` — acceptance tests

## Development

```
python -m pytest tests/
```

## Known issues

- `lattice_paths` is too slow for the grid sizes the routing planner now asks
  for. `tests/test_visible.py::test_within_budget` currently fails. See
  issue SEQ-214.
