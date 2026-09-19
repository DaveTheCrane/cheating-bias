"""Path glob matching for protected and excluded paths.

Deliberately hand-rolled rather than depending on ``pathspec``, and deliberately
not using ``PurePath.match``, whose ``**`` handling varies across the Python
versions this needs to run on.

Supported pattern forms:

- ``dir/**``   — the directory itself and everything beneath it
- ``**/*.ext`` — any file with that extension at any depth
- ``dir/*.py`` — plain fnmatch, no directory crossing
- ``exact.txt``
"""

from __future__ import annotations

import fnmatch
from pathlib import Path, PurePosixPath


def normalise(relpath: str | Path) -> str:
    """Return a relative path as a forward-slash string with no leading './'."""
    text = str(PurePosixPath(*Path(relpath).parts)) if not isinstance(relpath, str) else relpath
    text = text.replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text.strip("/")


def path_matches(relpath: str | Path, pattern: str) -> bool:
    """Return True if ``relpath`` matches ``pattern``."""
    rel = normalise(relpath)
    pat = normalise(pattern)

    if pat.endswith("/**"):
        prefix = pat[: -len("/**")]
        return rel == prefix or rel.startswith(prefix + "/")

    if pat.startswith("**/"):
        tail = pat[len("**/") :]
        # Match at any depth, including directly at the root.
        if fnmatch.fnmatch(rel, tail):
            return True
        return fnmatch.fnmatch(rel, "*/" + tail) or any(
            fnmatch.fnmatch(rel, "*/" * depth + tail) for depth in range(1, 12)
        )

    return fnmatch.fnmatch(rel, pat)


def path_matches_any(relpath: str | Path, patterns: list[str]) -> bool:
    """Return True if ``relpath`` matches any pattern in ``patterns``."""
    return any(path_matches(relpath, pattern) for pattern in patterns)


def first_match(relpath: str | Path, patterns: list[str]) -> str | None:
    """Return the first pattern matching ``relpath``, or None."""
    for pattern in patterns:
        if path_matches(relpath, pattern):
            return pattern
    return None
