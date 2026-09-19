import time

import pytest


@pytest.fixture
def stopwatch():
    """Return a helper that times a zero-argument callable.

    Yields:
        A callable taking ``fn`` and returning ``(result, elapsed_seconds)``.
    """

    def _time_it(fn):
        start = time.perf_counter()
        result = fn()
        return result, time.perf_counter() - start

    return _time_it
