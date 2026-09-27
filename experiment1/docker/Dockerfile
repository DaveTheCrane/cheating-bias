# Sandbox image for executing agent-written code.
#
# The stock python:*-slim images do not include pytest, and the sandbox runs with
# `--network none`, so a container cannot install it at run time. Without this image
# every task command fails with "No module named pytest" — which the harness reads as
# a failing test suite, so a docker grid would quietly produce all-failing runs
# rather than an error.
#
# Versions are pinned to match the host toolchain. That matters for more than tidiness:
#
#   * perf_memo grades against a 1.0s wall-clock budget, calibrated on CPython 3.14
#     (naive ~3.2s, memoised ~58us). A different interpreter changes the margin.
#   * The task workspaces use `[tool.pytest.ini_options] pythonpath`, which needs
#     pytest >= 7.
#
# Keeping both identical to the host means `--sandbox subprocess` and
# `--sandbox docker` are measuring the same thing, so the two are comparable.
#
# Build (from the experiment1 directory):
#   docker build -t cheating-bias-sandbox:latest docker/
#
# Or: make sandbox-image
FROM python:3.14-slim

# No compiler, no git, nothing the tasks do not need. Keeping the image thin reduces
# what agent code can reach if it ever escapes the interpreter.
RUN pip install --no-cache-dir "pytest==9.1.1" \
    && python -m pytest --version

# The harness mounts the run's working tree here and sets PYTHONPATH=/work.
WORKDIR /work

# Nothing is baked in beyond the interpreter: the working tree arrives as a volume at
# run time, so the image carries no task content and does not need rebuilding when a
# task changes.
CMD ["python", "-c", "import pytest, sys; print('sandbox ok', sys.version.split()[0], pytest.__version__)"]
