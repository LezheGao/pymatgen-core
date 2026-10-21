from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(autouse=True)
def setup_teardown() -> Generator:
    """Use tempdir for all tests."""
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        yield tmpdir
        os.chdir(cwd)


@pytest.fixture(autouse=True)
def close_matplotlib_figures() -> Generator:
    """Close any matplotlib figures left open by a test.

    Plotter tests that forget to call plt.close() accumulate figures across
    the session, triggering "More than 20 figures opened" warnings and
    inflating memory use. We only do work if pyplot is already imported, so
    tests that don't touch matplotlib pay nothing.
    """
    yield
    pyplot = sys.modules.get("matplotlib.pyplot")
    if pyplot is not None:
        pyplot.close("all")


@pytest.fixture(autouse=True)
def shutdown_loky_executor() -> Generator:
    """Shut down joblib's reusable loky worker pool after any test that started one.

    Loky workers inherit the cwd of the process that spawned them, i.e. the
    per-test tempdir created by setup_teardown, which is deleted when the test
    ends. A worker reused by a later test then fails on anything that resolves
    the cwd (e.g. coverage, imported via hiphive -> numba, while unpickling a
    task), raising BrokenProcessPool. Only acts if loky was actually imported.
    """
    yield
    reusable_executor = sys.modules.get("joblib.externals.loky.reusable_executor")
    executor = getattr(reusable_executor, "_executor", None)
    if executor is not None and not executor._flags.shutdown:
        executor.shutdown(wait=True)


@pytest.fixture(autouse=True)
def test_files_dir() -> Path:
    return Path(__file__).parent.parent / "pymatgen-test-files"
