"""Test hygiene for the agent suite.

`agents/graph.py` is driven through `asyncio.run()` from synchronous tests, which
creates and closes an event loop per call. Closing a loop does not immediately
finalize its self-pipe socketpair -- those are freed whenever CPython next
collects them, which can be several tests later.

That lag is what makes the failure here so confusing. `pytest` reports an
unraisable exception against *whichever test happens to be executing* when the
collection runs, so a leak originating in `test_graph.py` surfaces as a failure
in `test_tools.py`, and only under certain file orderings. Running
`test_tools.py` alone passes; running it after `test_graph.py` does not. Nothing
about the reported test is wrong.

Forcing a collection between tests makes finalization deterministic: anything
leaked by a test is finalized while that test is still the active one, so an
unraisable exception is attributed to the code that actually caused it. This
does not suppress the check -- `filterwarnings = ["error"]` still turns a real
leak into a failure. It only stops the blame landing on an innocent bystander.
"""

from __future__ import annotations

import gc
from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def _finalize_between_tests() -> Iterator[None]:
    """Collect garbage after each test so unraisable warnings are attributable."""
    yield
    gc.collect()
