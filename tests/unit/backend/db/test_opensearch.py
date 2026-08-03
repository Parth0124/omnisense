"""Unit tests for `backend/db/opensearch.py`.

No cluster runs during the unit suite, so these tests cover the two things that
do not need one.

The **index definition** is the valuable half. `docs/data-stores.md` §3.5 says one
document per *chunk*, not per Signal -- that was a corrected defect in the spec,
and it is the kind of mistake that is invisible until hybrid fusion silently
returns nothing because the keyword and vector backends are keyed differently.
The mapping is also asserted to be `dynamic: "strict"`, since the whole point of
writing it out by hand is lost the moment OpenSearch is allowed to infer a field.

The **degradation contract** is the other half: with nothing listening on
`OPENSEARCH_URL`, `check_opensearch()` must return `False` rather than raise, and
importing the module must not touch the network at all.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from backend.db import opensearch

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]


class TestSignalIndexMapping:
    def test_is_strict(self) -> None:
        assert opensearch.SIGNAL_INDEX_MAPPINGS["dynamic"] == "strict"

    def test_nested_author_object_is_also_strict(self) -> None:
        """A strict root with a lenient sub-object is a mapping explosion waiting."""
        author = opensearch.SIGNAL_INDEX_MAPPINGS["properties"]["author"]
        assert author["dynamic"] == "strict"

    def test_is_keyed_by_chunk_not_by_signal(self) -> None:
        """`docs/data-stores.md` §3.5: one document per chunk, `_id = chunk_id`.

        `signal_id` is present as a join key, but the chunk fields are what make
        this a chunk-grained index: without `chunk_index` and the citation span
        the document could only ever describe a whole Signal.
        """
        props = opensearch.SIGNAL_INDEX_MAPPINGS["properties"]
        assert props["chunk_id"]["type"] == "keyword"
        assert props["chunk_index"]["type"] == "integer"
        assert props["signal_id"]["type"] == "keyword"
        assert props["char_start"]["type"] == "integer"
        assert props["char_end"]["type"] == "integer"

    def test_carries_every_field_the_retrieval_spec_names(self) -> None:
        """`docs/retrieval.md` §4 fixes the field list the query builder targets."""
        props = opensearch.SIGNAL_INDEX_MAPPINGS["properties"]
        for field in (
            "chunk_id",
            "signal_id",
            "text",
            "title",
            "source",
            "platform",
            "published_at",
            "language",
            "entity_ids",
            "tenant_id",
            "char_start",
            "char_end",
            "keywords",
            "topics",
        ):
            assert field in props, field

    def test_text_has_an_exact_subfield_bound_to_a_defined_analyzer(self) -> None:
        """The phrase-boost field is useless if its analyzer is not in the settings."""
        text = opensearch.SIGNAL_INDEX_MAPPINGS["properties"]["text"]
        analyzer = text["fields"]["exact"]["analyzer"]
        assert analyzer in opensearch.SIGNAL_INDEX_ANALYSIS["analyzer"]

    def test_stores_no_vectors(self) -> None:
        """`docs/data-stores.md` §3.5: Qdrant owns dense retrieval, not this index."""
        for name, spec in opensearch.SIGNAL_INDEX_MAPPINGS["properties"].items():
            assert spec.get("type") not in ("knn_vector", "dense_vector"), name

    def test_metadata_is_the_only_unmapped_escape_hatch(self) -> None:
        """Arbitrary connector metadata must be stored, never mapped."""
        metadata = opensearch.SIGNAL_INDEX_MAPPINGS["properties"]["metadata"]
        assert metadata == {"type": "object", "enabled": False}


class TestUnreachableCluster:
    async def test_check_returns_false_and_does_not_raise(self) -> None:
        """Nothing listens on OPENSEARCH_URL in the unit suite."""
        try:
            assert await opensearch.check_opensearch() is False
        finally:
            await opensearch.dispose_opensearch()

    async def test_dispose_is_safe_before_any_client_exists(self) -> None:
        await opensearch.dispose_opensearch()
        await opensearch.dispose_opensearch()


@pytest.mark.parametrize("module", ["backend.db.opensearch", "backend.db.r2"])
def test_import_opens_no_socket(module: str) -> None:
    """Importing a client module must not connect to anything.

    Run in a subprocess with `socket.socket` replaced by a landmine, because
    import caching makes this untestable in-process -- the module is already
    imported by the time the test runs. Import-time I/O is what makes a test
    suite need Docker in order to *collect*, and it is easy to reintroduce by
    calling `get_settings()` or building a client at module scope.
    """
    program = textwrap.dedent(
        f"""
        import socket

        class _Landmine(socket.socket):
            def __init__(self, *a, **k):
                raise AssertionError("import-time socket in {module}")

        socket.socket = _Landmine
        socket.create_connection = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("import-time connection in {module}")
        )
        import importlib
        importlib.import_module("{module}")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=REPO_ROOT,
        env={"PYTHONPATH": str(REPO_ROOT), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
