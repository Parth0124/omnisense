"""Unit tests for `connectors/normalize/`.

These two modules sit on the identity path, which is the reason the assertions
below are as pedantic as they are. `docs/signal-model.md` §4.1 derives
`native_id` from the canonicalized URL (rule 2) and from the SimHash of the
cleaned text (rule 3), and §7 records that changing the derivation of `id` is
"not migratable in place". So a change that makes `canonicalize_url` or
`extract_readable` return something one character different is not a refactor --
it orphans every stored vector, index entry, graph edge and report citation for
the affected records.

What each concern below is defending:

- **canonicalization** -- two spellings of one URL must produce one string, and
  canonicalizing twice must change nothing;
- **extraction** -- all three backends must degrade rather than raise, and must
  agree on the *shape* of their output, or losing trafilatura re-hashes the
  corpus under dedup layer 2;
- **the rule ladder** -- it must be evaluated top-down and never re-entered;
- **error attribution** -- a `NormalizationError` without a `native_id` is a DLQ
  record nobody can replay, so identity is derived before requirements are
  enforced.

No network, no services: nothing here does I/O at all, which is what
`docs/architecture.md` §6.2 rule 2 buys.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, timezone
from typing import Any, ClassVar

import pytest

from connectors.exceptions import NormalizationError
from connectors.normalize import html as html_module
from connectors.normalize.html import (
    canonicalize_url,
    collapse_whitespace,
    extract_readable,
    is_tracking_param,
)
from connectors.normalize.mapper import (
    FieldMap,
    FieldSpec,
    MappingContext,
    MediaMap,
    build_lineage,
    derive_native_id,
    simhash64,
    to_utc_datetime,
)
from connectors.protocol import RawRecord
from models.enums import MediaKind, Platform, SignalStatus, SourceCategory
from models.signal import signal_id

pytestmark = pytest.mark.unit

T0 = datetime(2026, 7, 28, 14, 2, 11, tzinfo=UTC)

ARTICLE = """<!doctype html>
<html><head><title>Publisher name</title><style>.x{color:red}</style></head>
<body>
  <nav class="site-nav"><a href="/">Home</a> <a href="/about">About</a></nav>
  <article>
    <h1>Our observability bill tripled</h1>
    <p>We moved forty services onto self-hosted Grafana and Loki after the renewal
       quote came back at three times the previous figure, which was already the
       largest line item on the infrastructure budget.</p>
    <p>Ingest volume did not change at all. The migration took six weeks and two
       engineers, and the resulting stack is materially cheaper to run than the
       vendor contract it replaced.</p>
    <ul><li>six weeks</li><li>two engineers</li></ul>
  </article>
  <noscript><img src="https://t.example.com/pixel.gif?id=7" width="1" height="1"></noscript>
  <div class="newsletter-signup">Subscribe to our newsletter</div>
  <footer>Copyright 2026</footer>
</body></html>"""


# --------------------------------------------------------------------------- #
# html.py
# --------------------------------------------------------------------------- #


class TestCollapseWhitespace:
    """Whitespace is normalized, paragraph structure is not."""

    def test_preserves_paragraph_breaks(self) -> None:
        """`retrieval/chunking/splitter.py` splits on paragraphs; flattening them
        would push chunk boundaries into the middle of sentences."""
        assert collapse_whitespace("a\n\n\n\n\nb") == "a\n\nb"

    def test_collapses_runs_and_trims_line_edges(self) -> None:
        assert collapse_whitespace("  a \t\t b  \n   c   ") == "a b\nc"

    def test_normalizes_carriage_returns(self) -> None:
        assert collapse_whitespace("a\r\nb\rc") == "a\nb\nc"

    def test_removes_zero_width_characters(self) -> None:
        """They are invisible, survive `strip()`, and give two identically
        rendered bodies two different content hashes."""
        assert collapse_whitespace("bi​ll­i﻿on") == "billion"

    def test_folds_non_breaking_spaces(self) -> None:
        """`&nbsp;` decodes to U+00A0, which `str.split()`-shaped code misses."""
        assert collapse_whitespace("a\u00a0\u202f b") == "a b"

    def test_does_not_nfkc_normalize(self) -> None:
        """NFKC belongs in the dedup hash, not in a stored body: it would rewrite
        the text a report later quotes."""
        assert collapse_whitespace("ﬁle ①") == "ﬁle ①"


class TestCanonicalizeUrl:
    """Rule 2 of `native_id` derivation. Stability here is identity stability."""

    CANONICAL = "https://example.com/r/selfhosted/comments/1abcde/?a=1&b=2"

    @pytest.mark.parametrize(
        "spelling",
        [
            "https://example.com/r/selfhosted/comments/1abcde/?a=1&b=2",
            "https://example.com/r/selfhosted/comments/1abcde/?a=1&b=2&utm_source=share",
            "https://example.com/r/selfhosted/comments/1abcde/?a=1&b=2#comment-42",
            "https://EXAMPLE.COM/r/selfhosted/comments/1abcde/?a=1&b=2",
            "https://example.com:443/r/selfhosted/comments/1abcde/?a=1&b=2",
            "HTTPS://Example.Com:443/r/selfhosted/comments/1abcde/?b=2&a=1"
            "&utm_source=share&utm_medium=email#top",
            "https://example.com./r/selfhosted/comments/1abcde/?a=1&b=2",
            "https://example.com/r/selfhosted/x/../comments/1abcde/?a=1&b=2",
            "https://user:secret@example.com/r/selfhosted/comments/1abcde/?a=1&b=2",
        ],
    )
    def test_every_spelling_of_one_url_canonicalizes_identically(self, spelling: str) -> None:
        """The single assertion this module exists for.

        A tracking parameter, a fragment, an upper-case host, a redundant default
        port, a trailing root dot, a dot segment, embedded credentials and
        re-ordered parameters are all *spellings*. If any of them survives, one
        item gets two `native_id`s and therefore two Signals.
        """
        assert canonicalize_url(spelling) == self.CANONICAL

    def test_is_idempotent(self) -> None:
        """Canonicalizing a canonical URL must be a no-op, or re-processing a
        stored record would fork its identity."""
        once = canonicalize_url("HTTP://Example.com:80/a/%7Euser/b?q=a+b&utm_id=9#x")
        assert canonicalize_url(once) == once
        assert once == "http://example.com/a/~user/b?q=a%20b"

    def test_credentials_never_survive_into_an_identity(self) -> None:
        """`native_id` is stored, logged and cited. A password in it would be a
        credential leak that no redaction filter is looking for."""
        assert "secret" not in canonicalize_url("https://u:secret@example.com/a")

    def test_keeps_a_non_default_port(self) -> None:
        assert canonicalize_url("https://example.com:8443/a") == "https://example.com:8443/a"

    def test_does_not_strip_www(self) -> None:
        """A distinct DNS label that some hosts serve different content from.
        Sites that consider the two equivalent say so with a 301, which is what
        `resolver` is for."""
        assert canonicalize_url("https://www.example.com/a") != canonicalize_url(
            "https://example.com/a"
        )

    def test_encodes_internationalized_hosts(self) -> None:
        """`münchen.de` and `xn--mnchen-3ya.de` resolve to one server."""
        assert canonicalize_url("https://München.de/x") == "https://xn--mnchen-3ya.de/x"

    def test_does_not_decode_a_reserved_escape(self) -> None:
        """`%2F` is an encoded slash; decoding it moves the resource."""
        assert canonicalize_url("https://example.com/a%2fb") == "https://example.com/a%2Fb"

    def test_strips_embedded_newlines(self) -> None:
        """Feeds wrap long links across lines; the wrapped and unwrapped spellings
        must not get different ids."""
        assert canonicalize_url("https://example.com/a\n/b") == "https://example.com/a/b"

    def test_relative_reference_has_no_stable_identity(self) -> None:
        """`/index.html` is a different page in every feed that uses it, so the
        caller is expected to fall through to rule 3 rather than hash it."""
        assert canonicalize_url("/index.html") == ""
        assert canonicalize_url("") == ""

    def test_resolves_against_a_base_when_given_one(self) -> None:
        assert (
            canonicalize_url("/a?utm_source=x", base="https://example.com/feed")
            == "https://example.com/a"
        )

    def test_resolver_runs_before_canonicalization(self) -> None:
        """Connectors follow redirects in `fetch()` and pass the captured mapping
        in; this function performs no I/O, because identity derived from a live
        network answer would differ on replay."""
        captured = {"https://ex.am/abc": "https://example.com/full/article?utm_source=x"}
        assert (
            canonicalize_url("https://ex.am/abc", resolver=captured.get)
            == "https://example.com/full/article"
        )

    def test_unresolved_redirect_leaves_the_input_alone(self) -> None:
        assert canonicalize_url("https://ex.am/abc", resolver=lambda _: None) == (
            "https://ex.am/abc"
        )

    def test_opaque_schemes_are_left_addressing_the_same_thing(self) -> None:
        """Guessing further at a `mailto:` would change what it addresses."""
        assert canonicalize_url("MAILTO:Foo@Example.com#x") == "mailto:Foo@Example.com"

    def test_tracking_prefixes_and_the_conservative_exclusions(self) -> None:
        """`ref` and `cid` are content selectors on some sites. Merging two items
        into one id is unrecoverable; forking one item into two is caught by dedup
        layer 2, so the list errs toward keeping parameters."""
        assert is_tracking_param("utm_source") and is_tracking_param("FBCLID")
        assert not is_tracking_param("ref") and not is_tracking_param("id")


class TestExtractReadable:
    """Three backends, one contract: degrade, never raise."""

    @pytest.fixture(params=["trafilatura", "beautifulsoup", "regex"])
    def backend(self, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
        """Force each backend in turn by removing the ones ahead of it.

        The module holds its optional imports as module-level handles precisely so
        this is possible: otherwise the fallback paths would only ever be exercised
        on a machine that happened to be missing a wheel, which is to say never,
        until production.
        """
        if request.param in {"beautifulsoup", "regex"}:
            monkeypatch.setattr(html_module, "_TRAFILATURA", None)
        if request.param == "regex":
            monkeypatch.setattr(html_module, "_BS4", None)
        return str(request.param)

    def test_extracts_the_body_and_drops_the_chrome(self, backend: str) -> None:
        text = extract_readable(ARTICLE, url="https://example.com/a")
        assert "moved forty services" in text
        assert "Ingest volume did not change" in text
        assert "Copyright 2026" not in text
        assert "pixel.gif" not in text

    def test_keeps_paragraph_breaks(self, backend: str) -> None:
        """Every backend must agree on the *shape* of the output. They cannot
        agree on block selection, but if they disagree on whitespace too, a
        deployment that loses trafilatura re-hashes its whole corpus under dedup
        layer 2."""
        text = extract_readable(ARTICLE)
        assert "\n\n" in text
        assert "\n\n\n" not in text
        assert not text.startswith(" ") and not text.endswith(" ")

    def test_never_leaves_markup_in_the_body(self, backend: str) -> None:
        text = extract_readable(ARTICLE)
        assert "<" not in text and "&nbsp;" not in text

    def test_omits_the_document_title(self, backend: str) -> None:
        """`content.title` comes from the payload's own field. Extracting it into
        the body too would duplicate the headline on every article and skew both
        the language detector and the chunker."""
        assert "Publisher name" not in extract_readable(ARTICLE)

    def test_survives_malformed_markup(self, backend: str) -> None:
        """A failed extraction degrades to a worse body, never to a DLQ record:
        the feed entry is still a valid Signal (`docs/connector-spec.md` §11.2)."""
        assert "hello" in extract_readable("<div><p>hello</b></div")

    def test_empty_input_is_empty_output(self, backend: str) -> None:
        assert extract_readable("") == ""
        assert extract_readable("   \n  ") == ""

    def test_plain_text_is_not_parsed_as_markup(self) -> None:
        """An ampersand or a `<` in prose would be entity-decoded or swallowed as a
        malformed tag, quietly editing the body."""
        assert extract_readable("Costs rose & margins < 5% fell") == (
            "Costs rose & margins < 5% fell"
        )

    def test_decodes_legacy_bytes_rather_than_mangling_them(self) -> None:
        """The raw bytes are archived, but the cleaned text is what gets embedded
        and cited, so mojibake there is permanent."""
        payload = '<html><head><meta charset="cp1252"></head><body><p>caf\xe9</p></body></html>'
        assert "café" in extract_readable(payload.encode("cp1252"))

    def test_inline_markup_does_not_gain_spaces(self) -> None:
        assert extract_readable("<p>with <b>bold</b>.</p>") == "with bold."


# --------------------------------------------------------------------------- #
# mapper.py -- identity
# --------------------------------------------------------------------------- #


class TestDeriveNativeId:
    """The three-rule ladder of `docs/signal-model.md` §4.1."""

    def test_rule_1_returns_the_provider_id_verbatim(self) -> None:
        """Not hashed: a DLQ record for `t3_1abcde` can be pasted into the
        provider's own UI, and a 64-character digest cannot."""
        assert (
            derive_native_id(platform=Platform.REDDIT, item_id="t3_1abcde", url="https://x.y/a")
            == "t3_1abcde"
        )

    def test_rule_1_wins_even_when_a_url_is_present(self) -> None:
        """The rules are ranked by stability under re-fetch, so the ladder is
        evaluated top-down and never re-entered. A connector preferring the URL
        would fork every id the day a provider started emitting its guid."""
        with_url = derive_native_id(
            platform=Platform.RSS, item_id="guid-1", url="https://example.com/a"
        )
        without_url = derive_native_id(platform=Platform.RSS, item_id="guid-1")
        assert with_url == without_url == "guid-1"

    def test_rule_1_accepts_a_numeric_provider_id(self) -> None:
        """Half the catalogue sends ints; refusing them would silently drop to
        rule 2 for those sources."""
        assert derive_native_id(platform=Platform.YOUTUBE, item_id=12345) == "12345"

    def test_rule_2_hashes_the_canonical_url(self) -> None:
        expected = derive_native_id(platform=Platform.RSS, url="https://example.com/a")
        assert len(expected) == 64
        for spelling in (
            "https://EXAMPLE.com/a?utm_source=feed",
            "https://example.com:443/a#top",
        ):
            assert derive_native_id(platform=Platform.RSS, url=spelling) == expected

    def test_rule_2_is_skipped_for_a_url_with_no_host(self) -> None:
        """Hashing `/index.html` would collide across every feed that uses it."""
        derived = derive_native_id(
            platform=Platform.RSS, url="/index.html", timestamp=T0, text="a body"
        )
        assert derived == derive_native_id(platform=Platform.RSS, timestamp=T0, text="a body")

    def test_rule_3_combines_platform_author_timestamp_and_simhash(self) -> None:
        base: dict[str, Any] = {
            "platform": Platform.RSS,
            "timestamp": T0,
            "text": "the same body text",
        }
        derived = derive_native_id(**base, author_id="a1")
        assert len(derived) == 64
        assert derived == derive_native_id(**base, author_id="a1")
        assert derived != derive_native_id(**base, author_id="a2")
        assert derived != derive_native_id(
            platform=Platform.RSS, timestamp=T0 + timedelta(seconds=1), text=base["text"]
        )

    def test_rule_3_separates_two_items_posted_in_the_same_second(self) -> None:
        """Rule 3 exists for sources whose only distinguishing feature is their
        text, so the timestamp is not truncated to make ids prettier."""
        common: dict[str, Any] = {"platform": Platform.RSS, "author_id": "a1", "timestamp": T0}
        assert derive_native_id(**common, text="first comment") != derive_native_id(
            **common, text="a completely different comment"
        )

    def test_rule_3_is_stable_across_timezone_spellings(self) -> None:
        """The same instant written as +02:00 must not be a different item."""
        berlin = T0.astimezone(timezone(timedelta(hours=2)))
        assert derive_native_id(
            platform=Platform.RSS, timestamp=berlin, text="body"
        ) == derive_native_id(platform=Platform.RSS, timestamp=T0, text="body")

    def test_no_applicable_rule_raises_and_names_all_three(self) -> None:
        """No `native_id` is attached because none exists -- that is the failure --
        so the message has to carry the diagnosis instead."""
        with pytest.raises(NormalizationError) as caught:
            derive_native_id(platform=Platform.RSS, timestamp=T0, text="   ")
        assert caught.value.native_id is None
        for rule in ("rule 1", "rule 2", "rule 3"):
            assert rule in str(caught.value)

    def test_platform_is_part_of_the_rule_3_material(self) -> None:
        """The same press release on two platforms is two observations that get
        clustered, not one Signal (`docs/signal-model.md` §4.3)."""
        assert derive_native_id(
            platform=Platform.RSS, timestamp=T0, text="body"
        ) != derive_native_id(platform=Platform.GDELT, timestamp=T0, text="body")


class TestSimhash:
    """Frozen, because rule 3 makes it part of identity derivation."""

    def test_is_deterministic(self) -> None:
        assert simhash64("the quick brown fox") == simhash64("the quick brown fox")

    def test_ignores_whitespace_and_case_and_punctuation(self) -> None:
        """The same body extracted by two backends differs in punctuation and
        whitespace, not in words; identity must not notice."""
        assert simhash64("The  quick,\n\nbrown fox!") == simhash64("the quick brown fox")

    def test_differs_for_different_text(self) -> None:
        assert simhash64("observability costs tripled") != simhash64("nothing happened today")

    def test_empty_text_is_zero_not_an_error(self) -> None:
        """A media-only post has an empty body; rule 3 rejects it earlier, and
        this must not be the thing that raises."""
        assert simhash64("") == 0

    def test_fits_in_64_bits(self) -> None:
        assert 0 <= simhash64("a moderately long body of text here") < 2**64


# --------------------------------------------------------------------------- #
# mapper.py -- provenance
# --------------------------------------------------------------------------- #


def raw_record(payload: dict[str, Any], **overrides: Any) -> RawRecord:
    body = json.dumps(payload).encode()
    defaults: dict[str, Any] = {
        "native_id": "unused",
        "payload": payload,
        "fetched_at": T0,
        "raw_bytes": body,
        "request_fingerprint": "fp_1",
    }
    defaults.update(overrides)
    return RawRecord(**defaults)


CTX = MappingContext(connector_slug="rss", connector_version="0.1.0", sync_run_id="run_1")


class TestBuildLineage:
    """Provenance a connector can know: acquisition, raw payload, identity."""

    def test_records_the_acquisition_group(self) -> None:
        lineage = build_lineage(raw_record({"a": 1}), native_id="n1", ctx=CTX)
        assert (lineage.connector_slug, lineage.connector_version) == ("rss", "0.1.0")
        assert lineage.sync_run_id == "run_1"
        assert lineage.fetched_at == T0
        assert lineage.request_fingerprint == "fp_1"

    def test_digests_the_bytes_the_provider_returned(self) -> None:
        """Never a re-serialization of `payload`: `json.dumps` orders keys and
        escapes non-ASCII differently across versions, and the R2 key is
        content-addressed off exactly this value."""
        record = raw_record({"a": 1}, raw_bytes=b'{"a": 1}')
        lineage = build_lineage(record, native_id="n1", ctx=CTX)
        assert lineage.raw_sha256 == (
            "f9d86028c6e0d64e225186f96acb69338b2c59764df79162107f5c4bb34d1310"
        )
        assert lineage.raw_bytes == 8

    def test_leaves_the_object_key_unset(self) -> None:
        """The connector does not perform the R2 PUT (`docs/connector-spec.md`
        §2.6); a pointer to an object that was never written turns a failed
        upload into a 404 at citation time instead of a visible gap."""
        assert build_lineage(raw_record({}), native_id="n1", ctx=CTX).raw_object_key is None

    def test_stamps_the_zero_pipeline_version(self) -> None:
        """A connector has run no enrichment stage. Claiming "1.0.0" would assert
        an enrichment that never happened, and §7 makes that field the basis for
        deciding whether a stored Signal needs reprocessing."""
        lineage = build_lineage(raw_record({}), native_id="n1", ctx=CTX)
        assert lineage.pipeline_version == "0.0.0"
        assert lineage.status is SignalStatus.RAW

    def test_missing_raw_bytes_is_null_not_a_digest_of_nothing(self) -> None:
        lineage = build_lineage(raw_record({}, raw_bytes=None), native_id="n1", ctx=CTX)
        assert lineage.raw_sha256 is None and lineage.raw_bytes is None


# --------------------------------------------------------------------------- #
# mapper.py -- the field map
# --------------------------------------------------------------------------- #

REDDIT_PAYLOAD: dict[str, Any] = {
    "kind": "t3",
    "data": {
        "id": "t3_1abcde",
        "title": "  Our observability bill tripled  ",
        "selftext": "<p>We moved <b>forty</b> services.</p><p>Ingest did not change.</p>",
        "permalink": "/r/selfhosted/comments/1abcde/x/?utm_source=share",
        "created_utc": 1785000000,
        "author": "ops_gremlin",
        "author_fullname": "t2_9k2lx",
        "score": 412,
        "num_comments": 137,
        "is_self": True,
        "subreddit": "selfhosted",
        "preview": [{"u": "https://i.redd.it/abc.png", "t": "image/png"}],
    },
}


def reddit_map(**overrides: Any) -> FieldMap:
    defaults: dict[str, Any] = {
        "platform": Platform.REDDIT,
        "timestamp": FieldSpec.at("data.created_utc", required=True),
        "item_id": FieldSpec.at("data.id"),
        "url": FieldSpec.at(
            "data.permalink",
            transform=lambda v: f"https://www.reddit.com{v}" if v.startswith("/") else v,
        ),
        "title": FieldSpec.at("data.title"),
        "text": FieldSpec.at("data.selftext"),
        "text_is_html": True,
        "author_id": FieldSpec.at("data.author_fullname"),
        "author_handle": FieldSpec.at("data.author"),
        "engagement": {
            "score": FieldSpec.at("data.score"),
            "num_comments": FieldSpec.at("data.num_comments"),
            "is_self": FieldSpec.at("data.is_self"),
        },
        "metadata": {"reddit.subreddit": FieldSpec.at("data.subreddit")},
        "media": MediaMap(container="data.preview", url="u", mime_type="t"),
    }
    defaults.update(overrides)
    return FieldMap(**defaults)


class TestFieldMapDeclaration:
    """Import-time gates. Everything here is wrong with the map, not a payload."""

    def test_metadata_keys_must_be_namespaced_by_platform(self) -> None:
        """Un-namespaced keys collide across connectors in one jsonb column and
        one OpenSearch mapping (`docs/signal-model.md` §2)."""
        with pytest.raises(ValueError, match="namespaced"):
            reddit_map(metadata={"subreddit": FieldSpec.at("data.subreddit")})

    def test_a_map_that_can_never_derive_an_identity_is_refused(self) -> None:
        """Rule 1 needs an id, rule 2 a URL, rule 3 text. With none declared,
        every record it will ever see fails -- so fail now, with the map in hand."""
        with pytest.raises(ValueError, match="never derive a native_id"):
            FieldMap(platform=Platform.RSS, timestamp=FieldSpec.at("published"))

    def test_platform_unknown_is_refused(self) -> None:
        """`UNKNOWN` is a reader's fallback for a newer producer's value, not a
        declaration; identity under it belongs to no connector."""
        with pytest.raises(ValueError, match="unknown"):
            FieldMap(
                platform=Platform.UNKNOWN,
                timestamp=FieldSpec.at("t"),
                item_id=FieldSpec.at("id"),
            )

    def test_a_platform_string_is_not_a_platform(self) -> None:
        """`Platform` is a `StrEnum`, so `"reddit"` compares equal to the member
        and fails only where something calls `.value`."""
        with pytest.raises(ValueError, match="must be a Platform member"):
            FieldMap(platform="reddit", timestamp=FieldSpec.at("t"), item_id=FieldSpec.at("i"))  # type: ignore[arg-type]

    def test_an_engagement_counter_may_not_shadow_a_signal_field(self) -> None:
        """Resolution is keyed by name, so the counter would overwrite the field
        and the mapping would fail somewhere unrelated."""
        with pytest.raises(ValueError, match="may not reuse"):
            reddit_map(engagement={"title": FieldSpec.at("data.score")})

    def test_a_fieldspec_needs_a_path(self) -> None:
        with pytest.raises(ValueError, match="at least one path"):
            FieldSpec.at()


class TestFieldSpecResolution:
    """Which values count as present, and which paths win."""

    PAYLOAD: ClassVar[dict[str, Any]] = {
        "a": {"b": [{"c": "found"}]},
        "blank": "   ",
        "zero": 0,
        "false": False,
    }

    def test_walks_mappings_and_sequences(self) -> None:
        """Provider payloads nest lists inside dicts; a mapper that only handled
        mappings would push indexing back into every connector."""
        assert FieldSpec.at("a.b.0.c").resolve(self.PAYLOAD, name="x") == "found"

    def test_falls_through_to_the_next_path(self) -> None:
        """Atom carries `content`, RSS 2.0 carries `description`, and the same
        feedparser dict may hold either."""
        assert FieldSpec.at("missing", "a.b.0.c").resolve(self.PAYLOAD, name="x") == "found"

    def test_a_blank_string_is_absent_but_zero_is_not(self) -> None:
        """A score of zero is a fact about the item; `"summary": ""` is the
        provider saying it has none."""
        assert FieldSpec.at("blank", "a.b.0.c").resolve(self.PAYLOAD, name="x") == "found"
        assert FieldSpec.at("zero").resolve(self.PAYLOAD, name="x") == 0
        assert FieldSpec.at("false").resolve(self.PAYLOAD, name="x") is False

    def test_an_out_of_range_index_is_absent_not_an_error(self) -> None:
        assert FieldSpec.at("a.b.9.c", default="d").resolve(self.PAYLOAD, name="x") == "d"

    def test_required_raises_with_the_paths_it_tried(self) -> None:
        with pytest.raises(NormalizationError, match=r"\['nope'\]"):
            FieldSpec.at("nope", required=True).resolve(self.PAYLOAD, name="x", native_id="n")


class TestFieldMapToSignal:
    """The golden mapping: one payload, one canonical Signal."""

    @pytest.fixture
    def signal(self) -> Any:
        return reddit_map().to_signal(raw_record(REDDIT_PAYLOAD), CTX)

    def test_identity_is_derived_not_assigned(self, signal: Any) -> None:
        assert signal.id == signal_id(Platform.REDDIT, "t3_1abcde")
        assert signal.lineage.native_id == "t3_1abcde"

    def test_mapping_the_same_payload_twice_yields_the_same_id(self) -> None:
        """`docs/signal-model.md` §8 calls this out explicitly: re-fetching an
        item may never create a second Signal."""
        first = reddit_map().to_signal(raw_record(REDDIT_PAYLOAD), CTX)
        second = reddit_map().to_signal(raw_record(REDDIT_PAYLOAD), CTX)
        assert first.id == second.id

    def test_source_follows_the_platform(self, signal: Any) -> None:
        assert signal.source is SourceCategory.SOCIAL

    def test_url_is_canonicalized_with_the_function_identity_uses(self, signal: Any) -> None:
        """Otherwise `Signal.url` and `Signal.id` would disagree about which page
        this is."""
        assert signal.url == "https://www.reddit.com/r/selfhosted/comments/1abcde/x/"

    def test_body_is_cleaned_and_title_is_collapsed(self, signal: Any) -> None:
        assert "<b>" not in signal.content.text
        assert "forty services" in signal.content.text
        assert signal.content.title == "Our observability bill tripled"
        assert signal.content.char_count == len(signal.content.text)

    def test_engagement_keeps_counters_and_refuses_flags(self, signal: Any) -> None:
        """`True` is an `int` in Python; an unguarded coercion would file
        `is_self` as a counter of 1 and let it into a percentile cohort."""
        assert signal.engagement.raw == {"score": 412, "num_comments": 137}

    def test_normalized_axes_are_left_unset(self, signal: Any) -> None:
        """They are percentiles within a cohort (§3.4); a connector holding one
        record cannot know a percentile."""
        assert signal.engagement.available_axes() == {}

    def test_enrichment_fields_are_left_to_the_pipeline(self, signal: Any) -> None:
        assert (signal.entities, signal.topics, signal.keywords) == ([], [], [])
        assert signal.sentiment is None and signal.language.code == "und"
        assert signal.confidence == 0.0

    def test_author_carries_the_stable_id_not_the_handle(self, signal: Any) -> None:
        assert signal.author is not None
        assert signal.author.platform_author_id == "t2_9k2lx"
        assert signal.author.handle == "ops_gremlin"

    def test_a_handle_is_never_promoted_into_the_author_id(self) -> None:
        """Handles are renameable (§3.1); keying on one forks an author's history
        the first time they rename, silently."""
        payload = json.loads(json.dumps(REDDIT_PAYLOAD))
        del payload["data"]["author_fullname"]
        signal = reddit_map().to_signal(raw_record(payload), CTX)
        assert signal.author is None

    def test_media_is_classified_and_canonicalized(self, signal: Any) -> None:
        assert [m.kind for m in signal.media] == [MediaKind.IMAGE]
        assert signal.media[0].source_url == "https://i.redd.it/abc.png"

    def test_metadata_is_namespaced_and_sparse(self, signal: Any) -> None:
        assert signal.metadata == {"reddit.subreddit": "selfhosted"}

    def test_extra_metadata_is_merged(self) -> None:
        signal = reddit_map().to_signal(
            raw_record(REDDIT_PAYLOAD), CTX, extra_metadata={"reddit.listing": "new"}
        )
        assert signal.metadata["reddit.listing"] == "new"

    def test_content_digest_is_set_but_the_object_key_is_not(self, signal: Any) -> None:
        assert signal.content.raw_sha256 == signal.lineage.raw_sha256
        assert signal.content.raw_ref is None


class TestFieldMapFailureModes:
    """Every failure is one DLQ record, attributable, never a dead run."""

    def test_a_missing_timestamp_still_names_the_item(self) -> None:
        """Identity is derived before requirements are enforced precisely so that
        the most common malformed payload -- an entry with no date -- produces a
        DLQ record someone can replay."""
        payload = json.loads(json.dumps(REDDIT_PAYLOAD))
        del payload["data"]["created_utc"]
        with pytest.raises(NormalizationError) as caught:
            reddit_map().to_signal(raw_record(payload), CTX)
        assert caught.value.native_id == "t3_1abcde"
        assert caught.value.details["field"] == "timestamp"

    def test_an_unparseable_timestamp_is_a_different_message(self) -> None:
        """A feed with no date and a format this mapper does not know are
        different defects and need different fixes."""
        payload = json.loads(json.dumps(REDDIT_PAYLOAD))
        payload["data"]["created_utc"] = "last Tuesday"
        with pytest.raises(NormalizationError, match="unusable"):
            reddit_map().to_signal(raw_record(payload), CTX)

    def test_a_naive_timestamp_is_refused_unless_the_map_declares_a_zone(self) -> None:
        """Guessing UTC silently shifts a trend by hours, which is why
        `models/base.py` rejects naive datetimes at all."""
        payload = {"id": "x1", "published": "2026-07-28 14:02:11"}
        spec: dict[str, Any] = {
            "platform": Platform.RSS,
            "timestamp": FieldSpec.at("published", required=True),
            "item_id": FieldSpec.at("id"),
        }
        with pytest.raises(NormalizationError, match="assume_timezone"):
            FieldMap(**spec).to_signal(raw_record(payload), CTX)

        declared = FieldMap(**spec, assume_timezone=timezone(timedelta(hours=2)))
        signal = declared.to_signal(raw_record(payload), CTX)
        assert signal.timestamp == datetime(2026, 7, 28, 12, 2, 11, tzinfo=UTC)

    def test_a_required_field_error_carries_the_native_id(self) -> None:
        payload = json.loads(json.dumps(REDDIT_PAYLOAD))
        del payload["data"]["title"]
        with pytest.raises(NormalizationError) as caught:
            reddit_map(title=FieldSpec.at("data.title", required=True)).to_signal(
                raw_record(payload), CTX
            )
        assert caught.value.native_id == "t3_1abcde"

    def test_a_signal_validation_failure_becomes_one_dlq_record(self) -> None:
        """`BaseConnector.run()` catches only `NormalizationError`; letting
        `ValidationError` escape would abort the page and block every well-formed
        record behind it."""
        payload = json.loads(json.dumps(REDDIT_PAYLOAD))
        payload["data"]["deep"] = {"a": {"b": {"c": {"d": 1}}}}
        with pytest.raises(NormalizationError) as caught:
            reddit_map(metadata={"reddit.deep": FieldSpec.at("data.deep")}).to_signal(
                raw_record(payload), CTX
            )
        assert caught.value.native_id == "t3_1abcde"
        assert "metadata nests" in str(caught.value)
        # Model-level validators report an empty location; naming them keeps them
        # from vanishing out of the DLQ record entirely.
        assert caught.value.details["fields"] == ["<signal>"]

    def test_an_unidentifiable_payload_raises_without_an_id(self) -> None:
        payload = {"title": "no id, no link, no body"}
        unmappable = FieldMap(
            platform=Platform.RSS,
            timestamp=FieldSpec.at("published", required=True),
            text=FieldSpec.at("summary"),
        )
        with pytest.raises(NormalizationError) as caught:
            unmappable.to_signal(raw_record(payload), CTX)
        assert caught.value.native_id is None


class TestTimestampCoercion:
    """The five shapes providers actually send."""

    @pytest.mark.parametrize(
        "value",
        [
            datetime(2026, 7, 28, 14, 2, 11, tzinfo=UTC),
            datetime(2026, 7, 28, 16, 2, 11, tzinfo=timezone(timedelta(hours=2))),
            1785247331,
            1785247331000,
            "2026-07-28T14:02:11Z",
            "2026-07-28T16:02:11+02:00",
            "Tue, 28 Jul 2026 14:02:11 GMT",
            (2026, 7, 28, 14, 2, 11, 1, 209, 0),
        ],
    )
    def test_every_supported_shape_lands_on_the_same_instant(self, value: Any) -> None:
        """RSS 2.0 sends RFC 2822, feedparser sends a `struct_time`, JSON APIs
        send epochs in seconds or milliseconds, and all of them mean one moment."""
        assert to_utc_datetime(value) == datetime(2026, 7, 28, 14, 2, 11, tzinfo=UTC)

    def test_milliseconds_are_distinguished_from_seconds(self) -> None:
        """The boundary sits where no realistic event time falls, so a
        millisecond epoch is never read as a year-5138 second epoch."""
        assert to_utc_datetime(1785247331000).year == 2026
        assert to_utc_datetime(1785247331).year == 2026

    def test_a_boolean_is_not_a_timestamp(self) -> None:
        with pytest.raises(ValueError, match="boolean"):
            to_utc_datetime(True)

    def test_an_unknown_format_names_the_value(self) -> None:
        with pytest.raises(ValueError, match="last Tuesday"):
            to_utc_datetime("last Tuesday")
