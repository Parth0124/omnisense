"""Unit tests for the two deterministic Signal Engine stages: Clean and Language.

Both stages are pure functions of their input -- no network, no datastore, no
model -- which is exactly the property `docs/signal-model.md` §5.1 promises
("stages 1-3 reproduce byte-identical output for the same input and version") and
the reason `scripts/reindex.py` can rebuild the derived stores without drift. So
these tests need no fakes for infrastructure; the only injected doubles are a
language detector, which exists to prove the seam is real, and a PII redactor.

Five properties are load-bearing enough that a regression in one is a data
incident rather than a failing assertion, and each has a class here.

1. **Encoding is decided sceptically.** A mis-decoded body does not raise -- it
   succeeds and is wrong, forever, in five stores. The nastiest case has no
   exception anywhere in it: UTF-16 bytes labelled `iso-8859-1` decode *cleanly*
   under a single-byte codec, so a decoder that waits for a `UnicodeDecodeError`
   never notices. `TestEncodingLadder` pins the cases where the declaration is
   refuted rather than obeyed.
2. **Markup never reaches the body.** Not only for tidiness: `TestLanguageDetection`
   demonstrates the actual failure, a German article that a detector reads as
   English with 0.86 confidence because it is scoring `div`, `class` and `href`.
3. **Stage 1 raises rather than deciding it is fatal.** `FATAL_STAGES` is the
   pipeline's, and the tests drive a real `SignalPipeline` to show the raise
   becomes a quarantine, while a stage-3 raise becomes `partial` with the field
   at its documented empty value.
4. **Redaction is off by default and conservative when on.** It rewrites
   `content.text`, which feeds the dedup content hash and, under rule 3 of §4.1,
   `native_id` itself. A default-on redactor would fork identity silently, and an
   over-eager pattern would eat the dates and prices a report is grounded in.
5. **Language detection is seeded.** `TestSeeding` first shows an *unseeded*
   `langdetect` returning a different answer on nearly every run of the same
   text, so the seed in `LangdetectDetector` is demonstrably doing work rather
   than being defensive decoration.

Tests that drive a stage are `async def` and awaited, matching the rest of
`tests/unit/services/`. That is a requirement rather than a style preference:
`pytest-asyncio` runs in auto mode and owns the loop, and calling `asyncio.run()`
inside a sync test instead detaches the loop the plugin installed, leaving it
unclosed. Its `ResourceWarning` then fires at whatever later moment the collector
runs, and `filterwarnings = ["error"]` turns that into a failure in an unrelated
module.
"""

from __future__ import annotations

import codecs
import json
import random
from typing import Any

import pytest

from models.enums import Platform, SignalStatus, StageName, StageStatus
from models.lineage import Lineage
from models.signal import LANGUAGE_CONFIDENCE_FLOOR, Content, Signal
from services.signal_engine.cleaning import (
    CARD_PLACEHOLDER,
    EMAIL_PLACEHOLDER,
    PHONE_PLACEHOLDER,
    CleaningError,
    CleaningStage,
    ContentFamily,
    RegexRedactor,
    classify_content_type,
    clean_text,
    decode_bytes,
)
from services.signal_engine.language import (
    DETECTOR_SEED,
    MIN_ALPHA_CHARS,
    LangdetectDetector,
    LanguageStage,
    dominant_script,
    normalize_language_code,
)
from services.signal_engine.pipeline import EnrichmentContext, SignalPipeline, Stage

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Fixtures and doubles
# --------------------------------------------------------------------------- #

#: A realistic multi-paragraph article. Length matters: `trafilatura` treats a
#: one-paragraph document as a fragment and falls back to a baseline extractor
#: whose output is markedly worse, so a two-sentence fixture would be testing
#: that fallback rather than the path production traffic takes.
ARTICLE_HTML_EN = """<!doctype html>
<html><head><title>Observability costs</title>
<link rel="stylesheet" href="/static/site.css">
<script src="/static/analytics.js">var tracked = true;</script>
<style>.headline { font-size: 2rem; }</style></head>
<body class="article-page">
<nav class="navigation main-nav"><a href="/">Home</a> <a href="/business">Business</a>
<a href="/subscribe">Subscribe now</a></nav>
<article><h1>Observability costs are rising sharply</h1>
<p>The vendor raised prices by forty per cent this quarter, according to three
customers who spoke on condition of anonymity. They said the increase applied to
log ingestion as well as to metrics and traces.</p>
<p>One engineering director said the renewal quote arrived with no advance notice
at all, and that switching costs were the only reason the team did not move to a
competitor immediately after receiving it.</p></article>
<footer class="site-footer">Copyright 2026 Example Media. All rights reserved.</footer>
<noscript><img src="/pixel/track.gif?id=9" width="1" height="1"></noscript>
</body></html>"""

ARTICLE_HTML_DE = """<!doctype html>
<html><head><title>Preise</title>
<link rel="stylesheet" href="/static/site.css">
<script src="/static/app.js"></script></head>
<body class="article-page theme-light">
<nav class="navigation main-nav"><a href="/">Home</a><a href="/business">Business</a>
<a href="/technology">Technology</a><a href="/subscribe">Subscribe</a></nav>
<div id="content-wrapper" data-testid="article-body"><article>
<h1>Die Preise steigen weiter</h1>
<p>Der Anbieter hat die Preise in diesem Quartal um vierzig Prozent erhoeht.</p>
<p>Ein Kunde sagte, das Angebot sei ohne jede Vorankuendigung gekommen.</p>
</article></div>
<footer class="site-footer">Impressum Datenschutz Kontakt</footer></body></html>"""

ENGLISH_BODY = (
    "The vendor raised prices by forty per cent this quarter, according to three "
    "customers who spoke on condition of anonymity about the renewal."
)


def make_signal(text: str = "", *, title: str | None = None) -> Signal:
    """A minimal but genuinely valid Signal for stages 3+ to decorate.

    Built through `Signal.create` rather than `Signal(...)` so the derived-id
    invariant is exercised too: a factory that hand-assigned `id` would let a
    stage test pass against an object the real pipeline could never produce.
    """
    return Signal.create(
        platform=Platform.RSS,
        native_id="item-1",
        timestamp="2026-07-28T14:02:11Z",
        content=Content(text=text, title=title),
        lineage=Lineage(
            pipeline_version="1.0.0",
            connector_slug="rss",
            connector_version="0.1.0",
            sync_run_id="run_01J8XN5Q2P",
            fetched_at="2026-07-28T14:29:55Z",
            native_id="item-1",
        ),
    )


class FakeNormalize:
    """Stand-in for stage 2, so stage 1 and stage 3 can be driven in one pipeline.

    Exists only to put a Signal on the context -- `SignalPipeline` validates
    stage order, and stage 3 legitimately refuses to run without one.
    """

    name = StageName.NORMALIZE
    version = "0.0.0-test"

    @property
    def model_id(self) -> str | None:
        return None

    async def apply(self, ctx: EnrichmentContext) -> None:
        ctx.signal = make_signal(ctx.cleaned_text or "")


class StubDetector:
    """A `LanguageDetector` with a scripted answer.

    The whole point of the injected-detector seam: these tests decide what the
    detector says, so the *policy* under test (the confidence floor, the `und`
    fallback, the failure path) is exercised without depending on how langdetect
    happens to score a particular sentence.
    """

    def __init__(
        self,
        results: list[tuple[str, float]] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.id = "stub/1"
        self._results = results or []
        self._error = error
        self.calls: list[str] = []

    def probabilities(self, text: str) -> list[tuple[str, float]]:
        self.calls.append(text)
        if self._error is not None:
            raise self._error
        return self._results


class ShoutingRedactor:
    """A custom redactor, to prove the hook is a seam and not a hard-coded regex."""

    name = "shouting/1"

    def redact(self, text: str) -> str:
        return text.upper()


_WARMUP = "warm the profile tables up before any test in this module has run at all"

#: The one real detector for this module, built and warmed at **import** time.
#:
#: Two costs, and the second is not obvious. Loading langdetect's 55 language
#: profiles takes tens of milliseconds, so a per-test instance would dominate this
#: file's runtime. It also allocates several megabytes of long-lived nested dicts,
#: which is enough to change *when* CPython runs a full collection for the rest of
#: the session -- and with `filterwarnings = ["error"]` in `pyproject.toml`, a full
#: collection landing mid-test turns any third-party finalizer that emits a
#: `ResourceWarning` into a failure attributed to whatever test was running.
#: Paying the cost once, at import, keeps both out of the tests.
DETECTOR = LangdetectDetector()
DETECTOR.probabilities(_WARMUP)


def _unseeded_factory() -> Any:
    """A `langdetect` factory with seeding deliberately disabled.

    Only `TestSeeding` uses it, to demonstrate the failure the seed prevents;
    nothing in `services/` may ever construct one. It borrows the profile tables
    the detector above already loaded rather than loading a second copy -- they
    are read-only lookup data, and the independence this test needs is in the
    per-detector RNG state, not in the tables.
    """
    from langdetect.detector_factory import DetectorFactory

    loaded = DETECTOR._ensure_factory()  # see the docstring
    factory = DetectorFactory()
    factory.word_lang_prob_map = loaded.word_lang_prob_map
    factory.langlist = loaded.langlist
    factory.seed = None
    return factory


UNSEEDED_FACTORY = _unseeded_factory()


@pytest.fixture(scope="module")
def detector() -> LangdetectDetector:
    """The shared real detector. Stateless between calls, so sharing is safe."""
    return DETECTOR


# --------------------------------------------------------------------------- #
# 1. Encoding
# --------------------------------------------------------------------------- #


class TestEncodingLadder:
    """Which evidence wins, and why the obvious ladder is not enough.

    Every test here describes a real provider behaviour, not a hypothetical.
    """

    def test_declared_charset_is_trusted_first(self) -> None:
        """The provider is the only party that actually knows what it encoded."""
        body = "Café — naïve résumé"
        raw = body.encode("cp1252")
        assert decode_bytes(raw, content_type="text/html; charset=windows-1252") == body

    def test_a_wide_bom_overrides_a_declared_single_byte_charset(self) -> None:
        """The one case where the declaration is refutable, and the nastiest.

        UTF-16 bytes decode *without error* under any single-byte codec, so there
        is no exception to fall through on: an implementation that only reacts to
        `UnicodeDecodeError` stores NUL-interleaved garbage and reports success.
        """
        body = "Café — naïve résumé"
        raw = body.encode("utf-16")  # carries a BOM
        assert decode_bytes(raw, content_type="text/plain; charset=iso-8859-1") == body

    def test_utf32_bom_is_not_read_as_utf16(self) -> None:
        """UTF-32-LE begins with the UTF-16-LE mark, so BOM order is not cosmetic."""
        raw = codecs.BOM_UTF32_LE + "hello".encode("utf-32-le")
        assert decode_bytes(raw, content_type="text/plain") == "hello"

    def test_utf8_bom_is_stripped_from_the_body(self) -> None:
        """A leading U+FEFF is invisible and breaks every prefix comparison."""
        raw = codecs.BOM_UTF8 + b"Guten Morgen"
        assert decode_bytes(raw, content_type="text/plain; charset=utf-8") == "Guten Morgen"

    def test_latin1_declared_over_a_utf8_body_prefers_utf8(self) -> None:
        """RFC 2616 made ISO-8859-1 the default for `text/*`, so servers still emit
        it over UTF-8 bodies. Valid multi-byte UTF-8 does not arise by accident in
        genuine 8-bit text, so its presence outweighs a free-to-emit label."""
        body = "naïve café"
        raw = body.encode("utf-8")
        assert decode_bytes(raw, content_type="text/html; charset=ISO-8859-1") == body

    def test_pure_ascii_never_overrides_the_declaration(self) -> None:
        """ASCII is valid under both candidates and decodes identically, so it is
        no evidence at all -- a byte that is only meaningful under the declared
        codec must still be honoured."""
        raw = b"\x93quoted\x94"
        assert decode_bytes(raw, content_type="text/html; charset=iso-8859-1") == "“quoted”"

    def test_latin1_and_ascii_labels_decode_as_windows_1252(self) -> None:
        """WHATWG Encoding §4.2. True latin-1 renders smart quotes as C1 control
        characters, which then travel into the body, the embedding and any report
        that quotes it."""
        assert decode_bytes(b"Caf\xe9", content_type="text/plain; charset=us-ascii") == "Café"

    def test_in_document_declaration_is_used_when_the_transport_is_silent(self) -> None:
        """Feeds and archived files frequently arrive with no Content-Type."""
        document = "<html><head><meta charset='cp1251'></head><body>Привет</body></html>"
        assert "Привет" in decode_bytes(document.encode("cp1251"))

    def test_an_unknown_charset_label_is_ignored_rather_than_fatal(self) -> None:
        """Providers emit `charset=utf8mb4` and worse; none of that is a reason to
        quarantine an otherwise perfectly good record."""
        body = "naïve"
        raw = body.encode("utf-8")
        assert decode_bytes(raw, content_type="text/plain; charset=utf8mb4") == body

    def test_a_mislabelled_body_is_sniffed_when_there_is_evidence_to_sniff(self) -> None:
        """Statistical detection is the last rung before replacement, and it earns
        its place on genuinely non-Latin content."""
        body = "Здравствуйте, это тестовое сообщение о новых ценах. " * 3  # noqa: RUF001
        assert decode_bytes(body.encode("cp1251")).startswith("Здравствуйте")

    def test_a_short_body_is_not_sniffed(self) -> None:
        """Four bytes of French come back from a detector as two CJK ideographs.
        Below the high-byte threshold cp1252 is simply the better prior."""
        assert decode_bytes(b"Caf\xe9") == "Café"

    def test_undecodable_bytes_never_raise(self) -> None:
        """A body with replacement characters is repairable by reprocessing from
        R2; a decoder that raised would quarantine the record instead."""
        result = decode_bytes(b"\xc3\x28\xa0\xa1", content_type="text/plain; charset=utf-8")
        assert isinstance(result, str) and result

    def test_empty_bytes_decode_to_empty_string(self) -> None:
        """A media-only post is legal and must not cost an exception."""
        assert decode_bytes(b"", content_type="text/html") == ""

    def test_decoding_is_deterministic(self) -> None:
        """§5.1: stage 1 must reproduce byte-identical output for the same input."""
        raw = ("Le procès s'est tenu à Montréal. " * 6).encode("cp1252")
        assert len({decode_bytes(raw) for _ in range(10)}) == 1


# --------------------------------------------------------------------------- #
# 2. Markup, boilerplate and what survives
# --------------------------------------------------------------------------- #


class TestBodyExtraction:
    """`content.text` is "markup stripped, boilerplate removed, whitespace
    collapsed" (`docs/signal-model.md` §3.2). These pin what that means."""

    def test_navigation_scripts_and_footer_are_removed(self) -> None:
        cleaned = clean_text(ARTICLE_HTML_EN, content_type="text/html")
        assert "Subscribe now" not in cleaned
        assert "var tracked" not in cleaned
        assert "font-size" not in cleaned
        assert "All rights reserved" not in cleaned

    def test_the_article_body_survives_intact(self) -> None:
        cleaned = clean_text(ARTICLE_HTML_EN, content_type="text/html")
        assert "raised prices by forty per cent" in cleaned
        assert "switching costs were the only reason" in cleaned

    def test_paragraph_structure_is_preserved(self) -> None:
        """The chunker splits on blank lines. An extractor that returned one long
        line would push chunk boundaries into the middle of sentences for every
        long article in the corpus."""
        cleaned = clean_text(ARTICLE_HTML_EN, content_type="text/html")
        assert "\n\n" in cleaned
        assert "\n\n\n" not in cleaned

    def test_no_markup_survives(self) -> None:
        cleaned = clean_text(ARTICLE_HTML_EN, content_type="text/html")
        assert "<" not in cleaned and ">" not in cleaned

    def test_declared_plain_text_is_never_parsed_as_html(self) -> None:
        """A `text/plain` body containing `<` or `&` is prose. Handing it to an
        HTML parser would entity-decode or swallow it and silently edit the
        observation."""
        body = "Rating: 5 < 6 & 7 > 2 and that is final"
        assert clean_text(body, content_type="text/plain") == body

    def test_emoji_are_preserved(self) -> None:
        """They carry the polarity stage 5 reads. A body cleaned down to "This
        update is" has lost the entire observation."""
        cleaned = clean_text("This update is 🔥 and the pricing is 💸", content_type="text/plain")
        assert "🔥" in cleaned and "💸" in cleaned

    def test_whitespace_is_collapsed_without_losing_paragraphs(self) -> None:
        cleaned = clean_text("one   two\t\tthree\n\n\n\nnext para", content_type="text/plain")
        assert cleaned == "one two three\n\nnext para"

    def test_an_empty_document_cleans_to_an_empty_string(self) -> None:
        assert clean_text("   \n\n  ", content_type="text/html") == ""

    def test_bytes_are_decoded_before_extraction(self) -> None:
        """The two halves of stage 1 are one function, and the seam between them
        is load-bearing: an extractor handed raw bytes would either refuse them or
        decode them with its own guess, discarding the declared charset that
        `decode_bytes` was asked to honour.

        Pinned on a non-UTF-8 body specifically, because a UTF-8 fixture passes
        this test under an accidental `latin-1` decode too and would prove
        nothing.
        """
        document = "<html><body><p>Der Preis stieg um vierzig Prozent — sagte er.</p></body></html>"
        cleaned = clean_text(
            document.encode("cp1252"), content_type="text/html; charset=windows-1252"
        )
        assert cleaned == "Der Preis stieg um vierzig Prozent — sagte er."

    def test_a_plain_text_body_arrives_as_bytes_too(self) -> None:
        """Most records reach stage 1 as bytes off the wire, not as `str`. Only
        stage 2 passes a string, having pulled a body field out of a payload the
        JSON decoder already decoded."""
        assert clean_text("Café  ouvert".encode(), content_type="text/plain") == "Café ouvert"


# --------------------------------------------------------------------------- #
# 3. Content-type routing
# --------------------------------------------------------------------------- #


class TestContentTypeRouting:
    """Which of the four behaviours a media type earns."""

    @pytest.mark.parametrize(
        ("content_type", "expected"),
        [
            ("text/html; charset=utf-8", ContentFamily.MARKUP),
            ("application/xhtml+xml", ContentFamily.MARKUP),
            ("application/atom+xml", ContentFamily.MARKUP),
            ("text/plain", ContentFamily.PLAIN),
            ("text/markdown", ContentFamily.PLAIN),
            ("application/json", ContentFamily.STRUCTURED),
            ("application/vnd.api+json", ContentFamily.STRUCTURED),
            ("application/pdf", ContentFamily.BINARY),
            ("image/png", ContentFamily.BINARY),
            ("audio/mpeg", ContentFamily.BINARY),
            (None, ContentFamily.PLAIN),
            ("application/x-something-new", ContentFamily.PLAIN),
        ],
    )
    def test_classification(self, content_type: str | None, expected: ContentFamily) -> None:
        """Suffix rules matter: feeds arrive as `+xml` and APIs as `+json`, and an
        exact-string table sends both down the wrong path."""
        assert classify_content_type(content_type) is expected

    def test_a_binary_content_type_names_what_is_missing(self) -> None:
        """§4 of the build rules: no silent stub. A PDF genuinely cannot be cleaned
        without an extractor this deployment does not have, and the error says so
        rather than storing an empty body."""
        with pytest.raises(NotImplementedError, match="pdfminer"):
            clean_text(b"%PDF-1.4 ...", content_type="application/pdf")

    async def test_binary_bytes_arriving_unlabelled_are_rejected(self) -> None:
        """A NUL never appears in decoded text. Extracting "text" from a PNG would
        store a screenful of control characters as an observation."""
        ctx = EnrichmentContext(
            raw_bytes=b"\x89PNG\r\n\x1a\n\x00\x00\x00abc",
            content_type="application/octet-stream",
        )
        with pytest.raises(CleaningError, match="NUL"):
            await CleaningStage().apply(ctx)

    async def test_bomless_utf16_fails_loudly_rather_than_decoding_to_garbage(self) -> None:
        """The one body the encoding ladder cannot recover, pinned so the failure
        stays loud.

        With no BOM and no declaration, UTF-16 offers its only evidence as
        interleaved NULs -- which are not high bytes, so statistical detection is
        never consulted (deliberately: lowering that bar would hand every
        mislabelled PNG to a detector that answers confidently and decodes the
        NULs away). The body therefore reaches cp1252 and the NUL guard catches
        it. Fatal is the right outcome: the record is quarantined with its
        original intact in R2 and is repairable by reprocessing, whereas a
        NUL-interleaved "success" would be embedded and indexed forever.
        """
        ctx = EnrichmentContext(
            raw_bytes="Der Preis stieg um vierzig Prozent.".encode("utf-16-le"),
            content_type="text/plain",
        )
        with pytest.raises(CleaningError, match="NUL"):
            await CleaningStage().apply(ctx)

    async def test_a_json_record_is_parsed_rather_than_flattened(self) -> None:
        """There is no document body in a provider payload: the observation's text
        sits at a path only the connector's field map knows. Concatenating string
        leaves would put ids, URLs and timestamps into `content.text`."""
        payload = {"id": "t3_1abcde", "selftext": "the body", "url": "https://x.example"}
        ctx = EnrichmentContext(
            raw_bytes=json.dumps(payload).encode("utf-8"), content_type="application/json"
        )
        await CleaningStage().apply(ctx)
        assert ctx.payload == payload
        assert ctx.cleaned_text == ""

    async def test_an_existing_payload_is_never_overwritten(self) -> None:
        """The worker supplies the connector's verbatim payload; re-parsing the
        bytes over the top of it would discard whatever the connector resolved."""
        ctx = EnrichmentContext(
            raw_bytes=b'{"id": "from-bytes"}',
            content_type="application/json",
            payload={"id": "from-connector"},
        )
        await CleaningStage().apply(ctx)
        assert ctx.payload == {"id": "from-connector"}

    async def test_invalid_json_declared_as_json_is_fatal(self) -> None:
        ctx = EnrichmentContext(raw_bytes=b"{not json", content_type="application/json")
        with pytest.raises(CleaningError, match="not valid JSON"):
            await CleaningStage().apply(ctx)

    async def test_a_json_array_is_fatal_because_stage_2_maps_a_mapping(self) -> None:
        ctx = EnrichmentContext(raw_bytes=b"[1, 2, 3]", content_type="application/json")
        with pytest.raises(CleaningError, match="not an object"):
            await CleaningStage().apply(ctx)


# --------------------------------------------------------------------------- #
# 4. Stage 1 as a Stage
# --------------------------------------------------------------------------- #


class TestCleaningStageContract:
    """What the pipeline is entitled to assume about stage 1."""

    def test_satisfies_the_stage_protocol(self) -> None:
        assert isinstance(CleaningStage(), Stage)

    def test_model_id_is_none(self) -> None:
        """§5.1 records a model only for stages 4-6, whose output cannot be
        reproduced without knowing which model produced it. Naming one here would
        imply stage 1 needs a model to replay."""
        assert CleaningStage().model_id is None

    async def test_cleaned_text_is_populated_from_raw_bytes(self) -> None:
        ctx = EnrichmentContext(
            raw_bytes=ARTICLE_HTML_EN.encode("utf-8"), content_type="text/html; charset=utf-8"
        )
        await CleaningStage().apply(ctx)
        assert ctx.cleaned_text is not None
        assert "raised prices by forty per cent" in ctx.cleaned_text

    async def test_a_media_only_record_cleans_to_empty_rather_than_failing(self) -> None:
        """`Content.text` "may be empty for media-only posts". Raising here would
        quarantine every photo post in the corpus."""
        ctx = EnrichmentContext(raw_bytes=b"", content_type="text/html")
        await CleaningStage().apply(ctx)
        assert ctx.cleaned_text == ""

    async def test_missing_raw_bytes_is_fatal_for_a_text_record(self) -> None:
        ctx = EnrichmentContext(raw_bytes=None, content_type="text/html")
        with pytest.raises(CleaningError):
            await CleaningStage().apply(ctx)

    async def test_a_stage_1_failure_quarantines_the_record(self) -> None:
        """The stage raises; `FATAL_STAGES` -- not the stage -- makes it fatal, and
        the pipeline returns rather than raising so the DLQ record keeps the
        outcome list that makes a replay possible."""
        pipeline = SignalPipeline([CleaningStage(), FakeNormalize()])
        ctx = EnrichmentContext(raw_bytes=None, content_type="text/html")
        result = await pipeline.run(ctx)

        assert result.status is SignalStatus.QUARANTINED
        assert result.fatal_stage is StageName.CLEAN
        assert result.error == "CleaningError"
        assert not result.succeeded
        # Stage 2 never ran: a fatal stage stops the pass.
        assert [o.name for o in result.outcomes] == [StageName.CLEAN]

    async def test_a_clean_pass_reaches_enriched(self) -> None:
        pipeline = SignalPipeline([CleaningStage(), FakeNormalize()])
        ctx = EnrichmentContext(
            raw_bytes=ARTICLE_HTML_EN.encode("utf-8"), content_type="text/html; charset=utf-8"
        )
        result = await pipeline.run(ctx)

        assert result.status is SignalStatus.ENRICHED
        assert result.succeeded
        assert result.signal is not None
        assert "raised prices" in result.signal.content.text

    async def test_the_stage_is_reusable_across_records(self) -> None:
        """One instance per worker is driven concurrently, so nothing per-record
        may be kept on `self`."""
        stage = CleaningStage()
        first = EnrichmentContext(raw_bytes=b"first body here", content_type="text/plain")
        second = EnrichmentContext(raw_bytes=b"second body here", content_type="text/plain")
        await stage.apply(first)
        await stage.apply(second)
        assert first.cleaned_text == "first body here"
        assert second.cleaned_text == "second body here"


# --------------------------------------------------------------------------- #
# 5. The PII hook
# --------------------------------------------------------------------------- #


class TestPiiRedactionHook:
    """`docs/security-and-privacy.md` §6.1 -- and why the default is *off*."""

    async def test_redaction_is_off_by_default(self) -> None:
        """Redaction rewrites `content.text`, which feeds the layer-2 content hash
        and, under rule 3 of §4.1, `native_id`. Turning it on forks identity for
        every affected record, so it is a `pipeline_version` bump -- never
        something that happens because nobody passed an argument."""
        stage = CleaningStage()
        assert stage.redactor_name is None

        ctx = EnrichmentContext(
            raw_bytes=b"write to press@example.com", content_type="text/plain"
        )
        await stage.apply(ctx)
        assert ctx.cleaned_text == "write to press@example.com"

    def test_a_redactor_is_reported_by_name_when_one_is_installed(self) -> None:
        """"Was redaction on when this body was stored?" is otherwise unanswerable
        after the fact, and the answer changes what the stored text means."""
        assert CleaningStage(redactor=RegexRedactor()).redactor_name == "regex/v1"

    def test_the_default_redactor_replaces_emails(self) -> None:
        redacted = RegexRedactor().redact("mail alice.smith@example.co.uk today")
        assert redacted == f"mail {EMAIL_PLACEHOLDER} today"

    @pytest.mark.parametrize(
        "body",
        [
            "call +1 (415) 555-0132 now",
            "ring 415-555-0132 today",
            "the desk is on (020) 7946 0958 daily",
            "reach us at +44 20 7946 0958 always",
        ],
    )
    def test_the_default_redactor_replaces_phone_numbers(self, body: str) -> None:
        assert PHONE_PLACEHOLDER in RegexRedactor().redact(body)

    def test_the_default_redactor_replaces_luhn_valid_card_numbers(self) -> None:
        assert RegexRedactor().redact("charged 4111 1111 1111 1111 ok") == (
            f"charged {CARD_PLACEHOLDER} ok"
        )

    def test_a_digit_run_that_fails_luhn_is_left_alone(self) -> None:
        """Without the checksum the pattern eats every tracking number, IMEI and
        de-hyphenated ISBN-13 in the corpus."""
        body = "order 1234567890123456 shipped"
        assert RegexRedactor().redact(body) == body

    @pytest.mark.parametrize(
        "body",
        [
            "published on 2026-07-31 at noon",
            "upgrade to version 1.0.0 today",
            "the host is 192.168.0.1 internally",
            "revenue was $1,234.56 last quarter",
            "issue 2024 12 31 was reprinted",
        ],
    )
    def test_the_default_redactor_leaves_non_pii_alone(self, body: str) -> None:
        """Over-redaction is the worse failure: it corrupts the evidence a claim
        rests on, and unlike a missed phone number it is undetectable afterwards
        because the original only exists in R2."""
        assert RegexRedactor().redact(body) == body

    def test_cards_are_redacted_before_phones(self) -> None:
        """Otherwise a phone pattern consumes part of a card number and leaves the
        rest in the body -- worse than leaving all of it, because it looks
        redacted."""
        assert "1111" not in RegexRedactor().redact("4111-1111-1111-1111")

    def test_redaction_runs_after_markup_removal(self) -> None:
        """Publishers split addresses across elements. A detector run before
        extraction sees two fragments and matches neither."""
        document = (
            "<html><body><article>"
            "<p>The newsroom desk answers every weekday morning without exception, "
            "and the duty editor reads each message before the first conference.</p>"
            "<p>Write to <span>newsroom</span>@example.com if you have a tip about "
            "pricing, and someone will reply within one working day of receipt.</p>"
            "</article></body></html>"
        )
        cleaned = clean_text(document, content_type="text/html", redactor=RegexRedactor())
        assert "newsroom@example.com" not in cleaned
        assert EMAIL_PLACEHOLDER in cleaned

    async def test_a_custom_redactor_plugs_in(self) -> None:
        """The seam is the point: a real deployment supplies a detector with a
        model behind it, and `RegexRedactor` is only the fallback."""
        ctx = EnrichmentContext(raw_bytes=b"quiet body", content_type="text/plain")
        stage = CleaningStage(redactor=ShoutingRedactor())
        await stage.apply(ctx)
        assert ctx.cleaned_text == "QUIET BODY"
        assert stage.redactor_name == "shouting/1"


# --------------------------------------------------------------------------- #
# 6. Why the detector is seeded
# --------------------------------------------------------------------------- #


class TestSeeding:
    """The reproducibility that stage determinism is supposed to provide."""

    #: Short, Latin-script, and genuinely ambiguous between English and French --
    #: exactly the shape where langdetect's random n-gram sampling dominates the
    #: answer. Long unambiguous prose would pass this test with no seed at all
    #: and prove nothing.
    AMBIGUOUS = "Chat noir. Photo. Menu. Restaurant. Fin."

    def test_unseeded_langdetect_is_not_reproducible(self) -> None:
        """The failure the seed exists to prevent, demonstrated rather than
        asserted: the same body classifies differently across runs, so a Signal
        reprocessed on another worker silently changes language."""
        answers = set()
        for step in range(40):
            random.seed(step * 7919)
            detector = UNSEEDED_FACTORY.create()
            detector.append(self.AMBIGUOUS)
            answers.add(
                tuple((item.lang, round(item.prob, 6)) for item in detector.get_probabilities())
            )
        assert len(answers) > 1

    def test_the_seeded_detector_is_stable_under_hostile_global_rng(
        self, detector: LangdetectDetector
    ) -> None:
        """langdetect seeds by calling `random.seed()` on the process-global
        module, so the test perturbs that state between runs: a detector that
        merely inherited it would drift."""
        answers = set()
        for step in range(40):
            random.seed(step * 7919)
            answers.add(tuple(detector.probabilities(self.AMBIGUOUS)))
        assert len(answers) == 1

    def test_a_second_detector_over_the_same_profiles_agrees(
        self, detector: LangdetectDetector
    ) -> None:
        """Determinism has to hold across processes, not just within one instance
        -- that is what makes `scripts/reindex.py` free of drift. A fresh factory
        with a fresh RNG is the closest a unit test gets to a second worker."""
        from langdetect.detector_factory import DetectorFactory

        second = DetectorFactory()
        second.word_lang_prob_map = UNSEEDED_FACTORY.word_lang_prob_map
        second.langlist = UNSEEDED_FACTORY.langlist
        second.seed = DETECTOR_SEED

        run = second.create()
        run.append(self.AMBIGUOUS)
        replica = tuple((item.lang, item.prob) for item in run.get_probabilities())
        assert tuple(detector.probabilities(self.AMBIGUOUS)) == replica


# --------------------------------------------------------------------------- #
# 7. Detection policy
# --------------------------------------------------------------------------- #


class TestLanguageDetection:
    """§3.3: detect after cleaning, and refuse below the confidence floor."""

    def test_markup_skews_detection_which_is_why_stage_3_follows_stage_1(
        self, detector: LangdetectDetector
    ) -> None:
        """The concrete failure, not a general worry. A German article's markup is
        English-shaped ASCII in volume -- `div`, `class`, `href`, `Subscribe` --
        and the detector dutifully scores it."""
        raw_top = detector.probabilities(ARTICLE_HTML_DE)[0]
        cleaned = clean_text(ARTICLE_HTML_DE, content_type="text/html")
        cleaned_top = detector.probabilities(cleaned)[0]

        assert raw_top[0] == "en"  # wrong, and confidently so
        assert cleaned_top[0] == "de"
        assert cleaned_top[1] > LANGUAGE_CONFIDENCE_FLOOR

    async def test_a_confident_detection_is_recorded(self) -> None:
        signal = make_signal(ENGLISH_BODY)
        ctx = EnrichmentContext(signal=signal)
        await LanguageStage(StubDetector([("en", 0.99)])).apply(ctx)

        assert signal.language.code == "en"
        assert signal.language.confidence == pytest.approx(0.99)
        assert signal.language.is_determinate

    async def test_below_the_floor_records_und_but_keeps_the_measurement(self) -> None:
        """`und` is excluded from language-filtered retrieval, whereas a
        wrong-but-confident `en` puts a Portuguese review inside an English-only
        evidence set. The measured confidence is retained because it is what makes
        the floor tunable later from stored data."""
        signal = make_signal(ENGLISH_BODY)
        ctx = EnrichmentContext(signal=signal)
        below = LANGUAGE_CONFIDENCE_FLOOR - 0.05
        await LanguageStage(StubDetector([("pt", below)])).apply(ctx)

        assert signal.language.code == "und"
        assert signal.language.confidence == pytest.approx(below)
        assert not signal.language.is_determinate

    async def test_the_highest_scoring_candidate_wins(self) -> None:
        signal = make_signal(ENGLISH_BODY)
        ctx = EnrichmentContext(signal=signal)
        await LanguageStage(StubDetector([("fr", 0.2), ("es", 0.79)])).apply(ctx)
        assert signal.language.code == "es"

    async def test_a_detector_probability_above_one_does_not_fail_the_stage(self) -> None:
        """`Score` is a validated range, so floating-point noise from summing
        per-language probabilities would otherwise turn a successful detection
        into a stage failure."""
        signal = make_signal(ENGLISH_BODY)
        ctx = EnrichmentContext(signal=signal)
        await LanguageStage(StubDetector([("en", 1.0000001)])).apply(ctx)
        assert signal.language.confidence == 1.0

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("zh-cn", "zh-CN"), ("zh-tw", "zh-TW"), ("EN", "en"), ("pt-br", "pt-BR"), ("", "und")],
    )
    def test_language_codes_are_normalized_to_bcp47(self, raw: str, expected: str) -> None:
        """A retrieval filter comparing against `"zh-CN"` silently matches nothing
        if the stored value is `"zh-cn"`."""
        assert normalize_language_code(raw) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Guten Morgen alle", "Latn"),
            ("Здравствуйте всем", "Cyrl"),
            ("日本語のテキストです", "Jpan"),
            ("한국어 텍스트입니다", "Hang"),
            ("مرحبا بالعالم", "Arab"),
            ("12345 !!! 🔥", None),
        ],
    )
    def test_script_is_measured_from_the_text(self, text: str, expected: str | None) -> None:
        """Measured, not inferred from the language code: romanized Japanese is
        `ja` written in `Latn`, and a table keyed on the code would confidently
        report `Jpan` for a body with no Japanese characters in it."""
        assert dominant_script(text) == expected

    async def test_the_detector_sees_the_cleaned_body(self) -> None:
        signal = make_signal(ENGLISH_BODY)
        stub = StubDetector([("en", 0.99)])
        await LanguageStage(stub).apply(EnrichmentContext(signal=signal))
        assert stub.calls == [ENGLISH_BODY]


# --------------------------------------------------------------------------- #
# 8. Stage 3 as a Stage
# --------------------------------------------------------------------------- #


class TestLanguageStageContract:
    """Degradable: failure costs the field, never the Signal."""

    def test_satisfies_the_stage_protocol(self) -> None:
        assert isinstance(LanguageStage(StubDetector()), Stage)

    async def test_model_id_is_none_but_the_detector_is_still_recorded(self) -> None:
        """The detector's identity is not lost -- it lands on `language.detector`,
        which is where a consumer looking at a language would think to check."""
        stage = LanguageStage(StubDetector([("en", 0.99)]))
        assert stage.model_id is None
        assert stage.detector_id == "stub/1"

        signal = make_signal(ENGLISH_BODY)
        await stage.apply(EnrichmentContext(signal=signal))
        assert signal.language.detector == "stub/1"

    async def test_an_empty_body_is_und_without_consulting_the_detector(self) -> None:
        """A media-only post has no words by design. Letting the detector raise on
        it would demote a perfectly good photo Signal to `partial` and dock its
        confidence for containing a picture."""
        signal = make_signal("")
        stub = StubDetector([("en", 0.99)])
        await LanguageStage(stub).apply(EnrichmentContext(signal=signal))

        assert signal.language.code == "und"
        assert stub.calls == []

    async def test_a_very_short_body_is_und_without_consulting_the_detector(self) -> None:
        """On "ok" or "+1" there are almost no n-grams to weigh, so the detector
        returns the highest-prior language *with high reported probability* --
        sailing straight past the confidence floor."""
        signal = make_signal("ok " * 2)
        stub = StubDetector([("en", 0.99)])
        await LanguageStage(stub).apply(EnrichmentContext(signal=signal))

        assert len(signal.content.text.replace(" ", "")) < MIN_ALPHA_CHARS
        assert signal.language.code == "und"
        assert stub.calls == []

    async def test_a_detector_with_no_opinion_is_und_and_not_a_failure(self) -> None:
        signal = make_signal(ENGLISH_BODY)
        await LanguageStage(StubDetector([])).apply(EnrichmentContext(signal=signal))
        assert signal.language.code == "und"
        assert signal.language.confidence == 0.0

    async def test_running_before_normalize_fails_loudly(self) -> None:
        """A wiring bug, and an `AttributeError` on `None` three frames deeper is a
        poor way to discover it."""
        with pytest.raises(RuntimeError, match="no Signal on the context"):
            await LanguageStage(StubDetector()).apply(EnrichmentContext())

    async def test_a_detector_failure_degrades_to_und_and_marks_the_signal_partial(self) -> None:
        """The stage does not catch its own exception: it raises, and the pipeline
        decides. The Signal survives with `language` at its documented empty
        value, which is exactly what `Language()` already defaults to."""
        pipeline = SignalPipeline(
            [
                CleaningStage(),
                FakeNormalize(),
                LanguageStage(StubDetector(error=RuntimeError("profiles unavailable"))),
            ]
        )
        ctx = EnrichmentContext(
            raw_bytes=ENGLISH_BODY.encode("utf-8"), content_type="text/plain; charset=utf-8"
        )
        result = await pipeline.run(ctx)

        assert result.status is SignalStatus.PARTIAL
        assert result.succeeded  # partial is still retrievable and citable
        assert result.failed_stages == [StageName.LANGUAGE]
        assert result.signal is not None
        assert result.signal.language.code == "und"

        record = result.signal.lineage.latest_stages()[StageName.LANGUAGE]
        assert record.status is StageStatus.FAILED
        # Only the exception class -- a provider message can echo fetched content.
        assert record.error == "RuntimeError"

    async def test_a_full_deterministic_pass_with_the_real_detector(
        self, detector: LangdetectDetector
    ) -> None:
        """End to end over the two stages this module owns, with no fakes at all:
        HTML bytes in, a German-language Signal out."""
        pipeline = SignalPipeline([CleaningStage(), FakeNormalize(), LanguageStage(detector)])
        ctx = EnrichmentContext(
            raw_bytes=ARTICLE_HTML_DE.encode("utf-8"), content_type="text/html; charset=utf-8"
        )
        result = await pipeline.run(ctx)

        assert result.status is SignalStatus.ENRICHED
        assert result.signal is not None
        assert result.signal.language.code == "de"
        assert result.signal.language.script == "Latn"
        assert result.signal.language.detector is not None
        assert result.signal.language.detector.startswith("langdetect/")
        assert "Impressum" not in result.signal.content.text
