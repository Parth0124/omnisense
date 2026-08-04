"""Enrichment stage 4: entity mentions, topics and keywords.

`docs/signal-model.md` §5.1 gives this stage `content.text` and `language`, and
asks for three fields back -- `entities`, `topics`, `keywords` -- all three
degrading to `[]` together. Topics and keywords live in
`services/signal_engine/keywords.py`; this module owns extraction, the LLM call
that produces it, and the one thing that makes the difference between a citation
system and a plausible-looking one: **character offsets that are actually
correct**.

## Why offsets are the hard part

`EntityMention.start` and `.end` index into `Signal.content.text`. Everything
downstream trusts them absolutely: the UI highlights `text[start:end]` when a
report cites a Signal, and `agents/critic/` walks the same span to check that a
claim rests on the words it says it does. An offset that is wrong by four
characters does not fail -- it highlights the wrong span, forever, in every
citation of that Signal, and nothing anywhere raises.

Models get offsets wrong constantly, and for structural reasons rather than
sloppiness:

- they count in tokens, not characters, and reconstruct a character index by
  arithmetic;
- a model trained on JavaScript tooling counts UTF-16 code units, so every
  astral character -- emoji, rare CJK, mathematical alphanumerics -- shifts every
  subsequent offset by one, silently, only for documents that contain one;
- they normalize as they read: collapsed whitespace, straightened quotes, a
  stripped leading article. Each normalization moves the index it reports.

So this module treats a model-reported offset as a **hint, never as truth**.
Every mention is verified with `text[start:end] == surface`, and one of three
things happens:

1. it matches -- the offset is used as reported;
2. it does not match but the surface is findable -- the mention is re-located by
   search, using the reported offset only to disambiguate which occurrence was
   meant;
3. the surface is not in the text at all -- the mention is **dropped**.

Dropping is the important case, and it is why the model is asked for the surface
string alongside the offsets even though the offsets alone would be smaller: a
mention with no verifiable span is either a hallucination or an offset we cannot
repair, and there is no third possibility worth guessing at. A missing mention
costs recall on one Signal. A wrong span is a false citation, which is the one
failure this whole system exists to not produce.

## What this stage does not do

**No resolution.** `candidate_ids` are extraction *hints* -- blocking keys for
`graph/resolution/`, which is the only thing allowed to decide that a mention
refers to a canonical `Entity`. `resolved_id` and `link_score` are left `None`
here, permanently. The split is what lets resolution be re-run and corrected
without re-running extraction (`models/entity.py`).

**No provider construction, and no error handling.** The `LLMProvider` arrives as
a constructor argument (Design Doc §15) so every test drives a ten-line fake. And
when the provider fails, this stage raises: `services/signal_engine/pipeline.py`
owns the fatal-versus-degradable decision, reads it from `FATAL_STAGES`, and
records the failure in `lineage.stages[]`. A stage that caught its own
`LLMSchemaError` and returned quietly would report `status = "ok"` with no
entities, claim the full 0.35 of `extraction_quality` that `STAGE_QUALITY_WEIGHTS`
gives this stage, and make a prompt regression indistinguishable from a document
that genuinely mentions nobody.

The line between raising and tolerating is drawn once, deliberately:

| Failure | Response | Why |
| --- | --- | --- |
| Provider error, timeout, rate limit | raise | Retryable; the sweeper re-drives `partial` rows |
| Whole response fails the schema | raise | A prompt or schema defect, and must stay countable |
| One item inside a valid response is junk | drop the item | Four good mentions must not be lost to a fifth bad one |

## Cost

The fast tier, always (`LLM_MODEL_FAST`). This call happens once per ingested
Signal -- the highest-volume LLM call in the platform by orders of magnitude --
so tier choice here dominates the model bill. The stage calls the provider
directly rather than going through `services/llm/router.py`: the router's value
is tier shedding and a per-investigation token budget, and neither applies here.
The fast tier has nowhere to shed to (`_SHED_LADDER` terminates at it), and
ingestion has no `RunBudget` because it is not an investigation.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.core.config import LLMSettings, get_settings
from models.entity import EntityMention
from models.enums import EntityType, StageName
from services.llm.provider import LLMProvider
from services.signal_engine.keywords import (
    TOPIC_VOCABULARY,
    extract_keywords,
    select_topics,
)
from services.signal_engine.pipeline import EnrichmentContext

__all__ = [
    "EXTRACTION_PROMPT_VERSION",
    "MAX_EXTRACTION_CHARS",
    "MAX_EXTRACTION_OUTPUT_TOKENS",
    "MAX_MENTIONS",
    "STAGE_VERSION",
    "EntityExtraction",
    "EntityExtractionStage",
    "ExtractedMention",
    "ProposedTopic",
    "candidate_ids_for",
    "coerce_entity_type",
    "locate_span",
    "resolve_mentions",
]


STAGE_VERSION: Final = "1.0.0"
"""Semantic version of this stage implementation, written to `lineage.stages[]`.

Bumped for a prompt change as well as a code change. The prompt is an input to a
non-deterministic function, so two Signals extracted under different prompts are
not comparable, and `docs/signal-model.md` §5.1 records the stage version
precisely so that question is answerable afterwards.
"""

EXTRACTION_PROMPT_VERSION: Final = "signal_engine.entities/v1"
"""Identifier for the prompt below.

The prompt is a module constant rather than a file under `prompts/` because
`prompts/loader.py` -- which is what would give it an id, a version and a content
hash -- is not implemented yet. That is a real gap: today a prompt change is
invisible unless someone also remembers to bump `STAGE_VERSION`. When the loader
lands, this constant becomes the prompt id and the text moves to
`prompts/signal_engine/entities/v1.md` unchanged.
"""

MAX_EXTRACTION_CHARS: Final = 12_000
"""How much of `content.text` is sent to the model.

A **prefix**, never a middle slice, and that is load-bearing: offsets reported
against a prefix are valid offsets into the whole string, while offsets reported
against a window starting at character 5,000 would all be short by 5,000 and
would verify against the wrong spans. Entities in the tail of a very long
document are missed; that is a recall cost, taken knowingly, in exchange for a
bounded per-Signal price on a call that runs on every record ingested.
"""

MAX_EXTRACTION_OUTPUT_TOKENS: Final = 2048
"""Output ceiling for the extraction call.

Roughly `MAX_MENTIONS` mentions plus topics with room to spare. Bounding output
is the other half of cost control -- a model that decides to enumerate every noun
in a long article is otherwise billed for it. A response truncated by this
ceiling arrives as an incomplete tool call and surfaces as `LLMSchemaError`,
which the pipeline records as a failed stage: visible, not silent.
"""

MAX_MENTIONS: Final = 64
"""Cap on emitted mentions per Signal.

Not a quality judgement -- a bound. Every mention becomes a `MENTIONS` edge in
Neo4j (`docs/knowledge-graph.md`), so an unbounded list turns one pathological
document into thousands of graph writes.
"""

MAX_TOPICS: Final = 6
MAX_KEYWORDS: Final = 12

_MAX_OCCURRENCES: Final = 128
"""How many occurrences of one surface are considered when re-locating.

A short surface in a long document can occur hundreds of times; past the first
hundred the choice is arbitrary anyway, and the cap keeps repair linear in a
bounded constant rather than in the document.
"""


# --------------------------------------------------------------------------- #
# The wire schema: what the model is asked to return
# --------------------------------------------------------------------------- #


class ExtractedMention(BaseModel):
    """One mention as the *model* reported it. Unverified by construction.

    Deliberately permissive, and each looseness is there to stop one bad item
    from destroying a good response:

    - `extra="ignore"` -- a model that adds a `confidence` field it was not
      asked for should not fail the batch.
    - offsets are `int | None`, with junk coerced to `None` rather than raising.
      `None` means "no usable hint", which the locator already handles by
      searching. Letting a single `"start": "unknown"` raise would discard every
      other mention in the same response.
    - `type` is a free string, mapped by `coerce_entity_type()` afterwards, so a
      model answering `ORG` is understood rather than rejected.

    What is *not* loose: nothing here is trusted. Every field is re-derived or
    verified against the text before an `EntityMention` is built.
    """

    model_config = ConfigDict(extra="ignore")

    surface: str = ""
    type: str = ""
    start: int | None = None
    end: int | None = None
    candidates: list[str] = Field(default_factory=list)

    @field_validator("surface", "type", mode="before")
    @classmethod
    def _coerce_text(cls, value: object) -> str:
        """Non-strings become empty, which the caller then drops as unusable.

        Pydantic will not coerce `42` into `"42"` in lax mode, so without this a
        model that emitted a numeric surface would fail the *whole* response.
        """
        return value if isinstance(value, str) else ""

    @field_validator("start", "end", mode="before")
    @classmethod
    def _coerce_offset(cls, value: object) -> int | None:
        """A non-integer offset becomes `None`: an absent hint, not an error."""
        if isinstance(value, bool) or value is None:
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            try:
                return int(value.strip())
            except ValueError:
                return None
        return None

    @field_validator("candidates", mode="before")
    @classmethod
    def _coerce_candidates(cls, value: object) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, Sequence):
            return [item for item in value if isinstance(item, str)]
        return []


class ProposedTopic(BaseModel):
    """A topic the model proposes, before it is checked against the closed set."""

    model_config = ConfigDict(extra="ignore")

    topic: str = ""
    score: float = 0.0

    @field_validator("topic", mode="before")
    @classmethod
    def _coerce_topic(cls, value: object) -> str:
        return value if isinstance(value, str) else ""

    @field_validator("score", mode="before")
    @classmethod
    def _coerce_score(cls, value: object) -> float:
        """Out-of-range and unparseable scores collapse to 0.0.

        `select_topics()` clamps as well; this exists so a `"score": "high"`
        does not fail validation for the whole response.
        """
        if isinstance(value, bool):
            return 0.0
        if isinstance(value, int | float):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                return 0.0
        return 0.0


class EntityExtraction(BaseModel):
    """The full structured response for one Signal.

    Both lists default to empty so a document with nothing to extract can be
    answered with `{}`. That is a legitimate answer -- "no entities" is
    information -- and treating a missing key as a schema failure would turn the
    most common short-post outcome into a failed stage.
    """

    model_config = ConfigDict(extra="ignore")

    mentions: list[ExtractedMention] = Field(default_factory=list)
    topics: list[ProposedTopic] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# The prompt
# --------------------------------------------------------------------------- #


def _build_system_prompt() -> str:
    """Compose the system prompt, including the closed topic vocabulary.

    The vocabulary goes in the **system** prompt, not the user turn, for a cost
    reason: it is identical on every one of the millions of extraction calls this
    stage will make, so it belongs in the cacheable prefix where the provider
    bills it at cache-read rates (`LLMResponse.cached_tokens` exists to make that
    visible). Putting it in the user turn would re-bill ~200 tokens per Signal at
    full price for a string that never changes.
    """
    slugs = "\n".join(f"- {definition.slug}: {definition.label}" for definition in TOPIC_VOCABULARY)
    types = ", ".join(member.value for member in EntityType if member is not EntityType.UNKNOWN)
    return (
        "You extract structured facts from a single piece of text for a competitive "
        "intelligence system. You are precise and you never invent.\n"
        "\n"
        "MENTIONS. Report every mention of a real, named thing in the BODY text. For each:\n"
        f"- type: one of {types}. Use the closest fit.\n"
        "- surface: the exact substring as it appears in the BODY, copied character for "
        "character. Do not correct spelling, expand abbreviations, change case, or trim "
        "words. If you cannot copy it exactly, do not report it.\n"
        "- start / end: 0-based character offsets into the BODY such that "
        "BODY[start:end] is exactly the surface. Count characters, not tokens and not "
        "bytes. An emoji or a CJK character is ONE character.\n"
        "- candidates: up to three lowercase identifier guesses for the thing named, "
        "such as ent_datadog. These are hints for a later resolution step, not answers.\n"
        "\n"
        "Report each occurrence separately when a name appears more than once, and make the "
        "offsets distinguish them. Do not report pronouns, generic nouns, or anything that "
        "appears only in the TITLE. Skip a mention rather than guess at its position.\n"
        "\n"
        "TOPICS. Assign up to six topics from this closed list and nothing outside it. "
        "Score each 0.0-1.0 by how central it is to the text.\n"
        f"{slugs}\n"
        "\n"
        "Return no prose. If the text names nothing, return empty lists."
    )


SYSTEM_PROMPT: Final = _build_system_prompt()


def _build_user_prompt(*, title: str | None, language: str, body: str) -> str:
    """Frame one Signal for extraction.

    The title is given as context but explicitly fenced off, because
    `EntityMention` offsets are defined against `content.text` alone and there is
    no field able to hold an offset into a title. A mention the model takes from
    the title cannot be located in the body and is dropped by
    `resolve_mentions()` -- correct, but a wasted mention, so the prompt says so
    rather than relying on the drop.
    """
    parts = [f"LANGUAGE: {language}"]
    if title:
        parts.append(f"TITLE (context only, do not extract offsets from it):\n{title}")
    parts.append(f"BODY (offsets are into this text):\n{body}")
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# Offset verification and repair
# --------------------------------------------------------------------------- #


def locate_span(
    text: str,
    surface: str,
    *,
    reported_start: int | None = None,
    reported_end: int | None = None,
    claimed: Sequence[tuple[int, int]] = (),
) -> tuple[int, int] | None:
    """Find the true span of `surface` in `text`, or `None` if there is not one.

    The one function in this module that must never be wrong. Its contract: if it
    returns `(start, end)` then `text[start:end]` is a real, non-empty substring
    of `text` that corresponds to `surface`. If it cannot promise that, it returns
    `None` and the mention is dropped.

    Four passes, cheapest and most trustworthy first:

    1. **Verify.** If the reported offsets slice out exactly `surface`, the model
       was right; use them. This is the overwhelmingly common case and costs one
       slice and one comparison.
    2. **Exact search.** The surface is right and the arithmetic was wrong -- the
       UTF-16 drift case, and the token-to-character case. Every occurrence is
       collected and the one *nearest the reported offset* is chosen, because a
       model that is off by three characters meant the occurrence it was pointing
       at, not the first one in the document. Taking `text.find()` blindly is the
       subtle bug this avoids: in "Datadog costs more than Datadog used to", the
       second mention would be repaired onto the first one's span and the two
       citations would highlight the same words.
    3. **Case-insensitive search.** The model normalized capitalization. Matched
       with `re.IGNORECASE` rather than by lower-casing the text, because
       `str.lower()` and `str.casefold()` can change a string's *length* -- German
       "ß" folds to "ss" -- which would corrupt every offset after it. That is the
       identical class of bug this function exists to catch, and introducing it in
       the repair path would be an unusually poor joke.
    4. **Whitespace-flexible search.** The surface straddles a line break or a
       double space that the model collapsed. Only attempted for multi-word
       surfaces, since for a single word it is the same query as pass 3.

    `claimed` is the list of spans already taken by an earlier mention of the same
    surface, and they are excluded. That handles two distinct cases with one
    mechanism: a duplicate mention emitted twice, and two genuine mentions that
    both need repair and would otherwise collapse onto the same occurrence. When
    every occurrence is claimed the extra mention is a hallucination -- the model
    reported the name more often than it appears -- and `None` drops it.
    """
    needle = surface.strip()
    if not needle or not text:
        return None

    taken = frozenset(claimed)

    if (
        reported_start is not None
        and reported_end is not None
        and 0 <= reported_start < reported_end <= len(text)
        and text[reported_start:reported_end] == needle
    ):
        # Verified. Still refused if already claimed: an identical mention
        # reported twice is one mention, not two.
        return None if (reported_start, reported_end) in taken else (reported_start, reported_end)

    for spans in (
        _exact_spans(text, needle),
        _regex_spans(text, re.compile(re.escape(needle), re.IGNORECASE)),
        _flexible_spans(text, needle),
    ):
        free = [span for span in spans if span not in taken]
        if free:
            return _nearest(free, reported_start)
    return None


def _exact_spans(text: str, needle: str) -> list[tuple[int, int]]:
    """Every literal occurrence of `needle`, left to right, capped."""
    spans: list[tuple[int, int]] = []
    index = text.find(needle)
    while index != -1 and len(spans) < _MAX_OCCURRENCES:
        spans.append((index, index + len(needle)))
        index = text.find(needle, index + 1)
    return spans


def _regex_spans(text: str, pattern: re.Pattern[str]) -> list[tuple[int, int]]:
    """Match spans for `pattern`, capped, empty matches discarded.

    Spans come from the *original* text, so no case- or whitespace-folding ever
    touches the indices that get emitted.
    """
    spans: list[tuple[int, int]] = []
    for match in pattern.finditer(text):
        if match.end() > match.start():
            spans.append((match.start(), match.end()))
        if len(spans) >= _MAX_OCCURRENCES:
            break
    return spans


def _flexible_spans(text: str, needle: str) -> list[tuple[int, int]]:
    """Occurrences allowing any whitespace run where the surface had a space."""
    chunks = needle.split()
    if len(chunks) < 2:
        return []
    pattern = re.compile(r"\s+".join(re.escape(chunk) for chunk in chunks), re.IGNORECASE)
    return _regex_spans(text, pattern)


def _nearest(spans: Sequence[tuple[int, int]], reported_start: int | None) -> tuple[int, int]:
    """The span closest to the reported start, or the first when there is no hint."""
    if reported_start is None:
        return spans[0]
    return min(spans, key=lambda span: (abs(span[0] - reported_start), span[0]))


# --------------------------------------------------------------------------- #
# Types and candidate hints
# --------------------------------------------------------------------------- #

_TYPE_ALIASES: Final[Mapping[str, EntityType]] = {
    "org": EntityType.COMPANY,
    "organization": EntityType.COMPANY,
    "organisation": EntityType.COMPANY,
    "corporation": EntityType.COMPANY,
    "business": EntityType.COMPANY,
    "brand": EntityType.COMPANY,
    "vendor": EntityType.COMPANY,
    "service": EntityType.PRODUCT,
    "app": EntityType.PRODUCT,
    "application": EntityType.PRODUCT,
    "software": EntityType.PRODUCT,
    "tool": EntityType.PRODUCT,
    "per": EntityType.PERSON,
    "people": EntityType.PERSON,
    "individual": EntityType.PERSON,
    "theme": EntityType.TOPIC,
    "concept": EntityType.TOPIC,
    "subject": EntityType.TOPIC,
    "tech": EntityType.TECHNOLOGY,
    "framework": EntityType.TECHNOLOGY,
    "library": EntityType.TECHNOLOGY,
    "protocol": EntityType.TECHNOLOGY,
    "standard": EntityType.TECHNOLOGY,
    "language": EntityType.TECHNOLOGY,
    "loc": EntityType.REGION,
    "location": EntityType.REGION,
    "gpe": EntityType.REGION,
    "country": EntityType.REGION,
    "city": EntityType.REGION,
    "place": EntityType.REGION,
    "geo": EntityType.REGION,
    "conference": EntityType.EVENT,
    "incident": EntityType.EVENT,
    "launch": EntityType.EVENT,
}
"""Common NER-tagset spellings mapped onto the Neo4j node labels in `EntityType`.

Models answer in whatever tagset dominated their training data -- CoNLL's `ORG`
and `LOC`, OntoNotes' `GPE`, or plain English. `EntityType` is case-insensitive
already (`TolerantStrEnum`), so this table only needs the spellings that differ
in more than case.
"""


def coerce_entity_type(raw: str) -> EntityType:
    """Map a model-reported type onto the closed `EntityType` set.

    Falls back to `EntityType.UNKNOWN` rather than dropping the mention. The span
    is the valuable part -- it still highlights correctly, still counts as a
    mention, and `graph/resolution/` can type it later from the canonical entity
    it resolves to. Discarding a verified span because its label was unfamiliar
    would throw away the expensive, trustworthy half to punish the cheap half.
    """
    return _TYPE_ALIASES.get(raw.strip().casefold(), EntityType(raw))


_CANDIDATE_ID_RE: Final = re.compile(r"^[a-z0-9][a-z0-9._:-]{0,63}$")
_SLUG_NOISE: Final = re.compile(r"[^\w]+", re.UNICODE)


def candidate_ids_for(surface: str, proposed: Iterable[str] = (), *, limit: int = 4) -> list[str]:
    """Blocking hints for `graph/resolution/`, best first. Never a resolution.

    The first entry is derived deterministically from the surface -- `"Datadog"`
    becomes `ent_datadog` -- and it is first because it is the only one that is
    reproducible. It is a *lookup key*, and it may well name nothing: resolution
    treats a miss as "no candidate from this hint" and proceeds with its own
    blocking. Anything the model proposed follows, shape-checked, because a model
    id is occasionally better ("ent_datadog_inc") and occasionally a sentence.

    `resolved_id` and `link_score` stay `None`. Deciding which canonical entity a
    mention refers to needs the alias table, the graph and the corpus, none of
    which exist inside an enrichment stage, and a guess made here would be
    indistinguishable downstream from a real resolution
    (`docs/signal-model.md` §3, `models/entity.py`).
    """
    hints: list[str] = []
    derived = _slugify(surface)
    if derived:
        hints.append(f"ent_{derived}")
    for raw in proposed:
        candidate = raw.strip().casefold()
        if _CANDIDATE_ID_RE.fullmatch(candidate) and candidate not in hints:
            hints.append(candidate)
        if len(hints) >= limit:
            break
    return hints[:limit]


def _slugify(surface: str) -> str:
    """Lowercase, punctuation-free key for blocking. Empty when nothing survives."""
    return _SLUG_NOISE.sub("_", surface.strip().casefold()).strip("_")[:48]


# --------------------------------------------------------------------------- #
# Raw mentions -> verified EntityMentions
# --------------------------------------------------------------------------- #


def resolve_mentions(
    raw_mentions: Iterable[ExtractedMention],
    *,
    text: str,
    limit: int = MAX_MENTIONS,
) -> list[EntityMention]:
    """Verify, repair or drop every reported mention.

    The surface written onto the `EntityMention` is always `text[start:end]` --
    the document's own bytes -- never the string the model sent. Those differ
    whenever repair went through the case-insensitive or whitespace-flexible
    pass, and `EntityMention.surface` is documented as "the literal text as it
    appeared". Taking the model's version would make `surface` and the span
    disagree, which is the same lie as a wrong offset wearing a different hat.

    Emitted in document order, outer spans before the inner spans they contain
    (`Apple Vision Pro` before the `Apple` inside it). That is the order a nested
    highlighter needs in order to open tags correctly, and the model's own
    ordering carries no information the schema can hold -- there is no rank
    field. Sorting also makes the list stable, so reprocessing that returns the
    same set returns it identically.

    Truncation to `limit` happens *before* sorting, in the model's own order, so
    what survives a cap is what the model thought mattered most. Sorting first
    would quietly substitute "earliest in the document" for "most important".
    """
    accepted: list[EntityMention] = []
    claimed: dict[str, list[tuple[int, int]]] = {}

    for raw in raw_mentions:
        if len(accepted) >= limit:
            break
        needle = raw.surface.strip()
        if not needle:
            continue
        key = needle.casefold()
        span = locate_span(
            text,
            needle,
            reported_start=raw.start,
            reported_end=raw.end,
            claimed=claimed.get(key, ()),
        )
        if span is None:
            continue
        start, end = span
        surface = text[start:end]
        if not surface.strip():
            # Unreachable through `locate_span`, which only returns spans of a
            # stripped needle. Kept because `EntityMention.surface` is a
            # `NonEmptyStr` and would raise here, inside a stage that must fail
            # only for reasons worth recording.
            continue
        accepted.append(
            EntityMention(
                surface=surface,
                type=coerce_entity_type(raw.type),
                start=start,
                end=end,
                candidate_ids=candidate_ids_for(surface, raw.candidates),
            )
        )
        claimed.setdefault(key, []).append(span)

    accepted.sort(key=lambda mention: (mention.start, -mention.end))
    return accepted


# --------------------------------------------------------------------------- #
# The stage
# --------------------------------------------------------------------------- #


class EntityExtractionStage:
    """Stage 4. Satisfies `services/signal_engine/pipeline.Stage`.

    Stateless per record and safe to share across concurrent enrichments: the
    only instance state is configuration and the provider, and nothing about one
    Signal is remembered between `apply()` calls.
    """

    name: StageName = StageName.ENTITIES
    version: str = STAGE_VERSION

    def __init__(
        self,
        provider: LLMProvider,
        *,
        settings: LLMSettings | None = None,
        model: str | None = None,
        max_chars: int = MAX_EXTRACTION_CHARS,
        max_mentions: int = MAX_MENTIONS,
        max_topics: int = MAX_TOPICS,
        max_keywords: int = MAX_KEYWORDS,
    ) -> None:
        resolved = settings if settings is not None else get_settings().llm
        self._provider = provider
        self._model = model or resolved.model_fast
        self._max_chars = max_chars
        self._max_mentions = max_mentions
        self._max_topics = max_topics
        self._max_keywords = max_keywords

    @property
    def model_id(self) -> str | None:
        """The fast-tier model id, recorded in `lineage.stages[]`.

        Stage 4 is non-deterministic, so `docs/signal-model.md` §5.1 requires the
        model to be recorded: without it, "why did this Signal extract three
        entities in March and five in June" is unanswerable.
        """
        return self._model

    async def apply(self, ctx: EnrichmentContext) -> None:
        """Populate `entities`, `topics` and `keywords`, or raise trying."""
        signal = ctx.require_signal()
        text = signal.content.text

        if not text.strip():
            # Media-only posts are routine on social platforms. There is nothing
            # to extract, the three fields keep their documented `[]`, and --
            # the point of the branch -- no call is made. At one call per
            # ingested Signal, paying to be told a photo caption is empty is a
            # cost line nobody would defend.
            return

        window = text[: self._max_chars]
        extraction = await self._provider.structured(
            prompt=_build_user_prompt(
                title=signal.content.title,
                language=signal.language.code,
                body=window,
            ),
            schema=EntityExtraction,
            system=SYSTEM_PROMPT,
            model=self._model,
            max_tokens=MAX_EXTRACTION_OUTPUT_TOKENS,
        )

        # Located against `window`, not `text`: the model can only have meant a
        # span it was shown, and searching the tail it never saw would repair a
        # mention onto an occurrence that had nothing to do with the answer.
        # A prefix keeps the offsets valid against the full string regardless.
        mentions = resolve_mentions(extraction.mentions, text=window, limit=self._max_mentions)

        signal.entities = mentions
        signal.topics = select_topics(
            ((proposal.topic, proposal.score) for proposal in extraction.topics),
            limit=self._max_topics,
        )
        # Keywords read the *whole* text, not the window: they are free, so the
        # window's cost justification does not apply to them.
        signal.keywords = extract_keywords(
            text,
            entity_surfaces=[mention.surface for mention in mentions],
            limit=self._max_keywords,
        )
