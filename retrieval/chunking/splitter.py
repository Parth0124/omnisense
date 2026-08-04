"""Chunking: turning one Signal's body into the units retrieval actually returns.

The chunk -- not the Signal -- is the unit of retrieval, of citation and of the
evidence pack (`docs/retrieval.md` §8). Everything downstream of here is joined
on `chunk_id`, which is why chunking happens exactly **once**, during enrichment
stage 6 (`services/signal_engine/embeddings.py`), and the resulting ids travel
on `omnisense.signals.enriched` unchanged. If the vector indexer and the keyword
indexer each chunked independently they would derive divergent ids for the same
Signal, and hybrid fusion -- which joins the two candidate lists on that key --
would silently return two disjoint result sets that never reinforce each other.

The invariant this module exists to guarantee:

    chunk.text == source_text[chunk.char_start:chunk.char_end]

exactly, byte for byte. `services/evidence_service.py` verifies a report's
quotes by re-reading `[char_start, char_end)` out of the stored Signal and
confirming the quote is a substring of it (`docs/retrieval.md` §8). The moment a
chunker normalizes, trims or re-joins the text it emits, every citation it
produced becomes unverifiable -- and it fails *later*, in the Critic, against
Signals that were correct when they were written. Hence the whole implementation
works in **spans over the original string** and slices only at the end.

Boundaries are chosen, not taken. Three failure modes from §8 drive the rules:

- **Split negation.** "We evaluated Competitor X and rejected it" cut after
  "evaluated" yields a chunk that supports the opposite claim, and no reranker
  can see the half that is missing. Hence: never split mid-sentence, with a hard
  cut only for a single sentence longer than twice the target.
- **Orphaned referent.** A chunk opening "It has been unreliable since March"
  cites nothing checkable. Hence overlap, and hence section-aware grouping.
- **Merged sources.** Packing two comments by different authors into one chunk
  attributes one author's words to the other. Hence social and review content is
  never packed -- one item, one chunk.

Layer note: **L1** (`docs/architecture.md` §6.1). This package may import
`models/` and nothing else in the repository -- in particular not
`backend/core/config.py`. Chunk geometry therefore arrives as arguments;
`EMBEDDING_MAX_CHARS_PER_CHUNK` and `EMBEDDING_CHUNK_OVERLAP_CHARS` are read by
the L2 caller in `services/signal_engine/embeddings.py` and passed down. That is
not ceremony: it is what lets `scripts/reindex.py` re-chunk a corpus at the
geometry it was originally written with rather than at whatever the environment
happens to say today.

Sizes here are **characters**, while `docs/retrieval.md` §8 states targets in
tokens (512 target, 64 overlap, 128 minimum). The configuration is in characters
because the pipeline must not depend on a tokenizer that belongs to whichever
embedding vendor is configured this week, and the defaults line up at the usual
~4 chars/token: 2000 ≈ 512 tokens, 200 ≈ 50, and the derived 500-char minimum
≈ 128 tokens.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from models.enums import SourceCategory

__all__ = [
    "CHUNK_ID_SEPARATOR",
    "MIN_CHUNK_DIVISOR",
    "Chunk",
    "ChunkStrategy",
    "chunk_id",
    "split_text",
    "strategy_for",
]


CHUNK_ID_SEPARATOR: Final = ":"
"""Separator in `{signal_id}:{chunk_index}` (`docs/retrieval.md` §8).

A colon is safe in all three consumers: it is the OpenSearch `_id` verbatim, it
is hashed into the Qdrant point id rather than used as one, and `Signal.id`
carries a `sig_` prefix and hex only, so the split is unambiguous from the right.
"""

MIN_CHUNK_DIVISOR: Final = 4
"""Minimum chunk size as a fraction of the target -- `max_chars // 4`.

§8 fixes the minimum at 128 tokens against a 512-token target, i.e. a quarter.
Deriving it keeps one number configurable instead of two that can silently
contradict each other (a minimum above the target makes every chunk a runt and
merges the whole document back into one).
"""


class ChunkStrategy(StrEnum):
    """How a document is cut, per the source-class table in `docs/retrieval.md` §8.

    A closed set owned by this module -- no `UNKNOWN` member, because an
    unrecognized strategy is a wiring bug in the caller and must fail loudly
    rather than fall back to something that quietly indexes the corpus wrong.
    """

    WHOLE = "whole"
    """One item, one chunk: social posts, reviews, individual comments."""

    PARAGRAPH = "paragraph"
    """Paragraph-packing to the target, breaking on paragraph boundaries: news."""

    HEADING = "heading"
    """Split at headings first, then paragraph-pack within a section: papers, docs."""

    SPEAKER_TURN = "speaker_turn"
    """Transcripts, packed on speaker-turn boundaries. Not implemented -- see `split_text`."""


@dataclass(frozen=True, slots=True)
class Chunk:
    """One retrievable span of a Signal's body.

    Frozen because a chunk's identity is its span: mutating `char_start` after
    the id has been derived and the vector upserted would leave a citation
    pointing into the wrong region of a document that never changed.
    """

    index: int
    text: str
    char_start: int
    char_end: int

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError(f"chunk index must be non-negative, got {self.index}")
        if self.char_start < 0 or self.char_end <= self.char_start:
            raise ValueError(
                f"chunk span [{self.char_start}, {self.char_end}) is empty or negative; "
                "a chunk with no extent cannot be cited"
            )
        if not self.text.strip():
            raise ValueError("chunk text is blank; blank chunks cost an embedding call and "
                             "are rejected outright by most providers")

    @property
    def char_count(self) -> int:
        return self.char_end - self.char_start

    def id_for(self, signal_id: str) -> str:
        """This chunk's stable `{signal_id}:{chunk_index}` handle."""
        return chunk_id(signal_id, self.index)


def chunk_id(signal_id: str, chunk_index: int) -> str:
    """Derive the stable chunk handle (`docs/retrieval.md` §8, `docs/data-stores.md` §5.2).

    Pure and total. Re-chunking changes these ids and therefore invalidates every
    stored citation, which is why a chunker change is treated like an
    embedding-model change: rebuild into a new collection through
    `scripts/reindex.py` rather than mutating in place.
    """
    if not signal_id:
        raise ValueError("signal_id must be non-empty; a chunk id has no meaning without it")
    if chunk_index < 0:
        raise ValueError(f"chunk_index must be non-negative, got {chunk_index}")
    return f"{signal_id}{CHUNK_ID_SEPARATOR}{chunk_index}"


def strategy_for(source: SourceCategory) -> ChunkStrategy:
    """Pick the chunking strategy for a source category (`docs/retrieval.md` §8).

    Keyed on `SourceCategory` rather than on `Platform` deliberately: the table
    in §8 is about *document shape*, and a per-platform mapping would be one more
    place that has to be edited every time a connector is added -- exactly the
    platform-shaped code `models/signal.py` forbids above `connectors/`.

    `UNKNOWN` gets paragraph packing rather than `WHOLE`. Both are safe for short
    text (packing a single paragraph yields one chunk either way); they differ
    only for a long body, where packing degrades gracefully and `WHOLE` would
    hand a 40 KB string to an embedding provider.
    """
    if source in (SourceCategory.SOCIAL, SourceCategory.REVIEWS):
        return ChunkStrategy.WHOLE
    if source in (SourceCategory.RESEARCH, SourceCategory.ENTERPRISE):
        return ChunkStrategy.HEADING
    return ChunkStrategy.PARAGRAPH


def split_text(
    text: str,
    *,
    strategy: ChunkStrategy,
    max_chars: int,
    overlap_chars: int = 0,
    min_chars: int | None = None,
) -> list[Chunk]:
    """Split `text` into ordered, citable chunks.

    Returns `[]` for empty or whitespace-only input, which is a normal outcome
    for a media-only post rather than an error: stage 6 records zero embeddings
    and stage 6b scores the missing body through `content_integrity`.

    `min_chars` defaults to `max_chars // MIN_CHUNK_DIVISOR`. A chunk below it is
    merged into its predecessor within the same section, so the corpus never
    fills with two-word fragments that outrank real passages on BM25 term
    density while carrying no context a reader could act on.
    """
    if max_chars < 1:
        raise ValueError(f"max_chars must be positive, got {max_chars}")
    if overlap_chars < 0:
        raise ValueError(f"overlap_chars must be non-negative, got {overlap_chars}")
    if overlap_chars >= max_chars:
        # Mirrors the `EmbeddingSettings` validator. Repeated here because this
        # is a library with callers other than that stage -- `scripts/reindex.py`
        # passes the geometry a collection was built with, not today's settings.
        raise ValueError(
            f"overlap_chars ({overlap_chars}) must be smaller than max_chars "
            f"({max_chars}), otherwise every chunk restarts inside its predecessor"
        )
    if strategy is ChunkStrategy.SPEAKER_TURN:
        raise NotImplementedError(
            "speaker-turn chunking needs turn boundaries, and nothing produces them "
            "yet: `MediaRef.transcript_ref` points at an R2 object that no stage "
            "loads, and stage 1 collapses any speaker labels that survived into "
            "plain paragraphs. Implement transcript loading in "
            "services/signal_engine/cleaning.py first, then this branch."
        )

    floor = max_chars // MIN_CHUNK_DIVISOR if min_chars is None else min_chars
    if floor < 0:
        raise ValueError(f"min_chars must be non-negative, got {floor}")

    spans: list[_Span] = []
    for group in _groups_for(text, strategy, max_chars):
        units = _unit_spans(text, group, max_chars)
        if not units:
            continue
        # Runt merging and overlap both happen *inside* the group. Applying
        # either across a section boundary would undo the heading-awareness that
        # put the boundary there, and reintroduce the merged-sources failure for
        # the one strategy that exists to prevent it.
        group_spans = _merge_runts(_pack(units, max_chars), floor)
        if overlap_chars:
            group_spans = _apply_overlap(text, group_spans, overlap_chars)
        spans.extend(group_spans)

    return [
        Chunk(index=index, text=text[start:end], char_start=start, char_end=end)
        for index, (start, end) in enumerate(spans)
    ]


# --------------------------------------------------------------------------- #
# Internals -- everything below works in spans over the original string
# --------------------------------------------------------------------------- #

_Span = tuple[int, int]

_PARAGRAPH_BREAK: Final = re.compile(r"\n[ \t]*\n\s*")
"""A blank line. `connectors/normalize/html.py` emits exactly this between blocks.

Coupled to that module by convention rather than by import -- `retrieval/` may
not import `connectors/` (`docs/architecture.md` §6.1). The coupling is stated
in both docstrings because a change to either side pushes chunk boundaries into
the middle of sentences for every article in the corpus, and nothing raises.
"""

_HEADING_LINE: Final = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+\S", re.MULTILINE)
"""An ATX markdown heading.

Only markdown, deliberately. The HTML path flattens `<h2>` into an ordinary
paragraph with no marker left behind, so heading-awareness genuinely only fires
for sources whose cleaned body is markdown -- arXiv abstracts, Notion, GitHub,
Confluence. Inferring headings from short lines instead was rejected: it
promotes every one-line paragraph in a news article to a section break, which
fragments exactly the documents paragraph packing exists for.
"""

_TRAILING_CLOSERS: Final = "\"'\u201d\u2019)]}"
"""Characters that may follow sentence-final punctuation and still end the sentence.

Typographic quotes (U+201D, U+2019) as well as straight ones:
`connectors/normalize/html.py` preserves whatever the publisher wrote, and
treating a curly quote as ordinary text would put the boundary one character
early on every quoted sentence in the news corpus. Written as escapes because
the literal characters are visually ambiguous with ASCII in a diff -- which is
exactly the bug class this constant exists to handle.
"""

_SENTENCE_END: Final = re.compile(f"(?<=[.!?])[{re.escape(_TRAILING_CLOSERS)}]*\\s+")
"""Whitespace following sentence-final punctuation and any closing quotes."""

_ABBREVIATIONS: Final[frozenset[str]] = frozenset(
    {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st",
        "e.g", "i.e", "etc", "vs", "cf", "al", "fig", "eq", "ref", "no",
        "inc", "ltd", "co", "corp", "dept", "est", "approx", "min", "max",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct",
        "nov", "dec",
    }
)
"""Tokens whose trailing period does not end a sentence.

Short on purpose. A missed abbreviation costs one slightly-early chunk boundary;
an over-eager list costs *merged* sentences, and the merge is what produces the
split-negation failure this module is built to avoid.
"""


def _groups_for(text: str, strategy: ChunkStrategy, max_chars: int) -> list[list[_Span]]:
    """Block spans, grouped into units that may never be packed together."""
    body = _tighten(text, 0, len(text))
    if body is None:
        return []

    if strategy is ChunkStrategy.WHOLE:
        start, end = body
        if end - start <= max_chars:
            return [[body]]
        # A deliberate deviation from §8's "never split" for social and reviews.
        # That rule assumes a post fits, and a 40 KB Reddit self-post does not:
        # handing it to the provider whole earns a 400 for exceeding the model's
        # context, which fails the stage and loses *every* chunk of the Signal
        # rather than splitting one. Degrading to paragraph packing keeps the
        # Signal retrievable; the cost is that one long post is cited by section.
        return [_paragraph_spans(text, start, end)]

    if strategy is ChunkStrategy.HEADING:
        return [_paragraph_spans(text, start, end) for start, end in _section_spans(text, body)]

    start, end = body
    return [_paragraph_spans(text, start, end)]


def _section_spans(text: str, body: _Span) -> list[_Span]:
    """Split at markdown headings; the heading line stays with its section.

    The heading is kept in the chunk *text* rather than lifted into metadata
    because the span invariant forbids synthesizing text: a chunk whose body was
    prefixed with a section path would no longer equal its own source slice, and
    quote verification would fail on it.
    """
    lo, hi = body
    starts = [match.start() for match in _HEADING_LINE.finditer(text, lo, hi)]
    if not starts:
        return [body]
    boundaries = starts if starts[0] <= lo else [lo, *starts]
    sections: list[_Span] = []
    for index, start in enumerate(boundaries):
        end = boundaries[index + 1] if index + 1 < len(boundaries) else hi
        tightened = _tighten(text, start, end)
        if tightened is not None:
            sections.append(tightened)
    return sections


def _paragraph_spans(text: str, lo: int, hi: int) -> list[_Span]:
    """Non-blank paragraph spans within `[lo, hi)`, in document order."""
    spans: list[_Span] = []
    cursor = lo
    for match in _PARAGRAPH_BREAK.finditer(text, lo, hi):
        tightened = _tighten(text, cursor, match.start())
        if tightened is not None:
            spans.append(tightened)
        cursor = match.end()
    tightened = _tighten(text, cursor, hi)
    if tightened is not None:
        spans.append(tightened)
    return spans


def _unit_spans(text: str, blocks: list[_Span], max_chars: int) -> list[_Span]:
    """The smallest spans the packer is allowed to place a boundary between.

    A paragraph that fits is one unit. A paragraph that does not is broken into
    sentences; a sentence longer than twice the target is hard-cut, which §8
    permits only in that case. A sentence between one and two targets is left
    whole and becomes a single oversized chunk -- overshooting the target is
    cheap, and cutting a sentence in half is the expensive failure.
    """
    units: list[_Span] = []
    for start, end in blocks:
        if end - start <= max_chars:
            units.append((start, end))
            continue
        for sentence_start, sentence_end in _sentence_spans(text, start, end):
            if sentence_end - sentence_start <= 2 * max_chars:
                units.append((sentence_start, sentence_end))
            else:
                units.extend(_hard_cut(text, sentence_start, sentence_end, max_chars))
    return units


def _sentence_spans(text: str, lo: int, hi: int) -> list[_Span]:
    spans: list[_Span] = []
    cursor = lo
    for match in _SENTENCE_END.finditer(text, lo, hi):
        boundary = match.start()
        if boundary <= cursor or _ends_with_abbreviation(text, boundary):
            continue
        tightened = _tighten(text, cursor, boundary)
        if tightened is not None:
            spans.append(tightened)
        cursor = match.end()
    tightened = _tighten(text, cursor, hi)
    if tightened is not None:
        spans.append(tightened)
    return spans


def _ends_with_abbreviation(text: str, boundary: int) -> bool:
    """Whether the period at `boundary` closes an abbreviation, not a sentence.

    Also treats a single letter before the period as an abbreviation, which
    catches initials ("J. Doe") and spelled-out acronyms ("U. S. policy") without
    needing either in the table.
    """
    tail = text[max(0, boundary - 16) : boundary].rstrip(_TRAILING_CLOSERS)
    if not tail.endswith("."):
        return False
    token = tail[:-1].rsplit(maxsplit=1)[-1] if tail[:-1].strip() else ""
    token = token.lower().lstrip("(\"'")
    return len(token) == 1 or token in _ABBREVIATIONS


def _hard_cut(text: str, lo: int, hi: int, max_chars: int) -> list[_Span]:
    """Cut an unsplittable run into target-sized pieces, preferring word breaks.

    Reached only for a single sentence over twice the target -- minified JSON
    pasted into a post, a wall of concatenated log lines, a language this
    module's sentence regex does not segment. Cutting on the last space in the
    final fifth keeps the pieces from starting mid-word; if there is no space at
    all (a single 20 KB token), the cut is exact, because the alternative is
    emitting the whole thing and losing the Signal to a provider 400.
    """
    spans: list[_Span] = []
    cursor = lo
    while hi - cursor > max_chars:
        window_end = cursor + max_chars
        pivot = text.rfind(" ", cursor + (max_chars * 4) // 5, window_end)
        cut = pivot if pivot > cursor else window_end
        tightened = _tighten(text, cursor, cut)
        if tightened is not None:
            spans.append(tightened)
        cursor = cut
    tightened = _tighten(text, cursor, hi)
    if tightened is not None:
        spans.append(tightened)
    return spans


def _pack(units: list[_Span], max_chars: int) -> list[_Span]:
    """Greedily pack consecutive units up to the target.

    Size is measured as the *extent* of the packed span, not the sum of the unit
    lengths, because the chunk's text is the contiguous slice and therefore
    includes the whitespace between units. Summing the units instead would
    under-count a list-heavy document by exactly the blank lines it is made of.
    """
    packed: list[_Span] = []
    start, end = units[0]
    for unit_start, unit_end in units[1:]:
        if unit_end - start <= max_chars:
            end = unit_end
        else:
            packed.append((start, end))
            start, end = unit_start, unit_end
    packed.append((start, end))
    return packed


def _merge_runts(spans: list[_Span], min_chars: int) -> list[_Span]:
    """Fold any chunk below `min_chars` into its predecessor (`docs/retrieval.md` §8).

    A merge can push a chunk over the target by up to `min_chars`. That is the
    intended trade: the target is a soft budget, while a fragment below the
    minimum is a retrieval liability -- it wins on term density, carries no
    context, and produces a citation nobody can read.

    A group whose entire content is below the minimum keeps its single short
    chunk. There is nothing to merge into, and dropping it would make a
    two-sentence Signal unretrievable.
    """
    merged: list[_Span] = []
    for start, end in spans:
        if merged and end - start < min_chars:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def _apply_overlap(text: str, spans: list[_Span], overlap_chars: int) -> list[_Span]:
    """Extend each chunk's start backwards into its predecessor, within one group.

    Overlap is expressed as a *wider span*, never as prepended text, so
    `chunk.text == source[char_start:char_end]` survives it. The start snaps to
    the nearest sentence boundary in the overlap window, falling back to a word
    boundary, so the borrowed context is itself readable -- an overlap that
    begins mid-sentence reintroduces the orphaned-referent problem it exists to
    solve.

    The window is clamped to the *original* predecessor start, so a run of short
    chunks cannot chain each other's overlap backwards until the first chunk of
    the group has been swallowed whole.
    """
    extended: list[_Span] = [spans[0]]
    for index in range(1, len(spans)):
        start, end = spans[index]
        previous_start, _ = spans[index - 1]
        window_start = max(previous_start, start - overlap_chars)
        extended.append((_snap_to_boundary(text, window_start, start), end))
    return extended


def _snap_to_boundary(text: str, window_start: int, hard_start: int) -> int:
    """Earliest readable boundary in `[window_start, hard_start)`, else `hard_start`.

    Earliest rather than nearest, because the earliest boundary inside the
    window is the one that carries the most context back.

    A sentence start is preferred. The word-boundary fallback matters more than
    it looks: a long paragraph of one-clause sentences frequently has no
    sentence boundary inside a 200-character window, and returning `hard_start`
    there would silently drop overlap for exactly the documents that need it.
    Beginning the overlap mid-sentence is the lesser evil -- that sentence is
    still present in full in the preceding chunk, so nothing is lost, only
    repeated.
    """
    if window_start >= hard_start:
        return hard_start
    # The same abbreviation guard the splitter uses. Without it the two
    # disagree, and overlap begins one word into "e.g. the second clause" -- a
    # mid-sentence start produced by the very code meant to prevent one.
    sentence_starts = [
        match.end()
        for match in _SENTENCE_END.finditer(text, window_start, hard_start)
        if match.end() < hard_start and not _ends_with_abbreviation(text, match.start())
    ]
    if sentence_starts:
        return sentence_starts[0]
    space = text.find(" ", window_start, hard_start)
    if space != -1 and space + 1 < hard_start:
        return space + 1
    return hard_start


def _tighten(text: str, start: int, end: int) -> _Span | None:
    """Trim whitespace off both ends of a span. `None` when nothing is left.

    Every span in this module is tight, which is what keeps the slice invariant
    honest: a span that began on a newline would emit a chunk whose text has
    leading whitespace, and the first thing any consumer does with that is strip
    it -- at which point the offsets no longer describe the string being cited.
    """
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if end > start else None
