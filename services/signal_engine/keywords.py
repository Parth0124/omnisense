"""Topics from a closed vocabulary, keywords from the text itself.

This module is the second half of enrichment stage 4. It is deliberately not a
`Stage`: `docs/signal-model.md` §5 says keyword extraction is "a sub-step folded
into stage 4", and `StageName` has no `KEYWORDS` member, so there is nowhere for
a separate stage record to go. `services/signal_engine/entities.py` calls into
here while it holds the model's output, and all three fields -- `entities`,
`topics`, `keywords` -- succeed or degrade to `[]` together.

**Why topics and keywords are different things.** `TopicScore` is a *closed*
vocabulary and `Keyword` is open (`models/signal.py`). That is not a stylistic
split: topics are aggregated across sources ("mentions of `vendor-pricing` rose
40% this quarter"), and an aggregate is only meaningful if every Signal drew
from the same finite set. Keywords are never aggregated -- they exist to boost
BM25 in `retrieval/keyword/query_builder.py`, where an open vocabulary is the
whole point. Letting a model invent a topic slug would silently convert the
closed set into an open one and quietly break every count computed over it, so
`select_topics()` drops proposals that are not in `TOPIC_VOCABULARY` rather than
passing them through.

**Why keyword extraction does not call a model.** Stage 4 already makes one LLM
call per ingested Signal, and that call is on the hot path for every record the
platform will ever see; a second call for keywords would roughly double the
per-Signal extraction bill for the cheapest, least model-shaped part of the job.
Three reasons make a deterministic extractor the better answer rather than
merely the cheaper one:

1. **BM25 needs terms that actually occur.** A model asked for keywords
   abstracts -- it answers "pricing concerns" for a document that says "renewal
   quote". A term that is not in the document cannot help a lexical index match
   that document, so the model's better summary is the worse keyword.
2. **Reproducibility.** Stage 4 is already non-deterministic because of the
   model (`docs/signal-model.md` §5.1). Keeping keywords deterministic means
   reprocessing after a model swap changes `entities` and leaves keyword-boosted
   retrieval stable, which makes an A/B of two extraction models interpretable.
3. **It degrades independently of the provider.** Keywords still get produced
   when the model is rate limited -- see `extract_keywords()`, which entities.py
   may call on a path the provider never touched.

The algorithm is RAKE (Rose et al., 2010) with two deviations noted at
`extract_keywords()`. RAKE rather than TF-IDF because we have no corpus at
enrichment time -- a Signal is scored the moment it arrives, with no document
frequencies available and no way to obtain them without a round trip that would
cost more than the LLM call this module exists to avoid.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from models.signal import Keyword, TopicScore

__all__ = [
    "KEYWORD_EXTRACTOR_VERSION",
    "MAX_KEYWORD_PHRASE_WORDS",
    "TOPIC_VOCABULARY",
    "TOPIC_VOCABULARY_VERSION",
    "TopicDefinition",
    "extract_keywords",
    "normalize_topic",
    "select_topics",
]


# --------------------------------------------------------------------------- #
# The controlled topic vocabulary
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class TopicDefinition:
    """One member of the closed topic vocabulary.

    `aliases` exist because the *model* proposes these slugs and models are
    inconsistent about phrasing -- "vendor pricing", "pricing", `Vendor_Pricing`
    and `vendor-pricing` all mean the same member. Normalizing them onto one
    slug recovers a proposal that would otherwise be discarded, without widening
    the vocabulary: an alias is a curator's decision recorded here, not an
    invention accepted at runtime.
    """

    slug: str
    label: str
    aliases: tuple[str, ...] = ()


TOPIC_VOCABULARY_VERSION: Final = "2026.07.0"
"""Version of `TOPIC_VOCABULARY`, stamped so a stored `TopicScore` is interpretable.

Bumping this is *not* a schema migration -- `TopicScore.topic` stays a string --
but it does invalidate comparisons across the boundary. A trend computed over
`vendor-pricing` before a rename and after it is two different measurements
presented as one series.

**Ownership is an open gap, stated rather than papered over.**
`docs/signal-model.md` §9 open question 8 records it exactly: `topics` is
described as a closed set, "but nothing yet defines who curates that set, how it
is versioned, or what happens to stored `TopicScore` entries when a topic is
retired." Nothing in this module resolves that. What it does is make the gap
concrete and small:

- the set is *here*, in one versioned constant, rather than implied by whatever
  a prompt happened to say last;
- `select_topics()` enforces closure, so drift cannot enter through the model;
- the three unanswered questions are unchanged and still need an owner:
  **(a)** who approves an addition, **(b)** whether a retired slug is deleted
  from stored Signals or tombstoned, and **(c)** whether adding a member
  triggers a backfill so that older Signals can carry it. Until (c) is decided,
  a topic added today applies only to Signals ingested after today, and any
  count spanning the change is wrong in a way no field records.

The initial membership below is a starting point drawn from the source
categories in Design Doc §5, not a curated taxonomy. It is deliberately coarse:
a vocabulary fine enough to be interesting is fine enough to need the owner that
does not exist yet.
"""


TOPIC_VOCABULARY: Final[tuple[TopicDefinition, ...]] = (
    # -- product and technology ---------------------------------------------
    TopicDefinition("product-launch", "Product launch", ("launch", "release", "new-product")),
    TopicDefinition("product-quality", "Product quality", ("quality", "bugs", "defects")),
    TopicDefinition("feature-request", "Feature request", ("feature-gap", "missing-feature")),
    TopicDefinition(
        "performance-and-reliability",
        "Performance and reliability",
        ("performance", "reliability", "latency", "uptime"),
    ),
    TopicDefinition(
        "outage-and-incident", "Outage and incident", ("outage", "incident", "downtime")
    ),
    TopicDefinition(
        "security-and-vulnerability",
        "Security and vulnerability",
        ("security", "vulnerability", "breach", "cve"),
    ),
    TopicDefinition(
        "privacy-and-data-protection",
        "Privacy and data protection",
        ("privacy", "data-protection", "gdpr"),
    ),
    TopicDefinition(
        "developer-experience", "Developer experience", ("dx", "documentation", "api-usability")
    ),
    TopicDefinition(
        "integration-and-interoperability",
        "Integration and interoperability",
        ("integration", "interoperability", "api-integration"),
    ),
    TopicDefinition(
        "observability-tooling",
        "Observability tooling",
        ("observability", "monitoring", "logging", "tracing"),
    ),
    TopicDefinition(
        "ai-and-machine-learning",
        "AI and machine learning",
        ("ai", "machine-learning", "ml", "llm", "genai"),
    ),
    TopicDefinition(
        "infrastructure-and-cloud",
        "Infrastructure and cloud",
        ("infrastructure", "cloud", "hosting", "kubernetes"),
    ),
    TopicDefinition("open-source", "Open source", ("oss", "licensing-open-source")),
    # -- commercial ----------------------------------------------------------
    TopicDefinition(
        "vendor-pricing", "Vendor pricing", ("pricing", "price-increase", "cost", "billing")
    ),
    TopicDefinition(
        "contract-and-licensing", "Contract and licensing", ("contract", "licensing", "renewal")
    ),
    TopicDefinition(
        "vendor-migration", "Vendor migration", ("migration", "switching", "replatforming")
    ),
    TopicDefinition(
        "procurement-and-evaluation",
        "Procurement and evaluation",
        ("procurement", "evaluation", "rfp", "vendor-selection"),
    ),
    TopicDefinition(
        "customer-support", "Customer support", ("support", "service-quality", "helpdesk")
    ),
    TopicDefinition(
        "sales-and-marketing", "Sales and marketing", ("marketing", "sales", "advertising")
    ),
    TopicDefinition(
        "partnership-and-alliance", "Partnership and alliance", ("partnership", "alliance")
    ),
    # -- corporate -----------------------------------------------------------
    TopicDefinition(
        "funding-and-investment",
        "Funding and investment",
        ("funding", "investment", "venture-capital", "fundraising"),
    ),
    TopicDefinition(
        "mergers-and-acquisitions", "Mergers and acquisitions", ("m-and-a", "acquisition", "merger")
    ),
    TopicDefinition("hiring-and-talent", "Hiring and talent", ("hiring", "recruiting", "talent")),
    TopicDefinition(
        "layoffs-and-restructuring", "Layoffs and restructuring", ("layoffs", "restructuring")
    ),
    TopicDefinition(
        "leadership-change", "Leadership change", ("leadership", "executive-change", "ceo-change")
    ),
    TopicDefinition(
        "financial-results", "Financial results", ("earnings", "revenue", "financials")
    ),
    TopicDefinition("legal-and-litigation", "Legal and litigation", ("legal", "lawsuit")),
    TopicDefinition(
        "regulation-and-compliance",
        "Regulation and compliance",
        ("regulation", "compliance", "antitrust"),
    ),
    # -- market --------------------------------------------------------------
    TopicDefinition(
        "competitive-positioning",
        "Competitive positioning",
        ("competition", "competitor", "market-position"),
    ),
    TopicDefinition("market-adoption", "Market adoption", ("adoption", "market-share", "growth")),
    TopicDefinition("customer-churn", "Customer churn", ("churn", "cancellation", "attrition")),
    TopicDefinition(
        "research-and-publication",
        "Research and publication",
        ("research", "paper", "publication", "benchmark"),
    ),
    TopicDefinition(
        "sustainability", "Sustainability", ("esg", "energy-use", "environmental-impact")
    ),
)
"""The closed set. See `TOPIC_VOCABULARY_VERSION` for what is unresolved about it."""


def _build_index() -> Mapping[str, str]:
    """Map every slug and alias, normalized, onto its canonical slug.

    Built once at import. A collision -- the same alias claimed by two members --
    is raised rather than resolved by dict ordering, because silently binding
    "pricing" to whichever definition happens to be later in the tuple would make
    the vocabulary's behaviour depend on its source order.
    """
    index: dict[str, str] = {}
    for definition in TOPIC_VOCABULARY:
        for raw in (definition.slug, definition.label, *definition.aliases):
            key = _normalize_slug(raw)
            if not key:
                continue
            existing = index.get(key)
            if existing is not None and existing != definition.slug:
                raise ValueError(
                    f"topic alias {key!r} is claimed by both {existing!r} and "
                    f"{definition.slug!r}; an alias must name exactly one topic"
                )
            index[key] = definition.slug
    return index


_SLUG_SEPARATORS: Final = re.compile(r"[\s_/]+")
_SLUG_ILLEGAL: Final = re.compile(r"[^a-z0-9-]+")
_SLUG_COLLAPSE: Final = re.compile(r"-{2,}")


def _normalize_slug(raw: str) -> str:
    """Fold a proposal into the slug shape the vocabulary is keyed by."""
    folded = _SLUG_SEPARATORS.sub("-", raw.strip().casefold())
    folded = _SLUG_ILLEGAL.sub("-", folded)
    folded = _SLUG_COLLAPSE.sub("-", folded)
    return folded.strip("-")


_TOPIC_INDEX: Final[Mapping[str, str]] = _build_index()


def normalize_topic(raw: str) -> str | None:
    """Canonical slug for a proposed topic, or `None` if it is not in the set.

    `None` is the load-bearing return value. It is what stops a model's invented
    label from entering a vocabulary that everything downstream treats as closed.
    """
    return _TOPIC_INDEX.get(_normalize_slug(raw))


def select_topics(proposals: Iterable[tuple[str, float]], *, limit: int = 6) -> list[TopicScore]:
    """Filter model-proposed topics down to the controlled vocabulary.

    Out-of-vocabulary proposals are **dropped, not coerced**. There is no
    nearest-member fallback: mapping an unknown "pricing-pressure" onto
    `vendor-pricing` because they share a token would fabricate an assignment
    nobody made, and the count it feeds would be wrong in a direction no field
    records. A dropped topic costs one missing assignment on one Signal.

    Duplicates keep the highest score -- a model that proposes the same topic
    twice is expressing more confidence, not less -- and scores are clamped
    because `TopicScore.score` is a `Score` and a model that answers `1.2` would
    otherwise raise inside a stage whose whole contract is to degrade quietly.

    The result is sorted by score descending, ties broken by slug, so two runs
    over the same proposals produce byte-identical output.
    """
    best: dict[str, float] = {}
    for raw_topic, raw_score in proposals:
        slug = normalize_topic(raw_topic)
        if slug is None:
            continue
        score = min(1.0, max(0.0, float(raw_score)))
        if score > best.get(slug, -1.0):
            best[slug] = score
    ordered = sorted(best.items(), key=lambda item: (-item[1], item[0]))
    return [TopicScore(topic=slug, score=round(score, 4)) for slug, score in ordered[:limit]]


# --------------------------------------------------------------------------- #
# Open-vocabulary keywords
# --------------------------------------------------------------------------- #

MAX_KEYWORD_PHRASE_WORDS: Final = 3
"""Longest candidate phrase, in words.

RAKE itself has no length cap and will happily emit an eleven-word run between
two stopwords. Such a phrase is a sentence fragment: it is unique to this
document, so it can never match a query, which makes it worthless to the BM25
booster that consumes these terms.
"""

MAX_KEYWORD_PHRASE_CHARS: Final = 60
"""Longest candidate phrase, in characters.

A separate cap from the word cap because word counting assumes whitespace
separates words. In Chinese, Japanese and Thai it does not, so a "one word"
candidate can be an entire clause; this is the bound that catches it. It does
not make the extractor correct for those scripts -- see the note on
`extract_keywords()`.
"""

KEYWORD_EXTRACTOR_VERSION: Final = "rake-1.0.0"
"""Identifies the extraction algorithm for reproducibility.

Not the same thing as the stage version: the stage can be re-released for a
prompt change that leaves keywords byte-identical, and a term set that changed
for a reason other than a scoring change is worth being able to rule out.
"""

ENTITY_SURFACE_BOOST: Final = 1.5
"""Multiplier applied to a candidate that an entity mention already claimed.

Free information: stage 4 holds the extracted mentions when it calls in here, and
a span a model bothered to type as a Company is by construction more salient than
a phrase that merely survived stopword removal. The multiplier is applied before
normalization, so it changes ranking rather than absolute weight.
"""

STOPWORDS: Final[frozenset[str]] = frozenset(
    # A block of words, split at import, rather than a 200-element list literal.
    # This list is edited by hand whenever keyword quality is tuned, and the
    # prose form is the one a reviewer can actually scan for a duplicate or a
    # word that should not be here.
    """
    a about above after again against all also am an and any are aren't as at be because been
    before being below between both but by can cannot could couldn't did didn't do does doesn't
    doing don't down during each few for from further had hadn't has hasn't have haven't having
    he her here hers herself him himself his how i if in into is isn't it its itself just let's
    me more most mustn't my myself no nor not of off on once only or other ought our ours
    ourselves out over own same shan't she should shouldn't so some such than that that's the
    their theirs them themselves then there these they this those through to too under until up
    very was wasn't we were weren't what when where which while who whom why with won't would
    wouldn't you your yours yourself yourselves
    got get gets getting going gone im ive youre theyre thats whats isnt dont doesnt didnt cant
    wont wouldnt couldnt shouldnt havent hasnt hadnt wasnt werent arent
    one two three like really actually basically pretty much many lot lots thing things stuff
    way ways make makes made making need needs needed want wants wanted use used using
    said says say saying know knows knew think thinks thought see sees saw seen
    came come comes coming went take takes taking took taken gave give gives put puts
    still even ever never always often now then yeah okay ok etc via per upon within without
    """.split()  # noqa: SIM905 -- a scannable block beats a 200-element list literal
)
"""English stoplist. Sole language supported, and that is a known gap.

Applied unconditionally rather than switched on `Signal.language`. Applying an
English list to German text is a weaker filter, not a wrong one -- it removes
nothing that carries meaning in German -- whereas *skipping* extraction for
non-English text would strip the BM25 booster from every non-English Signal in
the corpus. Per-language stoplists are uncurated and unowned, which is the same
missing-owner problem `TOPIC_VOCABULARY_VERSION` describes; callers with a
better list pass one to `extract_keywords(stopwords=...)`.
"""

_TOKEN_RE: Final = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*", re.UNICODE)  # noqa: RUF001
"""A word: alphanumeric runs, internally hyphenated or apostrophed.

`[^\\W_]` rather than `\\w` because `\\w` admits `_`, and `snake_case_identifier`
as a keyword is one token that will never appear in a natural-language query.
Unicode-aware by default for `str` patterns, so accented Latin, Cyrillic and CJK
all tokenize rather than being dropped as non-words.

The typographic apostrophe is in the class deliberately (hence the `RUF001`
suppression): cleaners and CMSes emit U+2019 far more often than U+0027, and
without it every possessive in a news body splits into two tokens.
"""


@dataclass(frozen=True, slots=True)
class _Candidate:
    """One RAKE candidate phrase: the words it contains and its surface form."""

    words: tuple[str, ...]
    surface: str


def extract_keywords(
    text: str,
    *,
    entity_surfaces: Sequence[str] = (),
    limit: int = 12,
    stopwords: frozenset[str] | None = None,
) -> list[Keyword]:
    """Rank open-vocabulary phrases by RAKE, normalized into `Keyword` weights.

    RAKE with two deviations, both forced by what consumes the output:

    1. **Phrases are capped** at `MAX_KEYWORD_PHRASE_WORDS` /
       `MAX_KEYWORD_PHRASE_CHARS`. See those constants.
    2. **Scores are normalized to `[0, 1]`** by dividing by the best score in
       this document. `Keyword.weight` is a `Score`, so raw RAKE values -- which
       are unbounded and grow with phrase length -- cannot be stored. The
       consequence is that weights are comparable *within* a Signal and not
       across Signals, which is the correct semantics for a BM25 boost anyway:
       the booster ranks this document's terms against each other.

    Deterministic and total: no I/O, no clock, no randomness, and the sort is
    fully specified down to the tie-break, so replaying stage 4 on the same text
    produces the same list in the same order.

    **Known gap -- scripts without whitespace word boundaries.** For Chinese,
    Japanese and Thai a "word" here is whatever `_TOKEN_RE` matched, which is a
    whole run of script characters. `MAX_KEYWORD_PHRASE_CHARS` keeps the result
    bounded, so nothing downstream breaks, but the terms are clause-shaped rather
    than word-shaped and the BM25 boost they provide is weak. Fixing it needs a
    segmenter per script, which is a dependency and an owner this stage does not
    have yet. It is a quality gap, not a correctness one -- offsets and every
    other field are unaffected, because keywords carry no offsets.
    """
    if not text.strip():
        return []

    stops = STOPWORDS if stopwords is None else stopwords
    candidates = _candidate_phrases(text, stops)
    if not candidates:
        return []

    scores = _rake_word_scores(candidates)
    boosted = {surface.casefold() for surface in entity_surfaces if surface.strip()}

    ranked: dict[str, tuple[float, str, int]] = {}
    for candidate in candidates:
        raw = sum(scores[word] for word in candidate.words)
        key = candidate.surface.casefold()
        if key in boosted or any(term in key for term in boosted):
            raw *= ENTITY_SURFACE_BOOST
        previous = ranked.get(key)
        occurrences = 1 if previous is None else previous[2] + 1
        best = raw if previous is None else max(previous[0], raw)
        surface = candidate.surface if previous is None else previous[1]
        ranked[key] = (best, surface, occurrences)

    top = max(entry[0] for entry in ranked.values())
    if top <= 0.0:
        return []

    # Sort: weight desc, then occurrence count desc, then the term itself. The
    # last two exist purely so the order is total -- a partial sort would let an
    # equal-scoring pair swap between runs and make reprocessing look like drift.
    ordered = sorted(ranked.items(), key=lambda item: (-item[1][0], -item[1][2], item[0]))
    return [
        Keyword(term=surface, weight=round(min(1.0, score / top), 4))
        for _, (score, surface, _count) in ordered[:limit]
    ]


def _candidate_phrases(text: str, stops: frozenset[str]) -> list[_Candidate]:
    """Split text into maximal runs of non-stopword tokens.

    A run is broken by a stopword, by a pure-digit token, by a single-character
    token, or by any punctuation between two tokens. Punctuation matters: without
    it "Datadog, Grafana" would become the phrase "Datadog Grafana", which occurs
    nowhere in the document and is exactly the kind of fabricated term that
    cannot match a query.
    """
    phrases: list[_Candidate] = []
    current: list[str] = []
    current_start = 0
    current_end = 0
    previous_end: int | None = None

    def flush() -> None:
        nonlocal current
        if current:
            # Collapse internal whitespace: a phrase that straddles a line break
            # would otherwise carry the newline into `Keyword.term`, and the term
            # is matched against a query analyzer that has never seen one.
            # Safe to rewrite here in a way it would not be for an entity
            # mention, because keywords carry no offsets back into the text.
            surface = " ".join(text[current_start:current_end].split())
            if surface and len(surface) <= MAX_KEYWORD_PHRASE_CHARS:
                phrases.append(
                    _Candidate(words=tuple(w.casefold() for w in current), surface=surface)
                )
            current = []

    for match in _TOKEN_RE.finditer(text):
        token = match.group()
        gap = text[previous_end : match.start()] if previous_end is not None else ""
        previous_end = match.end()

        is_stop = token.casefold() in stops or token.isdigit() or len(token) < 2
        if is_stop or (gap and gap.strip()):
            flush()
            if is_stop:
                continue
            # Punctuation broke the run but this token is itself content, so it
            # opens the next phrase rather than being discarded with the break.
            current_start = match.start()
            current_end = match.end()
            current = [token]
            continue

        if not current:
            current_start = match.start()
        current.append(token)
        current_end = match.end()
        if len(current) >= MAX_KEYWORD_PHRASE_WORDS:
            flush()

    flush()
    return phrases


def _rake_word_scores(candidates: Sequence[_Candidate]) -> Mapping[str, float]:
    """RAKE word score: `degree(w) / frequency(w)`.

    Degree counts how many words a term co-occurs with across all its
    occurrences, so a term that only ever appears alone scores 1.0 while one that
    consistently anchors longer phrases scores higher. That is the property that
    makes RAKE prefer "contract renewal" over "renewal" without a corpus.
    """
    frequency: dict[str, int] = {}
    degree: dict[str, int] = {}
    for candidate in candidates:
        span = len(candidate.words)
        for word in candidate.words:
            frequency[word] = frequency.get(word, 0) + 1
            degree[word] = degree.get(word, 0) + span
    return {word: degree[word] / frequency[word] for word in frequency}
