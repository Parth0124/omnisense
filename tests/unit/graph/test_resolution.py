"""Unit tests for `graph/resolution/` -- blocking, scoring, clustering, un-merge.

Entity resolution has two failure modes that no other part of the platform
shares, and this file is organized around them rather than around the module
layout.

**A missed merge is invisible.** Nothing raises when "Föö-Bär, Ltd." and
"foo bar limited" fail to share a blocking key. The two entities simply never
meet, the graph carries two nodes for one company forever, and the only symptom
is that a competitor query returns half its evidence -- which looks like thin
data, not like a bug. There is no runtime assertion that can catch this, so it
has to be caught here: `TestBlockingRecall` fixes the punctuation-and-case case
in a test because production cannot detect it.

**A wrong merge is destructive and must be reversible.** Merging rewires edges
and changes the answer to every query touching either node. The correction path
is `unmerge()`, and it is only a correction if it restores the *exact*
pre-merge state -- an un-merge that loses an alias or resets a counter has
converted one wrong entity into two damaged ones. `TestUnmergeRoundTrip` asserts
object equality against the original inputs rather than field-by-field
approximate equality, because "close enough" is how those losses hide.

Underneath both sits determinism. Two graph workers consuming the same Kafka
partition must build the same graph, including *which id survives*, or they
write competing canonical nodes and the divergence compounds with every later
pass. `TestDeterminism` shuffles the input twenty ways and demands a
byte-identical plan, which is the only formulation that catches an accidental
dependency on `dict` or `set` iteration order.

Everything here is offline and clock-free: no Neo4j, no network, no `utcnow()`
inside an assertion. `decided_at` is injected everywhere for exactly that reason.
"""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from graph.resolution.blocking import (
    DEFAULT_MAX_BLOCK_SIZE,
    BlockingIndex,
    BlockKind,
    ResolutionRecord,
    blocking_keys,
    cosine_similarity,
    name_tokens,
    normalize_name,
    pair_key,
    phonetic_code,
)
from graph.resolution.entity_resolution import (
    MIN_WITHIN_CLUSTER_LINKAGE,
    elect_survivor,
    refine_component,
    resolve,
    resolve_async,
    unmerge,
)
from graph.resolution.matcher import (
    AUTO_MERGE_THRESHOLD,
    MATCH_WEIGHTS,
    NAME_BLEND_WEIGHTS,
    REVIEW_THRESHOLD,
    MatchDecision,
    MatchScorer,
    score_pair,
)
from models.entity import Entity
from models.enums import EdgeType, EntityType

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

T0 = datetime(2026, 1, 1, tzinfo=UTC)
"""One fixed decision timestamp. Injected so no assertion depends on the clock."""


def record(
    record_id: str,
    name: str,
    *,
    entity_type: EntityType = EntityType.COMPANY,
    aliases: tuple[str, ...] = (),
    identifiers: dict[str, str] | None = None,
    embedding: tuple[float, ...] | None = None,
    context: frozenset[str] = frozenset(),
    first_seen: datetime | None = None,
    mention_count: int = 1,
) -> ResolutionRecord:
    """A resolution record with everything optional defaulted away."""
    return ResolutionRecord(
        id=record_id,
        type=entity_type,
        name=name,
        aliases=aliases,
        identifiers=identifiers or {},
        embedding=embedding,
        context=context,
        first_seen=first_seen,
        last_seen=first_seen,
        mention_count=mention_count,
    )


def fingerprint(result: Any) -> str:
    """A total, order-sensitive serialization of a `ResolutionResult`.

    Compares the *plan* -- clusters, survivors, merged aliases, audit records and
    edges -- rather than a summary, because determinism bugs show up in exactly
    the fields a summary drops: which id survived, and in what order aliases were
    unioned.
    """
    return json.dumps(
        {
            "clusters": [
                {
                    "survivor": cluster.survivor_id,
                    "members": list(cluster.member_ids),
                    "aliases": list(cluster.canonical.aliases),
                    "mentions": cluster.canonical.mention_count,
                    "weakest": cluster.weakest_link,
                    "merged_from": list(cluster.canonical.merged_from),
                }
                for cluster in result.clusters
            ],
            "merges": [merge.as_dict() for merge in result.merges],
            "edges": [edge.as_dict() for edge in result.same_as_edges],
            "review": [
                [item.left_id, item.right_id, item.reason, round(item.score.combined, 9)]
                for item in result.review_items
            ],
        },
        default=str,
        sort_keys=True,
    )


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #


class TestNameNormalization:
    """The single input every string-shaped blocking key derives from.

    If two writings of one name normalize differently, every downstream key
    differs and the pair is never compared. These cases are the ones observed in
    real connector output: scraped HTML entities, accented company names, legal
    suffixes present on one side only.
    """

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            ("Acme Corp.", "acme corp"),
            ("Föö-Bär, Ltd.", "foo bar limited"),
            ("The Guardian", "guardian"),
            ("AT&T Inc.", "at and t"),
            ("  Datadog,  Inc.  ", "datadog"),
            ("Zoë's Kitchen", "zoes kitchen"),
        ],
    )
    def test_variants_normalize_identically(self, left: str, right: str) -> None:
        """Punctuation, case, accents, articles and legal form are all noise."""
        assert normalize_name(left) == normalize_name(right)

    def test_legal_suffix_stripping_never_empties_a_name(self) -> None:
        """An entity really named "Corp" must not normalize to the empty string.

        The empty string is a valid dictionary key, so every such entity would
        land in one block together and be compared against each other forever --
        the single most expensive block the index can contain, made entirely of
        pairs that cannot match.
        """
        assert normalize_name("Corp.") == "corp"
        assert normalize_name("The Company") == "company"
        assert normalize_name("Ltd") == "ltd"

    def test_unnameable_input_normalizes_to_empty(self) -> None:
        """Pure punctuation has no name, and pretending otherwise invents a key."""
        assert normalize_name("???") == ""
        assert normalize_name("   ") == ""
        assert name_tokens("!!!") == ()

    def test_token_order_is_preserved(self) -> None:
        """Sorting belongs in the key, not in normalization.

        The matcher's token-coverage measure aligns tokens positionally-agnostic
        but needs the original sequence for the character-level comparisons.
        """
        assert name_tokens("Acme Cloud Platform") == ("acme", "cloud", "platform")


class TestPhoneticCode:
    """The recall key of last resort: same sound, different spelling.

    Its correctness criterion is not fidelity to published Metaphone tables --
    it is that two spellings of one sound collide, and that unrelated names do
    not all collapse into one bucket.
    """

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            ("smith", "smyth"),
            ("colour", "kolour"),
            ("acme", "akme"),
            ("knight", "night"),
            ("philips", "filips"),
            ("wright", "right"),
        ],
    )
    def test_homophones_collide(self, left: str, right: str) -> None:
        assert phonetic_code(left) == phonetic_code(right)
        assert phonetic_code(left) != ""

    def test_unrelated_names_do_not_collide(self) -> None:
        assert phonetic_code("datadog") != phonetic_code("grafana")

    def test_non_latin_tokens_produce_no_code(self) -> None:
        """Empty rather than a shared code, and the difference is load-bearing.

        A shared empty code would put every CJK, Cyrillic and Arabic entity into
        one block -- which is then dropped as oversized, removing phonetic
        coverage from exactly the names least served by the character-level keys.
        `blocking_keys` checks for the empty string and emits no key at all.
        """
        assert phonetic_code("株式会社") == ""
        assert phonetic_code("123") == ""


# --------------------------------------------------------------------------- #
# Blocking
# --------------------------------------------------------------------------- #


class TestBlockingRecall:
    """The requirement this module exists for: never lose a true match.

    A blocking miss cannot be detected at runtime. It produces no error, no
    counter and no log line -- only a graph that quietly holds two nodes for one
    company. These tests are the only place that failure is observable.
    """

    def test_punctuation_and_case_variants_are_candidates(self) -> None:
        """The headline case: same company, different typography.

        Asserted through `candidate_pairs()` rather than by inspecting keys,
        because what matters is the end-to-end property -- the pair reaches the
        matcher -- not which particular key delivered it.
        """
        index = BlockingIndex()
        index.add_all(
            [
                record("ent_a", "Föö-Bär, Ltd."),
                record("ent_b", "foo bar limited"),
                record("ent_c", "ACME CORP."),
                record("ent_d", "  acme corp  "),
            ]
        )
        pairs = index.candidate_pairs()
        assert ("ent_a", "ent_b") in pairs
        assert ("ent_c", "ent_d") in pairs

    def test_word_order_variants_are_candidates(self) -> None:
        """`TOKEN_SET` covers what prefix and exact keys cannot."""
        index = BlockingIndex()
        index.add_all([record("ent_a", "Acme Cloud"), record("ent_b", "Cloud Acme")])
        assert index.candidate_pairs() == (("ent_a", "ent_b"),)
        kinds = {key.kind for key in index.shared_keys("ent_a", "ent_b")}
        assert BlockKind.TOKEN_SET in kinds

    def test_alias_table_bridges_unrelated_strings(self) -> None:
        """"Big Blue" and "IBM" share no characters; only the alias key links them.

        This is the one family that can bridge names with zero string overlap,
        and losing it means every nickname, ticker slang and product codename in
        the corpus resolves to its own node.
        """
        index = BlockingIndex()
        index.add_all(
            [
                record("ent_ibm", "IBM", aliases=("Big Blue",)),
                record("ent_bb", "Big Blue"),
            ]
        )
        assert index.candidate_pairs() == (("ent_bb", "ent_ibm"),)
        kinds = {key.kind for key in index.shared_keys("ent_bb", "ent_ibm")}
        assert BlockKind.ALIAS in kinds

    def test_misspelling_survives_via_phonetic_key(self) -> None:
        """A typo that destroys the exact, prefix and token keys still blocks."""
        index = BlockingIndex()
        index.add_all([record("ent_a", "Kolour Systems"), record("ent_b", "Colour Systems")])
        pairs = index.candidate_pairs()
        assert ("ent_a", "ent_b") in pairs
        kinds = {key.kind for key in index.shared_keys("ent_a", "ent_b")}
        assert BlockKind.PHONETIC in kinds

    def test_strong_identifier_blocks_regardless_of_name(self) -> None:
        """A shared domain is enough on its own; the names need not agree at all."""
        index = BlockingIndex()
        index.add_all(
            [
                record("ent_a", "Northwind", identifiers={"domain": "northwind.com"}),
                record("ent_b", "NW Traders", identifiers={"domain": "northwind.com"}),
            ]
        )
        assert index.candidate_pairs() == (("ent_a", "ent_b"),)

    def test_candidates_for_is_not_capped(self) -> None:
        """The streaming path bounds a linear cost, so capping it buys nothing.

        Capping `candidate_pairs()` prevents a quadratic blow-up. Capping this
        would trade recall for a saving that does not exist.
        """
        index = BlockingIndex(max_block_size=2)
        members = [record(f"ent_{i:03d}", f"Acme Systems {i}") for i in range(10)]
        index.add_all(members)
        probe = record("probe", "Acme Systems 3")
        assert len(index.candidates_for(probe)) == 10

    def test_unknown_type_still_blocks_against_a_typed_entity(self) -> None:
        """Degraded mentions must be resolvable, not quarantined.

        `services/signal_engine/entities.py` degrades an unrecognized label to
        `UNKNOWN` rather than dropping the mention. A type-partitioned index
        would put those in a partition where they can only ever meet each other.
        """
        index = BlockingIndex()
        index.add_all(
            [
                record("ent_typed", "Datadog", entity_type=EntityType.COMPANY),
                record("ent_untyped", "Datadog", entity_type=EntityType.UNKNOWN),
            ]
        )
        assert index.candidate_pairs() == (("ent_typed", "ent_untyped"),)


class TestBlockingCost:
    """Recall is bought with CPU, and the price has to stay visible."""

    def test_oversized_low_precision_blocks_are_skipped_and_recorded(self) -> None:
        """The cap is a knowing recall sacrifice, so it must leave a trace.

        A silent cap is indistinguishable from a blocking bug. `oversized_blocks`
        is the evidence that the pass chose latency over recall.
        """
        index = BlockingIndex(max_block_size=3)
        # A shared four-character prefix and no other shared key.
        index.add_all([record(f"ent_{i}", f"Interlink {i} Ventures") for i in range(8)])
        index.candidate_pairs()
        assert index.oversized_blocks
        assert all(kind_size > 3 for kind_size in index.oversized_blocks.values())
        assert index.stats().skipped_pairs > 0

    def test_high_precision_blocks_are_never_capped(self) -> None:
        """Ten thousand records sharing an exact name are not noise.

        They are ten thousand records that genuinely need comparing, and the cap
        exists for generative keys, not for values somebody wrote down.
        """
        index = BlockingIndex(max_block_size=2)
        index.add_all([record(f"ent_{i}", "Acme Corp") for i in range(6)])
        pairs = index.candidate_pairs()
        assert len(pairs) == 15  # 6 choose 2, nothing dropped
        assert not any(key.kind is BlockKind.EXACT for key in index.oversized_blocks)

    def test_removing_a_record_removes_its_keys(self) -> None:
        """Stale keys would let an un-merged entity re-collide on a lost alias."""
        index = BlockingIndex()
        index.add_all(
            [record("ent_a", "Acme", aliases=("Widgets Inc",)), record("ent_b", "Widgets")]
        )
        assert index.candidate_pairs()
        index.remove("ent_a")
        assert index.candidate_pairs() == ()
        assert "ent_a" not in index

    def test_reindexing_replaces_rather_than_accumulates(self) -> None:
        """`add` on a known id must not leave the previous surfaces indexed."""
        index = BlockingIndex()
        index.add(record("ent_a", "Acme", aliases=("Widgets",)))
        index.add(record("ent_a", "Acme"))
        index.add(record("ent_b", "Widgets"))
        assert index.candidate_pairs() == ()
        assert len(index) == 2


class TestEmbeddingBlocking:
    """Locality-sensitive hashing stands in for the ANN key of §6.

    It needs no vector store, which is what lets the semantic-alias path be
    tested with nothing running.
    """

    def test_near_vectors_share_a_band(self) -> None:
        base = tuple(float((i * 7) % 13) - 6.0 for i in range(32))
        near = tuple(value + 0.001 for value in base)
        left = set(blocking_keys(record("a", "Zeta", embedding=base)))
        right = set(blocking_keys(record("b", "Omega", embedding=near)))
        shared = {key for key in left & right if key.kind is BlockKind.EMBEDDING_LSH}
        assert shared, "near-identical vectors must collide in at least one band"

    def test_zero_vector_emits_no_key(self) -> None:
        """Every degenerate embedding would otherwise share one enormous bucket."""
        keys = blocking_keys(record("a", "Zeta", embedding=(0.0,) * 16))
        assert not any(key.kind is BlockKind.EMBEDDING_LSH for key in keys)

    def test_hyperplanes_are_stable_across_calls(self) -> None:
        """Two workers must bucket the same vector identically, or blocks diverge."""
        vector = tuple(float(i % 5) - 2.0 for i in range(24))
        first = blocking_keys(record("a", "Zeta", embedding=vector))
        second = blocking_keys(record("a", "Zeta", embedding=vector))
        assert first == second

    def test_cosine_returns_none_rather_than_zero_when_undefined(self) -> None:
        """`0.0` claims a measurement was made. `None` admits it was not."""
        assert cosine_similarity((1.0, 0.0), (1.0, 0.0)) == pytest.approx(1.0)
        assert cosine_similarity((1.0, 0.0), (0.0, 0.0)) is None
        assert cosine_similarity((1.0, 0.0), (1.0, 0.0, 0.0)) is None


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #


class TestNameSimilarity:
    """Where the "shares one word" false positive lives.

    `rapidfuzz.token_set_ratio` returns 100 whenever one token set is a subset of
    the other, and "Apple Inc" normalizes to "apple", a subset of
    "apple bakery". Every test here exists to keep the coverage penalty honest.
    """

    def test_apple_inc_and_apple_bakery_do_not_merge(self) -> None:
        """Two businesses sharing a word are not one business.

        This is the canonical false positive for name-based resolution and the
        reason the name feature is not `token_set_ratio` alone: on that measure
        this pair scores a perfect 1.0.
        """
        score = score_pair(record("ent_apple", "Apple Inc"), record("ent_bakery", "Apple Bakery"))
        assert score.decision is MatchDecision.DISTINCT
        assert score.combined < REVIEW_THRESHOLD
        name = score.value_of("name")
        assert name is not None and name < AUTO_MERGE_THRESHOLD

    def test_end_to_end_apple_pair_stays_two_entities(self) -> None:
        """The same case through the full pass: blocked together, then rejected.

        Blocking is *supposed* to propose this pair -- both names start "apple" --
        so the test also proves the rejection happens in the matcher rather than
        by an accident of blocking.
        """
        records = [record("ent_apple", "Apple Inc"), record("ent_bakery", "Apple Bakery")]
        index = BlockingIndex()
        index.add_all(records)
        assert ("ent_apple", "ent_bakery") in index.candidate_pairs()

        result = resolve(records, decided_at=T0)
        assert [cluster.member_ids for cluster in result.clusters] == [
            ("ent_apple",),
            ("ent_bakery",),
        ]
        assert result.merges == ()
        assert result.same_as_edges == ()

    def test_identical_normalized_names_score_exactly_one(self) -> None:
        """"Acme Corp." and "acme corp" must not sit fractionally below 0.92."""
        score = score_pair(record("a", "Acme Corp."), record("b", "acme corp"))
        assert score.combined == pytest.approx(1.0)
        assert score.decision is MatchDecision.MERGE

    def test_word_order_does_not_defeat_the_name_feature(self) -> None:
        score = score_pair(record("a", "Acme Cloud"), record("b", "Cloud Acme"))
        assert score.decision is MatchDecision.MERGE

    def test_unrelated_names_score_low(self) -> None:
        score = score_pair(record("a", "Acme"), record("b", "Globex"))
        assert score.combined < 0.3
        assert score.decision is MatchDecision.DISTINCT

    def test_extra_substantive_token_lands_in_review_not_merge(self) -> None:
        """"Acme Cloud" vs "Acme Cloud Platform" is a human's call, not a rule's."""
        score = score_pair(record("a", "Acme Cloud"), record("b", "Acme Cloud Platform"))
        assert score.decision is MatchDecision.REVIEW
        assert REVIEW_THRESHOLD <= score.combined < AUTO_MERGE_THRESHOLD


class TestFeatureAvailability:
    """Missing evidence must be neutral, never negative.

    Most incoming mentions have no embedding and no context. Scoring those as
    0.0 would drag every fresh mention below the merge band and silently switch
    resolution off for new data, while a suite written against fully-populated
    fixtures stayed green -- so the fixtures here are deliberately sparse.
    """

    def test_weights_sum_to_one(self) -> None:
        """A drifting weight total silently rescales every score in the system."""
        assert sum(MATCH_WEIGHTS.values()) == pytest.approx(1.0)
        assert sum(NAME_BLEND_WEIGHTS.values()) == pytest.approx(1.0)

    def test_absent_features_are_excluded_not_zeroed(self) -> None:
        """Name-only pairs must still be able to reach the merge band.

        With missing features scored as zero, the maximum achievable combined
        score for a name-only pair is `MATCH_WEIGHTS["name"]` -- 0.35 -- and the
        0.92 threshold becomes unreachable for every record without an embedding.
        """
        score = score_pair(record("a", "Acme Corp"), record("b", "acme corporation"))
        assert score.value_of("embedding") is None
        assert score.value_of("context") is None
        assert score.combined == pytest.approx(1.0)

    def test_available_features_are_reported_with_their_reason(self) -> None:
        """A reviewer needs to know *why* a feature is missing, not just that it is."""
        score = score_pair(record("a", "Acme"), record("b", "Acme"))
        embedding = score.feature("embedding")
        assert embedding is not None
        assert embedding.available is False
        assert "missing" in embedding.detail

    def test_embedding_dimension_mismatch_is_unavailable_not_zero(self) -> None:
        """Two embedding models produce incomparable vectors.

        Truncating to the shorter one would produce a number that looks
        authoritative and is meaningless.
        """
        score = score_pair(
            record("a", "Acme", embedding=(1.0, 0.0, 0.0)),
            record("b", "Acme", embedding=(1.0, 0.0)),
        )
        embedding = score.feature("embedding")
        assert embedding is not None
        assert embedding.value is None
        assert "dimension mismatch" in embedding.detail

    def test_alias_feature_absent_when_neither_side_declares_one(self) -> None:
        """Otherwise the two canonical names get counted twice under two weights."""
        score = score_pair(record("a", "Acme"), record("b", "Acme"))
        assert score.value_of("alias") is None

    def test_disjoint_context_does_not_veto_an_identical_name(self) -> None:
        """Sparse-set disagreement is not disagreement.

        Two articles about the same company routinely mention no other entity in
        common, so a Jaccard of 0.0 is the *ordinary* case for a true match.
        Carried at weight 0.20 it would drag a character-for-character identical
        pair from 1.00 down to 0.73 -- below the review band entirely -- and
        resolution would stop merging exactly the records that arrived from two
        different sources, which is every interesting merge in the system.
        """
        score = score_pair(
            record("a", "Acme Corp", context=frozenset({"sig_1", "sig_2"})),
            record("b", "acme corp.", context=frozenset({"sig_8", "sig_9"})),
        )
        assert score.value_of("context") is None
        assert score.combined == pytest.approx(1.0)
        assert score.decision is MatchDecision.MERGE

    def test_unrelated_aliases_do_not_veto_an_identical_name(self) -> None:
        """Recording a true alias must never make an entity harder to resolve.

        If a non-overlapping alias list scored 0.0, two records that each declare
        one alias would rank below two records that declare none -- so curating
        the alias table would degrade the system that reads it.
        """
        score = score_pair(
            record("a", "Acme Corp", aliases=("Acme Widgets",)),
            record("b", "acme corp.", aliases=("Acme Freight Systems",)),
        )
        alias = score.feature("alias")
        assert alias is not None
        assert alias.value is None
        assert alias.detail.startswith("no alias evidence")
        assert score.decision is MatchDecision.MERGE

    def test_overlapping_aliases_still_corroborate(self) -> None:
        """The feature is suppressed only when it says nothing, not always."""
        score = score_pair(
            record("a", "Acme Corp", aliases=("ACME",)),
            record("b", "Acme Holdings", aliases=("acme",)),
        )
        assert score.value_of("alias") == pytest.approx(1.0)

    def test_context_only_evidence_cannot_carry_a_merge(self) -> None:
        """Rivals discussed in the same threads share context and are not one thing."""
        shared = frozenset({"sig_1", "sig_2", "sig_3"})
        score = score_pair(
            record("a", "Postgres", context=shared),
            record("b", "MySQL", context=shared),
        )
        assert score.decision is not MatchDecision.MERGE

    def test_no_comparable_features_is_distinct(self) -> None:
        """Two unnameable records with nothing else are not evidence of anything."""
        score = score_pair(record("a", "???"), record("b", "###"))
        assert score.decision is MatchDecision.DISTINCT
        assert score.applied_rule == "no_comparable_features"

    def test_name_evidence_floor_demotes_a_merge_with_no_string_evidence(self) -> None:
        """High score, zero string evidence: review it, do not merge it.

        Both records are unnameable, so the name feature is unavailable and
        renormalization would otherwise let embedding plus context alone clear
        0.92 -- a merge founded entirely on "these two sit near each other".
        """
        vector = tuple(float(i % 5) + 1.0 for i in range(16))
        shared = frozenset({"sig_1", "sig_2"})
        score = score_pair(
            record("a", "???", embedding=vector, context=shared),
            record("b", "###", embedding=vector, context=shared),
        )
        assert score.combined >= AUTO_MERGE_THRESHOLD
        assert score.decision is MatchDecision.REVIEW
        assert score.applied_rule == "name_evidence_floor"


class TestHardRules:
    """Rules that override the arithmetic, in both directions (§6)."""

    def test_different_labels_never_merge(self) -> None:
        """"Stripe" the company and "Stripe" the product are two nodes."""
        score = score_pair(
            record("a", "Stripe", entity_type=EntityType.COMPANY),
            record("b", "Stripe", entity_type=EntityType.PRODUCT),
        )
        assert score.decision is MatchDecision.DISTINCT
        assert score.applied_rule is not None
        assert score.applied_rule.startswith("type_conflict")

    def test_unknown_type_is_a_wildcard(self) -> None:
        """A degraded label is an absence of information, not a conflicting one."""
        score = score_pair(
            record("a", "Datadog", entity_type=EntityType.COMPANY),
            record("b", "Datadog", entity_type=EntityType.UNKNOWN),
        )
        assert score.decision is MatchDecision.MERGE

    def test_shared_domain_forces_a_match(self) -> None:
        """Externally-assigned identifiers do not collide by accident."""
        score = score_pair(
            record("a", "Northwind", identifiers={"domain": "nw.com"}),
            record("b", "NW Traders Group", identifiers={"domain": "nw.com"}),
        )
        assert score.decision is MatchDecision.MERGE
        assert score.applied_rule == "identifier_match:domain"

    def test_conflicting_ticker_forces_a_non_match(self) -> None:
        """Identical names, different tickers: two listed entities, not one.

        Conflict is checked before agreement precisely so a stale shared domain
        cannot outvote a disagreeing ticker.
        """
        score = score_pair(
            record("a", "Acme", identifiers={"ticker": "acm", "domain": "acme.com"}),
            record("b", "Acme", identifiers={"ticker": "amc", "domain": "acme.com"}),
        )
        assert score.decision is MatchDecision.DISTINCT
        assert score.applied_rule == "identifier_conflict:ticker"

    def test_type_conflict_outranks_identifier_agreement(self) -> None:
        """A product hosted on its vendor's domain is still not the vendor."""
        score = score_pair(
            record(
                "a", "Acme", entity_type=EntityType.COMPANY, identifiers={"domain": "acme.com"}
            ),
            record(
                "b", "Acme", entity_type=EntityType.PRODUCT, identifiers={"domain": "acme.com"}
            ),
        )
        assert score.decision is MatchDecision.DISTINCT

    def test_exact_alias_hit_is_promoted_to_review(self) -> None:
        """The alias table's whole purpose, rescued from the weighted average.

        Promoted to review rather than merged: aliases are absorbed
        automatically by earlier merges, so auto-merging on one would let a
        single bad absorption cascade.
        """
        score = score_pair(
            record("a", "IBM", aliases=("Big Blue",)),
            record("b", "Big Blue"),
        )
        assert score.decision is MatchDecision.REVIEW
        assert score.applied_rule == "exact_alias_hit"


class TestExplainability:
    """`MatchScore`, not a float: a merge nobody can explain cannot be undone."""

    def test_score_carries_every_feature_including_the_missing_ones(self) -> None:
        score = score_pair(record("a", "Acme Corp"), record("b", "acme corp"))
        assert {feature.name for feature in score.features} == set(MATCH_WEIGHTS)

    def test_explain_names_the_decision_and_the_features(self) -> None:
        text = score_pair(record("a", "Acme"), record("b", "Globex")).explain()
        assert "distinct" in text
        assert "name=" in text
        assert "alias=n/a" in text

    def test_as_dict_preserves_the_missing_feature_distinction(self) -> None:
        """"We had no embedding" is the fact a future reviewer needs most."""
        payload = score_pair(record("a", "Acme"), record("b", "Acme")).as_dict()
        assert payload["features"]["embedding"] is None  # type: ignore[index]
        assert payload["decision"] == "merge"

    def test_scorer_memoizes_symmetric_pairs(self) -> None:
        """`score(a, b)` and `score(b, a)` are one decision, not two."""
        scorer = MatchScorer()
        left, right = record("a", "Acme"), record("b", "Acme")
        scorer.score(left, right)
        scorer.score(right, left)
        assert scorer.comparisons == 1

    def test_review_threshold_above_merge_threshold_is_rejected(self) -> None:
        """An inverted band would make every review item also an auto-merge."""
        with pytest.raises(ValueError, match="review_threshold"):
            MatchScorer(auto_merge_threshold=0.5, review_threshold=0.9)


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #


class TestSurvivorElection:
    """Which id survives decides how many references dangle. It must be stable."""

    def test_earliest_first_seen_wins(self) -> None:
        """The oldest id is the one the rest of the system already points at."""
        older = record("ent_z", "Acme", first_seen=T0, mention_count=1)
        newer = record("ent_a", "Acme", first_seen=T0 + timedelta(days=30), mention_count=99)
        assert elect_survivor([newer, older]).id == "ent_z"

    def test_mention_count_breaks_a_first_seen_tie(self) -> None:
        """Better-evidenced records usually carry the better name and aliases."""
        thin = record("ent_a", "Acme", first_seen=T0, mention_count=1)
        thick = record("ent_z", "Acme", first_seen=T0, mention_count=50)
        assert elect_survivor([thin, thick]).id == "ent_z"

    def test_id_breaks_the_final_tie(self) -> None:
        """Ids are unique, so the order is total and no pair ends undecided."""
        left = record("ent_a", "Acme", first_seen=T0, mention_count=5)
        right = record("ent_b", "Acme", first_seen=T0, mention_count=5)
        assert elect_survivor([right, left]).id == "ent_a"

    def test_missing_first_seen_sorts_last(self) -> None:
        """An absent timestamp is unknown, not infinitely old.

        Treating it as old would let an unstamped mention displace an
        established node with real history behind it.
        """
        stamped = record("ent_z", "Acme", first_seen=T0)
        unstamped = record("ent_a", "Acme", first_seen=None, mention_count=100)
        assert elect_survivor([unstamped, stamped]).id == "ent_z"

    def test_empty_cluster_is_a_programming_error(self) -> None:
        with pytest.raises(ValueError, match="empty cluster"):
            elect_survivor([])


class TestClusterRefinement:
    """Pairwise decisions are not transitive, and chaining is the classic bug.

    Plain connected components will happily fuse a hundred entities through a
    path of individually-defensible links. These tests pin the constrained
    version.
    """

    def test_a_weak_closing_edge_splits_the_component(self) -> None:
        """A~B 0.95 and B~C 0.95 does not make A~C.

        Average linkage of `{A, B}` to `{C}` is 0.525, below the floor, so C
        stays separate instead of being chained in.
        """
        similarity = {("a", "b"): 0.95, ("b", "c"): 0.95, ("a", "c"): 0.10}
        assert refine_component(["a", "b", "c"], similarity) == (("a", "b"), ("c",))

    def test_a_coherent_component_survives_intact(self) -> None:
        """The floor must not shatter clusters held together through a hub."""
        similarity = {("a", "b"): 0.95, ("b", "c"): 0.95, ("a", "c"): 0.80}
        assert refine_component(["a", "b", "c"], similarity) == (("a", "b", "c"),)

    def test_refinement_is_independent_of_member_order(self) -> None:
        """Ties on linkage are common; `max()` over a dict would resolve them
        by iteration order, and two workers would build different clusters."""
        similarity = {("a", "b"): 0.9, ("b", "c"): 0.9, ("a", "c"): 0.9}
        expected = refine_component(["a", "b", "c"], similarity)
        for order in (["c", "b", "a"], ["b", "a", "c"], ["c", "a", "b"]):
            assert refine_component(order, similarity) == expected

    def test_floor_of_zero_restores_plain_connected_components(self) -> None:
        """The degenerate setting, kept honest so the floor's effect is legible."""
        similarity = {("a", "b"): 0.95, ("b", "c"): 0.95, ("a", "c"): 0.0}
        assert refine_component(["a", "b", "c"], similarity, 0.0) == (("a", "b", "c"),)

    def test_singleton_component_is_returned_unchanged(self) -> None:
        assert refine_component(["a"], {}) == (("a",),)


# --------------------------------------------------------------------------- #
# The pass
# --------------------------------------------------------------------------- #


class TestResolveContract:
    """Invariants every pass must hold regardless of the data."""

    def test_every_input_appears_in_exactly_one_cluster(self) -> None:
        """A record silently dropped by resolution is a node that never appears."""
        records = [
            record("ent_a", "Acme Corp"),
            record("ent_b", "acme corp."),
            record("ent_c", "Globex"),
            record("ent_d", "Initech"),
        ]
        result = resolve(records, decided_at=T0)
        seen = [rid for cluster in result.clusters for rid in cluster.member_ids]
        assert sorted(seen) == sorted(r.id for r in records)
        assert len(seen) == len(set(seen))

    def test_duplicate_ids_are_rejected(self) -> None:
        """Two records sharing an id make survivor election ambiguous."""
        with pytest.raises(ValueError, match="duplicate record id"):
            resolve([record("ent_a", "Acme"), record("ent_a", "Acme")], decided_at=T0)

    def test_oversized_components_skip_refinement_and_are_reported(self) -> None:
        """Refinement is O(k^3); above the cap it degrades, but never silently.

        An unrefined component is the one most likely to contain a bad chain --
        it got that large by pulling in everything a popular name touched -- so
        the compromise is surfaced for a human rather than absorbed.
        """
        records = [
            record(f"ent_{i:03d}", "Acme Corp", first_seen=T0, mention_count=1)
            for i in range(65)
        ]
        result = resolve(records, decided_at=T0)
        assert len(result.unrefined_components) == 1
        assert len(result.unrefined_components[0]) == 65
        assert len(result.clusters) == 1
        assert result.clusters[0].survivor_id == "ent_000"

    def test_review_band_pairs_are_queued_not_merged(self) -> None:
        """The ambiguous middle is deferred, never guessed (§6)."""
        records = [record("ent_a", "Acme Cloud"), record("ent_b", "Acme Cloud Platform")]
        result = resolve(records, decided_at=T0)
        assert result.merges == ()
        assert len(result.review_items) == 1
        item = result.review_items[0]
        assert item.pair == ("ent_a", "ent_b")
        assert REVIEW_THRESHOLD <= item.score.combined < AUTO_MERGE_THRESHOLD

    def test_merge_produces_a_same_as_edge_and_an_audit_record(self) -> None:
        """The absorbed node is redirected, never deleted (§6, steps 4 and 5)."""
        records = [
            record("ent_keep", "Acme Corp", first_seen=T0, mention_count=9),
            record("ent_gone", "acme corp.", first_seen=T0 + timedelta(days=1)),
        ]
        result = resolve(records, decided_at=T0)
        assert len(result.same_as_edges) == 1
        edge = result.same_as_edges[0]
        assert (edge.from_id, edge.to_id) == ("ent_gone", "ent_keep")
        assert edge.edge_type is EdgeType.SAME_AS
        assert result.merges[0].canonical_id == "ent_keep"
        assert result.merges[0].absorbed_ids == ("ent_gone",)
        assert result.canonical_id_for("ent_gone") == "ent_keep"

    def test_edge_key_is_stable_across_replays(self) -> None:
        """At-least-once Kafka delivery replays batches; the writer `MERGE`s on
        this key, so an unstable key creates a second edge per replay (§7)."""
        records = [record("ent_a", "Acme Corp"), record("ent_b", "acme corp.")]
        first = resolve(records, decided_at=T0).same_as_edges[0]
        second = resolve(records, decided_at=T0).same_as_edges[0]
        assert first.edge_key == second.edge_key
        assert first.merge_id == second.merge_id

    def test_merged_entity_unions_aliases_and_sums_counts(self) -> None:
        """The absorbed spelling stays searchable; the evidence adds up."""
        records = [
            record("ent_a", "Acme Corp", aliases=("ACME",), first_seen=T0, mention_count=7),
            record(
                "ent_b",
                "acme corp.",
                aliases=("Acme Widgets",),
                first_seen=T0 + timedelta(days=2),
                mention_count=4,
            ),
        ]
        cluster = resolve(records, decided_at=T0).clusters[0]
        assert cluster.canonical.mention_count == 11
        assert "acme corp." in cluster.canonical.aliases
        assert "ACME" in cluster.canonical.aliases
        assert "Acme Widgets" in cluster.canonical.aliases
        assert cluster.canonical.merged_from == ("ent_b",)
        assert cluster.canonical.first_seen == T0
        assert cluster.canonical.last_seen == T0 + timedelta(days=2)

    def test_cluster_confidence_is_the_weakest_link(self) -> None:
        """A cluster is only as trustworthy as its least convincing pair.

        Averaging lets two excellent links hide a marginal third -- which is the
        merge most likely to be wrong and the one a reviewer most needs ranked
        low.
        """
        records = [
            record("ent_a", "Acme Corp"),
            record("ent_b", "acme corp."),
            record("ent_c", "Acme Corporation"),
        ]
        cluster = resolve(records, decided_at=T0).clusters[0]
        assert cluster.weakest_link is not None
        assert cluster.weakest_link == pytest.approx(1.0)

    def test_to_entity_projects_onto_the_graph_model(self) -> None:
        """The pass returns a plan; `graph/ingest/writer.py` writes these rows."""
        records = [
            record("ent_a", "Acme Corp", first_seen=T0, mention_count=3),
            record("ent_b", "acme corp.", first_seen=T0 + timedelta(days=1)),
        ]
        entities = resolve(records, decided_at=T0).entities()
        assert all(isinstance(entity, Entity) for entity in entities)
        merged = entities[0]
        assert merged.merged_from == ["ent_b"]
        assert merged.resolution_confidence == pytest.approx(1.0)
        assert merged.properties["source_count"] == 4

    def test_from_entity_round_trips_through_to_entity(self) -> None:
        """Resolution reads graph nodes and writes them back; the projection
        must not lose identifiers, counters or context on the way."""
        original = record(
            "ent_a",
            "Acme Corp",
            aliases=("ACME",),
            identifiers={"domain": "acme.com", "handle:reddit": "acmehq"},
            context=frozenset({"sig_1"}),
            first_seen=T0,
            mention_count=12,
        )
        restored = ResolutionRecord.from_entity(original.to_entity())
        assert restored.identifiers == original.identifiers
        assert restored.mention_count == original.mention_count
        assert restored.context == original.context
        assert restored.aliases == original.aliases

    def test_malformed_graph_properties_do_not_abort_a_batch(self) -> None:
        """Node properties come from other writers and may hold anything.

        A resolution pass that dies on one malformed node fails an entire batch
        for one bad row.
        """
        entity = Entity(
            id="ent_a",
            type=EntityType.COMPANY,
            canonical_name="Acme",
            properties={
                "embedding": "not-a-vector",
                "source_count": "seventeen",
                "context": {"not": "a list"},
                "domain": 42,
            },
        )
        projected = ResolutionRecord.from_entity(entity)
        assert projected.embedding is None
        assert projected.mention_count == 0
        assert projected.context == frozenset()
        assert projected.identifiers == {}

    async def test_resolve_async_matches_resolve(self) -> None:
        """The worker-facing entry point must not change the answer.

        It exists to keep a CPU-bound pass off the event loop -- blocking it for
        a whole batch stalls the Kafka heartbeat and triggers a consumer-group
        rebalance mid-batch -- not to resolve differently.
        """
        records = [record("ent_a", "Acme Corp"), record("ent_b", "acme corp.")]
        assert fingerprint(await resolve_async(records, decided_at=T0)) == fingerprint(
            resolve(records, decided_at=T0)
        )


class TestOverrides:
    """`must_link` / `must_not_link`: how a human corrects the resolver (§6)."""

    def test_must_not_link_prevents_an_otherwise_certain_merge(self) -> None:
        """The constraint is the durable half of a correction.

        Without it the next pass sees the same records, computes the same scores
        and rebuilds the merge -- the fix survives until the worker catches up.
        """
        records = [record("ent_a", "Acme Corp"), record("ent_b", "acme corp.")]
        result = resolve(records, decided_at=T0, must_not_link=[("ent_b", "ent_a")])
        assert result.merges == ()
        assert [cluster.member_ids for cluster in result.clusters] == [("ent_a",), ("ent_b",)]

    def test_must_link_merges_a_pair_blocking_never_proposed(self) -> None:
        """A human who asserts identity has information the index does not.

        Requiring them to also fix the blocking keys first would make the
        override useless in exactly the cases it exists for.
        """
        records = [record("ent_a", "Big Blue"), record("ent_b", "Watson Research")]
        index = BlockingIndex()
        index.add_all(records)
        assert index.candidate_pairs() == ()

        result = resolve(records, decided_at=T0, must_link=[("ent_a", "ent_b")])
        assert len(result.clusters) == 1
        assert result.clusters[0].member_ids == ("ent_a", "ent_b")

    def test_must_link_survives_cluster_refinement(self) -> None:
        """Average linkage must not overrule an explicit human decision.

        If it did, the reviewer would adjudicate the same pair every pass and
        their decision would never stick.
        """
        records = [
            record("ent_a", "Big Blue"),
            record("ent_b", "IBM"),
            record("ent_c", "International Business Machines"),
        ]
        result = resolve(
            records, decided_at=T0, must_link=[("ent_a", "ent_b"), ("ent_b", "ent_c")]
        )
        assert len(result.clusters) == 1
        assert result.clusters[0].member_ids == ("ent_a", "ent_b", "ent_c")

    def test_must_not_link_wins_over_must_link(self) -> None:
        """A contradictory override pair must resolve one way, deterministically.

        Refusing to merge is the safe direction: it leaves a duplicate node,
        which the next reviewer can fix, rather than an unwanted merge, which
        needs the un-merge path.
        """
        records = [record("ent_a", "Acme Corp"), record("ent_b", "acme corp.")]
        result = resolve(
            records,
            decided_at=T0,
            must_link=[("ent_a", "ent_b")],
            must_not_link=[("ent_a", "ent_b")],
        )
        assert result.merges == ()


class TestUnmergeRoundTrip:
    """The correction path. A merge that cannot be undone must not be made."""

    def test_merge_round_trips_exactly(self) -> None:
        """Restoration is by snapshot, so equality is exact, not approximate.

        A delta-based un-merge cannot invert an alias union -- it does not know
        which side contributed a shared alias -- and would lose one every round
        trip. Asserting full object equality is what catches that.
        """
        records = [
            record(
                "ent_keep",
                "Acme Corp",
                aliases=("ACME",),
                first_seen=T0,
                mention_count=9,
                context=frozenset({"sig_1"}),
            ),
            record(
                "ent_gone",
                "acme corp.",
                aliases=("Acme Widgets",),
                first_seen=T0 + timedelta(days=3),
                mention_count=4,
                context=frozenset({"sig_2"}),
            ),
        ]
        result = resolve(records, decided_at=T0)
        merge = result.merges[0]
        assert merge.absorbed_ids == ("ent_gone",)

        reversal = unmerge(merge, decided_at=T0)
        by_id = {r.id: r for r in records}
        assert list(reversal.restored) == [by_id["ent_gone"]]
        assert reversal.canonical == by_id["ent_keep"]
        assert reversal.retracted_edges == (("ent_gone", "ent_keep"),)
        assert reversal.must_not_link == (("ent_gone", "ent_keep"),)

    def test_unmerge_constraint_stops_the_next_pass_re_merging(self) -> None:
        """The full correction cycle: merge, reverse, feed the constraint back.

        Without this the graph oscillates -- the reviewer's correction is undone
        by the next ingest, silently, every time.
        """
        records = [record("ent_a", "Acme Corp"), record("ent_b", "acme corp.")]
        merge = resolve(records, decided_at=T0).merges[0]
        reversal = unmerge(merge, decided_at=T0)

        result = resolve(records, decided_at=T0, must_not_link=reversal.must_not_link)
        assert result.merges == ()
        assert [cluster.member_ids for cluster in result.clusters] == [("ent_a",), ("ent_b",)]

    def test_partial_unmerge_detaches_one_member_and_re_folds_the_rest(self) -> None:
        """A four-way cluster is usually wrong about one member, not all of them.

        Forcing a reviewer to explode the whole cluster and re-adjudicate every
        pair is how correction queues stop being used.
        """
        records = [
            record("ent_a", "Acme Corp", first_seen=T0, mention_count=5),
            record("ent_b", "acme corp.", first_seen=T0 + timedelta(days=1)),
            record("ent_c", "ACME  Corporation", first_seen=T0 + timedelta(days=2)),
        ]
        merge = resolve(records, decided_at=T0).merges[0]
        assert merge.absorbed_ids == ("ent_b", "ent_c")

        reversal = unmerge(merge, separate=["ent_b"], decided_at=T0)
        assert [r.id for r in reversal.restored] == ["ent_b"]
        assert reversal.canonical is not None
        assert reversal.canonical.id == "ent_a"
        assert reversal.canonical.merged_from == ("ent_c",)
        # Constraints only between what was detached and what was kept: saying
        # "B is not A" says nothing about whether B is C.
        assert reversal.must_not_link == (("ent_a", "ent_b"), ("ent_b", "ent_c"))

    def test_canonical_id_is_not_re_elected_on_reversal(self) -> None:
        """Everything now points at the canonical id; re-electing dangles it all."""
        records = [
            record("ent_z", "Acme Corp", first_seen=T0, mention_count=1),
            record("ent_a", "acme corp.", first_seen=T0 + timedelta(days=1), mention_count=99),
        ]
        merge = resolve(records, decided_at=T0).merges[0]
        assert merge.canonical_id == "ent_z"
        reversal = unmerge(merge, decided_at=T0)
        assert reversal.canonical is not None
        assert reversal.canonical.id == "ent_z"

    def test_detaching_the_survivor_is_rejected(self) -> None:
        """That is a re-election, which `resolve()` performs, not a reversal."""
        records = [record("ent_a", "Acme Corp"), record("ent_b", "acme corp.")]
        merge = resolve(records, decided_at=T0).merges[0]
        with pytest.raises(ValueError, match="canonical id"):
            unmerge(merge, separate=[merge.canonical_id], decided_at=T0)

    def test_detaching_an_unknown_id_is_rejected(self) -> None:
        """Silently ignoring it would report a correction that did not happen."""
        records = [record("ent_a", "Acme Corp"), record("ent_b", "acme corp.")]
        merge = resolve(records, decided_at=T0).merges[0]
        with pytest.raises(ValueError, match="never absorbed"):
            unmerge(merge, separate=["ent_nope"], decided_at=T0)

    def test_merge_record_is_self_sufficient(self) -> None:
        """Reversal must not depend on the graph, the matcher or the raw signals
        still being unchanged months later."""
        records = [record("ent_a", "Acme Corp"), record("ent_b", "acme corp.")]
        merge = resolve(records, decided_at=T0).merges[0]
        assert merge.snapshot_for("ent_b") == records[1]
        assert set(merge.member_ids) == {"ent_a", "ent_b"}
        payload = merge.as_dict()
        assert payload["canonical_id"] == "ent_a"
        assert len(payload["snapshots"]) == 2  # type: ignore[arg-type]

    def test_snapshot_for_unknown_id_raises(self) -> None:
        """A merge that cannot produce a snapshot for its own member is corrupt,
        and continuing would write a half-restored entity into the graph."""
        records = [record("ent_a", "Acme Corp"), record("ent_b", "acme corp.")]
        merge = resolve(records, decided_at=T0).merges[0]
        with pytest.raises(KeyError):
            merge.snapshot_for("ent_missing")


class TestDeterminism:
    """Two workers on one Kafka partition must build the same graph.

    Every assertion here is about *order independence*, because the realistic
    source of divergence is not a random number generator -- it is a `set` or
    `dict` iteration order that happens to differ between two processes holding
    the same records in a different sequence.
    """

    @staticmethod
    def _corpus() -> list[ResolutionRecord]:
        """A batch with merges, review items and singletons in one pass."""
        return [
            record("ent_acme", "Acme Corp.", first_seen=T0, mention_count=10),
            record("ent_acme_2", "acme corp", first_seen=T0, mention_count=3),
            record("ent_acme_3", "ACME Corporation", first_seen=T0, mention_count=1),
            record("ent_apple", "Apple Inc", first_seen=T0),
            record("ent_bakery", "Apple Bakery", first_seen=T0),
            record("ent_cloud", "Acme Cloud", first_seen=T0),
            record("ent_cloud_p", "Acme Cloud Platform", first_seen=T0),
            record("ent_globex", "Globex Ltd", first_seen=T0),
            record("ent_ibm", "IBM", aliases=("Big Blue",), first_seen=T0),
            record("ent_bb", "Big Blue", first_seen=T0),
        ]

    def test_shuffled_input_produces_an_identical_plan(self) -> None:
        """The headline determinism property, asserted on the whole plan.

        Comparing cluster membership alone would miss the divergence that
        actually matters -- which id survived, and therefore which node every
        future reference resolves to.
        """
        records = self._corpus()
        expected = fingerprint(resolve(records, decided_at=T0))
        for seed in range(20):
            shuffled = list(records)
            random.Random(seed).shuffle(shuffled)
            assert fingerprint(resolve(shuffled, decided_at=T0)) == expected, f"seed={seed}"

    def test_survivor_choice_is_stable_under_reordering(self) -> None:
        """Stated separately from the plan comparison because this is the field
        that silently splits a graph in two when it drifts."""
        records = self._corpus()
        expected = {
            cluster.survivor_id: cluster.member_ids
            for cluster in resolve(records, decided_at=T0).clusters
        }
        for seed in range(5):
            shuffled = list(records)
            random.Random(seed).shuffle(shuffled)
            actual = {
                cluster.survivor_id: cluster.member_ids
                for cluster in resolve(shuffled, decided_at=T0).clusters
            }
            assert actual == expected

    def test_blocking_keys_do_not_depend_on_alias_order(self) -> None:
        """Two writers listing the same aliases differently must block alike."""
        left = record("ent_a", "Acme", aliases=("ACME", "Acme Widgets", "Acme Co"))
        right = record("ent_a", "Acme", aliases=("Acme Co", "ACME", "Acme Widgets"))
        assert set(blocking_keys(left)) == set(blocking_keys(right))

    def test_pair_key_is_orientation_independent(self) -> None:
        """`(a, b)` and `(b, a)` must be one entry in every constraint table,
        or a `must_not_link` written one way fails to block the other."""
        assert pair_key("b", "a") == pair_key("a", "b") == ("a", "b")

    def test_decided_at_is_injectable(self) -> None:
        """The only non-deterministic input, and it is a parameter.

        A resolver that stamps `utcnow()` on its own audit records produces
        different records for identical inputs and cannot be replayed or diffed
        against a second worker.
        """
        records = [record("ent_a", "Acme Corp"), record("ent_b", "acme corp.")]
        first = resolve(records, decided_at=T0).merges[0]
        second = resolve(records, decided_at=T0 + timedelta(seconds=1)).merges[0]
        assert first.id != second.id
        assert first.decided_at == T0

    def test_defaults_are_the_documented_values(self) -> None:
        """The bands in `docs/knowledge-graph.md` §6 are the shipped behaviour.

        Pinned because a drifting default silently changes how many entities
        exist, and nothing else in the system would notice.
        """
        assert AUTO_MERGE_THRESHOLD == 0.92
        assert REVIEW_THRESHOLD == 0.75
        assert MIN_WITHIN_CLUSTER_LINKAGE == 0.60
        assert DEFAULT_MAX_BLOCK_SIZE == 200
