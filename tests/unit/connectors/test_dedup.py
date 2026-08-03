"""Unit tests for `connectors/dedup/`.

The dedup layer has one property that is both the most important and the easiest
to break by accident: **near-duplicates are clustered, never dropped**
(`docs/signal-model.md` §4.3). Six copies of a press release across six platforms
are the evidence of spread; a helpful-looking `drop_duplicates()` added six
months from now would silently delete five of them, take the trend volume with
it, and collapse the `corroboration` term in every affected Signal's confidence.
Nothing would fail. So there are tests here that assert on the *shape of the
module's API*, not just on its behaviour.

The rest targets the things a fingerprint implementation gets quietly wrong:

- a fingerprint that is really just a hash of the whole string, so a one-word
  edit looks as different as an unrelated document;
- a fingerprint that is not stable across processes, because `hash()` is salted;
- an unweighted feature set, so a press release and its one-line summary rank
  alike;
- banding that claims to be an exact index while missing pairs, because the
  threshold was widened without widening the bands;
- a canonical election that depends on input order, so the cluster's canonical
  member changes between two runs over identical data;
- a Redis store that fails *closed*, dropping real observations every time the
  cache blips.

Everything runs with no Redis, no network and no services
(`docs/architecture.md` §6.2 rule 2).
"""

from __future__ import annotations

import itertools
import logging
import random
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from connectors.dedup import hashing
from connectors.dedup.hashing import (
    SIMHASH_BITS,
    SIMHASH_DISTANCE_THRESHOLD,
    SIMHASH_LSH_BANDS,
    BandedIndex,
    ClusterMember,
    assign_clusters,
    canonicalize,
    cluster_id_for,
    content_key,
    content_sha256,
    elect_canonical,
    hamming,
    identity_key,
    is_near_duplicate,
    jaccard,
    minhash,
    minhash_band_keys,
    shingles,
    simhash64,
    simhash_band_keys,
    tokenize,
)
from connectors.dedup.store import InMemoryDedupStore, RedisDedupStore
from connectors.protocol import DedupStore
from models.enums import SignalStatus
from models.lineage import Lineage

pytestmark = pytest.mark.unit

T0 = datetime(2026, 7, 28, 12, 0, 0, tzinfo=UTC)


# --------------------------------------------------------------------------- #
# Corpus
#
# Real prose rather than lorem ipsum: SimHash behaviour depends on the shingle
# distribution of natural language, and a fixture built from repeated filler
# words would produce numbers that say nothing about the wire stories this is
# actually for.
# --------------------------------------------------------------------------- #

WIRE = (
    "Acme Corp today announced the general availability of Acme Observability Cloud, a fully "
    "managed platform that consolidates metrics, logs and traces into a single billing model. "
    "The company said early customers reduced observability spend by roughly forty percent "
    "during the beta programme, which ran for six months across eleven enterprise accounts. "
    "Acme Observability Cloud ingests telemetry through an OpenTelemetry collector and stores "
    "it in a columnar backend the company built in house, replacing the three separate vendors "
    "most of its customers were running before. Pricing is per gigabyte ingested with no "
    "per-host charge, a structure the company said was the single most requested change from "
    "customers renewing existing contracts. The product is available immediately in North "
    "America and Europe, with Asia Pacific regions planned for the first quarter of next year."
)
"""The press release, as it left the wire."""

SYNDICATED = (
    "By Jane Doe.\n\n"
    "ACME CORP TODAY ANNOUNCED the general availability of Acme Observability Cloud — a "
    "fully managed platform that consolidates metrics, logs and traces into a single billing "
    "model.\n\n"
    "The company said early customers reduced observability spend by roughly forty percent "
    "during the beta programme, which ran for six months across eleven enterprise accounts. "
    "Acme Observability Cloud ingests telemetry through an OpenTelemetry collector and stores "
    "it in a columnar backend the company built in house, replacing the three separate "
    "vendors most of its customers were running before.   Pricing is per gigabyte ingested "
    "with no per\u2010host charge, a structure the company said was the single most requested "
    "change from customers renewing existing contracts. The product is available immediately "
    "in North America and Europe, with Asia Pacific regions planned for the first quarter of "
    "next year. Acme did not respond to a request for comment.\n"
    "[Read more…]"
)
"""The same story as one outlet republished it: a byline, a headline in caps, an
em dash, a U+2010 hyphen, ragged whitespace, one added boilerplate sentence and a
feed-generator tail. This is the shape a real near-duplicate arrives in."""

UNRELATED = (
    "The city council voted on Tuesday to extend the downtown bike lane pilot for another "
    "year, citing a measurable drop in collisions and steady growth in weekday ridership "
    "since the barriers were installed last spring. Council members heard two hours of public "
    "comment before the vote, most of it supportive, and the transportation department was "
    "directed to report back with collision data every six months. Business owners along the "
    "corridor remain divided, with several restaurants reporting higher foot traffic and two "
    "hardware stores complaining about the loss of loading zones."
)


# --------------------------------------------------------------------------- #
# The rule the whole module exists to protect
# --------------------------------------------------------------------------- #


class TestNearDuplicatesAreNeverDropped:
    """`docs/signal-model.md` §4.3, enforced against the API surface itself."""

    def test_module_exposes_no_way_to_drop_a_near_duplicate(self) -> None:
        """The one-line "helper" that would destroy the trend signal.

        Deleting five of six copies of a press release removes the evidence of
        spread and collapses the `corroboration` component of `confidence`
        (§3.5) for the one copy left. Nothing downstream would raise; the counts
        would simply be wrong forever. Asserting on the exported names is the
        only check that fires *before* such a function acquires a caller.
        """
        forbidden = ("drop", "discard", "remove", "prune", "delete", "filter", "dedupe")
        offenders = [
            name
            for name in hashing.__all__
            if any(word in name.lower() for word in forbidden)
        ]
        assert not offenders, (
            f"{offenders} look like removal functions. Near-duplicates are "
            "clustered, never dropped (docs/signal-model.md §4.3)."
        )

    def test_clustering_returns_every_member_it_was_given(self) -> None:
        """A partition, in the strict sense: no member added, none lost."""
        members = [
            _member("sig_a", WIRE, minutes=0),
            _member("sig_b", SYNDICATED, minutes=5),
            _member("sig_c", UNRELATED, minutes=7),
        ]
        clusters = assign_clusters(members)

        returned = [m.signal_id for cluster in clusters for m in cluster.members]
        assert sorted(returned) == ["sig_a", "sig_b", "sig_c"]
        assert len(returned) == len(set(returned)), "a member appeared in two clusters"

    def test_a_six_platform_press_release_keeps_all_six_signals(self) -> None:
        """The worked example from §4.3, end to end.

        One story, six platforms. One canonical for retrieval; six Signals still
        in existence, because all six count toward trend volume and all six
        contribute graph edges.
        """
        platforms = ["rss", "news_api", "x", "reddit", "gdelt", "youtube"]
        members = [
            _member(f"sig_{i}", _reworded(WIRE, i), minutes=i, platform=platform)
            for i, platform in enumerate(platforms)
        ]
        clusters = assign_clusters(members)

        assert len(clusters) == 1, "the six copies should be one cluster"
        cluster = clusters[0]
        assert len(cluster.members) == 6
        assert len(cluster.duplicates()) == 5
        assert cluster.canonical_id == "sig_0"
        assert cluster.distinct_platforms() == tuple(sorted(platforms))

    def test_crossposts_from_one_platform_do_not_inflate_corroboration(self) -> None:
        """§4.3: per-platform dedup, so one source cannot corroborate itself.

        Three crossposts inside one subreddit are one platform's opinion, and the
        `corroboration` component must read them as such.
        """
        members = [
            _member(f"sig_{i}", _reworded(WIRE, i), minutes=i, platform="reddit")
            for i in range(3)
        ]
        cluster = assign_clusters(members)[0]
        assert len(cluster.members) == 3
        assert cluster.distinct_platforms() == ("reddit",)


# --------------------------------------------------------------------------- #
# Canonicalization -- layer 2's real work
# --------------------------------------------------------------------------- #


class TestCanonicalization:
    """`docs/connector-spec.md` §7: without this, one whitespace change defeats
    layer 2 and the same wire story is a new record in every feed forever."""

    @pytest.mark.parametrize(
        ("variant", "why"),
        [
            (WIRE.upper(), "case"),
            (WIRE.replace(" ", "\n\n  "), "whitespace and line breaks"),
            ("   " + WIRE + "\t\n", "leading and trailing whitespace"),
            (WIRE + " Read more", "feed-generator tail"),
            (WIRE + " [Read More…]", "bracketed tail"),
            (WIRE + " Continue reading … Read more", "stacked tails"),
            (WIRE.replace("Acme", "A\u200bcme"), "injected zero-width spaces"),
            (WIRE.replace("fi", "\ufb01"), "non-NFKC ligature"),
        ],
    )
    def test_cosmetic_differences_collapse(self, variant: str, why: str) -> None:
        assert variant != WIRE, f"the {why} fixture is identical to the original"
        assert content_sha256(variant) == content_sha256(WIRE), f"{why} survived"

    def test_campaign_parameters_are_stripped_but_content_parameters_are_not(self) -> None:
        """A false merge is more expensive than a missed one.

        `utm_*` selects nothing; `id` and `page` select the article. Stripping a
        parameter that chooses content would make two genuinely different pages
        hash alike, which no downstream layer can undo.
        """
        tracked = "See https://Example.COM/a?id=7&utm_source=nl&fbclid=z&page=2#top now"
        clean = "See https://example.com/a?id=7&page=2 now"
        assert content_sha256(tracked) == content_sha256(clean)

        different_article = "See https://example.com/a?id=8&page=2 now"
        assert content_sha256(tracked) != content_sha256(different_article)

    def test_a_malformed_url_does_not_lose_the_whole_hash(self) -> None:
        """One broken link in a 3,000-word article must not disable layer 2 for
        the source."""
        text = "before http://[oops:::/x?utm_source=a after"
        assert canonicalize(text)  # no exception, and something survives

    def test_line_separators_are_collapsed_not_deleted(self) -> None:
        """U+2028 is invisible like a zero-width space but is a *separator*.

        Deleting it would weld two words together and invent a shingle that
        appears in no other copy of the text.
        """
        assert canonicalize("alpha\u2028beta") == "alpha beta"

    def test_genuinely_different_text_does_not_collide(self) -> None:
        assert content_sha256(WIRE) != content_sha256(UNRELATED)

    def test_digest_is_a_full_sha256(self) -> None:
        digest = content_sha256(WIRE)
        assert len(digest) == 64 and int(digest, 16) >= 0


class TestDedupKeys:
    def test_content_keys_are_scoped_per_connector(self) -> None:
        """A global content key would let one connector's poll suppress another
        connector's first sighting -- destroying exactly the cross-platform
        recurrence layer 3 exists to preserve (§4.3)."""
        assert content_key("rss", WIRE) != content_key("news_api", WIRE)
        assert content_key("rss", WIRE) == content_key("rss", WIRE.upper())

    def test_identity_key_names_its_connector_and_signal(self) -> None:
        assert identity_key("rss", "sig_1") == "os:dedup:id:rss:sig_1"


# --------------------------------------------------------------------------- #
# SimHash
# --------------------------------------------------------------------------- #


class TestSimHash:
    def test_identical_text_collides_exactly(self) -> None:
        assert simhash64(WIRE) == simhash64(WIRE)
        assert hamming(simhash64(WIRE), simhash64(WIRE)) == 0

    def test_fingerprint_is_a_64_bit_unsigned_value(self) -> None:
        assert 0 <= simhash64(WIRE) < (1 << SIMHASH_BITS)

    def test_a_small_edit_stays_within_the_threshold(self) -> None:
        """The property the whole layer rests on.

        `SYNDICATED` is the same story with a byline, a caps headline, an em
        dash, ragged whitespace, one boilerplate sentence and a "Read more" tail.
        Layer 2 has already given up on it -- the cleaned texts are genuinely
        different -- and layer 3 still recognizes it. Asserting that the
        canonical forms differ is what stops this test passing for a
        reimplementation that is secretly just `sha256`.
        """
        assert canonicalize(WIRE) != canonicalize(SYNDICATED)
        assert content_sha256(WIRE) != content_sha256(SYNDICATED)

        distance = hamming(simhash64(WIRE), simhash64(SYNDICATED))
        assert distance <= SIMHASH_DISTANCE_THRESHOLD, distance
        assert is_near_duplicate(simhash64(WIRE), simhash64(SYNDICATED))

    def test_unrelated_text_does_not_collide(self) -> None:
        """Two random 64-bit values sit about 32 bits apart; unrelated prose must
        look like that, not like a near miss of the threshold."""
        distance = hamming(simhash64(WIRE), simhash64(UNRELATED))
        assert distance > 4 * SIMHASH_DISTANCE_THRESHOLD, distance
        assert not is_near_duplicate(simhash64(WIRE), simhash64(UNRELATED))

    def test_a_one_word_edit_moves_far_less_than_an_unrelated_document(self) -> None:
        """This is what separates a fingerprint from a checksum.

        `content_sha256` answers "different" identically for a one-word edit and
        for a different article. If this assertion ever fails, `simhash64` has
        become a hash of the whole string and Hamming distance over it means
        nothing.
        """
        edited = WIRE.replace("forty percent", "thirty percent")
        near = hamming(simhash64(WIRE), simhash64(edited))
        far = hamming(simhash64(WIRE), simhash64(UNRELATED))
        assert near < far / 3, (near, far)

    def test_features_are_shingles_not_whole_documents(self) -> None:
        """3-gram shingles, per `docs/connector-spec.md` §7.

        The separator check matters: joining shingles with a space would make
        ("ab", "c") and ("a", "bc") one feature and quietly inflate the
        similarity of unrelated text.
        """
        assert shingles(("a", "b", "c", "d")) == ("a\x1fb\x1fc", "b\x1fc\x1fd")
        assert shingles(("ab", "c")) != shingles(("a", "bc"))

    def test_features_are_weighted_by_frequency(self) -> None:
        """A phrase used forty times must pull harder than one used once.

        An unweighted (set-based) implementation would answer 0 here, because the
        repeated text contributes no shingle the original did not already have.
        """
        repeated = WIRE + " " + ("acme observability cloud " * 40)
        assert set(shingles(tokenize(canonicalize(WIRE)))) <= set(
            shingles(tokenize(canonicalize(repeated)))
        )
        assert hamming(simhash64(WIRE), simhash64(repeated)) > SIMHASH_DISTANCE_THRESHOLD

    def test_short_text_still_gets_a_real_fingerprint(self) -> None:
        """Text shorter than one shingle falls back to unigrams.

        Without the fallback every tweet, review title and one-line comment would
        fingerprint to the empty sentinel -- and short content is where a repost
        is most often the entire item.
        """
        assert simhash64("outage again") != 0
        assert simhash64("outage again") == simhash64("Outage  again!")
        assert simhash64("outage again") != simhash64("shipping delayed")

    def test_empty_text_yields_a_sentinel_that_never_matches(self) -> None:
        """0 sits within Hamming 3 of every fingerprint with three or fewer bits
        set. Treating it as comparable would collect every media-only post into
        one cluster and drag real items in behind them."""
        assert simhash64("") == 0
        assert simhash64("   \n\t ") == 0
        assert not is_near_duplicate(0, 0)
        assert not is_near_duplicate(0, simhash64(WIRE))

    def test_fingerprint_is_stable_across_processes_and_releases(self) -> None:
        """Pinned, deliberately.

        `docs/signal-model.md` §4.1 rule 3 feeds `simhash64` into `native_id` for
        sources with no ids, so changing this function forks identity -- a
        `schema_version` bump and a full re-ingest, not a refactor. It is also
        why the implementation may not use `hash()`, which is salted per
        interpreter by `PYTHONHASHSEED`: two workers would disagree about the
        same text and cross-process clustering would never fire, invisibly.
        """
        assert simhash64(WIRE) == 3506438261055845602


class TestHamming:
    def test_counts_differing_bits(self) -> None:
        assert hamming(0b1011, 0b1001) == 1
        assert hamming(0, 0b1111) == 4
        assert hamming(5, 5) == 0

    def test_is_symmetric(self) -> None:
        a, b = simhash64(WIRE), simhash64(UNRELATED)
        assert hamming(a, b) == hamming(b, a)

    def test_rejects_a_value_that_lost_its_sign(self) -> None:
        """A fingerprint stored in a Postgres `bigint` comes back signed.

        Masking it silently would answer with a plausible distance for the wrong
        pair, which is worse than refusing.
        """
        with pytest.raises(ValueError, match="unsigned fingerprint"):
            hamming(-1, 0)
        with pytest.raises(ValueError, match="unsigned fingerprint"):
            hamming(0, 1 << 64)

    def test_threshold_must_be_representable(self) -> None:
        with pytest.raises(ValueError, match="threshold"):
            is_near_duplicate(1, 2, threshold=65)


# --------------------------------------------------------------------------- #
# Banding -- the reason lookup is not O(n^2)
# --------------------------------------------------------------------------- #


class TestSimHashBanding:
    def test_produces_one_key_per_band(self) -> None:
        keys = simhash_band_keys(simhash64(WIRE))
        assert len(keys) == SIMHASH_LSH_BANDS
        assert len(set(keys)) == SIMHASH_LSH_BANDS
        assert all(key.startswith("os:dedup:sim:") for key in keys)

    def test_any_pair_within_the_threshold_shares_a_band(self) -> None:
        """The pigeonhole guarantee, exhaustively sampled.

        Three differing bits cannot touch all four disjoint bands, so banded
        lookup at threshold 3 has *no false negatives*. This is what makes the
        index exact rather than approximate, and it is the entire justification
        for probing four buckets instead of scanning the corpus.
        """
        rng = random.Random(20260728)
        for _ in range(2000):
            fingerprint = rng.getrandbits(SIMHASH_BITS)
            perturbed = fingerprint
            for bit in rng.sample(range(SIMHASH_BITS), rng.randint(1, 3)):
                perturbed ^= 1 << bit
            shared = set(simhash_band_keys(fingerprint)) & set(
                simhash_band_keys(perturbed)
            )
            assert shared, "a pair within Hamming 3 shared no band"

    def test_four_differing_bits_can_share_no_band(self) -> None:
        """The other half of the guarantee, and the reason `threshold < bands` is
        enforced rather than documented.

        At distance 4 the bands can miss the pair entirely. A threshold widened
        to catch the "related" 4-6 range (`docs/connector-spec.md` §7) without
        widening the banding would produce a lower duplicate rate and no error.
        """
        rng = random.Random(1)
        misses = 0
        for _ in range(2000):
            fingerprint = rng.getrandbits(SIMHASH_BITS)
            perturbed = fingerprint
            for bit in rng.sample(range(SIMHASH_BITS), 4):
                perturbed ^= 1 << bit
            if not set(simhash_band_keys(fingerprint)) & set(
                simhash_band_keys(perturbed)
            ):
                misses += 1
        assert misses > 0

    def test_bands_must_divide_the_fingerprint_evenly(self) -> None:
        with pytest.raises(ValueError, match="divide"):
            simhash_band_keys(simhash64(WIRE), bands=7)


class TestBandedIndex:
    def test_finds_candidates_without_scanning_every_member(self) -> None:
        index = BandedIndex()
        wire, syndicated, other = (
            simhash64(WIRE),
            simhash64(SYNDICATED),
            simhash64(UNRELATED),
        )
        index.add("sig_wire", simhash_band_keys(wire))
        index.add("sig_other", simhash_band_keys(other))

        assert index.candidates(simhash_band_keys(syndicated)) == {"sig_wire"}

    def test_excludes_the_member_asking(self) -> None:
        index = BandedIndex()
        fingerprint = simhash64(WIRE)
        index.add("sig_wire", simhash_band_keys(fingerprint))
        keys = simhash_band_keys(fingerprint)
        assert index.candidates(keys, exclude="sig_wire") == set()
        assert index.candidates(keys) == {"sig_wire"}

    def test_empty_index_answers_empty(self) -> None:
        assert BandedIndex().candidates(simhash_band_keys(simhash64(WIRE))) == set()


# --------------------------------------------------------------------------- #
# MinHash -- the long-document alternative
# --------------------------------------------------------------------------- #


class TestMinHash:
    def test_identical_text_estimates_perfect_similarity(self) -> None:
        assert jaccard(minhash(WIRE), minhash(WIRE)) == 1.0

    def test_a_reworded_copy_scores_high_and_an_unrelated_one_scores_low(self) -> None:
        near = jaccard(minhash(WIRE), minhash(SYNDICATED))
        far = jaccard(minhash(WIRE), minhash(UNRELATED))
        assert near > 0.8, near
        assert far < 0.1, far

    def test_signature_is_stable_across_processes(self) -> None:
        """Permutation coefficients are derived from BLAKE2b, not from a seeded
        RNG, so a signature written to a store today still means the same thing
        after a CPython upgrade."""
        assert minhash(WIRE)[0] == 2589488996860777
        assert minhash(WIRE) == minhash(WIRE)

    def test_signatures_of_different_lengths_are_not_comparable(self) -> None:
        """Silently comparing them produces a number that looks like an answer."""
        with pytest.raises(ValueError, match="lengths differ"):
            jaccard(minhash(WIRE), minhash(WIRE, permutations=64))

    def test_empty_text_is_not_similar_to_anything(self) -> None:
        assert minhash("") == ()
        assert jaccard(minhash(""), minhash("")) == 0.0
        assert jaccard(minhash(""), minhash(WIRE)) == 0.0

    def test_banding_puts_a_reworded_copy_in_a_shared_bucket(self) -> None:
        shared = set(minhash_band_keys(minhash(WIRE))) & set(
            minhash_band_keys(minhash(SYNDICATED))
        )
        assert shared
        assert not set(minhash_band_keys(minhash(WIRE))) & set(
            minhash_band_keys(minhash(UNRELATED))
        )

    def test_bands_must_divide_the_signature(self) -> None:
        with pytest.raises(ValueError, match="divide"):
            minhash_band_keys(minhash(WIRE), bands=7)


# --------------------------------------------------------------------------- #
# Canonical election
# --------------------------------------------------------------------------- #


class TestCanonicalElection:
    """§4.3: earliest `timestamp`, then highest `confidence`, then
    lexicographically smallest `id`. All three tiebreaks tested, because the
    third one is the one that looks redundant and is not."""

    def test_earliest_timestamp_wins(self) -> None:
        """The point of a cluster is to name the original -- whoever published
        before the story was syndicated."""
        members = [
            ClusterMember("sig_late", T0 + timedelta(hours=2), 1, confidence=0.99),
            ClusterMember("sig_early", T0, 1, confidence=0.10),
        ]
        assert elect_canonical(members).signal_id == "sig_early"

    def test_confidence_breaks_a_timestamp_tie(self) -> None:
        members = [
            ClusterMember("sig_a", T0, 1, confidence=0.40),
            ClusterMember("sig_b", T0, 1, confidence=0.90),
        ]
        assert elect_canonical(members).signal_id == "sig_b"

    def test_smallest_id_breaks_a_confidence_tie(self) -> None:
        """Two wire copies published in the same second with equal confidence is
        not exotic -- it is what a syndication burst looks like."""
        members = [
            ClusterMember("sig_zz", T0, 1, confidence=0.5),
            ClusterMember("sig_aa", T0, 1, confidence=0.5),
            ClusterMember("sig_mm", T0, 1, confidence=0.5),
        ]
        assert elect_canonical(members).signal_id == "sig_aa"

    def test_election_is_independent_of_input_order(self) -> None:
        """Without the id tiebreak this passes for some permutations and fails
        for others, which is the worst kind of failure to debug."""
        members = [
            ClusterMember("sig_c", T0, 1, confidence=0.5),
            ClusterMember("sig_a", T0, 1, confidence=0.5),
            ClusterMember("sig_b", T0, 1, confidence=0.5),
        ]
        elected = {
            elect_canonical(list(order)).signal_id
            for order in itertools.permutations(members)
        }
        assert elected == {"sig_a"}

    def test_an_empty_cluster_cannot_elect(self) -> None:
        with pytest.raises(ValueError, match="empty cluster"):
            elect_canonical([])

    def test_a_naive_timestamp_is_rejected_at_construction(self) -> None:
        """Naive datetimes compare happily among themselves and raise TypeError
        the moment one aware timestamp joins -- a crash in a sort, far from the
        connector that produced the bad value."""
        with pytest.raises(ValueError, match="naive"):
            ClusterMember("sig_a", datetime(2026, 7, 28, 12, 0, 0), 1)

    def test_a_signed_fingerprint_is_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError, match="unsigned"):
            ClusterMember("sig_a", T0, -1)


class TestClusterId:
    def test_is_derived_from_the_canonical_member(self) -> None:
        """Derived, not generated, so `scripts/reindex.py` rebuilds every derived
        store to the same ids with no coordination."""
        assert cluster_id_for("sig_a") == cluster_id_for("sig_a")
        assert cluster_id_for("sig_a") != cluster_id_for("sig_b")
        assert cluster_id_for("sig_a").startswith("dc_")

    def test_is_not_mistakeable_for_a_signal_id(self) -> None:
        """Embedding the Signal id verbatim would invite a join between two id
        spaces that must never be joined."""
        assert "sig_a" not in cluster_id_for("sig_a")


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #


class TestClustering:
    def test_a_signal_with_no_near_duplicate_is_a_cluster_of_one(self) -> None:
        """Singletons are clusters so callers have one code path, not a nullable
        one."""
        clusters = assign_clusters([_member("sig_a", WIRE)])
        assert len(clusters) == 1
        assert clusters[0].is_singleton
        assert clusters[0].canonical_id == "sig_a"

    def test_unrelated_documents_stay_apart(self) -> None:
        clusters = assign_clusters(
            [_member("sig_a", WIRE), _member("sig_b", UNRELATED, minutes=1)]
        )
        assert {c.canonical_id for c in clusters} == {"sig_a", "sig_b"}

    def test_clusters_are_transitive(self) -> None:
        """A near B and B near C puts all three together even when A and C are
        further apart than the threshold. Syndication chains, where each outlet
        lightly rewrites the previous one, are the reason."""
        a = 0b0
        b = 0b111  # 3 from a
        c = 0b111111  # 3 from b, 6 from a
        assert hamming(a | 1 << 63, c | 1 << 63) > SIMHASH_DISTANCE_THRESHOLD
        members = [
            ClusterMember("sig_a", T0, a | 1 << 63),
            ClusterMember("sig_b", T0 + timedelta(minutes=1), b | 1 << 63),
            ClusterMember("sig_c", T0 + timedelta(minutes=2), c | 1 << 63),
        ]
        clusters = assign_clusters(members)
        assert len(clusters) == 1
        assert len(clusters[0].members) == 3

    def test_empty_fingerprints_do_not_cluster_together(self) -> None:
        """Two media-only posts are not the same story because neither has
        text."""
        members = [
            ClusterMember("sig_a", T0, 0),
            ClusterMember("sig_b", T0 + timedelta(minutes=1), 0),
        ]
        clusters = assign_clusters(members)
        assert len(clusters) == 2
        assert all(c.is_singleton for c in clusters)

    def test_output_is_a_function_of_the_input_set_not_its_order(self) -> None:
        members = [
            _member("sig_a", WIRE, minutes=0),
            _member("sig_b", SYNDICATED, minutes=5),
            _member("sig_c", UNRELATED, minutes=7),
        ]
        shapes = {
            tuple(
                (c.cluster_id, c.canonical_id, tuple(m.signal_id for m in c.members))
                for c in assign_clusters(list(order))
            )
            for order in itertools.permutations(members)
        }
        assert len(shapes) == 1

    def test_a_repeated_signal_id_is_refused(self) -> None:
        """Merging them halves the cluster; keeping one discards a Signal. Both
        are worse than telling the caller."""
        with pytest.raises(ValueError, match="duplicate signal_id"):
            assign_clusters([_member("sig_a", WIRE), _member("sig_a", UNRELATED)])

    def test_a_threshold_the_banding_cannot_support_is_refused(self) -> None:
        """Silently returning more clusters than exist is the failure this
        prevents."""
        with pytest.raises(ValueError, match="bands"):
            assign_clusters([_member("sig_a", WIRE)], threshold=4, bands=4)

        # Widening both together is legal.
        assign_clusters([_member("sig_a", WIRE)], threshold=6, bands=8)

    def test_members_are_ordered_with_the_canonical_first(self) -> None:
        members = [
            _member("sig_b", SYNDICATED, minutes=5),
            _member("sig_a", WIRE, minutes=0),
        ]
        cluster = assign_clusters(members)[0]
        assert cluster.members[0].signal_id == cluster.canonical_id == "sig_a"

    def test_an_earlier_copy_arriving_later_re_elects_and_re_ids(self) -> None:
        """§4.3: election is re-run when a cluster gains an earlier member.

        The cluster id moves with the canonical, which is the documented cost of
        deriving it from the canonical -- and the reason `assignments()` covers
        every member rather than only the two that changed role.
        """
        first = assign_clusters([_member("sig_b", SYNDICATED, minutes=5)])[0]
        assert first.canonical_id == "sig_b"

        with_earlier = assign_clusters(
            [_member("sig_b", SYNDICATED, minutes=5), _member("sig_a", WIRE, minutes=0)]
        )[0]
        assert with_earlier.canonical_id == "sig_a"
        assert with_earlier.cluster_id != first.cluster_id
        assert {a.signal_id for a in with_earlier.assignments()} == {"sig_a", "sig_b"}


class TestClusterAssignments:
    def test_canonical_points_nowhere_and_duplicates_point_at_it(self) -> None:
        members = [
            _member("sig_a", WIRE, minutes=0),
            _member("sig_b", SYNDICATED, minutes=5),
        ]
        cluster = assign_clusters(members)[0]
        by_id = {a.signal_id: a for a in cluster.assignments()}

        assert by_id["sig_a"].duplicate_of is None
        assert by_id["sig_a"].is_canonical
        assert by_id["sig_b"].duplicate_of == "sig_a"
        assert not by_id["sig_b"].is_canonical

    def test_every_member_including_the_canonical_gets_the_cluster_id(self) -> None:
        """`models/lineage.py` rejects `duplicate_of` without `dedup_cluster_id`,
        and a canonical without one would be unfindable from its own duplicates.
        """
        members = [
            _member("sig_a", WIRE, minutes=0),
            _member("sig_b", SYNDICATED, minutes=5),
        ]
        cluster = assign_clusters(members)[0]
        assert {a.dedup_cluster_id for a in cluster.assignments()} == {
            cluster.cluster_id
        }

    def test_an_assignment_satisfies_the_lineage_validators(self) -> None:
        """The shape is only useful if `Lineage` accepts it -- proven rather than
        assumed, since the two files can drift."""
        cluster = assign_clusters(
            [_member("sig_a", WIRE, minutes=0), _member("sig_b", SYNDICATED, minutes=5)]
        )[0]
        duplicate = next(a for a in cluster.assignments() if not a.is_canonical)

        lineage = Lineage(
            pipeline_version="1.0.0",
            connector_slug="rss",
            connector_version="0.1.0",
            sync_run_id="run_1",
            fetched_at=T0,
            native_id="guid-1",
            status=SignalStatus.DUPLICATE,
            dedup_cluster_id=duplicate.dedup_cluster_id,
            duplicate_of=duplicate.duplicate_of,
        )
        assert lineage.duplicate_of == cluster.canonical_id


# --------------------------------------------------------------------------- #
# The seen-set stores
# --------------------------------------------------------------------------- #


class FakeRedis:
    """Twenty lines of Redis: `SET [EX] [NX]`, `EXISTS`, `DEL`.

    The exception message deliberately embeds credentials, so the fail-open tests
    can also prove the store never repeats it into a log.
    """

    def __init__(self, *, failing: bool = False) -> None:
        self.values: dict[str, str] = {}
        self.expiries: dict[str, int | None] = {}
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.failing = failing

    def _maybe_fail(self) -> None:
        if self.failing:
            raise ConnectionError("Error connecting to redis://default:hunter2@cache:6379")

    async def set(
        self, name: str, value: str, *, ex: int | None = None, nx: bool = False
    ) -> Any:
        self.calls.append(("set", (name, value), {"ex": ex, "nx": nx}))
        self._maybe_fail()
        if nx and name in self.values:
            return None
        self.values[name] = value
        self.expiries[name] = ex
        return True

    async def exists(self, *names: str) -> Any:
        self.calls.append(("exists", names, {}))
        self._maybe_fail()
        return sum(1 for name in names if name in self.values)

    async def delete(self, *names: str) -> Any:
        self.calls.append(("delete", names, {}))
        self._maybe_fail()
        return sum(1 for name in names if self.values.pop(name, None) is not None)


class TestRedisDedupStore:
    def test_satisfies_the_dedup_store_port(self) -> None:
        """The whole reason `SyncContext.dedup` is a Protocol: a connector gets
        a shared Redis seen-set without importing `backend/`."""
        assert isinstance(RedisDedupStore(FakeRedis()), DedupStore)

    async def test_marks_then_sees(self) -> None:
        store = RedisDedupStore(FakeRedis())
        key = identity_key("rss", "sig_a")
        assert await store.seen(key) is False
        await store.mark(key, 60)
        assert await store.seen(key) is True

    async def test_mark_sets_an_expiry_and_does_not_extend_it(self) -> None:
        """`NX` on a write nobody reads is not redundant: without it an item
        re-fetched on every poll inside the overlap window would keep its own
        seen-key alive forever and could never be legitimately re-emitted."""
        client = FakeRedis()
        store = RedisDedupStore(client)
        await store.mark("k", 60)
        assert client.expiries["k"] == 60
        assert client.calls[-1][2] == {"ex": 60, "nx": True}

    async def test_claim_is_a_single_atomic_round_trip(self) -> None:
        """`SET NX EX` is one command, so two workers racing on the same record
        cannot both be told they are first -- unlike `seen()` then `mark()`,
        which has a window between the calls in which both lose."""
        client = FakeRedis()
        store = RedisDedupStore(client)

        assert await store.claim("k", 60) is True
        assert await store.claim("k", 60) is False
        assert [call[0] for call in client.calls] == ["set", "set"]

    async def test_a_non_positive_ttl_writes_nothing(self) -> None:
        """A key with no expiry accumulates forever in a store built to be
        disposable. Fall through to PostgreSQL's unique index instead."""
        client = FakeRedis()
        store = RedisDedupStore(client)
        await store.mark("k", 0)
        assert await store.claim("k", 0) is True
        assert client.calls == []

    async def test_an_unreachable_redis_lets_records_through(self) -> None:
        """`docs/connector-spec.md` §2.5: dedup must never fail a run.

        A false "new" costs one redundant emit, absorbed by `ON CONFLICT (id) DO
        UPDATE`. A false "duplicate" drops an observation permanently -- the raw
        payload was never fetched, and posts get deleted.
        """
        store = RedisDedupStore(FakeRedis(failing=True))
        assert await store.seen("k") is False
        assert await store.claim("k", 60) is True
        assert await store.forget("k") is False
        await store.mark("k", 60)  # must not raise

    async def test_degraded_mode_logs_the_error_class_and_not_its_message(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """This logger does not pass through the kernel's redaction processor
        (`connectors/` may not import `backend/core/logging.py`), and a redis-py
        connection error is built from whatever connection string it was given.
        """
        caplog.set_level(logging.WARNING, logger="connectors.dedup.store")
        store = RedisDedupStore(FakeRedis(failing=True))
        await store.seen("os:dedup:id:rss:sig_a")

        assert "ConnectionError" in caplog.text
        assert "hunter2" not in caplog.text
        assert "redis://" not in caplog.text

    async def test_forget_removes_a_seen_key(self) -> None:
        """A repudiated run's seen-keys would otherwise suppress its own
        corrected re-ingest for a week."""
        store = RedisDedupStore(FakeRedis())
        await store.mark("k", 60)
        assert await store.forget("k") is True
        assert await store.seen("k") is False

    def test_refuses_a_default_ttl_that_never_expires(self) -> None:
        with pytest.raises(ValueError, match="default_ttl_seconds"):
            RedisDedupStore(FakeRedis(), default_ttl_seconds=0)


class TestInMemoryDedupStore:
    def test_satisfies_the_dedup_store_port(self) -> None:
        assert isinstance(InMemoryDedupStore(), DedupStore)

    async def test_marks_then_sees(self) -> None:
        store = InMemoryDedupStore()
        assert await store.seen("k") is False
        await store.mark("k", 60)
        assert await store.seen("k") is True
        assert len(store) == 1

    async def test_claim_is_true_once(self) -> None:
        store = InMemoryDedupStore()
        assert await store.claim("k", 60) is True
        assert await store.claim("k", 60) is False

    async def test_a_key_stops_suppressing_after_its_ttl(self) -> None:
        """Proven against an injected clock. A suite that slept to test a TTL
        would either take minutes or test a TTL nobody configures."""
        now = [1000.0]
        store = InMemoryDedupStore(time_source=lambda: now[0])

        await store.mark("k", 60)
        now[0] += 59
        assert await store.seen("k") is True
        now[0] += 2
        assert await store.seen("k") is False
        assert len(store) == 0

    async def test_re_marking_does_not_extend_the_window(self) -> None:
        """`SET NX` semantics, so a record re-fetched every poll cannot hold its
        own seen-key open indefinitely."""
        now = [1000.0]
        store = InMemoryDedupStore(time_source=lambda: now[0])

        await store.mark("k", 60)
        now[0] += 30
        await store.mark("k", 60)
        now[0] += 31
        assert await store.seen("k") is False

    async def test_a_non_positive_ttl_stores_nothing(self) -> None:
        store = InMemoryDedupStore()
        await store.mark("k", 0)
        assert await store.seen("k") is False
        assert await store.claim("k", 0) is True

    async def test_forget_removes_a_key(self) -> None:
        store = InMemoryDedupStore()
        await store.mark("k", 60)
        assert await store.forget("k") is True
        assert await store.forget("k") is False


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _member(
    signal_id: str,
    text: str,
    *,
    minutes: int = 0,
    confidence: float = 0.5,
    platform: str | None = None,
) -> ClusterMember:
    return ClusterMember(
        signal_id=signal_id,
        timestamp=T0 + timedelta(minutes=minutes),
        fingerprint=simhash64(text),
        confidence=confidence,
        platform=platform,
    )


def _reworded(text: str, index: int) -> str:
    """The same story as the nth outlet published it.

    Each variant differs in the ways a republication actually differs -- a
    byline, a headline case, a trailing affordance -- so the fixtures exercise
    canonicalization and fingerprinting together rather than being six copies of
    one string.
    """
    prefixes = ["", "By Jane Doe. ", "REPORTING: ", "", "Newsdesk — ", "Staff report. "]
    suffixes = ["", " Read more", "\n\n[Continue reading…]", " ", " Read the full story", ""]
    return prefixes[index % len(prefixes)] + text + suffixes[index % len(suffixes)]
