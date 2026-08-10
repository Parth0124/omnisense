"""Stage 2 -- Normalize: the payload plus the cleaned body become a `Signal`.

`docs/signal-model.md` §5.1 gives this stage the contract *cleaned text + raw
payload -> `Signal` with fields 1-8, 15 and 17 populated*, deterministic, **fatal**
on failure. Fatal for the same reason Clean is: nothing downstream can run
without a Signal. `EnrichmentContext.require_signal()` is the wall every later
stage hits, and until this stage assigns `ctx.signal` the pipeline can produce
nothing at all.

Why the pipeline *rebuilds* the Signal instead of receiving the connector's
-----------------------------------------------------------------------------
The connector already mapped this payload once, in `BaseConnector.normalize()`,
and threw the result away. That looks wasteful and is not.

`docs/data-stores.md` §5.1 puts the immutable raw payload in R2 and publishes
only its *address* on `omnisense.records.raw`. Read `RawRecordEvent`
(`services/events/schemas.py`) and notice what it does not carry: no `Signal`,
no `content`, no mapped fields -- just `raw_object_key`, `raw_sha256` and the
provenance needed to rebuild `Lineage`. That is deliberate (§5.1 step 2: "the
payload never travels on the bus; the reference does"), and it is what makes
`Content.raw_ref`'s promise true -- *"a cleaning bug is repairable by
reprocessing rather than re-fetching"*. Re-fetching is lossy: posts get deleted,
API windows expire, and a publisher's CMS rewrites the article under the same
URL. Reprocessing is not, because the bytes never changed.

So the mapping has to be re-runnable from the payload alone, by a process that
has no connector instance, no credentials and no cursor. That is exactly what
`connectors/normalize/mapper.py` was built for: `FieldMap` is *data*, and
`MappingContext` exists so the mapper "is then usable from `workers/dlq.py`
replaying a fixed field map against a historical payload, with no connector
instance in sight". This stage is the second such caller.

The connector-side mapping is not redundant either -- it is what dedup and the
cursor are computed from before anything is durable. The two runs agree because
they execute the *same* `FieldMap` object, not two copies of one.

Which map, and why it is injected
---------------------------------
The mapping is connector-specific and the payload alone does not identify its
connector -- a bare `{"data": {...}}` could be four different sources. So the
stage takes a `FieldMapResolver`, `(connector_slug, payload) -> FieldMap | None`,
keyed on the slug the event carries. It is a *function* rather than a dict
because one slug can need more than one map, and the choice is a property of the
payload: GitHub maps issues, discussions and releases differently -- three
different APIs' worth of field names for the same six concepts, chosen by the
stream the connector recorded in its envelope. That decision lives in the
connector that owns it; this module only asks.

An unmapped slug raises `NormalizationError` naming it. The tempting
alternative -- fall back to a generic map, emit what can be found -- produces a
Signal with a plausible id, an empty body and no author, which is indexed,
retrieved and eventually quoted in a report as evidence for a claim it does not
contain. A loud failure puts one record in the DLQ with the slug in the message.

Two properties this stage enforces beyond mapping
-------------------------------------------------
**Identity may not move.** `RawRecordEvent.native_id` already keyed the R2
object and the Kafka partition. If the rebuilt Signal derives a different one,
the same item now exists under two identities in five stores, and no reconciler
can tell which is real. The disagreement is caught here rather than discovered
as duplicate rows months later -- the same check `connectors/enterprise/github.py`
makes on its own side, against the same `node_id`.

**`pipeline_version` becomes real here.** Connectors stamp
`UNENRICHED_PIPELINE_VERSION` (`"0.0.0"`) because they have run no enrichment
and cannot know the version of a pipeline they never touch. `docs/signal-model.md`
§7 makes that field the basis for deciding whether a stored Signal needs
reprocessing, and `services/signal_engine/store.py` makes it the upsert guard, so
leaving `0.0.0` on a fully enriched Signal would make every write lose to every
other write forever. Stage 2 stamps `ctx.pipeline_version`, which
`SignalPipeline.run()` set from the pipeline that is actually executing.

Known limitation: post-map connector fix-ups are not reproduced
---------------------------------------------------------------
A connector may do work in `normalize()` *outside* its `FieldMap`, and a
declarative rebuild cannot see any of it. GitHub's is the live example: it drops
a draft release, which has a `node_id` and a body and maps perfectly well but has
not happened yet. Nothing here knows that, so a reprocessed draft would become a
Signal announcing an unpublished release.

In practice no draft reaches this stage, and the reason is worth being precise
about because it is incidental rather than designed: `_fetch_releases` sorts by
`published_at` and drops anything without a parseable one, so a draft is never
archived and there is no raw record to reprocess. The explicit `draft is True`
check in the connector's own `normalize()` is a second belt on the same
trousers. Neither is a guarantee this stage holds -- a stream added later that
archives an unpublished object would rebuild it here with nothing to stop it.

The fix belongs in `FieldMap` (a declarative drop predicate) rather than in a
per-connector branch here, which `models/signal.py` forbids on principle: no
platform-shaped code above `connectors/`.

Layer note: `services/` (L2), which `docs/architecture.md` §6.1 permits to import
`connectors/`. The import runs the other way round from the connector's own
mapping and never the reverse -- `connectors/` may not import `services/`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import lru_cache
from typing import Any, Final

from connectors.exceptions import NormalizationError
from connectors.normalize.mapper import FieldMap, MappingContext
from connectors.protocol import RawRecord
from models.enums import StageName
from models.signal import Content, Signal, signal_id
from services.events.schemas import RawRecordEvent
from services.signal_engine.pipeline import EnrichmentContext

__all__ = [
    "NORMALIZE_STAGE_VERSION",
    "FieldMapResolver",
    "NormalizeStage",
    "default_field_map_resolver",
]

NORMALIZE_STAGE_VERSION: Final = "1.0.0"
"""Version of this stage's implementation, recorded in `lineage.stages[]`.

Bumping it is not cosmetic: stage 2 is one of the deterministic stages, so its
version is part of the answer to "would reprocessing this Signal change it?"
(`docs/signal-model.md` §5.1).
"""


FieldMapResolver = Callable[[str, Mapping[str, Any]], FieldMap | None]
"""`(connector_slug, payload) -> FieldMap | None`.

`None` means "this slug is not mapped here", which the stage turns into a
`NormalizationError`. It is not an error return in the resolver's own right --
a resolver that raised would deny `NormalizeStage` the chance to attach the
`native_id`, and a DLQ record nobody can attribute is one nobody can replay
(`docs/connector-spec.md` §6).
"""


# --------------------------------------------------------------------------- #
# The default resolver
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def _shipped_selectors() -> Mapping[str, Callable[[Mapping[str, Any]], FieldMap | None]]:
    """Slug -> "pick this payload's map", for every connector that maps by table.

    Only GitHub is here, and the shortness of the table is the honest answer
    rather than a gap. A `FieldMap` is worth building when one connector emits
    several payload shapes that differ only in where the same fields live -- an
    issue, a discussion and a release are three such shapes. The other shipped
    connectors (Jira, Slack, Confluence, Notion, arXiv, Semantic Scholar, Papers
    with Code) each emit one shape and construct their `Signal` directly in
    `normalize()`, so there is no map to rebuild and nothing for this resolver to
    return. `None` for those slugs is correct: it means "this connector does not
    normalize by field map", not "this connector is unknown".

    Built lazily and cached rather than at module import so that
    `import services.signal_engine.normalize` stays free of `httpx` and every
    other connector dependency -- a deployment that assembles the pipeline with
    its own resolver should not pay for connectors it will never run. It also
    keeps the import cycle shallow: `services/` importing connector modules at
    definition time makes any future connector that reaches for a `services/`
    helper an import-time crash instead of a review comment.

    **`_FIELD_MAPS` is a private name in its own module and is imported anyway.**
    The alternative is a second copy of the maps living in `services/`, and a
    `FieldMap` decides `truncated`, the metadata namespace and -- through the
    body it produces -- rule 3 of `native_id`. Two copies that drift by one path
    give the same item two identities depending on which side rebuilt it, which
    is precisely the failure this stage exists to prevent. One shared definition,
    imported across a boundary the architecture matrix already permits, is the
    lesser evil; the maps are `Final` and never rebound.
    """
    from connectors.enterprise.github import _FIELD_MAPS, ENVELOPE_KEY, GitHubConnector

    def github(payload: Mapping[str, Any]) -> FieldMap | None:
        # The stream lives in the envelope the connector attached at fetch time,
        # not in the GitHub object -- an issue and a discussion are structurally
        # similar enough that guessing from the payload would misclassify one of
        # them. An unmapped stream resolves to `None` and fails loudly, because a
        # stream this table has never seen is a payload shape nobody mapped.
        envelope = payload.get(ENVELOPE_KEY)
        if not isinstance(envelope, Mapping):
            return None
        stream = envelope.get("stream")
        return _FIELD_MAPS.get(stream) if isinstance(stream, str) else None

    return {GitHubConnector.slug: github}


def default_field_map_resolver(
    connector_slug: str, payload: Mapping[str, Any]
) -> FieldMap | None:
    """The shipped `FieldMapResolver`.

    Keyed on `connector_slug` rather than on `platform` because the slug is what
    `RawRecordEvent` carries and what `Lineage.connector_slug` records, and
    because two connectors can legitimately target one platform (a public poller
    and an authenticated partner feed are different code with different payload
    shapes and the same `Platform`).

    Returns `None` for anything else, including a slug whose connector exists but
    whose payload shape this resolver cannot classify. `NormalizeStage` turns
    that into the error.
    """
    select = _shipped_selectors().get(connector_slug)
    if select is None:
        return None
    return select(payload)


# --------------------------------------------------------------------------- #
# The stage
# --------------------------------------------------------------------------- #


class NormalizeStage:
    """Stage 2. Satisfies `Stage`; **fatal** per `FATAL_STAGES`.

    Holds a resolver and nothing per-record, so one instance is shared by a
    worker and driven concurrently -- everything about one Signal lives on the
    context or on the stack.

    Like every other stage it never decides that its own failure is fatal. It
    raises; `SignalPipeline` consults `FATAL_STAGES`. That is what stops a stage
    from promoting itself and taking ingestion down with it.
    """

    name: StageName = StageName.NORMALIZE
    version: str = NORMALIZE_STAGE_VERSION

    def __init__(
        self,
        resolver: FieldMapResolver | None = None,
        *,
        url_resolver: Callable[[str], str | None] | None = None,
    ) -> None:
        """Take the resolver; never guess one.

        `url_resolver` is threaded straight through to `FieldMap.to_signal` and
        is normally `None`. It exists because rule 2 of the identity ladder
        hashes the *canonicalized* URL, and a deployment that resolves shortener
        redirects at ingest must resolve them identically here or every
        shortened link forks its Signal on reprocessing. It is a hook, not a
        network call this stage makes on its own.
        """
        self._resolve = resolver or default_field_map_resolver
        self._url_resolver = url_resolver

    @property
    def model_id(self) -> str | None:
        """Always `None`: stage 2 is deterministic and calls no model.

        `docs/signal-model.md` §5.1 records a model id only for stages 4-6, whose
        output cannot be reproduced without knowing which model produced it.
        Naming one here would imply this stage needs a model to replay, which is
        the property that makes reprocessing cheap.
        """
        return None

    async def apply(self, ctx: EnrichmentContext) -> None:
        """Build the Signal and assign it to `ctx.signal`.

        Order matters: the map is chosen, the Signal is mapped, identity is
        checked against the event, and only then is the body swapped and the R2
        address attached. Checking identity before touching the body means the
        check is made against exactly what the mapper derived, which is the value
        every other consumer of this payload derives too.
        """
        record = ctx.record
        if record is None:
            raise NormalizationError(
                "no RawRecordEvent on the context: stage 2 rebuilds the Signal from "
                "the connector payload plus the event's provenance, and cannot "
                "invent a connector slug, a sync run or a fetch time. The "
                "enrichment worker must set EnrichmentContext.record."
            )

        if not ctx.payload:
            # Distinguished from an unmapped slug because the remedies differ: a
            # missing payload is a worker that did not read R2 (or a deferred
            # PUT, `docs/architecture.md` §7.3), while an unmapped slug is code
            # that was never written.
            raise NormalizationError(
                f"no payload to map for connector {record.connector_slug!r}; the raw "
                f"object at {record.raw_object_key!r} was never read onto the context",
                native_id=record.native_id,
                connector=record.connector_slug,
            )

        field_map = self._resolve(record.connector_slug, ctx.payload)
        if field_map is None:
            raise NormalizationError(
                f"no FieldMap for connector {record.connector_slug!r} and this payload "
                "shape; an unmapped connector must fail loudly rather than emit a "
                "half-built Signal with an empty body that retrieval would still "
                "return and a report would still quote",
                native_id=record.native_id,
                connector=record.connector_slug,
                details={"connector_slug": record.connector_slug},
            )

        signal = field_map.to_signal(
            self._raw_record(ctx, record),
            self._mapping_context(ctx, record),
            url_resolver=self._url_resolver,
        )

        self._check_identity(signal, record)
        signal.content = self._content(ctx, signal, record)
        self._attach_raw_address(signal, record)

        ctx.signal = signal

    # ------------------------------------------------------------ internals --

    @staticmethod
    def _raw_record(ctx: EnrichmentContext, record: RawRecordEvent) -> RawRecord:
        """Reassemble the `RawRecord` the mapper expects.

        `RawRecord` is the connector-side value type and this stage has no
        connector, so the fields are taken from the event -- which was populated
        from exactly this object one hop earlier, and whose field names mirror
        `Lineage` for that reason.

        `content_type` is the *record's* declared type from the event, not
        `ctx.content_type`. They are normally the same; when they are not it is
        because a worker overrode the type to steer stage 1, and `Lineage` must
        keep recording what the provider actually returned.
        """
        return RawRecord(
            native_id=record.native_id,
            payload=ctx.payload,
            fetched_at=record.fetched_at,
            raw_bytes=ctx.raw_bytes,
            content_type=record.raw_content_type,
            source_url=record.source_url,
            request_fingerprint=record.request_fingerprint,
        )

    @staticmethod
    def _mapping_context(ctx: EnrichmentContext, record: RawRecordEvent) -> MappingContext:
        """Provenance for `build_lineage`, with the *real* pipeline version.

        `MappingContext` defaults `pipeline_version` to `"0.0.0"` for connectors,
        which have run no enrichment. This is the call site where that becomes
        the version of the pipeline actually executing -- see the module
        docstring for why leaving the zero version would break the store guard.
        """
        return MappingContext(
            connector_slug=record.connector_slug,
            connector_version=record.connector_version,
            sync_run_id=record.sync_run_id,
            pipeline_version=ctx.pipeline_version,
        )

    @staticmethod
    def _check_identity(signal: Signal, record: RawRecordEvent) -> None:
        """Refuse a rebuild that renames the record.

        `RawRecordEvent.native_id` already keyed the R2 object and the Kafka
        partition; `Signal.id` is derived from `(platform, native_id)` and keys
        all five stores. If the rebuild disagrees, the item exists twice and
        nothing downstream can tell which row is the real one -- so this fails
        the record into the DLQ, where the payload and both ids are visible,
        rather than committing the second identity.

        Both halves are checked at once by comparing the derived id: a map
        declared for the wrong `Platform` fails here too, and a platform
        mismatch is otherwise invisible because the native_id would match.
        """
        expected = signal_id(record.platform, record.native_id)
        if signal.id != expected:
            raise NormalizationError(
                "the rebuilt Signal does not match the record it came from: mapped "
                f"({signal.platform.value}, {signal.lineage.native_id!r}) but the raw "
                f"record event says ({record.platform.value}, {record.native_id!r}). "
                "Committing this would give one item two identities across five stores",
                native_id=record.native_id,
                connector=record.connector_slug,
                details={"mapped_signal_id": signal.id, "expected_signal_id": expected},
            )

    @staticmethod
    def _content(ctx: EnrichmentContext, signal: Signal, record: RawRecordEvent) -> Content:
        """The final `Content`: stage 1's body if it produced one, else the map's.

        Stage 1's output wins when it is non-blank, and the reason is not that it
        is better prose. It is the only body that has been through the
        deployment's `Redactor` (`services/signal_engine/cleaning.py`,
        `docs/security-and-privacy.md` §6.1). Preferring the payload-mapped body
        would silently route un-redacted text into `content.text`, and from there
        into the embedding, where PII is "neither readable nor auditable nor
        deletable".

        `""` is not a body. `CleaningStage` writes exactly that for a structured
        record -- "cleaned, and there is no body", because a provider JSON
        payload has no document to extract and the observation's text sits at a
        path only the field map knows. Treating that `""` as authoritative would
        empty `content.text` for every JSON source in the system, which is every
        source that reaches this stage today.

        Known limitation: a connector whose archived bytes are a *container*
        document -- one RSS feed carrying twenty entries -- must not present them
        as this record's own document, or stage 1 cleans the container and this
        method adopts the whole feed as one entry's body. `docs/data-stores.md`
        §5.1 archives the per-record payload as JSON (`{payload_sha256}.json`),
        which is the shape that makes this correct; a worker that substitutes the
        transport document is the bug.

        `Content` is rebuilt rather than assigned into because `char_count` is
        derived by a `mode="before"` model validator, which a field assignment
        does not re-run -- a 4000-character body would keep the excerpt's count
        and every consumer that reads it would be wrong.
        """
        cleaned = ctx.cleaned_text
        body = cleaned if cleaned and cleaned.strip() else signal.content.text
        return Content(
            title=signal.content.title,
            text=body,
            truncated=signal.content.truncated,
            content_type=signal.content.content_type,
            # The R2 key restated for readers that hold only a `Content`. The
            # connector could not fill this in -- it does not perform the PUT
            # (`docs/connector-spec.md` §2.6) -- and this stage is the first
            # component that knows the address.
            raw_ref=record.raw_object_key,
            raw_sha256=record.raw_sha256 or signal.content.raw_sha256,
        )

    @staticmethod
    def _attach_raw_address(signal: Signal, record: RawRecordEvent) -> None:
        """Point `lineage` at the archived original.

        `build_lineage` deliberately leaves `raw_object_key` `None`: the
        connector does not perform the PUT, and "a lineage pointer to an object
        that was never written is worse than a null one -- it turns a failed
        upload into a 404 at citation time instead of a visible gap". By the time
        this stage runs the PUT has happened (§5.1 step 1 precedes step 2), so
        the address is known and is attached.

        The digest and size come from the event in preference to a recount over
        `ctx.raw_bytes`: the event's values were recorded at the PUT and are what
        the content-addressed key was derived from, and they survive a worker
        that enriched from the parsed payload without re-reading the object.
        """
        lineage = signal.lineage
        lineage.raw_object_key = record.raw_object_key
        if record.raw_sha256 is not None:
            lineage.raw_sha256 = record.raw_sha256
        if record.raw_bytes is not None:
            lineage.raw_bytes = record.raw_bytes
        lineage.raw_content_type = record.raw_content_type
