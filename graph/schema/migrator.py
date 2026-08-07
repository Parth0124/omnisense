"""Forward-only graph schema migration, recorded on a singleton `(:_SchemaVersion)`.

Neo4j has no `alembic_version` table, no `CREATE SCHEMA`, and no notion of a
migration at all. Schema versioning here is therefore a discipline the
application imposes, and a discipline that is not mechanised is a discipline that
is skipped the first time someone is in a hurry. This module is the mechanism.

**Forward only, on purpose.** There is no `downgrade()`. The PostgreSQL side has
`make downgrade` and Alembic's `down_revision`; the graph deliberately does not,
because a graph down-migration is not the inverse of an up-migration. Dropping a
constraint restores the previous schema but not the previous *data*: the rows
that the constraint prevented from existing were never written, and no
down-migration invents them. Reversing a change means writing `v002` that undoes
it, which leaves the history intact and honest.

**Three failures this refuses to paper over.**

*Applying a version that was already applied* is a no-op. Every statement in
every version is `IF NOT EXISTS`, so re-running is harmless -- but "harmless" is
not "silent". The version is reported as skipped so an operator watching a deploy
can tell "already up to date" from "applied four versions".

*Applying versions out of order* is an error, not a warning. If `v003` is
recorded as applied and `v002` then appears on disk, someone merged two branches
that each added a version, and the schema in the database is not the schema the
files describe. Applying `v002` on top would produce a state no version file
describes and no other environment shares. The only safe move is to stop and make
a human renumber.

*A checksum that no longer matches* is an error for the same reason. A version
file that has been edited after being applied means this database and every other
one now disagree about what `v001` was, and nothing detects it afterwards --
`IF NOT EXISTS` makes the re-run of an edited file quietly do nothing.

**Why one node with parallel lists.** `docs/knowledge-graph.md` §8 asks for a
singleton `(:_SchemaVersion)` recording `version`, `applied_at`, `checksum` and
`duration_ms` per version. Neo4j property values may only be primitives or arrays
of primitives -- there is no map, and no list of maps -- so per-version records on
a single node have to be parallel arrays. The invariant that all four arrays are
the same length is checked on every read: if they have drifted, the mapping from
version to checksum is guesswork, and guessing is how a mismatched checksum gets
attributed to the wrong version.

**Why each statement runs in its own transaction.** Neo4j refuses to mix schema
commands with data writes in one transaction, so a version that adds a constraint
*and* backfills a property cannot be atomic no matter how it is written. Rather
than pretend, every statement is applied independently and every statement is
idempotent, which makes a partially-applied version safe to re-run -- the property
that actually matters when a migration dies halfway.

Layer note: **L1 library** -- `models/` and the standard library only. The caller
supplies the runner; see `CypherRunner`.
"""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

from graph.schema.nodes import GraphSchemaError

__all__ = [
    "SCHEMA_VERSION_NODE_ID",
    "VERSIONS_DIR",
    "AppliedVersion",
    "CypherRunner",
    "GraphMigrator",
    "MigrationError",
    "MigrationResult",
    "VersionFile",
    "checksum_of",
    "discover_versions",
    "split_statements",
]


class MigrationError(GraphSchemaError):
    """A migration cannot proceed safely. Always fatal -- never retried."""


VERSIONS_DIR: Final[Path] = Path(__file__).parent / "versions"

SCHEMA_VERSION_NODE_ID: Final[str] = "omnisense"
"""Identity property of the singleton.

A `MERGE` needs *something* to key on, and keying on the label alone
(`MERGE (v:_SchemaVersion)`) matches any existing node of that label, which is
fine while there is one and silently picks an arbitrary one once there are two.
An explicit id makes "there should be exactly one" checkable, which
`_read_record()` does.
"""

_VERSION_FILENAME: Final[re.Pattern[str]] = re.compile(r"^v(\d{3})_([a-z0-9_]+)\.cypher$")


@runtime_checkable
class CypherRunner(Protocol):
    """How this module reaches Neo4j. Supplied by the caller, never constructed here.

    `graph/` is an L1 library and may not import `backend/db/neo4j.py`
    (`docs/architecture.md` §6.1), so there is deliberately no default. The two
    call sites wire it in one line:

        from backend.db.neo4j import run_write
        migrator = GraphMigrator(run_write)

    `run_write` runs each statement in a managed transaction, which retries the
    transient failures a leader election produces. That retry is only safe
    because every statement here is idempotent -- which is a property this module
    guarantees and the runner assumes.
    """

    async def __call__(
        self,
        query: str,
        parameters: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]: ...


# --------------------------------------------------------------------------- #
# Reading version files
# --------------------------------------------------------------------------- #


def split_statements(text: str) -> tuple[str, ...]:
    """Split a `.cypher` file into individual statements.

    `str.split(";")` is wrong and the ways it is wrong are not theoretical. A
    Cypher file contains `//` line comments (the header of every version file),
    and a data migration contains string literals -- `SET n.note = 'a; b'` --
    where a semicolon is data. Splitting naively yields two fragments, neither of
    which parses, and the error the driver returns points at a syntax problem
    rather than at the splitter.

    So this is a small scanner rather than a split: it tracks single-quoted,
    double-quoted and backtick-quoted spans (backticks quote identifiers with
    awkward characters, and a semicolon inside one is legal), honours backslash
    escapes inside quotes, and skips `//` line comments and `/* … */` blocks.
    Comments are dropped from the returned statements, and empty statements --
    a trailing semicolon, a comment-only file -- are omitted rather than being
    sent to the server as an empty query.
    """
    statements: list[str] = []
    current: list[str] = []
    quote: str | None = None
    i = 0
    length = len(text)

    while i < length:
        char = text[i]

        if quote is not None:
            current.append(char)
            if char == "\\" and i + 1 < length:
                # Consume the escaped character wholesale so an escaped quote
                # does not close the span.
                current.append(text[i + 1])
                i += 2
                continue
            if char == quote:
                quote = None
            i += 1
            continue

        if char in "'\"`":
            quote = char
            current.append(char)
            i += 1
            continue

        if text.startswith("//", i):
            end = text.find("\n", i)
            i = length if end == -1 else end
            continue

        if text.startswith("/*", i):
            end = text.find("*/", i + 2)
            if end == -1:
                raise MigrationError(
                    "unterminated /* block comment; the rest of the file would be "
                    "silently discarded"
                )
            i = end + 2
            continue

        if char == ";":
            statement = "".join(current).strip()
            if statement:
                statements.append(statement)
            current = []
            i += 1
            continue

        current.append(char)
        i += 1

    if quote is not None:
        raise MigrationError(
            f"unterminated {quote!r} literal; the file cannot be split reliably"
        )

    trailing = "".join(current).strip()
    if trailing:
        # A final statement with no terminating semicolon. Accepted -- the file is
        # unambiguous -- but it is the shape that makes a truncated file look
        # valid, so it is worth knowing this branch exists.
        statements.append(trailing)

    return tuple(statements)


def checksum_of(text: str) -> str:
    """Content hash of a version file.

    Over the *raw file text*, comments and whitespace included, not over the
    parsed statements. Reformatting an applied version is exactly as forbidden as
    changing it: two environments whose `v001` files differ by so much as a
    comment are two environments where nobody can say which one was applied
    where. Normalising before hashing would make the check quietly tolerant of
    the edits it exists to catch.

    Line endings are normalised, because a checkout on Windows would otherwise
    report every version as modified.
    """
    normalised = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class VersionFile:
    """One `vNNN_slug.cypher` on disk."""

    version: int
    slug: str
    path: Path
    text: str

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def checksum(self) -> str:
        return checksum_of(self.text)

    @property
    def statements(self) -> tuple[str, ...]:
        return split_statements(self.text)


@dataclass(frozen=True, slots=True)
class AppliedVersion:
    """One version as recorded in the graph."""

    version: int
    checksum: str
    applied_at: datetime | None
    duration_ms: int


@dataclass(frozen=True, slots=True)
class MigrationResult:
    """What a run did. Both tuples are in application order."""

    applied: tuple[int, ...]
    skipped: tuple[int, ...]

    @property
    def changed(self) -> bool:
        return bool(self.applied)


def discover_versions(directory: Path | None = None) -> tuple[VersionFile, ...]:
    """Read every version file, in numeric order.

    Numeric and not lexical. They agree while the numbers are zero-padded to
    three digits, and stop agreeing at `v010` versus `v009` the moment someone
    writes `v10_`. Sorting by the parsed integer means the ordering does not
    depend on a naming convention holding forever.

    A filename that does not match `vNNN_slug.cypher` raises rather than being
    ignored. A migration silently skipped because it was named `v2_thing.cypher`
    is worse than a failed deploy: the deploy succeeds and the schema is wrong.
    """
    root = directory if directory is not None else VERSIONS_DIR
    if not root.is_dir():
        raise MigrationError(f"version directory {root} does not exist")

    files: list[VersionFile] = []
    seen: dict[int, str] = {}
    for path in sorted(root.iterdir()):
        if path.name.startswith(".") or not path.is_file():
            continue
        if path.suffix != ".cypher":
            continue
        match = _VERSION_FILENAME.match(path.name)
        if match is None:
            raise MigrationError(
                f"{path.name} is not a valid version filename. Versions are "
                "vNNN_slug.cypher with a three-digit number and a lowercase "
                "slug; a file that does not match would be skipped silently."
            )
        version = int(match.group(1))
        if version in seen:
            raise MigrationError(
                f"two files claim version {version:03d}: {seen[version]} and "
                f"{path.name}. Which one was applied is unknowable."
            )
        seen[version] = path.name
        files.append(
            VersionFile(
                version=version,
                slug=match.group(2),
                path=path,
                text=path.read_text(encoding="utf-8"),
            )
        )

    files.sort(key=lambda f: f.version)
    return tuple(files)


# --------------------------------------------------------------------------- #
# The migrator
# --------------------------------------------------------------------------- #

_READ_RECORD = """
MATCH (v:_SchemaVersion)
RETURN v.id AS id,
       coalesce(v.versions, [])     AS versions,
       coalesce(v.checksums, [])    AS checksums,
       coalesce(v.applied_at, [])   AS applied_at,
       coalesce(v.durations_ms, []) AS durations_ms
"""

# Recording is itself idempotent. Without the `is_new` guard, a managed
# transaction that commits and then loses its acknowledgement would be retried by
# the driver and append the same version a second time, leaving the parallel
# arrays describing a history that never happened.
_RECORD_VERSION = """
MERGE (v:_SchemaVersion {id: $node_id})
ON CREATE SET v.versions = [],
              v.checksums = [],
              v.applied_at = [],
              v.durations_ms = [],
              v.created_at = datetime()
WITH v, NOT $version IN coalesce(v.versions, []) AS is_new
SET v.versions     = CASE WHEN is_new THEN coalesce(v.versions, []) + [$version]
                          ELSE v.versions END,
    v.checksums    = CASE WHEN is_new THEN coalesce(v.checksums, []) + [$checksum]
                          ELSE v.checksums END,
    v.applied_at   = CASE WHEN is_new THEN coalesce(v.applied_at, []) + [$applied_at]
                          ELSE v.applied_at END,
    v.durations_ms = CASE WHEN is_new THEN coalesce(v.durations_ms, []) + [$duration_ms]
                          ELSE v.durations_ms END,
    v.current_version = $version,
    v.updated_at = datetime()
RETURN v.current_version AS current_version
"""


class GraphMigrator:
    """Applies pending version files and records what it applied.

    Not a module-level function because a run has state worth naming -- the
    runner, the directory -- and because tests need to point it at a temporary
    directory without monkeypatching a module global.
    """

    def __init__(
        self,
        runner: CypherRunner,
        *,
        versions_dir: Path | None = None,
        node_id: str = SCHEMA_VERSION_NODE_ID,
    ) -> None:
        self._run = runner
        self._versions_dir = versions_dir
        self._node_id = node_id

    # ---------------------------------------------------------------- state --

    def discover(self) -> tuple[VersionFile, ...]:
        """Version files on disk, in numeric order."""
        return discover_versions(self._versions_dir)

    async def applied_versions(self) -> tuple[AppliedVersion, ...]:
        """Read the singleton, in the order the versions were applied.

        Returns empty against a graph that has never been migrated. That is not
        an error and must not be treated as one: a fresh database is the normal
        case on the first deploy and in every integration test.
        """
        rows = await self._run(_READ_RECORD, {})
        if not rows:
            return ()
        if len(rows) > 1:
            ids = sorted(str(row.get("id")) for row in rows)
            raise MigrationError(
                f"found {len(rows)} (:_SchemaVersion) nodes ({', '.join(ids)}); "
                "there must be exactly one, and with several there is no way to "
                "know which records the truth. Resolve by hand before migrating."
            )
        return self._parse_record(rows[0])

    def _parse_record(self, row: Mapping[str, Any]) -> tuple[AppliedVersion, ...]:
        versions = list(row.get("versions") or [])
        checksums = list(row.get("checksums") or [])
        applied_at = list(row.get("applied_at") or [])
        durations = list(row.get("durations_ms") or [])

        lengths = {
            "versions": len(versions),
            "checksums": len(checksums),
            "applied_at": len(applied_at),
            "durations_ms": len(durations),
        }
        if len(set(lengths.values())) != 1:
            raise MigrationError(
                "(:_SchemaVersion) parallel arrays have drifted "
                f"({lengths}); version-to-checksum mapping is no longer "
                "recoverable, so a checksum check here would be a guess"
            )

        return tuple(
            AppliedVersion(
                version=int(version),
                checksum=str(checksum),
                applied_at=_as_datetime(when),
                duration_ms=int(duration),
            )
            for version, checksum, when, duration in zip(
                versions, checksums, applied_at, durations, strict=True
            )
        )

    async def pending(self) -> tuple[VersionFile, ...]:
        """Version files not yet applied, in numeric order.

        Raises rather than returning a filtered list when the two sides disagree:
        an out-of-order version or a changed checksum means the database and the
        repository describe different schemas, and there is no subset of files
        whose application reconciles them.
        """
        on_disk = self.discover()
        applied = await self.applied_versions()
        self._check_consistency(on_disk, applied)
        applied_numbers = {record.version for record in applied}
        return tuple(f for f in on_disk if f.version not in applied_numbers)

    def _check_consistency(
        self,
        on_disk: Sequence[VersionFile],
        applied: Sequence[AppliedVersion],
    ) -> None:
        by_version = {f.version: f for f in on_disk}
        applied_numbers = {record.version for record in applied}

        for record in applied:
            version_file = by_version.get(record.version)
            if version_file is None:
                raise MigrationError(
                    f"v{record.version:03d} is recorded as applied but no file "
                    "for it exists. Either the file was deleted or this database "
                    "was migrated by a different branch."
                )
            if version_file.checksum != record.checksum:
                raise MigrationError(
                    f"{version_file.name} has changed since it was applied "
                    f"(recorded {record.checksum[:12]}…, on disk "
                    f"{version_file.checksum[:12]}…). An applied version is "
                    "never edited in place -- add a new version instead. "
                    "Re-running the edited file would do nothing, because every "
                    "statement is IF NOT EXISTS, and the environments would stay "
                    "silently divergent."
                )

        if not applied_numbers:
            return
        highest_applied = max(applied_numbers)
        stragglers = [
            f.name
            for f in on_disk
            if f.version < highest_applied and f.version not in applied_numbers
        ]
        if stragglers:
            raise MigrationError(
                f"{', '.join(stragglers)} would be applied after v{highest_applied:03d}, "
                "which is already applied. Migrations are forward-only and "
                "ordered; applying an older version now produces a schema no "
                "version file describes and no other environment shares. "
                "Renumber the straggler above the highest applied version."
            )

    # ---------------------------------------------------------------- apply --

    async def apply_all(self) -> MigrationResult:
        """Apply every pending version, in order. Returns what was done.

        Stops at the first failure rather than continuing. Versions are ordered
        because they depend on each other; applying `v003` after `v002` failed
        means running statements against a schema they were never written for.
        """
        on_disk = self.discover()
        applied = await self.applied_versions()
        self._check_consistency(on_disk, applied)
        already = {record.version for record in applied}

        applied_now: list[int] = []
        skipped: list[int] = []
        for version_file in on_disk:
            if version_file.version in already:
                skipped.append(version_file.version)
                continue
            await self._apply_one(version_file)
            applied_now.append(version_file.version)

        return MigrationResult(applied=tuple(applied_now), skipped=tuple(skipped))

    async def apply(self, version_file: VersionFile) -> bool:
        """Apply one version. Returns False when it was already applied.

        The already-applied check is a read of the recorded history, not a guess
        from the state of the schema. "Does this constraint exist" cannot answer
        "was this version applied": a constraint may exist because a *later*
        version created it, or because someone created it by hand.
        """
        applied = await self.applied_versions()
        self._check_consistency(self.discover(), applied)
        if any(record.version == version_file.version for record in applied):
            return False
        await self._apply_one(version_file)
        return True

    async def _apply_one(self, version_file: VersionFile) -> None:
        statements = version_file.statements
        if not statements:
            raise MigrationError(
                f"{version_file.name} contains no statements. An empty version is "
                "almost always a file whose contents were never written, and "
                "recording it as applied would hide that permanently."
            )

        started = time.monotonic()
        for index, statement in enumerate(statements, start=1):
            try:
                await self._run(statement, {})
            except MigrationError:
                raise
            except Exception as exc:
                # Naming the statement matters more here than anywhere else in
                # the codebase: a migration failure is read by whoever is holding
                # a broken deploy, and "statement 7 of 14" turns a stack trace
                # into a place to look. The version is deliberately *not*
                # recorded, so the next run retries from the beginning -- safe,
                # because every statement is idempotent.
                raise MigrationError(
                    f"{version_file.name} failed at statement {index} of "
                    f"{len(statements)}: {statement.splitlines()[0][:120]}"
                ) from exc

        duration_ms = int((time.monotonic() - started) * 1000)
        await self._run(
            _RECORD_VERSION,
            {
                "node_id": self._node_id,
                "version": version_file.version,
                "checksum": version_file.checksum,
                # Client clock rather than the server's `datetime()`, so the
                # recorded instant matches the duration measured on this side.
                # The two would otherwise disagree by the clock skew between
                # application and database, which is small and confusing.
                "applied_at": datetime.now(UTC),
                "duration_ms": duration_ms,
            },
        )


def _as_datetime(value: Any) -> datetime | None:
    """Coerce whatever the driver returned for a datetime into a `datetime`.

    The Neo4j driver returns `neo4j.time.DateTime`, which is not a
    `datetime.datetime` and only converts through `.to_native()`. Importing the
    driver here to name that type would drag `neo4j` into an L1 library for one
    isinstance check, so this duck-types instead: anything with `to_native()` is
    converted, a real `datetime` passes through, and anything else -- including
    the string a fake driver in a test might hand back -- becomes `None` rather
    than propagating a type nobody downstream expects.
    """
    if isinstance(value, datetime):
        return value
    to_native = getattr(value, "to_native", None)
    if callable(to_native):
        native = to_native()
        return native if isinstance(native, datetime) else None
    return None
