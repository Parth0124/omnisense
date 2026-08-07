"""Stamping the index-state columns, and reading the backlog they expose.

`docs/data-stores.md` §5.1 step 8 and §6 make `signals.indexed_vector_at`,
`signals.indexed_keyword_at` and `signals.graphed_at` the reconciliation
mechanism for a system that writes five stores without a distributed
transaction. The columns are not telemetry. They are the *only* record of what
PostgreSQL believes exists elsewhere, and every self-healing property the
pipeline claims rests on two rules that this module exists to enforce in one
place rather than in three workers:

**Stamp after the derived write, never before.** A stamp is a claim that a
vector, a document or a subgraph exists. Written first, a crash between the
stamp and the upsert leaves a row that says "indexed" and a store that holds
nothing -- and because every reconciler looks for `NULL`, nothing will ever
revisit it. The Signal is then permanently unsearchable and permanently
invisible to the job whose entire purpose is to find that condition. Written
after, the same crash leaves `NULL`, the sweeper finds it, the idempotent
derived write is redone, and the system converges. The asymmetry is total: one
ordering self-heals, the other corrupts silently.

**Do not touch `updated_at`.** The reconciler's predicate is
`indexed_vector_at IS NULL AND updated_at < now() - interval '15 minutes'`, so
`updated_at` is the staleness clock for *every* derived store. A stamp that
bumped it would push the row's other still-`NULL` columns back out of the
window, and the graph backlog would be delayed by fifteen minutes every time the
indexing worker touched a row -- a starvation that grows with throughput and
looks exactly like Neo4j being slow.

The stamp value itself is deliberately Python-side, unlike `updated_at`, which
`services/signal_engine/store.py` takes from the server clock. Nothing compares
these three columns against a clock: they are read as `IS NULL` / `IS NOT NULL`
by the sweepers, and as a value only by a human or a drift audit asking "when did
this last get indexed". A skewed worker clock therefore costs an inaccurate
diagnostic, not an inverted predicate, and taking `func.now()` per stamp would
cost a round trip's worth of nothing on the hottest write in the fan-out.

Layer note: `workers/` (L4). Imports `models/` and SQLAlchemy only -- no client
is constructed here, so a worker under test stamps against the same statement it
would issue in production, on SQLite.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Final, cast

from sqlalchemy import CursorResult, Table, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.core.logging import get_logger
from models.base import utcnow
from models.orm.signal import SignalRow

__all__ = ["IndexState", "clear_index_state", "stamp_index_state"]

logger = get_logger(__name__)

_SIGNALS: Final[Table] = cast(Table, SignalRow.__table__)
"""The `signals` table as Core.

Core rather than the ORM for the same reason `services/signal_engine/store.py`
gives: this is a targeted single-column `UPDATE` on a row this process may not
have loaded, and routing it through the unit of work would either require a
`SELECT` first or silently flush unrelated pending changes on the session.
"""


class IndexState(enum.StrEnum):
    """The three columns, named by the store each one speaks for.

    A `StrEnum` whose *values are the column names* so a caller cannot pass a
    column that does not exist -- a typo becomes an `AttributeError` at import
    instead of an `UPDATE` that matches no column and is rejected at execute
    time, deep inside a handler, per message.

    Not in `models/enums.py`: this never crosses a process boundary and is never
    serialized. It is dispatch inside `workers/`.
    """

    VECTOR = "indexed_vector_at"
    """Qdrant holds this Signal's chunk vectors."""

    KEYWORD = "indexed_keyword_at"
    """OpenSearch holds this Signal's chunk documents."""

    GRAPH = "graphed_at"
    """Neo4j holds this Signal's entities, stub and `MENTIONS` edges."""


async def stamp_index_state(
    session_factory: async_sessionmaker[AsyncSession],
    signal_id: str,
    *columns: IndexState,
    at: datetime | None = None,
) -> bool:
    """Record that a derived store now holds this Signal. Returns whether a row moved.

    Call this **only after** the derived write has been acknowledged. See the
    module docstring: the ordering is the whole safety property, and it is the
    one thing about this function that a reviewer should check.

    `False` means the row is gone -- erased between the derived write and this
    statement. That is not an error and not worth raising over: the Signal no
    longer exists, so there is nothing left to reconcile, and the derived copy
    will be removed by the erasure path that deleted the row rather than by
    whoever happened to be indexing at the time.

    Idempotent by construction. A redelivery re-stamps the same columns with a
    later timestamp, which changes nothing any reader depends on -- both the
    sweeper predicate and the drift audit only ask whether the value is `NULL`.
    """
    if not columns:
        raise ValueError(
            "stamp_index_state() was called with no columns; a stamp that names "
            "no store would issue an UPDATE with an empty SET clause"
        )
    stamped_at = at or utcnow()
    values = {column.value: stamped_at for column in columns}
    return await _apply(session_factory, signal_id, values, action="stamped")


async def clear_index_state(
    session_factory: async_sessionmaker[AsyncSession],
    signal_id: str,
    *columns: IndexState,
) -> bool:
    """Withdraw the claim that a derived store holds this Signal.

    The inverse operation, and it exists for one situation: a derived write that
    is discovered to have been *wrong* rather than merely absent -- a collection
    rebuilt under a new embedding model, an index whose alias was swapped, a
    Signal re-chunked so its old chunk ids no longer describe it. Setting the
    column back to `NULL` is what puts the row into the sweeper's backlog again.

    `services/signal_engine/store.py` already writes all three as `NULL` on every
    upsert, so the ingest path needs nothing from this. Reprocessing is not the
    case it is for.
    """
    if not columns:
        raise ValueError("clear_index_state() was called with no columns")
    values: dict[str, datetime | None] = {column.value: None for column in columns}
    return await _apply(session_factory, signal_id, values, action="cleared")


async def _apply(
    session_factory: async_sessionmaker[AsyncSession],
    signal_id: str,
    values: dict[str, datetime | None],
    *,
    action: str,
) -> bool:
    """Issue the one-row `UPDATE` and report whether it matched.

    `updated_at` is deliberately absent from `values` and must stay absent --
    see the module docstring. `synchronize_session=False` is implied by using a
    Core statement, which is the other half of why this is not an ORM update: an
    ORM `update()` would try to reconcile the change against instances in the
    session's identity map, and the calling worker has none.
    """
    async with session_factory() as session:
        result = cast(
            CursorResult[None],
            await session.execute(
                update(_SIGNALS).where(_SIGNALS.c.id == signal_id).values(**values)
            ),
        )
        await session.commit()

    matched = result.rowcount != 0
    if not matched:
        logger.info(
            "worker.index_state.row_missing",
            signal_id=signal_id,
            action=action,
            columns=sorted(values),
            reason="the signals row was deleted between the derived write and the stamp",
        )
    return matched
