"""Datastore clients and session management.

Every client in this package follows one shape, established by `session.py`:
a lazily-created module-level singleton, no I/O at import time, and a
`check_*() -> bool` that never raises so `/readyz` can aggregate them.

`HEALTH_PROBE_TIMEOUT_SECONDS` lives here because the budget must be *shared*.
A readiness endpoint is only as fast as its slowest probe, and an unbounded one
does not merely report late -- it stalls the whole aggregate. On a half-open
network path (a blackholed host, a dropped NAT mapping, a security group that
silently discards rather than rejects) a TCP connect does not fail, it hangs
until some timeout fires. If that timeout is the driver's default rather than
ours, `/readyz` can block for a minute, which is long enough for the *liveness*
probe to conclude the process is dead and restart a pod whose only problem was
that one dependency was unreachable.

So each probe bounds itself with `asyncio.timeout(HEALTH_PROBE_TIMEOUT_SECONDS)`
regardless of what its client library would otherwise do. Measured against a
blackholed host before this was enforced: Redis 1.0s, Qdrant 5.0s, OpenSearch
5.6s, R2 5.0s, Neo4j 30.0s, PostgreSQL 60.0s.
"""

from typing import Final

__all__ = ["HEALTH_PROBE_TIMEOUT_SECONDS"]

HEALTH_PROBE_TIMEOUT_SECONDS: Final[float] = 5.0
"""Wall-clock ceiling for a single readiness probe.

Deliberately a constant rather than a setting. It is a property of how Kubernetes
probes behave, not something an operator should tune per environment -- and a
value large enough to be worth configuring is already large enough to break the
liveness probe it shares a deadline with.
"""
