"""The registry of MCP server definitions: what may be connected to, and how.

`docs/agent-system.md` §9 splits MCP into two files, and the split is a security
boundary rather than a tidiness one. This file says *which servers exist and what
they are allowed to offer*; `client.py` does the talking. Nothing here performs
I/O, so the question "what third-party code can this deployment reach?" is
answered by reading one table instead of tracing a call graph.

Three rules shape every field below.

**Nothing is enabled by default.** `DEFAULT_MCP_SERVERS` is empty. An MCP server
is a third-party process that will be handed text scraped from the internet and
whose replies land in a context window holding tools; a server that ships enabled
is third-party code running in every tenant's investigation because a default was
never revisited. Enabling one is a composition-root edit, visible in a diff.

**A server offers only what it was named for.** `exposes` is an explicit set of
tool names, and discovery is checked *against* it rather than trusted to produce
it. This is the second half of §9 rule 1: discovery never auto-grants. A server
that quietly starts advertising `run_shell` after an upgrade gets it ignored and
logged, and -- just as important -- a server that quietly *stops* advertising a
tool this deployment depends on produces a loud missing-tool warning instead of
an agent silently losing a capability. Registration is still not permission: a
registered MCP tool becomes callable only when an allowlist names
`mcp:<server>:<tool>` (`ToolRegistry.with_mcp_tools`).

**Secrets are named, never carried.** A stdio server needs environment variables
and an HTTP server needs headers; both are held here as *variable names* and
resolved from the process environment at spawn time. A definition that carried
the values would put a decrypted token into every log line, error payload and
checkpoint that ever serialised a server definition -- and this object is passed
around a package whose entire job is handling hostile text.

Timeouts are per-server rather than global because MCP servers are not
comparable: a local stdio process answers in milliseconds and a remote SaaS
server can take ten seconds, and one timeout that suits both is either useless or
spends a run's budget waiting.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Final

from backend.core.exceptions import ConfigurationError
from backend.core.logging import get_logger

__all__ = [
    "DEFAULT_MCP_SERVERS",
    "MAX_SERVER_NAME_CHARS",
    "MAX_TIMEOUT_SECONDS",
    "SERVER_NAME_RE",
    "TOOL_NAME_RE",
    "MCPServerDef",
    "MCPServerRegistry",
    "MCPTransport",
]

logger = get_logger(__name__)

MAX_SERVER_NAME_CHARS: Final = 40

SERVER_NAME_RE: Final = re.compile(r"^[a-z0-9](?:[a-z0-9_-]{0,38}[a-z0-9])?$")
"""Server names are lowercase slugs, and deliberately cannot contain `:`.

`:` is the namespace separator in `mcp:<server>:<tool>`. A server called `a:b`
would make `mcp:a:b:c` parse as server `a`, tool `b:c` -- two different servers
able to produce one tool name, which is an allowlist entry that means two things.
Rejecting the character is cheaper than making the parser clever.
"""

TOOL_NAME_RE: Final = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
"""What an MCP server may call a tool.

Restrictive because the name travels into the provider's tool schema (via
`ToolSpec.wire_name`), into allowlists and into log lines. A server-chosen name
containing whitespace, quotes or newlines formats badly in exactly the places an
operator reads under pressure.
"""

DEFAULT_TIMEOUT_SECONDS: Final = 20.0
DEFAULT_CONNECT_TIMEOUT_SECONDS: Final = 10.0

MAX_TIMEOUT_SECONDS: Final = 60.0
"""Ceiling on any single MCP operation.

A minute is already most of what one node can spend without threatening the
investigation timeout (`AgentSettings.timeout_seconds`), and an MCP server that
needs longer is doing work that belongs in a connector and a background sync
rather than inline in a reasoning step.
"""


class MCPTransport(str, Enum):
    """How the client reaches a server.

    The three the `mcp` package supports. `STDIO` spawns a local subprocess and
    is the only one that runs code on our host, which is why `MCPServerDef`
    validates its command separately and why an operator reviewing this table
    should read a `STDIO` entry as a deployment decision rather than a
    configuration one.
    """

    STDIO = "stdio"
    STREAMABLE_HTTP = "streamable_http"
    SSE = "sse"


@dataclass(frozen=True, slots=True)
class MCPServerDef:
    """One MCP server this deployment may connect to.

    Frozen: the client reads a definition on every call, and a mutable one would
    let a bug -- or code that an injected instruction talked into running --
    widen `exposes` at runtime. Widening is a source edit.
    """

    name: str
    transport: MCPTransport = MCPTransport.STDIO
    enabled: bool = False
    """Off unless a deployment says otherwise -- see the module docstring."""

    exposes: frozenset[str] = frozenset()
    """Tool names this server may contribute, deny-by-default.

    Empty means the server contributes nothing. That is a usable state -- it
    connects, discovery runs, and the log records what the server advertised --
    and it is the right default, because the alternative ("expose everything
    discovered") makes the set of callable capabilities a decision taken by the
    third party rather than by us.
    """

    # -- stdio ---------------------------------------------------------------
    command: str = ""
    args: tuple[str, ...] = ()
    env_keys: tuple[str, ...] = ()
    """Names of environment variables the child process needs. Never values."""

    cwd: str | None = None

    # -- http / sse ----------------------------------------------------------
    url: str = ""
    header_env_keys: Mapping[str, str] = field(default_factory=dict)
    """Header name -> environment variable holding its value.

    Same reason as `env_keys`: an `Authorization` header written into this table
    is a bearer token in the source tree, and this table is the one an operator
    is most likely to paste into a ticket.
    """

    # -- budgets -------------------------------------------------------------
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    connect_timeout_seconds: float = DEFAULT_CONNECT_TIMEOUT_SECONDS

    max_result_bytes: int = 16_384
    """This server's share of a context window.

    Half the registry default, because MCP results are the least predictable
    payload in the system: our own tools return shapes we designed, and an MCP
    server returns whatever it likes at whatever length it likes.
    """

    description: str = ""
    """An operator-facing note, written by us. An MCP server's own
    self-description is third-party text and is handled as such in `client.py`."""

    def __post_init__(self) -> None:
        if not SERVER_NAME_RE.match(self.name):
            raise ConfigurationError(
                f"invalid MCP server name {self.name!r}: lowercase letters, digits, "
                f"'_' and '-' only, at most {MAX_SERVER_NAME_CHARS} characters, and "
                "never ':' (the namespace separator in mcp:<server>:<tool>)."
            )
        bad_tools = sorted(name for name in self.exposes if not TOOL_NAME_RE.match(name))
        if bad_tools:
            raise ConfigurationError(
                f"MCP server {self.name!r} exposes unusable tool names: {bad_tools}"
            )
        if self.transport is MCPTransport.STDIO:
            self._validate_stdio()
        else:
            self._validate_remote()
        for label, value in (
            ("timeout_seconds", self.timeout_seconds),
            ("connect_timeout_seconds", self.connect_timeout_seconds),
        ):
            if not 0 < value <= MAX_TIMEOUT_SECONDS:
                raise ConfigurationError(
                    f"MCP server {self.name!r}: {label} must be in (0, "
                    f"{MAX_TIMEOUT_SECONDS}]; got {value}"
                )
        if self.max_result_bytes <= 0:
            raise ConfigurationError(f"MCP server {self.name!r}: max_result_bytes must be > 0")

    def _validate_stdio(self) -> None:
        if not self.command:
            raise ConfigurationError(
                f"MCP server {self.name!r} uses stdio transport but declares no command"
            )
        if self.url:
            # Both set means one of them is stale. The stale one is what an
            # operator will read when deciding whether this server is remote.
            raise ConfigurationError(
                f"MCP server {self.name!r} declares both a command and a url"
            )

    def _validate_remote(self) -> None:
        if not self.url:
            raise ConfigurationError(
                f"MCP server {self.name!r} uses {self.transport.value} transport but "
                "declares no url"
            )
        if self.command:
            raise ConfigurationError(
                f"MCP server {self.name!r} declares both a url and a command"
            )
        if not (self.url.startswith("https://") or self.url.startswith("http://localhost")):
            # Plaintext to a remote host puts the query, the evidence we send and
            # the server's replies on the wire in clear. Loopback is exempt
            # because a local dev server has no wire to be on.
            raise ConfigurationError(
                f"MCP server {self.name!r} must use https:// (or http://localhost for "
                f"local development); got {self.url!r}"
            )

    # ----------------------------------------------------------- resolution --

    def resolve_env(self, environ: Mapping[str, str] | None = None) -> dict[str, str]:
        """Pull this server's declared environment variables from the process env.

        Missing variables are *omitted* rather than defaulted to empty strings: a
        server handed `API_TOKEN=""` fails somewhere inside its own auth code with
        a message nobody can act on, whereas one handed no token at all fails at
        the handshake and lands in `MCPClient.unreachable` with its name attached.
        """
        source = os.environ if environ is None else environ
        resolved = {key: source[key] for key in self.env_keys if key in source}
        missing = [key for key in self.env_keys if key not in resolved]
        if missing:
            # Names only. Logging a value here would defeat the entire point of
            # holding variable names instead of secrets.
            logger.warning("agent.mcp.env_missing", server=self.name, missing=missing)
        return resolved

    def resolve_headers(self, environ: Mapping[str, str] | None = None) -> dict[str, str]:
        """Build request headers from the declared variable names."""
        source = os.environ if environ is None else environ
        headers = {
            header: source[key] for header, key in self.header_env_keys.items() if key in source
        }
        missing = sorted(
            header for header, key in self.header_env_keys.items() if key not in source
        )
        if missing:
            logger.warning("agent.mcp.header_env_missing", server=self.name, headers=missing)
        return headers

    def permits(self, tool_name: str) -> bool:
        """Whether a discovered tool may be registered at all."""
        return tool_name in self.exposes


DEFAULT_MCP_SERVERS: Final[tuple[MCPServerDef, ...]] = ()
"""No servers. Deliberately.

Shipping a default server would mean every deployment that never thought about
MCP is running one. The empty tuple makes "we connect to nothing third-party" the
state you get by doing nothing, which is the only safe direction for a default to
point.
"""


class MCPServerRegistry:
    """An immutable table of server definitions, keyed by name.

    Immutable for the reason `ToolRegistry` is: every method that would widen the
    reachable set returns a new registry, so no code path -- and therefore no
    injected instruction that reaches code holding one -- can add a server to a
    running investigation.
    """

    __slots__ = ("_servers",)

    def __init__(self, servers: Sequence[MCPServerDef] = DEFAULT_MCP_SERVERS) -> None:
        by_name: dict[str, MCPServerDef] = {}
        for server in servers:
            if server.name in by_name:
                raise ConfigurationError(
                    f"duplicate MCP server name {server.name!r}: two servers sharing one "
                    "name make every mcp:<server>:<tool> allowlist entry ambiguous."
                )
            by_name[server.name] = server
        self._servers: Mapping[str, MCPServerDef] = MappingProxyType(by_name)

    def __len__(self) -> int:
        return len(self._servers)

    def __iter__(self) -> Iterator[MCPServerDef]:
        """Sorted by name, so the order MCP tools join the tool list -- and
        therefore the prompt-cache prefix (§4) -- does not depend on how this
        table happened to be written."""
        return iter(sorted(self._servers.values(), key=lambda server: server.name))

    def __contains__(self, name: object) -> bool:
        return name in self._servers

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._servers))

    def get(self, name: str) -> MCPServerDef | None:
        return self._servers.get(name)

    def require(self, name: str) -> MCPServerDef:
        server = self._servers.get(name)
        if server is None:
            raise ConfigurationError(f"no MCP server named {name!r} is registered")
        return server

    def enabled(self) -> tuple[MCPServerDef, ...]:
        """The servers a client should actually try to reach, sorted by name."""
        return tuple(server for server in self if server.enabled)

    def with_servers(self, servers: Sequence[MCPServerDef]) -> MCPServerRegistry:
        """A new registry with `servers` added. Never mutates this one."""
        return MCPServerRegistry([*self._servers.values(), *servers])
