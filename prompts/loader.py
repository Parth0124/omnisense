"""Composes and hashes the versioned prompt text that produced a run's output.

A report generated six months ago has to be explainable, and "which words did we
send the model?" is the first question anybody asks. Storing a version string
alone does not answer it: a version string is a promise, and a file on disk can
be edited without anybody bumping it. So this module returns the **content
hash** of the fully-composed prompt, and that hash -- not the filename -- is what
`InvestigationState.prompt_versions` records (`docs/agent-system.md` §11).

Three decisions here are load-bearing.

**The hash covers the shared fragments too.** `prompts/shared/citation_rules.md`
is included by nine agents; editing it changes what nine agents were told. If the
hash covered only `prompts/<agent>/vN.md`, that edit would be invisible in every
run record -- claims produced under the new citation rules would be
indistinguishable from claims produced under the old ones. Composing first and
hashing the composite is what makes the blast radius of a fragment edit visible.

**Composition order is fixed, not derived.** `docs/agent-system.md` §4 requires a
cache-stable prefix, and the prompt cache keys on a byte prefix: reordering two
fragments is a cache miss for every agent that shares them, at full price. The
order lives in `_FRAGMENT_ORDER` as a tuple for exactly that reason -- a set or a
directory listing would reorder itself on a different filesystem and nobody would
notice until the bill arrived.

**Nothing per-run enters the text.** No tenant id, no investigation id, no
timestamp. Those belong in the message turns after the cache breakpoint; putting
one here would make every prompt unique, defeat caching entirely, and -- worse --
put a tenant identifier inside a string that gets hashed and shared across
tenants in logs.

Layer note: `prompts/` sits below `agents/`. It deliberately does **not** import
`agents.state.PromptRef`, because `agents/base.py` imports this module and the
reverse edge would be a cycle. `RenderedPrompt` carries exactly the three fields
`PromptRef` needs (`agent`, `version`, `sha256`); the agent constructs the ref.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final

from backend.core.exceptions import ConfigurationError
from models.enums import AgentName

__all__ = [
    "FRAGMENT_NAMES",
    "PROMPT_ROOT",
    "PromptError",
    "RenderedPrompt",
    "available_versions",
    "clear_prompt_cache",
    "fragments_for",
    "latest_version",
    "load_fragment",
    "load_prompt",
]


PROMPT_ROOT: Final[Path] = Path(__file__).resolve().parent
"""Where the markdown lives. Resolved from `__file__` rather than from settings.

Prompts are source artifacts that ship inside the package; making the location
configurable would mean a deployment could point at a different prompt tree and
still record the same version string, which is precisely the reproducibility hole
this module exists to close.
"""

_VERSION_PATTERN: Final = re.compile(r"^v(\d+)$")

_SHARED_DIR: Final = "shared"

FRAGMENT_NAMES: Final[tuple[str, ...]] = (
    "system",
    "safety",
    "citation_rules",
    "confidence_rubric",
)
"""Every shared fragment, in composition order. See the module docstring."""

_ALWAYS: Final[tuple[str, ...]] = ("system", "safety")
"""Fragments every agent gets.

`safety` is unconditional on purpose. It is the fragment that says retrieved
third-party text is data and never an instruction, and the agents most exposed to
injected text are exactly the mechanical ones (Collector, Retriever) that a
"claims-only" rule would have excluded.
"""

_EVIDENCE_BEARING: Final[frozenset[AgentName]] = frozenset(
    {
        AgentName.RETRIEVER,
        AgentName.INSIGHT,
        AgentName.STRATEGY,
        AgentName.CRITIC,
        AgentName.REPORT,
    }
)
"""Agents whose output asserts something about the world.

These get `citation_rules` and `confidence_rubric`. The Planner and Collector do
not: a plan step and a connector parameter set are decisions about what to *do*,
and attaching a citation requirement to them would train the model to invent
citations for statements that have no evidential content.
"""


class PromptError(ConfigurationError):
    """A prompt could not be loaded or is unusable.

    `ConfigurationError` rather than `NotFoundError`: a missing or empty prompt
    is never a caller's mistake and never recoverable at runtime. It means the
    deployment shipped without a file it needs, and the only fix is a redeploy.
    Surfacing it as a 404 would invite a retry loop that can never succeed.
    """

    code = "prompt_error"
    default_message = "A prompt template could not be loaded."


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    """One agent's fully-composed prompt, plus the identity of the text.

    Frozen because callers hold it across a node's lifetime and record its hash
    afterwards; a mutable rendering would let the text drift away from the digest
    that claims to describe it, which is the one failure this whole module is
    built to prevent.
    """

    agent: AgentName
    version: str
    text: str
    sha256: str
    fragments: tuple[str, ...]
    """Which shared fragments were composed in, in order. Recorded so a hash
    change can be attributed to a fragment edit rather than to the agent file."""

    @property
    def ref_fields(self) -> dict[str, object]:
        """The three fields `agents.state.PromptRef` needs.

        Returned as a mapping rather than as a `PromptRef` to keep `prompts/`
        from importing `agents/` -- see the module docstring. The caller writes
        `PromptRef(**rendered.ref_fields)`.
        """
        return {"agent": self.agent, "version": self.version, "sha256": self.sha256}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _read(path: Path) -> str:
    """Read a prompt file, normalising line endings and rejecting empties.

    Line endings are normalised because the hash is over bytes: a file that round
    trips through a Windows editor would otherwise change every run's recorded
    prompt hash without a single word changing, and every such false positive
    makes the real ones easier to ignore.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise PromptError(
            f"prompt file not found: {path}. Prompts ship with the package; a "
            "missing file means the build is incomplete, not that the caller "
            "asked for the wrong thing.",
            details={"path": str(path)},
            cause=exc,
        ) from exc
    text = raw.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        raise PromptError(
            f"prompt file is empty: {path}. An empty template would be composed "
            "into a prompt that silently asks the model for nothing, and the run "
            "would fail later at schema validation with no hint of the cause.",
            details={"path": str(path)},
        )
    return text


@lru_cache(maxsize=None)
def load_fragment(name: str) -> str:
    """Load one shared fragment by bare name (`"safety"`, not `"safety.md"`).

    Cached: a fragment is read once per process and shared by every agent that
    includes it. `docs/agent-system.md` §11 makes a version in use immutable, so
    the file cannot change underneath the cache during a run. Tests that write
    prompt files call `clear_prompt_cache()`.
    """
    if name not in FRAGMENT_NAMES:
        raise PromptError(
            f"unknown shared fragment {name!r}; known fragments are "
            f"{', '.join(FRAGMENT_NAMES)}. Add it to FRAGMENT_NAMES so the "
            "composition order stays explicit rather than alphabetical.",
            details={"fragment": name},
        )
    return _read(PROMPT_ROOT / _SHARED_DIR / f"{name}.md")


def fragments_for(agent: AgentName) -> tuple[str, ...]:
    """Which shared fragments compose into this agent's prompt, in order.

    Exposed rather than private because the evaluation harness and the prompt
    review checklist both need to answer "did the Critic actually receive the
    confidence rubric?" without re-deriving the rule from the rendered text.
    """
    ordered = list(_ALWAYS)
    if agent in _EVIDENCE_BEARING:
        ordered.extend(("citation_rules", "confidence_rubric"))
    # Re-project through FRAGMENT_NAMES so composition order is the declared one
    # even if someone appends to `_ALWAYS` in the wrong place.
    return tuple(name for name in FRAGMENT_NAMES if name in ordered)


def available_versions(agent: AgentName) -> tuple[str, ...]:
    """Every `vN.md` present for an agent, ascending by N.

    Sorted numerically rather than lexically: `v10` sorts before `v2` as a
    string, so a lexical sort would silently pin every new run to v1's successor
    the moment a tenth version shipped.
    """
    directory = PROMPT_ROOT / agent.value
    if not directory.is_dir():
        raise PromptError(
            f"no prompt directory for agent {agent.value!r} at {directory}.",
            details={"agent": agent.value, "path": str(directory)},
        )
    numbered: list[tuple[int, str]] = []
    for path in directory.glob("v*.md"):
        match = _VERSION_PATTERN.match(path.stem)
        if match is not None:
            numbered.append((int(match.group(1)), path.stem))
    return tuple(stem for _, stem in sorted(numbered))


def latest_version(agent: AgentName) -> str:
    """The highest-numbered version on disk."""
    versions = available_versions(agent)
    if not versions:
        raise PromptError(
            f"agent {agent.value!r} has no vN.md prompt file.",
            details={"agent": agent.value},
        )
    return versions[-1]


def _compose(agent: AgentName, version: str) -> tuple[str, tuple[str, ...]]:
    """Build the composite text: shared fragments first, then the agent template.

    Shared first because the shared block is the cache-stable prefix -- it is
    byte-identical across every agent that shares the same fragment set, so the
    provider's prefix cache can hit across agents within a run rather than only
    across runs of the same agent.
    """
    names = fragments_for(agent)
    blocks = [load_fragment(name) for name in names]
    blocks.append(_read(PROMPT_ROOT / agent.value / f"{version}.md"))
    return "\n\n---\n\n".join(blocks) + "\n", names


@lru_cache(maxsize=None)
def _load_cached(agent: AgentName, version: str) -> RenderedPrompt:
    text, names = _compose(agent, version)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return RenderedPrompt(
        agent=agent,
        version=version,
        text=text,
        sha256=digest,
        fragments=names,
    )


def load_prompt(agent: AgentName | str, version: str | None = None) -> RenderedPrompt:
    """Compose, hash and return one agent's prompt.

    `version=None` resolves to the latest on disk. That is safe *because* the
    resolved version and the content hash are both recorded on the run: a caller
    that does not pin still leaves behind enough to reconstruct what it sent.
    Callers reproducing a historical run must pass the recorded version
    explicitly -- "latest" is a moving target by definition.
    """
    resolved_agent = AgentName(agent) if isinstance(agent, str) else agent
    if resolved_agent is AgentName.UNKNOWN:
        raise PromptError(
            "AgentName.UNKNOWN has no prompt. It exists so that a newer producer's "
            "agent name does not raise on read; it is not a loadable agent.",
            details={"agent": str(agent)},
        )
    return _load_cached(resolved_agent, version or latest_version(resolved_agent))


def clear_prompt_cache() -> None:
    """Drop the memoised fragments and renderings.

    Only tests need this -- and they need it because a test that writes a prompt
    file into a tmp tree and then reads through a warm cache passes against the
    previous test's bytes, which is a green result for the wrong reason.
    """
    load_fragment.cache_clear()
    _load_cached.cache_clear()
