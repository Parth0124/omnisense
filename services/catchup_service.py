"""What happened while you were away.

The first thing in this system a person would actually run. Everything before it
was plumbing: a shape for artifacts, tables to hold them, a sync to fill the
tables. This reads a window of that history back and writes a briefing.

The bar is deliberately higher than "summarise the rows"
------------------------------------------------------
`git log --oneline` already lists commit titles, and nobody needs a second one.
What a person cannot get from a list is the shape of the work -- that three
commits on one afternoon were one push to production, that a pipeline went in and
came out again in a day. Those facts live *between* artifacts, in their order and
their spacing, and they are the only reason to spend a model call here.

Which is also the danger. Reading a narrative into a list of commits is exactly
what a language model does eagerly and badly. During development this module's
own author looked at four commits -- `establishing ci/cd`, `testing ci/cd` twice,
`Delete .github/workflows/deploy.yml` -- and wrote "CI/CD attempted and abandoned;
deployment fought back". Every workflow run had in fact **succeeded**; the author
deleted a working pipeline on purpose. The story was fluent, plausible, and
false, and nothing in the artifacts said otherwise.

So: **every sentence must cite, and every citation is checked.**

Citations are numbers, not ids
------------------------------
The prompt numbers each artifact `[1]`, `[2]`, and the model cites those. Asking
for `art_9f3e21c8-...` back instead invites a plausible-looking id that belongs to
nothing, or to the wrong row -- and a citation that resolves to the wrong artifact
is worse than none, because it survives inspection. A small integer is either in
range or it is not, and `_resolve` throws away anything that is not.

A phase that ends up with no surviving citation is **dropped**, not softened. The
alternative -- printing it unattributed, or hedged -- is how a fabrication ends up
in front of somebody who then repeats it in a standup.

Layer note: **L2 service.** Takes its provider as a constructor argument like
every other model caller (`services/llm/provider.py`); constructs nothing itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from pydantic import Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.core.config import LLMSettings, get_settings
from backend.core.exceptions import NotFoundError
from backend.core.logging import get_logger
from models.artifact import ArtifactKind
from models.base import StrictModel
from models.orm.artifact import ArtifactRow, SourceRow
from models.orm.project import ProjectRow
from services.llm.provider import LLMProvider

__all__ = [
    "CatchupBrief",
    "CatchupPhase",
    "CatchupService",
    "build_catchup_service",
]

_log = get_logger(__name__)

DEFAULT_TENANT = "local"

MAX_ARTIFACTS = 400
"""How many artifacts one briefing may consider.

A bound, not a target. Four hundred one-line digests is roughly twenty thousand
tokens of prompt -- large but affordable; four thousand is neither. When the
window holds more than this the briefing covers the most recent `MAX_ARTIFACTS`
and **says so in `omitted`**, because a summary that quietly drops three weeks
reads exactly like a summary of a quiet three weeks.
"""

MAX_BODY_CHARS = 240
"""How much of a commit message or pull-request body reaches the prompt.

Whole bodies are mostly boilerplate -- checklists, templates, stack traces -- and
they crowd out the thing that matters, which is *how many* artifacts the model can
see at once. The first couple of lines carry the intent.
"""


@dataclass(frozen=True, slots=True)
class Citation:
    """One artifact a sentence rests on."""

    artifact_id: str
    kind: ArtifactKind
    title: str
    occurred_at: datetime
    url: str | None = None


@dataclass(frozen=True, slots=True)
class CatchupPhase:
    """A stretch of work with a shape to it."""

    label: str
    period: str
    narrative: str
    citations: tuple[Citation, ...] = ()


@dataclass(frozen=True, slots=True)
class CatchupBrief:
    """The briefing, and enough honesty about it to be trusted."""

    project: str
    since: datetime
    until: datetime
    headline: str
    phases: tuple[CatchupPhase, ...] = ()

    considered: int = 0
    """Artifacts that reached the model."""

    omitted: int = 0
    """Artifacts inside the window the bound left out. Printed, never hidden."""

    dropped_phases: tuple[str, ...] = field(default_factory=tuple)
    """Phases removed for citing nothing that exists.

    Surfaced rather than swallowed: a model that keeps inventing a phase is
    telling us the prompt is wrong, and a silent filter would hide that.
    """

    @property
    def is_empty(self) -> bool:
        return not self.phases


class _DraftPhase(StrictModel):
    """One phase, as the model is asked to produce it."""

    label: str = Field(
        min_length=1,
        max_length=80,
        description="A few words naming what this stretch of work was. Not a date range.",
    )
    period: str = Field(
        min_length=1,
        max_length=40,
        description="When, in plain language. For example 'Nov 8-10' or '3 March'.",
    )
    narrative: str = Field(
        min_length=1,
        max_length=700,
        description=(
            "Two or three sentences on what happened and what it amounted to. "
            "State only what the cited artifacts support."
        ),
    )
    refs: list[int] = Field(
        min_length=1,
        max_length=40,
        description=(
            "The bracketed numbers of the artifacts this paragraph rests on. "
            "Every claim must be traceable to one of them."
        ),
    )


class _Draft(StrictModel):
    """What comes back from the model, before any of it is believed."""

    headline: str = Field(
        min_length=1,
        max_length=200,
        description="One sentence for the whole window. What, overall, happened.",
    )
    phases: list[_DraftPhase] = Field(min_length=1, max_length=8)


_SYSTEM_PROMPT = """\
You brief a developer returning to a project after time away.

You are given a numbered, chronological list of things that happened: commits,
pull requests, reviews, CI runs. Write a short narrative of the work.

Rules, in order of importance:

1. Cite everything. Every phase lists the `refs` it rests on. A sentence you
   cannot attach to a numbered artifact does not belong in the briefing.

2. Never infer an outcome that is not recorded. A commit named "fix login" does
   not tell you login now works. A deleted config file does not tell you the
   thing it configured had failed -- people delete working code on purpose. If
   the artifacts do not say how something turned out, do not say how it turned
   out.

3. Say what the list cannot. Grouping, order and spacing are the value here:
   several commits in one afternoon are one push; a thing added and removed in a
   day is one decision, not two events. If a window genuinely holds nothing but
   unrelated commits, say that plainly in one phase rather than inventing an arc.

4. Be short. Two or three sentences a phase. No preamble, no restating the
   question, no closing summary.

Write plainly, in past tense, as a colleague would."""


class CatchupService:
    """Reads a window of a project's history and briefs on it."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        provider: LLMProvider,
        settings: LLMSettings | None = None,
        model: str | None = None,
        tenant_id: str = DEFAULT_TENANT,
    ) -> None:
        self._session_factory = session_factory
        self._provider = provider
        resolved = settings if settings is not None else get_settings().llm
        self._model = model or resolved.model_worker
        self._tenant_id = tenant_id

    # ----------------------------------------------------------- the window --

    async def _window(
        self, slug: str, since: datetime, until: datetime, limit: int
    ) -> tuple[list[ArtifactRow], int]:
        """Artifacts for one project in one window, newest first, plus the overflow.

        Scoped by project rather than by source, which is the whole reason
        projects exist: a product spread over four repositories is one question,
        and asking it four times and stitching the answers is not the same thing.
        """
        async with self._session_factory() as session:
            project = (
                await session.execute(
                    select(ProjectRow).where(
                        ProjectRow.slug == slug,
                        ProjectRow.tenant_id == self._tenant_id,
                    )
                )
            ).scalar_one_or_none()
            if project is None:
                raise NotFoundError.for_resource("project", slug)

            base = (
                select(ArtifactRow)
                .join(SourceRow, SourceRow.id == ArtifactRow.source_id)
                .where(
                    SourceRow.project_id == project.id,
                    ArtifactRow.occurred_at >= since,
                    ArtifactRow.occurred_at <= until,
                )
            )
            rows = list(
                (
                    await session.execute(
                        # Newest first so the bound keeps the *recent* end of a
                        # long window -- "what happened while I was away" is
                        # asked about the near past, and truncating from the
                        # front would answer about the far one.
                        base.order_by(ArtifactRow.occurred_at.desc()).limit(limit + 1)
                    )
                )
                .scalars()
                .all()
            )

        omitted = 0
        if len(rows) > limit:
            # One row over the limit is how we learn there were more without
            # paying for a second COUNT over the same join.
            rows = rows[:limit]
            omitted = -1  # exact count filled in below, only when it matters

        if omitted:
            async with self._session_factory() as session:
                total = len(
                    (
                        await session.execute(
                            select(ArtifactRow.id)
                            .join(SourceRow, SourceRow.id == ArtifactRow.source_id)
                            .where(
                                SourceRow.project_id == project.id,
                                ArtifactRow.occurred_at >= since,
                                ArtifactRow.occurred_at <= until,
                            )
                        )
                    )
                    .scalars()
                    .all()
                )
            omitted = max(0, total - limit)

        # Reversed into chronological order for the prompt: a narrative is read
        # forwards, and handing the model a backwards list is asking it to
        # reverse time in its head while also not making things up.
        rows.reverse()
        return rows, omitted

    # ------------------------------------------------------------- briefing --

    async def brief(
        self,
        slug: str,
        *,
        since: datetime,
        until: datetime | None = None,
        limit: int = MAX_ARTIFACTS,
    ) -> CatchupBrief:
        """Brief on one project's window. Never invents; may return nothing."""
        until = until or datetime.now(UTC)
        rows, omitted = await self._window(slug, since, until, limit)

        if not rows:
            return CatchupBrief(
                project=slug,
                since=since,
                until=until,
                headline="Nothing happened in this window.",
                considered=0,
                omitted=0,
            )

        digest = _digest(rows)
        draft = await self._provider.structured(
            prompt=(
                f"Project: {slug}\n"
                f"Window: {since:%d %b %Y} to {until:%d %b %Y}\n"
                f"{len(rows)} artifacts, oldest first.\n\n"
                f"{digest}"
            ),
            schema=_Draft,
            system=_SYSTEM_PROMPT,
            model=self._model,
        )

        phases, dropped = _resolve(draft, rows)

        if dropped:
            _log.warning(
                "catchup.phases_dropped",
                project=slug,
                dropped=len(dropped),
                labels=list(dropped),
            )

        return CatchupBrief(
            project=slug,
            since=since,
            until=until,
            headline=draft.headline,
            phases=tuple(phases),
            considered=len(rows),
            omitted=omitted,
            dropped_phases=tuple(dropped),
        )


def _digest(rows: Sequence[ArtifactRow]) -> str:
    """The numbered list the model reads.

    One line per artifact, because the number of artifacts visible at once is
    what decides whether the model can see a shape at all -- and a shape is the
    only thing worth paying for here.
    """
    lines = []
    for index, row in enumerate(rows, start=1):
        parts = [f"[{index}]", f"{row.occurred_at:%Y-%m-%d}", str(row.kind.value)]
        if row.state:
            parts.append(f"({row.state.value})")
        if row.outcome:
            parts.append(f"[{row.outcome.value}]")
        parts.append((row.title or "(no title)").strip().splitlines()[0][:MAX_BODY_CHARS])
        lines.append(" ".join(parts))
    return "\n".join(lines)


def _resolve(draft: _Draft, rows: Sequence[ArtifactRow]) -> tuple[list[CatchupPhase], list[str]]:
    """Turn cited numbers back into artifacts, discarding what does not resolve.

    The check that makes the rest of this module trustworthy. A number outside
    the range is a fabrication; a phase left with none is a paragraph resting on
    nothing, and it is removed rather than printed unattributed.
    """
    phases: list[CatchupPhase] = []
    dropped: list[str] = []

    for phase in draft.phases:
        citations = tuple(
            Citation(
                artifact_id=rows[ref - 1].id,
                kind=rows[ref - 1].kind,
                title=(rows[ref - 1].title or "").strip().splitlines()[0],
                occurred_at=rows[ref - 1].occurred_at,
                url=rows[ref - 1].url,
            )
            # Deduplicated and ordered: a model that cites [3] twice should not
            # produce the same footnote twice.
            for ref in sorted(set(phase.refs))
            if 1 <= ref <= len(rows)
        )
        if not citations:
            dropped.append(phase.label)
            continue
        phases.append(
            CatchupPhase(
                label=phase.label,
                period=phase.period,
                narrative=phase.narrative,
                citations=citations,
            )
        )

    return phases, dropped


def parse_since(value: str, *, now: datetime | None = None) -> datetime:
    """`2w`, `10d`, `36h` or a date -- whichever somebody types first.

    Relative forms exist because the question is almost always "since I left",
    and counting back to a date to express that is a small tax on the one command
    a person runs every morning.
    """
    now = now or datetime.now(UTC)
    text = value.strip().lower()

    units = {"h": "hours", "d": "days", "w": "weeks"}
    if len(text) > 1 and text[-1] in units and text[:-1].isdigit():
        return now - timedelta(**{units[text[-1]]: int(text[:-1])})

    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise ValueError(
            f"{value!r} is not a date or a duration. Try 2w, 10d, 36h, or 2026-08-01."
        ) from error
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def build_catchup_service(
    session_factory: async_sessionmaker[AsyncSession] | None = None,
    *,
    provider: LLMProvider | None = None,
    tenant_id: str = DEFAULT_TENANT,
) -> CatchupService:
    if session_factory is None:
        from backend.db.session import get_sessionmaker

        session_factory = get_sessionmaker()
    if provider is None:
        from agents.composition import build_llm_provider

        provider = build_llm_provider(get_settings().llm)
    return CatchupService(session_factory, provider=provider, tenant_id=tenant_id)
