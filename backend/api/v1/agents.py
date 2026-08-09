"""`/api/v1/agents` -- introspection over the agent system (§4.6).

Read-only, and deliberately so. There is no endpoint here that *runs* an agent.
Agents are graph nodes; they execute inside an investigation, against a
checkpointed state, under a step ceiling and a token budget, with a Critic
downstream. An endpoint that invoked one directly would produce output with the
appearance of the system's rigour and none of it -- no plan, no evidence, no
critique, no citations that anyone verified.

What this exposes is the *configuration*: which agents exist, what each is
allowed to call, and which prompt version is live. That matters for a reason
`docs/agent-system.md` §9 makes explicit -- the tool allowlist is deny-by-default
and security-relevant, so it has to be inspectable without reading two Python
files side by side. An operator asking "can the Collector reach the graph?"
should be able to answer it from the running system rather than from the source
they hope is deployed.

**The prompt hash is published, the prompt text is not.** The hash is what makes
"this claim was produced by exactly this prompt" checkable a year later
(`prompts/README.md` rule 2). The text is a different matter: it contains the
injection defences and the exact phrasing an attacker would want in order to work
around them, and there is no operator question that needs the body rather than
the identity.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import Field

from backend.api.deps import Principal, require_scopes
from backend.core.exceptions import NotFoundError
from backend.schemas.common import ResponseModel, problem_responses
from models.enums import AgentName

__all__ = ["AgentDescriptor", "PromptDescriptor", "router"]

router = APIRouter(prefix="/agents", tags=["agents"])

RunnerPrincipal = Annotated[Principal, Depends(require_scopes("agents:run"))]
"""`agents:run` from the §3.1 vocabulary.

The scope is named for running even though these routes only read, because there
is no `agents:read` in the closed vocabulary and `require_scopes` raises at
import on an unknown name -- deliberately, since a typo there produces a
requirement no token can satisfy and a 403 indistinguishable from a lockout.
"""


class PromptDescriptor(ResponseModel):
    """Which prompt an agent is running. Identity, never body.

    `sha256` covers the shared fragments as well as the agent template, so a
    change to `prompts/shared/safety.md` moves every agent's hash -- which is
    correct, because it changed what every agent was told.
    """

    version: str
    sha256: str = Field(
        description=(
            "Digest of the composed prompt, shared fragments included. What makes "
            "'this claim came from exactly this prompt' checkable a year later."
        )
    )
    fragments: list[str] = Field(
        description="Shared fragments composed in, in order."
    )


class AgentDescriptor(ResponseModel):
    """One agent's configuration."""

    name: AgentName
    tier: str = Field(description="Model tier: planner, worker or fast.")
    blocking: bool = Field(
        description=(
            "Whether this node's failure leaves the run unable to continue "
            "honestly. Only the Planner is: without a plan there are no branches, "
            "no sub-questions and nothing for the Critic to check coverage against."
        )
    )
    tools: list[str] = Field(
        description=(
            "Exactly what this agent may call. Deny-by-default: absent means "
            "forbidden, and the empty list is a real answer distinct from 'not "
            "declared'."
        )
    )
    prompt: PromptDescriptor | None = None
    output_schema: str = Field(description="Name of the Pydantic model it must return.")


def _load_agents() -> dict[AgentName, type]:
    """The ten implementations, imported lazily.

    Inside the function rather than at module scope so that importing this router
    does not pull in the whole agent package -- which reaches LangGraph, the LLM
    providers and the tool registry. A route table should not cost that.
    """
    from agents.collector.agent import CollectorAgent
    from agents.competitor.agent import CompetitorAgent
    from agents.critic.agent import CriticAgent
    from agents.forecast.agent import ForecastAgent
    from agents.insight.agent import InsightAgent
    from agents.planner.agent import PlannerAgent
    from agents.report.agent import ReportAgent
    from agents.retriever.agent import RetrieverAgent
    from agents.strategy.agent import StrategyAgent
    from agents.trend.agent import TrendAgent

    return {
        agent.name: agent
        for agent in (
            PlannerAgent,
            CollectorAgent,
            RetrieverAgent,
            TrendAgent,
            CompetitorAgent,
            ForecastAgent,
            InsightAgent,
            StrategyAgent,
            CriticAgent,
            ReportAgent,
        )
    }


def _describe(agent_cls: type) -> AgentDescriptor:
    """Build one descriptor, tolerating a missing prompt file.

    A prompt that fails to load leaves `prompt=None` rather than failing the
    request. This endpoint's main use is diagnosing a broken deployment, and an
    introspection endpoint that dies on the thing you are trying to diagnose is
    the least useful possible behaviour.
    """
    prompt: PromptDescriptor | None = None
    try:
        from prompts.loader import load_prompt

        rendered = load_prompt(agent_cls.name, agent_cls.prompt_version)
        prompt = PromptDescriptor(
            version=rendered.version,
            sha256=rendered.sha256,
            fragments=list(rendered.fragments),
        )
    except Exception:  # noqa: BLE001 -- see the docstring
        prompt = None

    return AgentDescriptor(
        name=agent_cls.name,
        tier=getattr(agent_cls.tier, "value", str(agent_cls.tier)),
        blocking=bool(agent_cls.blocking),
        tools=sorted(agent_cls.tools),
        prompt=prompt,
        output_schema=agent_cls.output_model.__name__,
    )


@router.get(
    "",
    summary="Every agent, its tool allowlist and its live prompt version.",
    response_model=list[AgentDescriptor],
    responses=problem_responses(401, 403),
)
async def list_agents(principal: RunnerPrincipal) -> list[AgentDescriptor]:
    """The agent roster.

    Ordered by the graph's execution order rather than alphabetically, because
    that is the order someone reading it is thinking in -- planner first, report
    last.
    """
    return [_describe(agent_cls) for agent_cls in _load_agents().values()]


@router.get(
    "/{agent_name}",
    summary="One agent's configuration.",
    response_model=AgentDescriptor,
    responses=problem_responses(401, 403, 404),
)
async def get_agent(agent_name: str, principal: RunnerPrincipal) -> AgentDescriptor:
    """One agent.

    An unrecognised name is a 404 rather than degrading through `AgentName`'s
    tolerant lookup. That tolerance exists so a *reader* survives a checkpoint
    written by a newer build; here it would turn a typo into a description of
    `UNKNOWN`, which is not an agent and has no implementation.
    """
    agents = _load_agents()
    resolved = next(
        (name for name in agents if name.value == agent_name.strip().casefold()), None
    )
    if resolved is None:
        raise NotFoundError.for_resource("agent", agent_name)
    return _describe(agents[resolved])
