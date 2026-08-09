"""The composition root: builds ten agents, wires them into the graph, compiles it.

Everything else in `agents/` is deliberately unwired. `BaseAgent` takes a
provider and a registry; `build_investigation_graph()` takes a mapping of node
names to callables; neither constructs anything. That is what makes each piece
testable in isolation, and it is also why, until this module existed, the agents
and the graph had never been connected -- both halves were individually correct
and nothing ran.

This is the one place that knows how the whole thing fits together. It is
deliberately the *only* such place: a second construction path is how a
deployment ends up running nine agents because someone forgot to add the tenth
to their copy.

**Graph Expansion is a node, not an agent.** `agents/router.py` records this in
`NODE_AGENT` by mapping it to `AgentName.UNKNOWN`. It makes no LLM call, has no
prompt and has no allowlist -- it is a bounded read against `graph/` that widens
the evidence set. So it is supplied here as a plain callable rather than a
`BaseAgent`, and `_graph_expansion_node` below is its whole implementation.

**`CRITIC_FINAL` is the same agent bound to a second node id.** LangGraph node
ids are unique, and the final pass has different rules: `docs/agent-system.md`
§13 makes it annotate-only, so binding it to its own id is what makes "the final
pass cannot re-open the loop" a property of the topology rather than a flag
somebody has to remember to check.

**Nothing here reaches a datastore at construction.** The provider, the tool
registry and the checkpointer are all injectable, and the defaults are lazy. A
test builds the whole graph with fakes and no network; production passes the
real ones. That is what lets `test_composition.py` assert the wiring is complete
without a database.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agents.base import BaseAgent, PromptSource, TraceSink
from agents.collector.agent import CollectorAgent
from agents.competitor.agent import CompetitorAgent
from agents.critic.agent import CriticAgent
from agents.forecast.agent import ForecastAgent
from agents.graph import NodeCallable, build_investigation_graph
from agents.insight.agent import InsightAgent
from agents.planner.agent import PlannerAgent
from agents.report.agent import ReportAgent
from agents.retriever.agent import RetrieverAgent
from agents.router import NODE_AGENT, NodeName
from agents.state import GraphContext, InvestigationState
from agents.strategy.agent import StrategyAgent
from agents.trend.agent import TrendAgent
from backend.core.config import Settings, get_settings
from backend.core.logging import get_logger
from models.enums import AgentName

if TYPE_CHECKING:  # pragma: no cover
    from agents.base import ToolRegistry
    from services.llm.provider import LLMProvider

__all__ = [
    "AGENT_CLASSES",
    "AgentBundle",
    "build_agents",
    "build_default_bundle",
    "build_nodes",
    "compile_investigation_graph",
]

logger = get_logger(__name__)

AGENT_CLASSES: tuple[type[BaseAgent], ...] = (
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
"""Every agent implementation, in graph execution order.

A tuple rather than a discovered set, for the reason `connectors/__init__.py`
gives about its own explicit list: a directory walk registers whatever happens to
be on disk, including a half-finished module. A list you have to edit is a list
you notice editing -- and `build_nodes` cross-checks it against `NODE_AGENT`, so
a new agent that is not added here fails loudly at construction.
"""


@dataclass(frozen=True, slots=True)
class AgentBundle:
    """The constructed agents, keyed by name, plus what they were built from.

    Returned rather than kept in module state so a worker can hold one bundle per
    tenant configuration if it ever needs to, and so a test can inspect what was
    wired without reaching into a global.
    """

    agents: Mapping[AgentName, BaseAgent]
    provider: Any
    registry: Any

    def agent(self, name: AgentName) -> BaseAgent:
        try:
            return self.agents[name]
        except KeyError:
            raise KeyError(
                f"no agent constructed for {name.value}. AGENT_CLASSES in "
                "agents/composition.py is the single list; add it there."
            ) from None


def build_agents(
    *,
    provider: LLMProvider,
    registry: ToolRegistry,
    prompts: PromptSource | None = None,
    trace: TraceSink | None = None,
    settings: Settings | None = None,
) -> AgentBundle:
    """Construct all ten agents against one provider and one tool registry.

    One provider instance shared by every agent, deliberately. The provider owns
    an HTTP client with a connection pool; constructing one per agent would open
    ten pools to the same host and defeat the keep-alive that makes a
    twelve-node run affordable.

    One registry likewise: the allowlist is per-agent and lives *in* the registry
    (`AGENT_TOOL_ALLOWLIST`), so a registry per agent would let the two copies
    drift -- which is exactly the divergence `BaseAgent.use_tool` checks for.
    """
    resolved = settings if settings is not None else get_settings()
    agents: dict[AgentName, BaseAgent] = {}
    for agent_cls in AGENT_CLASSES:
        agents[agent_cls.name] = agent_cls(
            provider,
            registry,
            prompts=prompts,
            trace=trace,
            settings=resolved,
        )
    logger.info(
        "agents.constructed",
        count=len(agents),
        names=sorted(name.value for name in agents),
    )
    return AgentBundle(agents=agents, provider=provider, registry=registry)


async def _graph_expansion_node(state: InvestigationState) -> dict[str, Any]:
    """Widen the evidence set through the knowledge graph. Not an agent.

    Makes no LLM call, so it has no prompt, no tier and no tool allowlist --
    `NODE_AGENT` maps it to `AgentName.UNKNOWN` precisely so its failures are not
    attributed to the Retriever, whose failure domain is entirely different.

    Degrades to a no-op. `docs/architecture.md` §7.3 makes Neo4j optional, and
    this node is the clearest case: expansion widens recall, so losing it costs
    evidence rather than correctness. Returning an empty delta lets the run
    continue with vector and keyword hits, and the empty `GraphContext` records
    that expansion happened and found nothing -- which is different from
    expansion never running.
    """
    seeds = [ref.signal_id for ref in (state.get("evidence") or [])][:20]
    if not seeds:
        return {"graph_context": GraphContext()}

    try:
        from services.graph_service import build_graph_service

        service = build_graph_service()
        entity_ids: list[str] = []
        for signal_id in seeds[:5]:
            mentions = await service.signals_for_entity(
                tenant_id=state["tenant_id"], entity_id=signal_id, limit=10
            )
            entity_ids.extend(mention.signal_id for mention in mentions)
    except Exception as error:  # noqa: BLE001 -- expansion is optional by design
        logger.warning(
            "graph_expansion.degraded",
            error=type(error).__name__,
            consequence="run continues on vector and keyword evidence only",
        )
        return {"graph_context": GraphContext(seed_entity_ids=tuple(seeds))}

    return {
        "graph_context": GraphContext(
            seed_entity_ids=tuple(seeds),
            expanded_entity_ids=tuple(dict.fromkeys(entity_ids))[:100],
            fact_count=len(entity_ids),
        )
    }


def build_nodes(bundle: AgentBundle) -> dict[NodeName, NodeCallable]:
    """Map every graph node onto the callable that runs it.

    Cross-checked against `NODE_AGENT` rather than written out twice. That table
    is the topology's own statement of which agent backs which node, and building
    the mapping *from* it means a node added to the graph without an
    implementation fails here -- at construction, naming the node -- instead of
    at the moment the router first tries to dispatch it, mid-run, in a worker.
    """
    nodes: dict[NodeName, NodeCallable] = {}
    for node, agent_name in NODE_AGENT.items():
        if agent_name is AgentName.UNKNOWN:
            # Graph Expansion. See `_graph_expansion_node`.
            nodes[node] = _graph_expansion_node
            continue
        try:
            nodes[node] = bundle.agent(agent_name)
        except KeyError as error:
            raise ValueError(
                f"graph node {node.value!r} needs agent {agent_name.value!r}, which "
                f"was not constructed: {error}"
            ) from None

    missing = [node for node in NodeName if node not in nodes]
    if missing:
        raise ValueError(
            "NODE_AGENT does not cover every graph node; missing "
            f"{', '.join(n.value for n in missing)}. A node with no callable would "
            "be a stage the run silently skips."
        )
    return nodes


def compile_investigation_graph(
    bundle: AgentBundle,
    *,
    checkpointer: Any | None = None,
    settings: Settings | None = None,
) -> Any:
    """Build and compile the runnable graph. What a worker executes.

    `checkpointer=None` compiles without durability, which is what
    `AGENT_CHECKPOINT_ENABLED=false` produces and what a unit test wants. In
    production the worker passes a Postgres saver, because a run that cannot
    resume has to restart from the Planner after any crash -- paying for every
    model call again.
    """
    resolved = settings if settings is not None else get_settings()
    graph = build_investigation_graph(
        build_nodes(bundle), checkpointer=checkpointer, settings=resolved.agents
    )
    logger.info(
        "graph.compiled",
        nodes=len(NodeName),
        checkpointed=checkpointer is not None,
    )
    return graph


def build_default_bundle(
    *,
    provider: LLMProvider | None = None,
    registry: ToolRegistry | None = None,
    settings: Settings | None = None,
) -> AgentBundle:
    """Construct the production bundle from settings.

    Imports live inside the function so importing this module does not pull in
    the Anthropic client, the retrieval stack and every datastore driver. A test
    that only wants `build_nodes` should not need `anthropic` installed --
    the same reasoning `backend/main.py` applies to its disposers.

    Each toolset is built independently and a failure is *logged and skipped*
    rather than raised. `build_default_registry` trims the allowlist to whatever
    was actually registered, so a deployment without a graph service gets a
    working registry in which graph tools are simply absent -- and an agent
    asking for one gets the loud denial that says the capability is missing,
    which is far better than a startup crash that says nothing at all.
    """
    resolved = settings if settings is not None else get_settings()

    if provider is None:
        provider = build_llm_provider(resolved.llm)

    if registry is None:
        from agents.tools.registry import build_default_registry

        toolsets: dict[str, Any] = {}
        for name, factory in _toolset_factories().items():
            try:
                toolsets[name] = factory(resolved)
            except Exception as error:  # noqa: BLE001 -- see the docstring
                logger.warning(
                    "toolset.unavailable",
                    toolset=name,
                    error=type(error).__name__,
                    consequence=f"{name} tools are not registered for any agent",
                )
        registry = build_default_registry(**toolsets)
        logger.info("tools.registered", toolsets=sorted(toolsets))

    return build_agents(provider=provider, registry=registry, settings=resolved)


def build_llm_provider(settings: Any) -> LLMProvider:
    """Pick the model backend from `LLM_PROVIDER`.

    The one place the choice is made. `docs/architecture.md` treats the AI layer
    as model-agnostic, and that only holds if selection happens once -- a second
    site that constructed Anthropic directly would silently ignore the setting,
    and the symptom would be an unexplained bill against a provider the
    deployment thought it had switched away from.

    Anthropic has its own module because it speaks the Messages API.
    Everything else in the ecosystem -- OpenRouter, OpenAI, Ollama, LiteLLM,
    vLLM -- accepts the OpenAI chat-completions shape, so they share one
    provider and differ only by `LLM_BASE_URL`.
    """
    from backend.core.config import LLMProvider as ProviderName

    name = settings.provider

    if name is ProviderName.ANTHROPIC:
        from services.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider(settings=settings)

    if name in (
        ProviderName.OPENAI,
        ProviderName.OLLAMA,
        ProviderName.LITELLM,
        ProviderName.AZURE_OPENAI,
    ):
        from services.llm.openai_compatible import OpenAICompatibleProvider

        key = settings.api_key or settings.anthropic_api_key
        return OpenAICompatibleProvider(
            settings=settings,
            api_key=key.get_secret_value() if key else None,
        )

    raise NotImplementedError(
        f"LLM_PROVIDER={name.value!r} has no implementation. Supported today: "
        "anthropic (native), and openai / ollama / litellm / azure_openai "
        "(OpenAI chat-completions shape, selected by LLM_BASE_URL). "
        "google, bedrock and vertex are declared in the enum but not built."
    )


def _toolset_factories() -> dict[str, Any]:
    """How each toolset is constructed. Separated so the failure is per-toolset.

    A dict of thunks rather than four try/except blocks inline, because the
    handling is identical for every one and four copies of it is four chances for
    one to lose its logging.
    """

    def retrieval(settings: Settings) -> Any:
        from agents.tools.retrieval_tools import build_retrieval_toolset

        return build_retrieval_toolset()

    def graph(settings: Settings) -> Any:
        from agents.tools.graph_tools import GraphToolset, load_graph_service

        return GraphToolset(reader=load_graph_service(), tenant_id="default")

    def analytics(settings: Settings) -> Any:
        from agents.tools.analytics_tools import build_analytics_toolset

        return build_analytics_toolset()

    def connectors(settings: Settings) -> Any:
        from agents.tools.connector_tools import build_connector_toolset

        return build_connector_toolset()

    return {
        "retrieval": retrieval,
        "graph": graph,
        "analytics": analytics,
        "connectors": connectors,
    }
