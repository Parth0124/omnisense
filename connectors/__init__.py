"""Connector layer: Authenticate -> Fetch -> Rate Limit -> Normalize -> Deduplicate -> Emit.

Importing this package registers every shipped connector with
`connectors.registry`. That registration has to happen somewhere explicit:
`registry.get("github")` can only resolve a slug whose module has been imported,
and nothing else imports `connectors/enterprise/github.py` on its own. Without
this block a fresh process reports `registry.slugs() == ()`, and no
operator-facing path -- the sync endpoint, the scheduler, the CLI -- can discover
a connector that demonstrably works.

Explicit imports rather than walking the package tree, for the reason
`connectors/registry.py` already argues: a directory walk registers whatever
happens to be on disk, including a half-finished module someone is mid-way
through writing, and it turns an import error in one connector into a failure to
discover all of them. A list you have to edit is a list you notice editing.

**Registration is not enablement.** A connector appears here as soon as it is
implemented; whether a deployment runs it is `<SLUG>_ENABLED` in the environment.
The two are separate because the catalogue needs to distinguish "this build
cannot do that" from "this deployment has not turned it on" -- and because a
connector requiring credentials nobody has configured should be visible and off
rather than absent.

**Scope.** These are the systems a developer's project lives in. The social,
review and news connectors from the market-intelligence product were removed when
OmniSense became a developer platform: several had no lawful third-party API for
the data that product needed, which is what prompted the pivot in the first
place. They remain in git history.
"""

from connectors import registry
from connectors.enterprise.confluence import ConfluenceConnector
from connectors.enterprise.github import GitHubConnector
from connectors.enterprise.jira import JiraConnector
from connectors.enterprise.notion import NotionConnector
from connectors.enterprise.slack import SlackConnector
from connectors.research.arxiv import ArxivConnector
from connectors.research.papers_with_code import PapersWithCodeConnector
from connectors.research.semantic_scholar import SemanticScholarConnector

__all__ = [
    "ArxivConnector",
    "ConfluenceConnector",
    "GitHubConnector",
    "JiraConnector",
    "NotionConnector",
    "PapersWithCodeConnector",
    "SemanticScholarConnector",
    "SlackConnector",
    "registry",
]

#: Every connector shipped in this package. Registration is idempotent so that
#: re-importing under pytest, or a test that registered one directly, does not
#: trip the registry's duplicate-slug guard.
SHIPPED = (
    # Where the work happens.
    GitHubConnector,
    JiraConnector,
    SlackConnector,
    # Where the writing happens.
    ConfluenceConnector,
    NotionConnector,
    # Where the reading happens.
    ArxivConnector,
    SemanticScholarConnector,
    PapersWithCodeConnector,
)

for _connector in SHIPPED:
    if _connector.slug not in registry.slugs():
        registry.register(_connector)
del _connector
