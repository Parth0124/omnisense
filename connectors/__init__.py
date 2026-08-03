"""Connector layer (Design Doc S5): Authenticate -> Fetch -> Rate Limit -> Normalize -> Deduplicate -> Emit.

Importing this package registers every shipped connector with
`connectors.registry`. That registration has to happen somewhere explicit:
`registry.get("reddit")` can only resolve a slug whose module has been imported,
and nothing else imports `connectors/social/reddit.py` on its own. Without this
block a fresh process reports `registry.slugs() == ()`, and no operator-facing
path -- `POST /connectors/sync`, `scripts/sync_connector.py`, the scheduler --
can discover a connector that demonstrably works.

Explicit imports rather than walking the package tree, for the reason
`connectors/registry.py` already argues: a directory walk registers whatever
happens to be on disk, including a half-finished module someone is mid-way
through writing, and it turns an import error in one connector into a failure to
discover all of them. A list you have to edit is a list you notice editing.

`tests/unit/connectors/test_registry.py` asserts this list stays in step with the
connector modules actually present, so a new connector cannot ship unregistered.
"""

from connectors import registry
from connectors.news.gdelt import GdeltConnector
from connectors.news.news_api import NewsApiConnector
from connectors.news.rss import RssConnector
from connectors.social.reddit import RedditConnector

__all__ = [
    "GdeltConnector",
    "NewsApiConnector",
    "RedditConnector",
    "RssConnector",
    "registry",
]

#: Every connector shipped in this package. Registration is idempotent so that
#: re-importing under pytest, or a test that registered one directly, does not
#: trip the registry's duplicate-slug guard.
SHIPPED = (RssConnector, RedditConnector, NewsApiConnector, GdeltConnector)

for _connector in SHIPPED:
    if _connector.slug not in registry.slugs():
        registry.register(_connector)
del _connector
