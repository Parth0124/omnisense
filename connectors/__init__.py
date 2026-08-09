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

**Registration is not enablement.** A connector appears here as soon as it is
implemented; whether a deployment actually runs it is `<SLUG>_ENABLED` in the
environment. The two are separate because the catalogue endpoint needs to
distinguish "this build cannot do that" from "this deployment has not turned it
on" -- and because a connector that requires credentials nobody has configured
should be visible and off rather than absent.
"""

from connectors import registry
from connectors.enterprise.confluence import ConfluenceConnector
from connectors.enterprise.github import GitHubConnector
from connectors.enterprise.hubspot import HubSpotConnector
from connectors.enterprise.jira import JiraConnector
from connectors.enterprise.notion import NotionConnector
from connectors.enterprise.salesforce import SalesforceConnector
from connectors.enterprise.slack import SlackConnector
from connectors.news.gdelt import GdeltConnector
from connectors.news.news_api import NewsApiConnector
from connectors.news.rss import RssConnector
from connectors.research.arxiv import ArxivConnector
from connectors.research.papers_with_code import PapersWithCodeConnector
from connectors.research.semantic_scholar import SemanticScholarConnector
from connectors.reviews.amazon import AmazonConnector
from connectors.reviews.app_store import AppStoreConnector
from connectors.reviews.google_reviews import GoogleReviewsConnector
from connectors.reviews.play_store import PlayStoreConnector
from connectors.reviews.trustpilot import TrustpilotConnector
from connectors.social.instagram import InstagramConnector
from connectors.social.linkedin import LinkedInConnector
from connectors.social.reddit import RedditConnector
from connectors.social.tiktok import TikTokConnector
from connectors.social.x import XConnector
from connectors.social.youtube import YouTubeConnector

__all__ = [
    "AmazonConnector",
    "AppStoreConnector",
    "ArxivConnector",
    "ConfluenceConnector",
    "GdeltConnector",
    "GitHubConnector",
    "GoogleReviewsConnector",
    "HubSpotConnector",
    "InstagramConnector",
    "JiraConnector",
    "LinkedInConnector",
    "NewsApiConnector",
    "NotionConnector",
    "PapersWithCodeConnector",
    "PlayStoreConnector",
    "RedditConnector",
    "RssConnector",
    "SalesforceConnector",
    "SemanticScholarConnector",
    "SlackConnector",
    "TikTokConnector",
    "TrustpilotConnector",
    "XConnector",
    "YouTubeConnector",
    "registry",
]

#: Every connector shipped in this package. Registration is idempotent so that
#: re-importing under pytest, or a test that registered one directly, does not
#: trip the registry's duplicate-slug guard.
SHIPPED = (
    # Open: no credential, or an optional key that only raises the rate limit.
    RssConnector,
    GdeltConnector,
    AppStoreConnector,
    ArxivConnector,
    PapersWithCodeConnector,
    SemanticScholarConnector,
    # Key or token required, obtainable by anyone.
    NewsApiConnector,
    RedditConnector,
    YouTubeConnector,
    TrustpilotConnector,
    GitHubConnector,
    NotionConnector,
    SlackConnector,
    JiraConnector,
    ConfluenceConnector,
    HubSpotConnector,
    SalesforceConnector,
    PlayStoreConnector,
    XConnector,
    # `requires_tos_review`: registered and visible, but refused at dispatch
    # until someone has reviewed the terms. Registering them anyway is the
    # point -- the catalogue can then distinguish "this build cannot do that"
    # from "this deployment has not turned it on", which an absent connector
    # cannot.
    LinkedInConnector,
    InstagramConnector,
    TikTokConnector,
    AmazonConnector,
    GoogleReviewsConnector,
)

for _connector in SHIPPED:
    if _connector.slug not in registry.slugs():
        registry.register(_connector)
del _connector
