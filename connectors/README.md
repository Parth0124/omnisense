# connectors/

*Source integrations. One module per platform, one contract for all of them.*

Every connector implements the same six-stage lifecycle (Design Doc §5):

```
Authentication → Fetch → Rate Limit → Normalize → Deduplicate → Emit Signal
```

## Layout

| Path | Purpose |
| --- | --- |
| `base.py` | `BaseConnector` — the abstract contract every connector implements. |
| `protocol.py` | Structural typing protocols for connectors, paginators, cursors. |
| `registry.py` | Discovery and instantiation by slug. |
| `auth/` | OAuth2, API keys, encrypted credential storage and rotation. |
| `ratelimit/` | Redis token bucket, exponential backoff with jitter. |
| `dedup/` | Content hashing and near-duplicate detection. |
| `normalize/` | Source payload → `Signal` mapping, HTML/boilerplate stripping. |
| `social/` | Reddit, X, YouTube, Instagram, TikTok, LinkedIn. |
| `reviews/` | Amazon, Play Store, App Store, TrustPilot, Google Reviews. |
| `enterprise/` | Slack, Jira, Confluence, Notion, GitHub, Salesforce, HubSpot. |
| `research/` | arXiv, Semantic Scholar, Papers with Code. |
| `news/` | RSS, GDELT, News APIs. |

## Rules

- A connector may import `models/` and `connectors/*`. It must **not** import
  `backend/`, `agents/` or `services/` — connectors are libraries, not services.
- Emit `Signal` objects and nothing else. Enrichment is the Signal Engine's job.
- Never log credentials or raw response bodies containing personal data.
- Connectors that scrape rather than use an official API require a documented
  ToS/robots.txt review before they ship.

## Adding a connector

See the worked example at the end of
[`docs/connector-spec.md`](../docs/connector-spec.md).

## See also

[`docs/connector-spec.md`](../docs/connector-spec.md) ·
[`docs/security-and-privacy.md`](../docs/security-and-privacy.md)
