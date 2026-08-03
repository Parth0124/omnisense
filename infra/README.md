# infra/

*Deployment manifests and monitoring configuration.*

> **Nothing here is applied automatically.** There is no CI/CD in this
> repository and no cloud account is configured. Every one of these is a
> template for a future, manually performed deployment.

| Path | Purpose |
| --- | --- |
| `k8s/base/` | Base Kubernetes manifests for the API, workers and scheduler. |
| `k8s/overlays/{dev,staging,prod}/` | Per-environment kustomize overlays. |
| `modal/` | Modal app definitions for GPU inference (embeddings, cross-encoder rerank). |
| `railway/` | Railway service configuration. |
| `vercel/` | Vercel project configuration for the frontend. |
| `prometheus/` | Scrape configuration and recording rules. |
| `grafana/dashboards/` | Dashboard definitions. |
| `grafana/alerts/` | Alert rules and thresholds. |

## See also

[`docs/deployment.md`](../docs/deployment.md) ·
[`docs/observability.md`](../docs/observability.md)
