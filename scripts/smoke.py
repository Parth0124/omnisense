"""Drive one investigation end to end and report what happened.

This is the acceptance test for a local stack: it mints a token, creates an
investigation over HTTP, waits for the worker to finish it, and fetches the
report. If all of that works, the API, the database, the agent graph, the model
provider and the report store are all wired correctly -- which is a much stronger
claim than `make doctor` can make, because doctor only proves each piece answers
when poked.

**It is deliberately one command.** The manual version is four curl calls where
the output of one has to be pasted into the next, and the id is easy to lose
between them. Anything that has to be assembled by hand at a prompt gets
assembled wrong at least once.

**It costs real money**, because the investigation is real: the whole point is to
exercise the configured models rather than a stub. A `quick` run against the
default tiers is roughly ten to fifty cents.

There is deliberately no `--cheap` flag here, because it could not work: the
models are chosen by the *worker* from its own environment, and this script only
talks to the API over HTTP. To run it cheaply, start the worker with the override
instead -- `LLM_MODEL_PLANNER=openai/gpt-4o-mini LLM_MODEL_WORKER=openai/gpt-4o-mini
LLM_MODEL_FAST=openai/gpt-4o-mini make investigator` -- which is honest about
where the setting actually lives.

Two failures are common enough to be diagnosed explicitly rather than left as a
timeout: the API not running, and the worker not running. They look identical
from the outside -- nothing happens -- and mean completely different things.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
import typer

from backend.core.config import get_settings
from backend.core.security import encode_jws
from models.enums import InvestigationStatus

app = typer.Typer(add_completion=False, help="Run one investigation end to end.")

TERMINAL_STATES: frozenset[str] = frozenset(
    status.value for status in InvestigationStatus if status.is_terminal
)
"""Which states mean "stop polling", asked of the enum rather than listed here.

A hand-written list of terminal states was wrong within a day: it named
`completed`, `failed` and `cancelled` and omitted `completed_with_findings` --
which is the state a run reaches whenever the Critic files a finding, and
therefore the *most common* successful outcome. The script then polled a
finished investigation until it timed out and reported failure for a run that
had succeeded and stored its report.

`InvestigationStatus.is_terminal` already encodes this, and a status added later
is picked up here for free.
"""

QUEUED_GRACE_SECONDS = 20
"""How long a run may sit in `queued` before we blame the worker.

The worker claims rows on a poll, so a few seconds of `queued` is normal. Twenty
is long enough that a claim has certainly been attempted, and short enough that
nobody sits watching a spinner wondering whether to Ctrl-C.
"""


def _token(base: str) -> str:
    settings = get_settings()
    now = int(time.time())
    return encode_jws(
        {
            "sub": "smoke",
            "tenant": "local",
            "role": "admin",
            "iat": now,
            "exp": now + 3600,
        },
        secret=settings.security.secret_key.get_secret_value(),
    )


@app.command()
def main(
    base_url: str = typer.Option("http://localhost:8000", help="Where the API is."),
    query: str = typer.Option("What changed recently?", help="The question to ask."),
    timeout: int = typer.Option(180, help="Seconds to wait for completion."),
) -> None:
    """Create an investigation, wait for it, fetch its report."""
    token = _token(base_url)
    headers = {"Authorization": f"Bearer {token}"}

    with httpx.Client(base_url=base_url, headers=headers, timeout=30.0) as client:
        try:
            client.get("/health").raise_for_status()
        except Exception:
            typer.secho(f"The API is not answering at {base_url}.", fg=typer.colors.RED, bold=True)
            typer.echo("  Start it with:  make dev      (API + worker + frontend)")
            typer.echo("  or:             make api      (API only -- but then nothing")
            typer.echo("                                 will run the investigation)")
            raise typer.Exit(code=1) from None

        typer.secho(f"asking: {query!r}", fg=typer.colors.CYAN)
        response = client.post("/api/v1/investigations", json={"query": query, "depth": "quick"})
        if response.status_code == 401:
            typer.secho("401 from the API.", fg=typer.colors.RED, bold=True)
            typer.echo("  The token is signed with SECRET_KEY; if you changed it in .env")
            typer.echo("  after starting the API, restart the API so it verifies against")
            typer.echo("  the same value.")
            raise typer.Exit(code=1)
        response.raise_for_status()

        investigation_id = response.json()["id"]
        typer.echo(f"  id: {investigation_id}")

        deadline = time.monotonic() + timeout
        started = time.monotonic()
        warned = False
        state = "queued"

        while time.monotonic() < deadline:
            body = client.get(f"/api/v1/investigations/{investigation_id}").json()
            state = body["state"]

            if (
                state == "queued"
                and not warned
                and time.monotonic() - started > QUEUED_GRACE_SECONDS
            ):
                typer.secho(
                    f"\n  Still queued after {QUEUED_GRACE_SECONDS}s -- nothing is running it.",
                    fg=typer.colors.YELLOW,
                )
                typer.echo("  The API accepts investigations; the *worker* executes them.")
                typer.echo("  Start it with:  make investigator      (or use make dev)")
                warned = True

            if state in TERMINAL_STATES:
                break

            typer.echo(f"  {state}...")
            time.sleep(3)
        else:
            typer.secho(f"\nStill {state!r} after {timeout}s.", fg=typer.colors.RED, bold=True)
            raise typer.Exit(code=1)

        elapsed = time.monotonic() - started
        if state not in {
            InvestigationStatus.COMPLETED.value,
            InvestigationStatus.COMPLETED_WITH_FINDINGS.value,
        }:
            typer.secho(f"\nFinished as {state!r}.", fg=typer.colors.RED, bold=True)
            typer.echo(f"  error: {body.get('error')}")
            raise typer.Exit(code=1)

        report_id = body.get("report_id")
        typer.secho(f"\n  completed in {elapsed:.0f}s", fg=typer.colors.GREEN)

        if not report_id:
            typer.secho("  ...but stored no report.", fg=typer.colors.RED, bold=True)
            typer.echo("  The graph ran and produced nothing to store, or storing failed.")
            typer.echo("  Check the worker log for `investigation.report_store_failed`.")
            raise typer.Exit(code=1)

        report = client.get(f"/api/v1/reports/{report_id}").json()
        typer.echo(f"  report: {report_id}")
        typer.echo(f"  title : {report.get('title')}")
        typer.echo(
            f"  status: {report.get('status')}   "
            f"confidence: {report.get('confidence')} ({report.get('confidence_band')})"
        )
        typer.echo(f"  sections: {len(report.get('sections') or [])}")

        typer.secho(
            "\nEnd to end: API -> database -> agent graph -> model -> report. All working.",
            fg=typer.colors.GREEN,
            bold=True,
        )


if __name__ == "__main__":
    app()
