"""Mint a local development JWT.

There is no other way in. `backend/api/deps.py` supports two credentials: a JWT
signed with `SECRET_KEY`, and an API key -- and the API-key branch is
deliberately dead, because verifying one needs a bcrypt hash in an `api_keys`
table that does not exist. So every authenticated call in local development is
made with a token from here, and without this script the first thing anyone
meets is an undiagnosable 401.

Signed with the same `SECRET_KEY` the API verifies against, so a token stops
working the moment that value changes -- which is correct, and worth knowing
before you spend ten minutes wondering why a token from yesterday is refused.

Local development only. This mints an admin token against a secret sitting in a
`.env` file; it is not an issuer and must never become one.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import typer

from backend.core.config import get_settings
from backend.core.security import encode_jws

app = typer.Typer(add_completion=False, help="Print a local development JWT.")


@app.command()
def main(
    subject: str = typer.Option("local-dev", help="The `sub` claim."),
    tenant: str = typer.Option("local", help="The tenant this token can read."),
    role: str = typer.Option("admin", help="Expands via ROLE_SCOPES into scopes."),
    hours: int = typer.Option(24, help="Lifetime in hours."),
) -> None:
    """Print a token. Nothing else, so it can be captured directly into a variable."""
    settings = get_settings()
    now = int(time.time())
    typer.echo(
        encode_jws(
            {
                "sub": subject,
                "tenant": tenant,
                "role": role,
                "iat": now,
                "exp": now + hours * 3600,
            },
            secret=settings.security.secret_key.get_secret_value(),
        )
    )


if __name__ == "__main__":
    app()
