"""`python -m cli` -- so the CLI runs without the package being installed.

`pyproject.toml` deliberately has no `[project]` table: OmniSense runs from the
repository root rather than being installed, so there is no console-script entry
point to hang an `omnisense` command on. This module is what makes
`python -m cli` work, and `bin/omnisense` is a two-line wrapper around it that
gives the documented command without reversing that decision.
"""

from cli.main import main

main()
