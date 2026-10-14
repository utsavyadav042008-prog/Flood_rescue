"""One-time setup: activate the team's shared git hooks.

    python scripts/install_git_hooks.py

Hooks live in ``.githooks/`` INSIDE the repo (so they are versioned and
reviewed like code).  Git only runs hooks from ``.git/hooks`` by default,
which is not versioned; the trick is ``git config core.hooksPath .githooks``,
which points git at the tracked folder.  Every teammate runs this once.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

PRE_COMMIT = """#!/bin/sh
# FloodResQ pre-commit hook (installed by scripts/install_git_hooks.py).
# Blocks big files, raw data, model checkpoints and secrets.
if command -v python3 >/dev/null 2>&1; then PY=python3
elif command -v python >/dev/null 2>&1; then PY=python
else PY="py -3"; fi
exec $PY scripts/check_repo_hygiene.py --staged
"""


def write_hook_file(repo: Path = REPO_ROOT) -> Path:
    hook = repo / ".githooks" / "pre-commit"
    hook.parent.mkdir(exist_ok=True)
    # newline="\n": a CRLF shebang line makes the hook fail on Linux/macOS/Git Bash.
    hook.write_text(PRE_COMMIT, encoding="utf-8", newline="\n")
    hook.chmod(hook.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return hook


def main() -> int:
    hook = write_hook_file()
    try:
        subprocess.run(["git", "config", "core.hooksPath", ".githooks"], cwd=REPO_ROOT, check=True)
        # Record the executable bit in the index so Linux/macOS teammates get it too.
        subprocess.run(["git", "update-index", "--add", "--chmod=+x", str(hook.relative_to(REPO_ROOT))],
                       cwd=REPO_ROOT, check=False)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"Could not configure git: {exc}\nIs this folder a git repository? (git init)")
        return 1
    print(f"Hooks active: {hook.relative_to(REPO_ROOT)}  (core.hooksPath=.githooks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
