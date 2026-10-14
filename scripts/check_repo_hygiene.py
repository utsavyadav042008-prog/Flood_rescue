"""Git hygiene guard: block big files, raw data, model checkpoints and secrets.

Runs as a pre-commit hook (see ``scripts/install_git_hooks.py``) and by hand::

    python scripts/check_repo_hygiene.py --staged     # what you are about to commit
    python scripts/check_repo_hygiene.py --all        # every tracked file

Standard library only, so it works with ANY Python, even outside the venv.

Why these rules? A single 500 MB GeoTIFF or a leaked password in git history
is painful to remove; the hook stops it before the commit exists.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

MAX_BYTES = 5 * 1024 * 1024               # general limit per file
RELEASE_PREFIX = "models/release/"        # final model weights live here (reviewed by the team)
RELEASE_MAX_BYTES = 50 * 1024 * 1024      # GitHub rejects > 100 MB
TEXT_SCAN_MAX_BYTES = 1024 * 1024         # do not regex-scan huge files

FORBIDDEN_DIR_PREFIXES = ("data/raw/", "data/processed/", "data/reference/", "outputs/")
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".h5",
                      ".zip", ".tar", ".7z", ".nc", ".jp2", ".safe"}
RASTER_SUFFIXES = {".tif", ".tiff"}
RASTER_ALLOWED_PREFIXES = ("data/sample/", "docs/")
SECRET_FILE_NAMES = {".env", "id_rsa", "id_ed25519"}
SECRET_FILE_SUFFIXES = {".pem", ".key"}

SECRET_PATTERNS = (
    ("private key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")),
    ("AWS access key id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("hard-coded credential", re.compile(
        r"""\b(?:api[_-]?key|secret|token|passw(?:or)?d|pwd)\b\s*[:=]\s*['"][^'"\s]{8,}['"]""",
        re.IGNORECASE)),
)
ALLOW_MARKER = "hygiene: allow"          # put in a comment to silence one line


def classify_path(path: str, size: int) -> list[str]:
    """Return human-readable violations for one repo-relative path and its size."""
    p = path.replace("\\", "/")
    name = p.rsplit("/", 1)[-1]
    suffix = Path(name).suffix.lower()
    problems: list[str] = []

    if p.startswith(FORBIDDEN_DIR_PREFIXES):
        problems.append("lives in a git-ignored data/output folder (raw data never goes in git)")
    if name in SECRET_FILE_NAMES or suffix in SECRET_FILE_SUFFIXES:
        problems.append("looks like a secret/credential file")
    in_release = p.startswith(RELEASE_PREFIX)
    if suffix in FORBIDDEN_SUFFIXES and not in_release:
        problems.append(f"'{suffix}' files (checkpoints/archives/raw scenes) are not allowed; "
                        f"final weights go to {RELEASE_PREFIX} after a team decision")
    if suffix in RASTER_SUFFIXES and not p.startswith(RASTER_ALLOWED_PREFIXES):
        problems.append("GeoTIFFs are only allowed under data/sample/ or docs/")
    limit = RELEASE_MAX_BYTES if in_release else MAX_BYTES
    if size > limit:
        problems.append(f"is {size / 1_048_576:.1f} MB; limit is {limit // 1_048_576} MB")
    return problems


def find_secrets(text: str) -> list[tuple[int, str]]:
    """Return ``[(line_number, label), ...]`` for lines that look like secrets."""
    hits = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if ALLOW_MARKER in line:
            continue
        for label, pattern in SECRET_PATTERNS:
            if pattern.search(line):
                hits.append((lineno, label))
                break
    return hits


# -- git plumbing --------------------------------------------------------------
def _git(args: list[str], cwd: Path, check: bool = True) -> bytes:
    return subprocess.run(["git", *args], cwd=cwd, check=check,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


def repo_root() -> Path:
    out = subprocess.run(["git", "rev-parse", "--show-toplevel"], check=True,
                         stdout=subprocess.PIPE).stdout.decode().strip()
    return Path(out)


def staged_files(root: Path) -> list[str]:
    # --diff-filter=ACMR excludes deletions (D): removing a bad file must never be blocked.
    raw = _git(["diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"], root)
    return [n for n in raw.decode("utf-8", "replace").split("\0") if n]


def tracked_files(root: Path) -> list[str]:
    raw = _git(["ls-files", "-z"], root)
    return [n for n in raw.decode("utf-8", "replace").split("\0") if n]


def staged_size(root: Path, path: str) -> int:
    # ":path" addresses the STAGED blob, not the working-tree file, which may differ.
    return int(_git(["cat-file", "-s", f":{path}"], root).decode().strip())


def staged_bytes(root: Path, path: str) -> bytes:
    return _git(["show", f":{path}"], root)


def check(root: Path, staged: bool) -> list[str]:
    violations: list[str] = []
    for path in (staged_files(root) if staged else tracked_files(root)):
        if staged:
            size = staged_size(root, path)
        else:
            f = root / path
            if not f.exists():
                continue
            size = f.stat().st_size
        for problem in classify_path(path, size):
            violations.append(f"{path}: {problem}")
        if size <= TEXT_SCAN_MAX_BYTES:
            data = staged_bytes(root, path) if staged else (root / path).read_bytes()
            if b"\0" in data[:8000]:
                continue                                  # binary, skip the secret scan
            for lineno, label in find_secrets(data.decode("utf-8", "replace")):
                violations.append(f"{path}:{lineno}: possible {label} "
                                  f"(add '# {ALLOW_MARKER}' if it is a false alarm)")
    return violations


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--staged", action="store_true", help="check files staged for commit")
    mode.add_argument("--all", action="store_true", help="check all tracked files")
    args = ap.parse_args(argv)

    root = repo_root()
    violations = check(root, staged=args.staged)
    if not violations:
        print("hygiene: OK")
        return 0
    print("hygiene: commit blocked, please fix:\n")
    for v in violations:
        print(f"  - {v}")
    print("\nUnstage a file with:  git restore --staged <file>")
    print("(Emergency bypass: git commit --no-verify. Do not use it for secrets.)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
