#!/usr/bin/env python3
"""Fail when staged/tracked files contain common credential literals.

Only credential type, path, and line are reported; values are never printed.
The default scans exact Git index blobs (the next commit), avoiding a partially
staged file bypass. ``--worktree`` is available for pre-staging inspection.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path
from typing import Iterable, Tuple

ROOT = Path(__file__).resolve().parents[1]
RULES = {
    "telegram_bot_token": re.compile(rb"\b[0-9]{8,12}:[A-Za-z0-9_-]{30,}\b"),
    "telegram_chat_id_literal": re.compile(
        rb"TELEGRAM_CHAT_ID\s*[:=]\s*[\"']?-?[0-9]{6,}[\"']?"
    ),
    "tavily_api_key": re.compile(rb"\btvly-[A-Za-z0-9_-]{16,}\b", re.IGNORECASE),
    "xai_api_key": re.compile(rb"\bxai-[A-Za-z0-9_-]{32,}\b"),
    "helius_query_api_key": re.compile(
        rb"helius[^\s\"']*[?&]api-key=[A-Za-z0-9_-]{16,}", re.IGNORECASE
    ),
    "helius_api_key_literal": re.compile(
        rb"HELIUS_API_KEY\s*[:=]\s*[\"'][A-Za-z0-9_-]{16,}[\"']"
    ),
    "private_key_pem": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


def git_paths() -> list[str]:
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "-z"], cwd=ROOT
    )
    return [item.decode() for item in output.split(b"\0") if item]


def blobs(*, worktree: bool) -> Iterable[Tuple[str, bytes]]:
    for relative in git_paths():
        if worktree:
            try:
                yield relative, (ROOT / relative).read_bytes()
            except (OSError, IsADirectoryError):
                continue
        else:
            try:
                content = subprocess.check_output(
                    ["git", "show", f":{relative}"],
                    cwd=ROOT,
                    stderr=subprocess.DEVNULL,
                )
            except subprocess.CalledProcessError:
                continue
            yield relative, content


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--worktree", action="store_true",
        help="scan working-tree bytes instead of exact staged/index blobs",
    )
    args = parser.parse_args()
    findings: list[tuple[str, str, int]] = []
    for relative, content in blobs(worktree=args.worktree):
        for rule_name, pattern in RULES.items():
            for match in pattern.finditer(content):
                line = content.count(b"\n", 0, match.start()) + 1
                findings.append((rule_name, relative, line))
    if findings:
        for rule_name, path, line in findings:
            print(f"SECRET_PATTERN_DETECTED type={rule_name} path={path} line={line}")
        return 1
    target = "working-tree files" if args.worktree else "staged/index blobs"
    print(f"No configured credential patterns found in {target}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
