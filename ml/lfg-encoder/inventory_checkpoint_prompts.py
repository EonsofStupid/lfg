#!/usr/bin/env python3
"""Inventory explicitly marked prompts in one pinned Git history, without labeling them."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

OID = re.compile(r"^[0-9a-f]{40}$")
MARKER = re.compile(r"^User prompt:\s*$", re.MULTILINE)
PROMPT = re.compile(
    r"(?:^|\n\n)User prompt:\n\n([\s\S]*?)\n\n"
    r"(Replit-Commit-Author: [^\n]+(?:\nReplit-[^\n]+)*\s*)$"
)
SESSION = re.compile(r"^Replit-Commit-Session-Id:\s*(\S+)\s*$", re.MULTILINE)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parse_history(raw: bytes, project: str, revision: str) -> tuple[dict, list[dict]]:
    parts = raw.split(b"\0")
    if parts[-1].strip(b"\r\n") or (len(parts) - 1) % 3:
        raise ValueError("incomplete Git history stream")
    records = []
    commits = marked = rejected = missing_session = 0
    rejected_commits: list[str] = []
    seen: set[str] = set()
    duplicate_texts = 0
    sessions: set[str] = set()
    for index in range(0, len(parts) - 1, 3):
        oid = parts[index].strip().decode("ascii")
        timestamp = parts[index + 1].decode("utf-8")
        message = parts[index + 2].decode("utf-8")
        if not OID.fullmatch(oid):
            raise ValueError("invalid commit identity")
        commits += 1
        if not MARKER.search(message):
            continue
        marked += 1
        match = PROMPT.search(message)
        if not match or len(MARKER.findall(message)) != 1 or not match.group(1).strip():
            rejected += 1
            rejected_commits.append(oid)
            continue
        prompt, trailer = match.groups()
        text_sha = digest(prompt)
        duplicate = text_sha in seen
        duplicate_texts += int(duplicate)
        seen.add(text_sha)
        session = SESSION.search(trailer)
        session_sha = digest(session.group(1)) if session else None
        if session_sha:
            sessions.add(session_sha)
        else:
            missing_session += 1
        records.append({
            "schema_version": "lfg.unreviewed-source-prompt.v1",
            "id": oid,
            "text": prompt,
            "text_sha256": text_sha,
            "exact_text_seen_earlier": duplicate,
            "session_sha256": session_sha,
            "labels": None,
            "verified": False,
            "split": None,
            "source": {"project": project, "kind": "git-checkpoint-user-prompt",
                       "commit": oid, "commit_time": timestamp, "history_revision": revision},
        })
    return ({"schema_version": "lfg.source-inventory.v1", "project": project,
             "history_revision": revision, "commits": commits, "marked_commits": marked,
             "extracted": len(records), "format_rejected": rejected,
             "format_rejected_commits": rejected_commits,
             "duplicate_exact_texts": duplicate_texts, "distinct_sessions": len(sessions),
             "missing_session": missing_session, "verified": 0, "split_assigned": 0}, records)


def read_history(repo: Path, revision: str, max_bytes: int) -> bytes:
    if not OID.fullmatch(revision) or max_bytes < 1:
        raise ValueError("revision must be a full lowercase SHA-1 and max bytes must be positive")
    repo = repo.resolve(strict=True)
    command = ["git", "--no-replace-objects", "-c", f"safe.directory={repo}", "-C", str(repo),
               "log", "--no-notes", "--no-show-signature", "--format=%H%x00%cI%x00%B%x00", revision]
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
        assert process.stdout is not None
        raw = process.stdout.read(max_bytes + 1)
        if len(raw) > max_bytes:
            process.kill()
            process.wait()
            raise ValueError("Git history exceeds the configured byte limit")
        if process.wait() != 0:
            raise ValueError("Git could not read the requested history revision")
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--revision", required=True, help="full immutable Git commit ID")
    parser.add_argument("--project", required=True)
    parser.add_argument("--max-history-bytes", type=int, default=100 * 1024 * 1024)
    parser.add_argument("--output", type=Path, help="new private JSONL file; contains raw prompts")
    args = parser.parse_args()
    if not args.project.strip():
        parser.error("project must not be empty")
    try:
        raw = read_history(args.repo, args.revision, args.max_history_bytes)
        report, records = parse_history(raw, args.project, args.revision)
        report["history_stream_sha256"] = hashlib.sha256(raw).hexdigest()
        report["extractor_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if args.output:
            # Exclusive creation prevents accidental replacement of a prior review artifact.
            with args.output.open("x", encoding="utf-8", newline="\n") as output:
                for record in records:
                    output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps(report, indent=2))
    except (OSError, UnicodeError, ValueError) as error:
        print(f"inventory failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
