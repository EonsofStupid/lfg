#!/usr/bin/env python3
"""Teacher-trace validation and SFT dataset builder for LFG code distillation."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_TRACES = ROOT / "dataset" / "teacher-traces.v1.jsonl"
DEFAULT_OUTPUT = ROOT / "dataset" / "sft-train.v1.jsonl"


@dataclass(frozen=True)
class TeacherTrace:
    trace_id: str
    prompt: str
    response: str
    language: str
    difficulty: str
    checks_passed: bool
    source_model: str
    tags: tuple[str, ...]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"trace file not found: {path}")
    rows: list[dict[str, Any]] = []
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {idx} is not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"line {idx} must be a JSON object")
        rows.append(parsed)
    return rows


def _parse_trace(row: dict[str, Any], index: int, seen_ids: set[str]) -> TeacherTrace:
    required = {
        "traceId",
        "prompt",
        "response",
        "language",
        "difficulty",
        "checksPassed",
        "sourceModel",
        "tags",
    }
    missing = required - row.keys()
    if missing:
        raise ValueError(f"record {index} missing required fields: {sorted(missing)}")

    trace_id = str(row["traceId"]).strip()
    prompt = str(row["prompt"]).strip()
    response = str(row["response"]).strip()
    language = str(row["language"]).strip()
    difficulty = str(row["difficulty"]).strip()
    source_model = str(row["sourceModel"]).strip()

    if not trace_id:
        raise ValueError(f"record {index} has empty traceId")
    if trace_id in seen_ids:
        raise ValueError(f"duplicate traceId: {trace_id}")
    seen_ids.add(trace_id)

    if not prompt:
        raise ValueError(f"record {trace_id} has empty prompt")
    if not response:
        raise ValueError(f"record {trace_id} has empty response")
    if language not in {"rust", "typescript", "python", "go", "javascript", "shell", "mixed"}:
        raise ValueError(f"record {trace_id} has unsupported language: {language}")
    if difficulty not in {"easy", "medium", "hard"}:
        raise ValueError(f"record {trace_id} has invalid difficulty: {difficulty}")
    if not isinstance(row["checksPassed"], bool):
        raise ValueError(f"record {trace_id} checksPassed must be boolean")
    if not source_model:
        raise ValueError(f"record {trace_id} has empty sourceModel")
    if not isinstance(row["tags"], list) or any(not str(tag).strip() for tag in row["tags"]):
        raise ValueError(f"record {trace_id} tags must be a non-empty string list")

    return TeacherTrace(
        trace_id=trace_id,
        prompt=prompt,
        response=response,
        language=language,
        difficulty=difficulty,
        checks_passed=row["checksPassed"],
        source_model=source_model,
        tags=tuple(str(tag).strip() for tag in row["tags"]),
    )


def load_traces(path: Path) -> list[TeacherTrace]:
    rows = _read_jsonl(path)
    seen_ids: set[str] = set()
    traces = [_parse_trace(row, i + 1, seen_ids) for i, row in enumerate(rows)]
    return traces


def command_validate(args: argparse.Namespace) -> None:
    traces = load_traces(args.traces)
    verified = [trace for trace in traces if trace.checks_passed]
    difficulty = {"easy": 0, "medium": 0, "hard": 0}
    langs: dict[str, int] = {}
    for trace in traces:
        difficulty[trace.difficulty] += 1
        langs[trace.language] = langs.get(trace.language, 0) + 1
    report = {
        "schemaVersion": "lfg.distill-trace-report.v1",
        "traces": len(traces),
        "verifiedChecksPassed": len(verified),
        "coverageByDifficulty": difficulty,
        "coverageByLanguage": dict(sorted(langs.items())),
        "sourceSha256": hashlib.sha256(args.traces.read_bytes()).hexdigest(),
    }
    print(json.dumps(report, indent=2))


def _to_sft_row(trace: TeacherTrace) -> dict[str, Any]:
    return {
        "id": trace.trace_id,
        "messages": [
            {"role": "system", "content": "You are LFG-Code, a precise software engineering assistant."},
            {"role": "user", "content": trace.prompt},
            {"role": "assistant", "content": trace.response},
        ],
        "metadata": {
            "language": trace.language,
            "difficulty": trace.difficulty,
            "sourceModel": trace.source_model,
            "checksPassed": trace.checks_passed,
            "tags": list(trace.tags),
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True))
            fh.write("\n")


def command_build_sft(args: argparse.Namespace) -> None:
    traces = load_traces(args.traces)
    selected = [trace for trace in traces if trace.checks_passed or args.include_failed]
    if not selected:
        raise SystemExit("no traces selected; pass --include-failed if needed")

    random.Random(args.seed).shuffle(selected)
    split_index = int(len(selected) * args.train_ratio)
    if split_index <= 0 or split_index >= len(selected):
        raise SystemExit(f"train_ratio {args.train_ratio} yields empty split; keep between 0 and 1 with enough samples")

    train_rows = [_to_sft_row(trace) for trace in selected[:split_index]]
    eval_rows = [_to_sft_row(trace) for trace in selected[split_index:]]

    _write_jsonl(args.train_output, train_rows)
    _write_jsonl(args.eval_output, eval_rows)

    summary = {
        "schemaVersion": "lfg.distill-sft-build.v1",
        "sourceTraces": len(traces),
        "selectedTraces": len(selected),
        "includeFailed": bool(args.include_failed),
        "trainRows": len(train_rows),
        "evalRows": len(eval_rows),
        "trainOutput": str(args.train_output),
        "evalOutput": str(args.eval_output),
    }
    print(json.dumps(summary, indent=2))


def command_init_template(args: argparse.Namespace) -> None:
    if args.output.exists() and not args.force:
        raise SystemExit(f"{args.output} already exists; pass --force to overwrite")
    template = [
        {
            "traceId": "trace-0001",
            "prompt": "Implement a Rust function that sums even numbers from a slice.",
            "response": "```rust\npub fn sum_even(values: &[i64]) -> i64 {\n    values.iter().copied().filter(|n| n % 2 == 0).sum()\n}\n```",
            "language": "rust",
            "difficulty": "easy",
            "checksPassed": True,
            "sourceModel": "teacher-7b",
            "tags": ["implementation", "rust", "pure-function"],
        }
    ]
    _write_jsonl(args.output, template)
    print(f"wrote template traces to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LFG distillation dataset utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate-traces", help="validate teacher traces and print a report")
    validate.add_argument("--traces", type=Path, default=DEFAULT_TRACES)
    validate.set_defaults(func=command_validate)

    build = sub.add_parser("build-sft", help="build train/eval SFT JSONL datasets")
    build.add_argument("--traces", type=Path, default=DEFAULT_TRACES)
    build.add_argument("--train-output", type=Path, default=DEFAULT_OUTPUT)
    build.add_argument("--eval-output", type=Path, default=ROOT / "dataset" / "sft-eval.v1.jsonl")
    build.add_argument("--train-ratio", type=float, default=0.9)
    build.add_argument("--seed", type=int, default=731)
    build.add_argument("--include-failed", action="store_true")
    build.set_defaults(func=command_build_sft)

    template = sub.add_parser("init-template", help="write a starter teacher-traces JSONL file")
    template.add_argument("--output", type=Path, default=DEFAULT_TRACES)
    template.add_argument("--force", action="store_true")
    template.set_defaults(func=command_init_template)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
