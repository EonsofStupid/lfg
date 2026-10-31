#!/usr/bin/env python3
"""Refresh SBOM policy baseline package count with explicit operator confirmation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CONFIRM_TOKEN = "I_ACKNOWLEDGE_BASELINE_UPDATE"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON at {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"expected JSON object at {path}")
    return data


def _count_packages(sbom: dict[str, Any]) -> int:
    packages = sbom.get("packages")
    if isinstance(packages, list):
        return len(packages)
    artifacts = sbom.get("artifacts")
    if isinstance(artifacts, list):
        return len(artifacts)
    raise SystemExit("SBOM missing recognizable package list ('packages' or 'artifacts')")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh baselinePackageCount in security/sbom-policy.json"
    )
    parser.add_argument("--sbom", type=Path, required=True, help="SBOM JSON source")
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("security") / "sbom-policy.json",
        help="SBOM policy file to update",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Persist updates to the policy file (otherwise dry-run only)",
    )
    parser.add_argument(
        "--confirm",
        default="",
        help=f"Required with --write. Must equal: {CONFIRM_TOKEN}",
    )
    parser.add_argument(
        "--reason",
        default="",
        help="Human-readable reason for baseline update (required with --write)",
    )
    args = parser.parse_args()

    sbom = _load_json(args.sbom)
    policy = _load_json(args.policy)
    if "baselinePackageCount" not in policy:
        raise SystemExit("policy missing baselinePackageCount")

    current_baseline = int(policy["baselinePackageCount"])
    new_baseline = _count_packages(sbom)
    delta = new_baseline - current_baseline

    report = {
        "schemaVersion": "lfg.sbom-baseline-refresh-report.v1",
        "policy": str(args.policy),
        "sbom": str(args.sbom),
        "currentBaselinePackageCount": current_baseline,
        "proposedBaselinePackageCount": new_baseline,
        "delta": delta,
        "writeRequested": bool(args.write),
    }
    print(json.dumps(report, indent=2))

    if not args.write:
        return

    if args.confirm != CONFIRM_TOKEN:
        raise SystemExit(
            f"--write requires --confirm {CONFIRM_TOKEN}"
        )
    if not args.reason.strip():
        raise SystemExit("--write requires a non-empty --reason")

    policy["baselinePackageCount"] = new_baseline
    policy["baselineUpdatedAt"] = datetime.now(timezone.utc).isoformat()
    policy["baselineUpdateReason"] = args.reason.strip()
    args.policy.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    print(f"Updated {args.policy} baselinePackageCount -> {new_baseline}")


if __name__ == "__main__":
    main()
