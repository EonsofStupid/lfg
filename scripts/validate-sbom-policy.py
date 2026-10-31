#!/usr/bin/env python3
"""Validate SPDX SBOM output against LFG supply-chain drift policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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


def _normalize_namespace(value: str | None) -> str:
    if value is None:
        return "<none>"
    value = str(value).strip()
    return value if value else "<none>"


def _parse_purl(purl: str) -> tuple[str, str]:
    if not purl.startswith("pkg:"):
        return ("unknown", "<none>")
    body = purl[4:]
    body = body.split("?", 1)[0]
    body = body.split("#", 1)[0]
    body = body.split("@", 1)[0]
    parts = body.split("/")
    if not parts:
        return ("unknown", "<none>")
    pkg_type = parts[0].strip() or "unknown"
    if len(parts) >= 3:
        namespace = _normalize_namespace(parts[1])
    else:
        namespace = "<none>"
    return (pkg_type, namespace)


def _artifact_should_be_ignored(
    artifact: dict[str, Any], ignored_location_patterns: list[str]
) -> bool:
    if not ignored_location_patterns:
        return False
    locations = artifact.get("locations")
    if not isinstance(locations, list):
        return False
    for location in locations:
        if not isinstance(location, dict):
            continue
        path = location.get("path")
        if not isinstance(path, str):
            continue
        if any(pattern in path for pattern in ignored_location_patterns):
            return True
    return False


def _collect_package_kinds(
    sbom: dict[str, Any], ignored_location_patterns: list[str]
) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    artifacts = sbom.get("artifacts")
    if isinstance(artifacts, list):
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                continue
            if _artifact_should_be_ignored(artifact, ignored_location_patterns):
                continue
            pkg_type = str(artifact.get("type", "unknown")).strip() or "unknown"
            namespace = _normalize_namespace(artifact.get("namespace"))
            found.append((pkg_type, namespace))
    packages = sbom.get("packages")
    if isinstance(packages, list):
        for package in packages:
            if not isinstance(package, dict):
                continue
            purl = ""
            refs = package.get("externalRefs")
            if isinstance(refs, list):
                for ref in refs:
                    if not isinstance(ref, dict):
                        continue
                    locator = ref.get("referenceLocator")
                    if isinstance(locator, str) and locator.startswith("pkg:"):
                        purl = locator
                        break
            if not purl:
                continue
            found.append(_parse_purl(purl))
    return found


def _validate_schema(policy: dict[str, Any]) -> None:
    required = {
        "schemaVersion",
        "baselinePackageCount",
        "maxPackageGrowthPercent",
        "maxPackageGrowthAbsolute",
        "allowedPackageTypes",
        "allowedNamespacesByType",
    }
    missing = required - policy.keys()
    if missing:
        raise SystemExit(f"policy missing required fields: {sorted(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate SBOM against drift policy")
    parser.add_argument("--sbom", type=Path, required=True, help="Path to SPDX JSON SBOM")
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("security") / "sbom-policy.json",
        help="Path to SBOM policy JSON",
    )
    args = parser.parse_args()

    sbom = _load_json(args.sbom)
    policy = _load_json(args.policy)
    _validate_schema(policy)

    baseline = int(policy["baselinePackageCount"])
    max_growth_percent = float(policy["maxPackageGrowthPercent"])
    max_growth_absolute = int(policy["maxPackageGrowthAbsolute"])
    allowed_types = {str(value) for value in policy["allowedPackageTypes"]}
    allowed_namespaces_by_type = {
        str(key): {str(value) for value in values}
        for key, values in policy["allowedNamespacesByType"].items()
    }
    ignored_location_patterns = [
        str(value) for value in policy.get("ignoredLocationPatterns", [])
    ]

    package_count = _count_packages(sbom)
    growth = package_count - baseline
    allowed_growth_percent = int(round((baseline * max_growth_percent) / 100.0))
    allowed_growth = max(max_growth_absolute, allowed_growth_percent)
    kinds = _collect_package_kinds(sbom, ignored_location_patterns)
    observed_types = sorted({pkg_type for pkg_type, _ in kinds})
    unexpected_types = sorted([pkg_type for pkg_type in observed_types if pkg_type not in allowed_types])

    unexpected_namespaces: dict[str, list[str]] = {}
    for pkg_type, namespace in kinds:
        allowed_namespaces = allowed_namespaces_by_type.get(pkg_type)
        if allowed_namespaces is None:
            continue
        if namespace not in allowed_namespaces:
            unexpected_namespaces.setdefault(pkg_type, [])
            if namespace not in unexpected_namespaces[pkg_type]:
                unexpected_namespaces[pkg_type].append(namespace)

    unexpected_namespaces = {
        key: sorted(values) for key, values in sorted(unexpected_namespaces.items())
    }

    report = {
        "schemaVersion": "lfg.sbom-policy-report.v1",
        "policy": str(args.policy),
        "sbom": str(args.sbom),
        "baselinePackageCount": baseline,
        "currentPackageCount": package_count,
        "packageGrowth": growth,
        "maxAllowedGrowth": allowed_growth,
        "allowedPackageTypes": sorted(allowed_types),
        "observedPackageTypes": observed_types,
        "unexpectedPackageTypes": unexpected_types,
        "unexpectedNamespacesByType": unexpected_namespaces,
        "ignoredLocationPatterns": ignored_location_patterns,
    }
    print(json.dumps(report, indent=2))

    if growth > allowed_growth:
        raise SystemExit(
            f"SBOM package drift exceeds policy: growth={growth}, allowed={allowed_growth}"
        )
    if unexpected_types:
        raise SystemExit(
            f"SBOM package type policy violation: unexpected types {unexpected_types}"
        )
    if unexpected_namespaces:
        raise SystemExit(
            f"SBOM namespace policy violation: {unexpected_namespaces}"
        )


if __name__ == "__main__":
    main()
