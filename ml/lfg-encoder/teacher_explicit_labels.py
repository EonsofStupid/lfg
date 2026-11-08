#!/usr/bin/env python3
"""Pure schema and parser for explicit, source-bound teacher label claims.

This module validates a wire contract only. It does not collect examples,
establish truth, or train a model. Missing labels always remain unknown.
"""
from __future__ import annotations

import copy
import itertools
import json
import re
from typing import Any, Iterable

SCHEMA_VERSION = "lfg.teacher-explicit-labels.v1"
DECISIONS = ("positive", "negative", "unknown")
MAX_SELECTIONS = 512
MAX_RESPONSE_BYTES = 1024 * 1024
MAX_EXTENSION_BYTES = 64 * 1024
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ContractError(ValueError):
    """A sanitized contract failure; messages never include response values."""


class ParsedResponse(dict[str, Any]):
    """Sanitized public result with an explicit private extension channel.

    The dict mapping, repr/str, and default JSON encoding contain only the
    public summary. Call `private_extensions()` deliberately to retrieve a
    detached copy of accepted extension values.
    """

    __slots__ = ("__private_extensions",)

    def __init__(self, public: dict[str, Any], private_extensions: dict[str, Any] | None = None):
        super().__init__(public)
        self.__private_extensions = copy.deepcopy(private_extensions or {})

    def private_extensions(self) -> dict[str, Any]:
        """Return a detached copy of sensitive, inert teacher extension data."""
        return copy.deepcopy(self.__private_extensions)

    def __reduce_ex__(self, _protocol: int) -> Any:
        raise TypeError("ParsedResponse serialization is disabled; use a sanitized mapping")

    def __reduce__(self) -> Any:
        raise TypeError("ParsedResponse serialization is disabled; use a sanitized mapping")


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise ContractError("value is not bounded JSON data") from exc


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise ContractError("non-finite JSON number")


def _validate_selection(selected: Iterable[dict[str, str]], *, max_selections: int = MAX_SELECTIONS
                        ) -> list[dict[str, str]]:
    if type(max_selections) is not int or not 1 <= max_selections <= MAX_SELECTIONS:
        raise ContractError("selection limit is invalid")
    try:
        iterator = iter(selected)
        items = list(itertools.islice(iterator, max_selections + 1))
    except (TypeError, ValueError, RecursionError) as exc:
        raise ContractError("selected identities are invalid") from exc
    if not items or len(items) > max_selections:
        raise ContractError("selected identity count is outside bounds")
    normalized: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        if not isinstance(item, dict) or set(item) != {"field", "label"}:
            raise ContractError("selected identity must contain field and label")
        field, label = item["field"], item["label"]
        if (not isinstance(field, str) or not field.strip() or
                not isinstance(label, str) or not label.strip()):
            raise ContractError("selected identity values must be nonempty strings")
        identity = (field, label)
        if identity in seen:
            raise ContractError("selected identities contain a duplicate")
        seen.add(identity)
        normalized.append({"field": field, "label": label})
    return normalized


def build_response_schema(selected: Iterable[dict[str, str]], taxonomy_sha256: str, *,
                          max_selections: int = MAX_SELECTIONS,
                          max_schema_bytes: int = MAX_RESPONSE_BYTES) -> dict[str, Any]:
    """Return a bounded JSON Schema for decisions over the pinned identities.

    The schema keeps field/label pairs atomic through `oneOf`; parsing still
    validates every response because a teacher can return schema-invalid data.
    """
    columns = _validate_selection(selected, max_selections=max_selections)
    if not isinstance(taxonomy_sha256, str) or not SHA256_RE.fullmatch(taxonomy_sha256):
        raise ContractError("taxonomy digest is invalid")
    if type(max_schema_bytes) is not int or not 1 <= max_schema_bytes <= MAX_RESPONSE_BYTES:
        raise ContractError("schema size limit is invalid")
    decision_schema = {
        "oneOf": [
            {
                "type": "object",
                "properties": {
                    "field": {"const": item["field"]},
                    "label": {"const": item["label"]},
                    "value": {"enum": list(DECISIONS)},
                },
                "required": ["field", "label", "value"],
                "additionalProperties": False,
            }
            for item in columns
        ]
    }
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION},
            "taxonomy_sha256": {"const": taxonomy_sha256},
            "decisions": {"type": "array", "items": decision_schema,
                          "maxItems": len(columns)},
            "abstain": {"type": "boolean"},
            # Extension data is inert and retained. It can never override decisions.
            "extensions": {"type": "object", "additionalProperties": True},
        },
        "required": ["schema_version", "taxonomy_sha256", "decisions", "abstain"],
        "additionalProperties": False,
    }
    if len(_canonical_bytes(schema)) > max_schema_bytes:
        raise ContractError("generated schema exceeds configured bound")
    return schema


def parse_response(raw: str | bytes, selected: Iterable[dict[str, str]], taxonomy_sha256: str, *,
                   outcome: str = "complete", max_response_bytes: int = MAX_RESPONSE_BYTES,
                   max_extension_bytes: int = MAX_EXTENSION_BYTES,
                   max_selections: int = MAX_SELECTIONS) -> ParsedResponse:
    """Parse one bounded response into ordered states, filling omissions unknown.

    Non-answer outcomes never carry asserted decisions. Invalid JSON and
    contract violations fail closed with a generic `invalid` status. No raw
    prompt, response or exception content is copied to output.
    """
    columns = _validate_selection(selected, max_selections=max_selections)
    if not isinstance(taxonomy_sha256, str) or not SHA256_RE.fullmatch(taxonomy_sha256):
        raise ContractError("taxonomy digest is invalid")
    if type(max_response_bytes) is not int or not 1 <= max_response_bytes <= MAX_RESPONSE_BYTES or \
            type(max_extension_bytes) is not int or not 0 <= max_extension_bytes <= MAX_EXTENSION_BYTES:
        raise ContractError("response size limit is invalid")
    if not isinstance(outcome, str) or outcome not in {"complete", "refusal", "truncated", "incomplete", "invalid"}:
        raise ContractError("response outcome is unsupported")

    def result(status: str, values: dict[tuple[str, str], str] | None = None,
               private_extensions: dict[str, Any] | None = None,
               extensions_present: bool = False) -> ParsedResponse:
        values = values or {}
        public = {
            "schema_version": SCHEMA_VERSION,
            "taxonomy_sha256": taxonomy_sha256,
            "status": status,
            "decisions": [
                {"field": item["field"], "label": item["label"],
                 "value": values.get((item["field"], item["label"]), "unknown")}
                for item in columns
            ],
            "extensions_present": extensions_present,
        }
        return ParsedResponse(public, private_extensions)

    if outcome != "complete":
        return result(outcome)
    try:
        if isinstance(raw, str):
            if len(raw) > max_response_bytes:
                return result("truncated")
            payload = raw.encode("utf-8")
        elif isinstance(raw, bytes):
            payload = raw
        else:
            return result("invalid")
        if len(payload) > max_response_bytes:
            return result("truncated")
        value = json.loads(payload.decode("utf-8", errors="strict"),
                           object_pairs_hook=_pairs_no_duplicates,
                           parse_constant=_reject_constant)
        if not isinstance(value, dict):
            return result("invalid")
        allowed = {"schema_version", "taxonomy_sha256", "decisions", "abstain", "extensions"}
        if set(value) - allowed:
            return result("invalid")
        if value.get("schema_version") != SCHEMA_VERSION or \
                value.get("taxonomy_sha256") != taxonomy_sha256:
            return result("invalid")
        abstain = value.get("abstain")
        if type(abstain) is not bool:
            return result("invalid")
        extensions = value.get("extensions", {})
        if not isinstance(extensions, dict) or len(_canonical_bytes(extensions)) > max_extension_bytes:
            return result("invalid")
        decisions = value.get("decisions")
        if not isinstance(decisions, list) or len(decisions) > len(columns):
            return result("invalid")
        expected = {(item["field"], item["label"]) for item in columns}
        parsed: dict[tuple[str, str], str] = {}
        for decision in decisions:
            if not isinstance(decision, dict) or set(decision) != {"field", "label", "value"}:
                return result("invalid")
            identity = (decision["field"], decision["label"])
            state = decision["value"]
            if identity not in expected or identity in parsed or state not in DECISIONS:
                return result("invalid")
            parsed[identity] = state
        if abstain and any(state in {"positive", "negative"} for state in parsed.values()):
            return result("invalid")
        if abstain:
            return result("abstain", private_extensions=extensions,
                          extensions_present="extensions" in value)
        return result("answered", parsed, private_extensions=extensions,
                      extensions_present="extensions" in value)
    except (UnicodeError, json.JSONDecodeError, ContractError, TypeError, ValueError,
            RecursionError, OverflowError):
        return result("invalid")
