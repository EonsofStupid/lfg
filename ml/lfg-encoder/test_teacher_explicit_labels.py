import contextlib
import io
import json
import pickle
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import teacher_explicit_labels as contract


TAXONOMY = "a" * 64
SELECTED = [
    {"field": "intent", "label": "build"},
    {"field": "effort", "label": "high"},
    {"field": "models", "label": "local:small"},
]


def response(decisions=None, **extra):
    return json.dumps({
        "schema_version": contract.SCHEMA_VERSION,
        "taxonomy_sha256": TAXONOMY,
        "decisions": decisions or [],
        "abstain": False,
        **extra,
    })


class ExplicitLabelContractTests(unittest.TestCase):
    def test_schema_round_trip_is_deterministic_and_preserves_pair_identity(self):
        schema = contract.build_response_schema(SELECTED, TAXONOMY)
        self.assertEqual(schema, contract.build_response_schema(SELECTED, TAXONOMY))
        choices = schema["properties"]["decisions"]["items"]["oneOf"]
        self.assertEqual(len(choices), len(SELECTED))
        self.assertEqual(choices[2]["properties"]["field"]["const"], "models")
        self.assertEqual(choices[2]["properties"]["label"]["const"], "local:small")

    def test_positive_negative_unknown_are_distinct_and_ordered(self):
        parsed = contract.parse_response(response([
            {"field": "intent", "label": "build", "value": "positive"},
            {"field": "effort", "label": "high", "value": "negative"},
            {"field": "models", "label": "local:small", "value": "unknown"},
        ]), SELECTED, TAXONOMY)
        self.assertEqual(parsed["status"], "answered")
        self.assertEqual([item["value"] for item in parsed["decisions"]],
                         ["positive", "negative", "unknown"])

    def test_omitted_labels_are_unknown_not_negative(self):
        parsed = contract.parse_response(response([
            {"field": "intent", "label": "build", "value": "positive"},
        ]), SELECTED, TAXONOMY)
        self.assertEqual([item["value"] for item in parsed["decisions"]],
                         ["positive", "unknown", "unknown"])

    def test_empty_decision_list_is_all_unknown(self):
        parsed = contract.parse_response(response(), SELECTED, TAXONOMY)
        self.assertEqual([item["value"] for item in parsed["decisions"]], ["unknown"] * 3)

    def test_extensions_are_inert_and_retained(self):
        decisions = [{"field": "intent", "label": "build", "value": "positive"}]
        parsed = contract.parse_response(response(decisions, extensions={"trace_id": "opaque-1"}),
                                         SELECTED, TAXONOMY)
        without_extensions = contract.parse_response(response(decisions), SELECTED, TAXONOMY)
        self.assertTrue(parsed["extensions_present"])
        self.assertEqual(parsed.private_extensions(), {"trace_id": "opaque-1"})
        self.assertEqual(parsed["decisions"], without_extensions["decisions"])

    def test_absent_extensions_are_valid_at_zero_and_one_byte_budgets(self):
        for limit in (0, 1):
            for abstain in (False, True):
                with self.subTest(limit=limit, abstain=abstain):
                    raw = response([], abstain=True) if abstain else response([])
                    parsed = contract.parse_response(raw, SELECTED, TAXONOMY,
                                                     max_extension_bytes=limit)
                    self.assertEqual(parsed["status"], "abstain" if abstain else "answered")
                    self.assertFalse(parsed["extensions_present"])
                    self.assertEqual(parsed.private_extensions(), {})
            present_empty = response([], extensions={})
            parsed = contract.parse_response(present_empty, SELECTED, TAXONOMY,
                                             max_extension_bytes=limit)
            self.assertEqual(parsed["status"], "invalid")
            self.assertFalse(parsed["extensions_present"])

    def test_present_empty_extensions_require_two_bytes_and_remain_distinct(self):
        absent = contract.parse_response(response([]), SELECTED, TAXONOMY,
                                         max_extension_bytes=2)
        present = contract.parse_response(response([], extensions={}), SELECTED, TAXONOMY,
                                          max_extension_bytes=2)
        self.assertEqual(absent["status"], "answered")
        self.assertFalse(absent["extensions_present"])
        self.assertEqual(present["status"], "answered")
        self.assertTrue(present["extensions_present"])
        self.assertEqual(present.private_extensions(), {})

    def test_nested_extension_is_private_by_default_and_retrieval_is_detached(self):
        sentinel = "NESTED-PRIVATE-SENTINEL"
        extensions = {"metadata": {"items": ["safe-name", {"opaque": sentinel}]}}
        parsed = contract.parse_response(response([], extensions=extensions), SELECTED, TAXONOMY)
        public = json.dumps(parsed, sort_keys=True)
        self.assertNotIn(sentinel, public)
        self.assertNotIn(sentinel, repr(parsed))
        self.assertNotIn(sentinel, str(parsed))
        self.assertNotIn(sentinel, repr({"summary": parsed}))
        self.assertNotIn(sentinel, json.dumps({"summary": parsed}))
        self.assertEqual(parsed.private_extensions(), extensions)
        retrieved = parsed.private_extensions()
        retrieved["metadata"]["items"].append("caller mutation")
        self.assertEqual(parsed.private_extensions(), extensions)
        with self.assertRaises(TypeError):
            vars(parsed)

    def test_parser_diagnostics_and_contract_errors_do_not_echo_private_values(self):
        sentinel = "NEVER-IN-ERROR-OR-LOG"
        raw = response([], extensions={"private": {"note": sentinel}})
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            parsed = contract.parse_response(raw, SELECTED, TAXONOMY)
        self.assertNotIn(sentinel, repr(parsed))
        self.assertNotIn(sentinel, stdout.getvalue() + stderr.getvalue())
        bad_selection = [{"field": "", "label": "x"}]
        with self.assertRaises(contract.ContractError) as raised:
            contract.parse_response(raw, bad_selection, TAXONOMY)
        self.assertNotIn(sentinel, str(raised.exception))

    def test_pickle_protocols_fail_closed_without_private_extension_echo(self):
        sentinel = "PICKLE-PRIVATE-SENTINEL"
        extensions = {"nested": ["value", {"secret": sentinel}]}
        parsed = contract.parse_response(response([], extensions=extensions), SELECTED, TAXONOMY)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol), self.assertRaises(TypeError) as raised:
                pickle.dumps(parsed, protocol=protocol)
            self.assertNotIn(sentinel, str(raised.exception))
            self.assertNotIn(sentinel, repr(raised.exception))
        with self.assertRaises(TypeError) as direct_reduce:
            parsed.__reduce__()
        self.assertNotIn(sentinel, str(direct_reduce.exception))
        self.assertEqual(parsed.private_extensions(), extensions)
        sanitized = dict(parsed)
        self.assertNotIn(sentinel, pickle.dumps(sanitized).decode("latin1"))

    def test_abstention_preserves_private_extensions_without_public_echo(self):
        sentinel = "ABSTAINED-PRIVATE-SENTINEL"
        extensions = {"note": {"opaque": sentinel}}
        parsed = contract.parse_response(response([], abstain=True, extensions=extensions),
                                         SELECTED, TAXONOMY)
        self.assertEqual(parsed["status"], "abstain")
        self.assertTrue(parsed["extensions_present"])
        self.assertEqual(parsed.private_extensions(), extensions)
        self.assertNotIn(sentinel, json.dumps(parsed))
        self.assertNotIn(sentinel, repr(parsed))

    def test_nonanswer_and_invalid_paths_do_not_retain_private_extensions(self):
        sentinel = "DO-NOT-RETAIN-ON-FAILURE"
        raw = response([], extensions={"opaque": sentinel})
        for outcome in ("refusal", "incomplete", "truncated", "invalid"):
            parsed = contract.parse_response(raw, SELECTED, TAXONOMY, outcome=outcome)
            self.assertFalse(parsed["extensions_present"])
            self.assertEqual(parsed.private_extensions(), {})
            self.assertNotIn(sentinel, json.dumps(parsed))
        malformed = contract.parse_response(raw.replace('"abstain": false', '"abstain": "bad"'),
                                            SELECTED, TAXONOMY)
        self.assertEqual(malformed["status"], "invalid")
        self.assertFalse(malformed["extensions_present"])
        self.assertEqual(malformed.private_extensions(), {})

    def test_unknown_top_level_field_fails_closed_without_echo(self):
        sentinel = "PRIVATE-SENTINEL-NOT-TO-ECHO"
        parsed = contract.parse_response(response([], unexpected=sentinel), SELECTED, TAXONOMY)
        self.assertEqual(parsed["status"], "invalid")
        self.assertNotIn(sentinel, repr(parsed))

    def test_refusal_incomplete_truncated_and_invalid_are_unknown(self):
        raw = response([{"field": "intent", "label": "build", "value": "positive"}])
        for status in ("refusal", "incomplete", "truncated", "invalid"):
            with self.subTest(status=status):
                parsed = contract.parse_response(raw, SELECTED, TAXONOMY, outcome=status)
                self.assertEqual(parsed["status"], status)
                self.assertEqual([item["value"] for item in parsed["decisions"]], ["unknown"] * 3)

    def test_oversize_response_is_truncated_and_unknown(self):
        parsed = contract.parse_response(response(), SELECTED, TAXONOMY, max_response_bytes=2)
        self.assertEqual(parsed["status"], "truncated")
        self.assertEqual([item["value"] for item in parsed["decisions"]], ["unknown"] * 3)

    def test_malformed_and_truncated_json_fail_closed_without_response_echo(self):
        sentinel = "PRIVATE-RAW-RESPONSE"
        for raw in ("{" + sentinel, "\"unterminated", b"\xff"):
            with self.subTest(raw=type(raw).__name__):
                parsed = contract.parse_response(raw, SELECTED, TAXONOMY)
                self.assertEqual(parsed["status"], "invalid")
                self.assertNotIn(sentinel, repr(parsed))

    def test_duplicate_json_keys_at_any_depth_fail_closed(self):
        raw = (f'{{"schema_version":"{contract.SCHEMA_VERSION}",'
               f'"taxonomy_sha256":"{TAXONOMY}","decisions":[],"abstain":false,'
               '"extensions":{"x":1,"x":2}}')
        parsed = contract.parse_response(raw, SELECTED, TAXONOMY)
        self.assertEqual(parsed["status"], "invalid")

    def test_duplicate_and_contradictory_decisions_fail_closed(self):
        decision = {"field": "intent", "label": "build", "value": "positive"}
        parsed = contract.parse_response(response([decision, dict(decision, value="negative")]),
                                         SELECTED, TAXONOMY)
        self.assertEqual(parsed["status"], "invalid")
        self.assertTrue(all(item["value"] == "unknown" for item in parsed["decisions"]))

    def test_unknown_identity_and_delimiter_lookalikes_fail_closed(self):
        cases = [
            {"field": "other", "label": "build", "value": "positive"},
            {"field": "models", "label": "small", "value": "positive"},
            {"field": "intent:models", "label": "build", "value": "positive"},
        ]
        for decision in cases:
            with self.subTest(decision=decision):
                parsed = contract.parse_response(response([decision]), SELECTED, TAXONOMY)
                self.assertEqual(parsed["status"], "invalid")

    def test_invalid_state_and_invalid_identity_types_fail_closed(self):
        for decision in (
            {"field": "intent", "label": "build", "value": "maybe"},
            {"field": ["intent"], "label": "build", "value": "positive"},
            {"field": "intent", "label": "build", "value": None},
        ):
            with self.subTest(decision=decision):
                parsed = contract.parse_response(response([decision]), SELECTED, TAXONOMY)
                self.assertEqual(parsed["status"], "invalid")

    def test_abstention_cannot_assert_positive_or_negative(self):
        parsed = contract.parse_response(response([
            {"field": "intent", "label": "build", "value": "positive"},
        ], abstain=True), SELECTED, TAXONOMY)
        self.assertEqual(parsed["status"], "invalid")
        self.assertTrue(all(item["value"] == "unknown" for item in parsed["decisions"]))

    def test_abstention_with_no_assertions_is_all_unknown(self):
        parsed = contract.parse_response(response([], abstain=True), SELECTED, TAXONOMY)
        self.assertEqual(parsed["status"], "abstain")
        self.assertTrue(all(item["value"] == "unknown" for item in parsed["decisions"]))

    def test_taxonomy_mismatch_and_nonfinite_numbers_fail_closed(self):
        parsed = contract.parse_response(response([], taxonomy_sha256="b" * 64), SELECTED, TAXONOMY)
        self.assertEqual(parsed["status"], "invalid")
        raw = (f'{{"schema_version":"{contract.SCHEMA_VERSION}",'
               f'"taxonomy_sha256":"{TAXONOMY}","decisions":[],"abstain":false,'
               '"extensions":{"x":NaN}}')
        self.assertEqual(contract.parse_response(raw, SELECTED, TAXONOMY)["status"], "invalid")

    def test_invalid_selections_reject_empty_duplicate_and_malformed_values(self):
        for selected in ([], SELECTED + [dict(SELECTED[0])],
                         [{"field": "", "label": "x"}],
                         [{"field": "x", "label": " "}],
                         [{"field": "x", "label": "y", "extra": True}]):
            with self.subTest(selected=selected), self.assertRaises(contract.ContractError):
                contract.build_response_schema(selected, TAXONOMY)

    def test_invalid_taxonomy_and_schema_limits_reject(self):
        with self.assertRaises(contract.ContractError):
            contract.build_response_schema(SELECTED, "not-a-digest")
        with self.assertRaises(contract.ContractError):
            contract.build_response_schema(SELECTED, TAXONOMY, max_schema_bytes=8)
        with self.assertRaises(contract.ContractError):
            contract.build_response_schema(SELECTED, TAXONOMY, max_selections=2)
        with self.assertRaises(contract.ContractError):
            contract.build_response_schema(SELECTED, TAXONOMY,
                                           max_selections=contract.MAX_SELECTIONS + 1)

    def test_extension_limit_and_unknown_outcome_fail_closed(self):
        parsed = contract.parse_response(response([], extensions={"x": "long-value"}),
                                         SELECTED, TAXONOMY, max_extension_bytes=2)
        self.assertEqual(parsed["status"], "invalid")
        self.assertEqual([decision["value"] for decision in parsed["decisions"]],
                         ["unknown"] * len(SELECTED))
        with self.assertRaises(contract.ContractError):
            contract.parse_response(response(), SELECTED, TAXONOMY, outcome="teacher_says_so")
        with self.assertRaises(contract.ContractError):
            contract.parse_response(response(), SELECTED, TAXONOMY,
                                    max_response_bytes=contract.MAX_RESPONSE_BYTES + 1)

    def test_schema_serialization_is_stable(self):
        first = json.dumps(contract.build_response_schema(SELECTED, TAXONOMY),
                           sort_keys=True, separators=(",", ":"))
        second = json.dumps(contract.build_response_schema(SELECTED, TAXONOMY),
                            sort_keys=True, separators=(",", ":"))
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
