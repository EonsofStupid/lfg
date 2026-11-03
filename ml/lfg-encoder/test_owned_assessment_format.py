import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

AVAILABLE = all(importlib.util.find_spec(name) for name in ('torch', 'safetensors'))
if AVAILABLE:
    import evaluate_owned_assessment_format as evaluation


@unittest.skipUnless(AVAILABLE, 'torch and safetensors required')
class AssessmentFormatTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / 'dataset'
        self.dataset.mkdir()
        self.policy = self.root / 'policy.json'
        self.policy.write_text(json.dumps({'schema_version': 'lfg.private-corpus-screen.v1', 'revision': 'fixture',
            'limits': {'max_policy_bytes': 65536, 'max_plan_bytes': 65536, 'max_source_bytes': 100000,
                       'max_total_source_bytes': 100000, 'max_total_output_bytes': 100000, 'max_records': 100,
                       'max_text_bytes': 10000, 'max_matches_per_record': 100},
            'rules': [{'id': 'fixture-secret', 'pattern': 'fixture-secret-[0-9]+', 'replacement': '[REDACTED]'}]}))
        rows = []
        for i in range(3):
            text = f'Fixture validation request {i} fixture-secret-{i}'
            rows.append({'schema_version': 'lfg.screened-text-candidate.v1', 'split': 'validation', 'id': f'id-{i}',
                         'text': text, 'screened_text_sha256': evaluation.sha(text.encode()), 'extra': True})
        raw = b''.join((json.dumps(row) + '\n').encode() for row in rows)
        (self.dataset / 'validation.jsonl').write_bytes(raw)
        (self.dataset / 'manifest.json').write_text(json.dumps({'schema_version': 'lfg.private-corpus-screen-manifest.v1',
            'outputs': {'validation': evaluation.sha(raw), 'train': 'a' * 64}}))
        self.config = {'schema_version': 'lfg.owned-assessment-format-evaluation.v1', 'seed': 'fixture',
            'system_prompt': 'Assess the request.', 'framing': {'system_prefix': '<s>', 'system_suffix': '</s>',
                'user_prefix': '<u>', 'user_suffix': '</u>', 'assistant_prefix': '<a>'},
            'required_fields': ['intent', 'complexity'], 'declared_subscriptions': [],
            'max_validation_bytes': 100000, 'max_manifest_bytes': 65536, 'max_records': 2,
            'min_prompt_bytes': 1, 'max_prompt_bytes': 1000, 'max_context': 1536, 'max_new_bytes': 16,
            'max_response_bytes': 65536, 'timeout_seconds': 20, 'startup_timeout_seconds': 5, 'extra': True}
        self.config_path = self.root / 'config.json'
        self.config_path.write_text(json.dumps(self.config))

    def test_selects_screened_validation_without_opening_train_or_test(self):
        config, _ = evaluation.config_file(self.config_path)
        rows, identity = evaluation.select_validation(self.dataset, self.policy, config)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all('fixture-secret-' not in row for row in rows))
        self.assertFalse((self.dataset / 'train.jsonl').exists())
        self.assertFalse((self.dataset / 'test.jsonl').exists())
        self.assertEqual(identity['source_records'], 3)
        self.assertEqual(evaluation.select_validation(self.dataset, self.policy, config), (rows, identity))
        (self.dataset / 'validation.jsonl').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'digest mismatch'):
            evaluation.select_validation(self.dataset, self.policy, config)

    def test_format_measurement_does_not_claim_correctness(self):
        fields = ['intent', 'complexity']
        self.assertEqual(evaluation.response_format(b'not json', fields), 'invalid_json')
        self.assertEqual(evaluation.response_format(b'{"intent":"x"}', fields), 'missing_required_fields')
        self.assertEqual(evaluation.response_format(b'{"intent":null,"complexity":"x","extra":true}', fields),
                         'required_fields_present')

    def test_prompt_framing_and_context_limit(self):
        config, _ = evaluation.config_file(self.config_path)
        self.assertEqual(evaluation.prompt('request', config), '<s>Assess the request.</s><u>request</u><a>')
        config['max_context'] = 4
        with self.assertRaisesRegex(ValueError, 'exceeds context'):
            evaluation.prompt('request', config)

    def test_stop_sequence_is_checked_before_accepting_a_response(self):
        predictor = object.__new__(evaluation.FormatGenerator)
        predictor.stop = '\n'; predictor.max_new_bytes = 8
        value = {'schema_version': 'lfg.owned-byte-generation.v1', 'id': 'test',
                 'input_bytes': 1, 'generated_bytes': [123, 125, 10], 'stop_reason': 'stop_sequence'}
        predictor.validate_response(value, 'test', 'x')
        value['generated_bytes'] = [123, 125]
        with self.assertRaisesRegex(ValueError, 'bounds'):
            predictor.validate_response(value, 'test', 'x')


if __name__ == '__main__':
    unittest.main()
