"""Software fixtures only; no fixture is an LFG training or evaluation example."""
import hashlib
import json
import subprocess
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import screen_private_corpus as subject


class PrivateScreenTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.plan_path = self.root / 'plan.json'
        self.policy_path = self.root / 'policy.json'
        self.output = self.root / 'output'
        self.policy = {'schema_version': subject.POLICY_VERSION, 'revision': 'fixture',
                       'limits': {'max_policy_bytes': 4096, 'max_plan_bytes': 4096,
                                  'max_source_bytes': 4096, 'max_total_source_bytes': 8192,
                                  'max_total_output_bytes': 16384, 'max_records': 20,
                                  'max_text_bytes': 2048, 'max_matches_per_record': 5},
                       'rules': [{'id': 'fixture-secret', 'pattern': 'fixture-secret-[0-9]+',
                                  'replacement': '[removed]'}]}
        self.policy_path.write_text(json.dumps(self.policy), encoding='utf-8')
        self.source = self.root / 'source.jsonl'
        self.row = {'id': 'id-1', 'family': 'family-1', 'text': 'hello fixture-secret-123 🚀',
                    'source': {'project': 'project-1', 'private_path': 'not-for-output'},
                    'verified': False, 'labels_pending': True}
        self.source.write_text(json.dumps(self.row, ensure_ascii=False) + '\n', encoding='utf-8')
        self.plan = {'schema_version': 'lfg.encoder-development-corpus-plan.v2', 'review_ledger': None,
                     'sources': [{'path': str(self.source), 'project': 'project-1',
                                  'project_group': 'group-1', 'split': 'train'}]}
        self.write_plan()

    def write_plan(self):
        self.plan_path.write_text(json.dumps(self.plan), encoding='utf-8')

    def prepare(self):
        return subject.prepare(self.plan_path, self.policy_path, self.output, 8192)

    def test_redaction_is_unreviewed_and_provenance_is_bound(self):
        result = self.prepare()
        text = (self.output / 'train.jsonl').read_text(encoding='utf-8')
        record = json.loads(text)
        self.assertNotIn('fixture-secret-123', text)
        self.assertNotIn('not-for-output', text)
        self.assertEqual(record['text'], 'hello [removed] 🚀')
        self.assertEqual(record['source_text_sha256'], hashlib.sha256(self.row['text'].encode()).hexdigest())
        self.assertEqual(record['candidate_file_sha256'], hashlib.sha256(self.source.read_bytes()).hexdigest())
        self.assertEqual(record['redaction_counts'], {'fixture-secret': 1})
        self.assertEqual(record['review_status'], 'unreviewed')
        self.assertFalse(record['redaction_complete'])
        self.assertFalse(result['claims']['human_reviewed'])
        self.assertFalse(result['claims']['promotion_eligible'])
        self.assertEqual(result['counts'], {'train': 1, 'validation': 0})
        self.assertEqual(result['outputs']['train'], hashlib.sha256((self.output / 'train.jsonl').read_bytes()).hexdigest())
        with self.assertRaisesRegex(subject.ScreenError, 'output_exists'):
            self.prepare()

    def test_heldout_is_rejected_before_source_open(self):
        self.plan['sources'].append({'path': str(self.root / 'must-not-open'), 'project': 'project-2', 'split': 'test'})
        self.write_plan()
        with self.assertRaisesRegex(subject.ScreenError, 'heldout_split'):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_cross_split_after_redaction_is_rejected(self):
        second = self.root / 'second.jsonl'
        other = {**self.row, 'id': 'id-2', 'family': 'family-2', 'text': 'hello fixture-secret-456 🚀',
                 'source': {'project': 'project-2'}}
        second.write_text(json.dumps(other, ensure_ascii=False) + '\n', encoding='utf-8')
        self.plan['sources'].append({'path': str(second), 'project': 'project-2', 'split': 'validation'})
        self.write_plan()
        with self.assertRaisesRegex(subject.ScreenError, 'screened_text_crosses_split'):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_zero_length_and_match_limits_fail_without_output(self):
        self.policy['rules'][0]['pattern'] = '(?=hello)'
        self.policy_path.write_text(json.dumps(self.policy), encoding='utf-8')
        with self.assertRaisesRegex(subject.ScreenError, 'empty_policy_match'):
            self.prepare()
        self.policy['rules'][0]['pattern'] = 'fixture'
        self.policy_path.write_text(json.dumps(self.policy), encoding='utf-8')
        self.source.write_text(json.dumps({**self.row, 'text': 'fixture ' * 6}) + '\n', encoding='utf-8')
        with self.assertRaisesRegex(subject.ScreenError, 'match_limit_exceeded'):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_changed_source_is_detected_before_output(self):
        original = subject._read
        calls = 0

        def mutate(path, maximum, code):
            nonlocal calls
            value = original(path, maximum, code)
            if path == self.source:
                calls += 1
                if calls == 1:
                    self.source.write_text(json.dumps({**self.row, 'text': 'changed'}) + '\n', encoding='utf-8')
            return value

        with patch.object(subject, '_read', side_effect=mutate):
            with self.assertRaisesRegex(subject.ScreenError, 'input_changed'):
                self.prepare()
        self.assertFalse(self.output.exists())

    def test_same_inputs_reproduce_the_screened_bytes(self):
        first = self.prepare()
        other = self.root / 'second-output'
        second = subject.prepare(self.plan_path, self.policy_path, other, 8192)
        self.assertEqual((self.output / 'train.jsonl').read_bytes(), (other / 'train.jsonl').read_bytes())
        self.assertEqual(first, second)

    def test_duplicate_json_keys_and_sensitive_metadata_are_rejected(self):
        self.plan_path.write_text('{"schema_version":"lfg.encoder-development-corpus-plan.v2",'
                                  '"sources":[],"sources":[]}', encoding='utf-8')
        with self.assertRaisesRegex(subject.ScreenError, 'invalid_plan'):
            self.prepare()
        self.write_plan()
        self.source.write_text(json.dumps({**self.row, 'id': 'fixture-secret-123'}) + '\n', encoding='utf-8')
        with self.assertRaisesRegex(subject.ScreenError, 'sensitive_metadata_detected'):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_example_policy_finds_common_credential_shapes(self):
        path = Path(subject.__file__).with_name('private-corpus-screen.example.json')
        policy, rules, _ = subject.load_policy(path, 65536)
        raw = 'Authorization: Bearer ' + 'fixtureSecret0123456789' + '\n' + ('gh' + 'p_' + 'x' * 24)
        screened, counts = subject._screen(raw, rules, policy['limits'])
        self.assertNotIn('fixtureSecret0123456789', screened)
        self.assertEqual(counts['bearer-credential'], 1)
        self.assertEqual(counts['common-token-prefix'], 1)

    def test_overlapping_rules_cannot_reintroduce_a_match(self):
        self.policy['rules'].append({'id': 'later', 'pattern': 'removed',
                                     'replacement': 'fixture-secret-999'})
        self.policy_path.write_text(json.dumps(self.policy), encoding='utf-8')
        with self.assertRaisesRegex(subject.ScreenError, 'residual_policy_match'):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_all_output_files_must_be_ignored(self):
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True)
        (self.root / '.gitignore').write_text('/output/train.jsonl\n', encoding='utf-8')
        with self.assertRaisesRegex(subject.ScreenError, 'output_not_private'):
            self.prepare()
        self.assertFalse(self.output.exists())
        (self.root / '.gitignore').write_text('/output/\n', encoding='utf-8')
        self.prepare()
        self.assertTrue((self.output / 'manifest.json').exists())

    def test_ignored_but_tracked_output_is_rejected(self):
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True)
        (self.root / '.gitignore').write_text('/output/\n', encoding='utf-8')
        self.output.mkdir()
        tracked = self.output / 'train.jsonl'
        tracked.write_text('fixture', encoding='utf-8')
        subprocess.run(['git', '-C', str(self.root), 'add', '-f', 'output/train.jsonl'], check=True)
        tracked.unlink()
        self.output.rmdir()
        with self.assertRaisesRegex(subject.ScreenError, 'output_tracked_or_unverifiable'):
            self.prepare()
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
