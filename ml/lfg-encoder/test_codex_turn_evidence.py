"""Fixture-only tests; usage evidence is not a successful-outcome label."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import codex_turn_evidence as subject
import corpus_audit


class TurnEvidenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = Path(subject.__file__).with_name('codex-turn-evidence.example.json')
        self.plan = self.root / 'plan.json'
        self.source = self.root / 'session.jsonl'
        self.candidates = self.root / 'candidates.jsonl'
        self.output = self.root / 'evidence.jsonl'
        self.values = {'schema_version': corpus_audit.DEVELOPMENT_PLAN_SCHEMA,
                       'sources': [{'project': 'fixture', 'path': str(self.candidates), 'split': 'train'}]}
        self.plan.write_text(json.dumps(self.values), encoding='utf-8')

    def make_source(self, contexts, user_turn='turn-one', candidate_text='private fixture request'):
        events = [{'type': 'turn_context', 'payload': {'turn_id': turn, 'model': model, 'effort': effort}}
                  for turn, model, effort in contexts]
        events.append({'type': 'response_item', 'payload': {'type': 'message', 'role': 'user',
                       'internal_chat_message_metadata_passthrough': {'content_item_kinds': ['user.text'], 'turn_id': user_turn},
                       'content': [{'type': 'input_text', 'text': 'private fixture request'}]}})
        lines = [(json.dumps(item) + '\n').encode() for item in events]
        raw = b''.join(lines)
        self.source.write_bytes(raw)
        provenance = {'project': 'fixture', 'kind': 'codex-session-user-text', 'source_file': str(self.source),
                      'source_file_sha256': subject.sha(raw), 'source_line': len(lines),
                      'source_line_sha256': subject.sha(lines[-1].rstrip(b'\r\n')),
                      'source_prefix_bytes': len(raw), 'source_prefix_sha256': subject.sha(raw),
                      'text_sha256': subject.sha(b'private fixture request')}
        self.candidates.write_text(json.dumps({'id': 'record-one', 'family': 'family-one',
                                                'text': candidate_text, 'source': provenance}) + '\n', encoding='utf-8')

    def prepare(self):
        return subject.prepare(self.plan, self.config, self.output)

    def test_exact_turn_and_repeated_consistent_context(self):
        self.make_source([('turn-one', 'fixture-model', 'medium'), ('turn-one', 'fixture-model', 'medium')])
        report = self.prepare()
        raw = self.output.read_text(encoding='utf-8')
        row = json.loads(raw)
        self.assertEqual(report['matched_model'], 1)
        self.assertEqual(row['context']['consistent_contexts'], 2)
        self.assertEqual(row['status'], 'matched_turn_context')
        self.assertEqual(row['outcome'], 'unknown')
        self.assertIsNone(row['agent_count'])
        self.assertEqual(row['review_status'], 'unreviewed')
        self.assertFalse(row['promotion_eligible'])
        self.assertNotIn('private fixture request', raw)
        self.assertNotIn(str(self.source), raw)

    def test_conflicting_or_unmatched_context_never_becomes_usage(self):
        self.make_source([('turn-one', 'model-a', 'low'), ('turn-one', 'model-b', 'high')])
        self.prepare()
        row = json.loads(self.output.read_text())
        self.assertEqual(row['status'], 'ambiguous_turn_context')
        self.assertIsNone(row['context'])
        other = self.root / 'other.jsonl'
        self.make_source([('another-turn', 'model-a', 'low')])
        subject.prepare(self.plan, self.config, other)
        self.assertEqual(json.loads(other.read_text())['status'], 'turn_context_missing')

    def test_changed_candidate_or_source_not_misattributed(self):
        self.make_source([('turn-one', 'model-a', 'low')], candidate_text='edited text')
        self.prepare()
        row = json.loads(self.output.read_text())
        self.assertEqual(row['status'], 'candidate_text_changed')
        self.assertIsNone(row['context'])
        other = self.root / 'other.jsonl'
        self.source.write_bytes(self.source.read_bytes().replace(b'model-a', b'model-z'))
        with self.assertRaisesRegex(ValueError, 'prefix'):
            subject.prepare(self.plan, self.config, other)
        self.assertFalse(other.exists())

    def test_heldout_rejected_before_source_access(self):
        self.values['sources'].append({'project': 'holdout', 'path': 'must-not-open', 'split': 'test'})
        self.plan.write_text(json.dumps(self.values))
        with self.assertRaisesRegex(ValueError, 'development plan'):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_declared_project_group_and_family_cannot_cross_splits(self):
        self.make_source([('turn-one', 'model-a', 'low')])
        second = self.root / 'validation.jsonl'
        candidate = json.loads(self.candidates.read_text())
        candidate['id'] = 'other-id'
        candidate['source']['project'] = 'other-project'
        second.write_text(json.dumps(candidate) + '\n')
        self.values['sources'].append({'project': 'other-project', 'path': str(second), 'split': 'validation'})
        self.plan.write_text(json.dumps(self.values))
        with self.assertRaisesRegex(ValueError, 'family or normalized text crosses split'):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_git_ignored_but_tracked_output_rejected(self):
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True)
        (self.root / '.gitignore').write_text('evidence.jsonl\n')
        self.output.write_text('fixture')
        subprocess.run(['git', '-C', str(self.root), 'add', '-f', 'evidence.jsonl'], check=True)
        self.output.unlink()
        with self.assertRaisesRegex(ValueError, 'untracked and ignored'):
            subject.require_private_output(self.output)


if __name__ == '__main__':
    unittest.main()
