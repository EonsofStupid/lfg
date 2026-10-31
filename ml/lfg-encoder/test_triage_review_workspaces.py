import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import corpus_audit
import review_corpus
import triage_review_workspaces as triage


class TriageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        (root / 'taxonomy.json').write_text(json.dumps({'schema_version': 'lfg.encoder-evaluation-taxonomy.v1',
                                                       'intents': ['build'], 'capabilities': ['write'], 'risks': ['sensitive']}))
        rows = [{'id': str(i), 'text': text, 'family': f'family-{i}', 'verified': False,
                 'intents': [], 'capabilities': [], 'risks': [], 'labels_pending': True,
                 'source': {'project': 'fixture'}}
                for i, text in enumerate(['ordinary private fixture', '<in-app-browser-context>private fixture</in-app-browser-context>', 'done'])]
        (root / 'source.jsonl').write_bytes(review_corpus.encoded_rows(rows))
        plan = {'schema_version': corpus_audit.PLAN_SCHEMA, 'taxonomy': 'taxonomy.json', 'review_ledger': None,
                'sources': [{'path': 'source.jsonl', 'project': 'fixture', 'split': 'train'}]}
        (root / 'plan.json').write_text(json.dumps(plan))
        self.workspace = root / 'workspace'
        review_corpus.init_workspace(root / 'plan.json', self.workspace)
        self.config = Path(triage.__file__).with_name('review-triage.example.json')

    def test_triage_reports_only_markers_and_never_changes_workspace(self):
        before = {p.name: p.read_bytes() for p in self.workspace.iterdir() if p.is_file()}
        result = triage.triage([self.workspace], self.config)
        self.assertEqual(result['pending'], 3)
        self.assertEqual(result['flagged'], 2)
        self.assertEqual(result['reason_counts'], {'ambient-browser-context': 1, 'short-acknowledgement': 1})
        self.assertEqual([item['pending_offset'] for item in result['items']], [1, 2])
        self.assertNotIn('private fixture', json.dumps(result))
        self.assertFalse(result['review_decisions_made'])
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.workspace.iterdir() if p.is_file()})

    def test_test_split_rejected_before_records_open(self):
        path = self.workspace / 'plan.json'
        plan = json.loads(path.read_bytes())
        plan['sources'][0]['split'] = 'test'
        path.write_text(json.dumps(plan))
        with patch.object(review_corpus, 'load_workspace', side_effect=AssertionError('records opened')):
            with self.assertRaisesRegex(ValueError, 'only training sources'):
                triage.triage([self.workspace], self.config)

    def test_configurable_patterns_and_scan_limit(self):
        config = json.loads(self.config.read_bytes())
        config['max_text_bytes'] = 30
        config['patterns'] = [{'id': 'selected-word', 'expression': 'ordinary', 'ignore_case': False}]
        path = self.workspace.parent / 'config.json'
        path.write_text(json.dumps(config))
        result = triage.triage([self.workspace], path)
        self.assertEqual(result['reason_counts'], {'scan-limit-exceeded': 1, 'selected-word': 1})


if __name__ == '__main__':
    unittest.main()
