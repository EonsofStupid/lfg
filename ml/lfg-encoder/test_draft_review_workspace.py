import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import corpus_audit
import draft_review_workspace as draft
import evaluate
import review_corpus


class DraftWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        taxonomy = {'schema_version': 'lfg.encoder-evaluation-taxonomy.v1',
                    'intents': ['build'], 'capabilities': ['write'], 'risks': ['sensitive']}
        (self.root / 'taxonomy.json').write_text(json.dumps(taxonomy))
        labels, taxonomy_sha = evaluate.load_taxonomy(self.root / 'taxonomy.json')
        guide = {'schema_version': 'lfg.review-label-guide.v1', 'revision': 'fixture', 'notice': 'test only',
                 'labels': {field: {label: {'apply': 'apply', 'boundary': 'boundary'} for label in values}
                            for field, values in labels.items()}}
        self.guide = self.root / 'guide.json'
        self.guide.write_text(json.dumps(guide))
        rows = [{'id': str(i), 'text': f'test request {i}', 'family': f'family-{i}', 'verified': False,
                 'intents': [], 'capabilities': [], 'risks': [], 'labels_pending': True,
                 'source': {'project': 'fixture'}, 'review_note': 'original note'} for i in range(2)]
        (self.root / 'source.jsonl').write_bytes(review_corpus.encoded_rows(rows))
        plan = {'schema_version': corpus_audit.PLAN_SCHEMA, 'taxonomy': 'taxonomy.json', 'review_ledger': None,
                'sources': [{'path': 'source.jsonl', 'project': 'fixture', 'split': 'train'}]}
        (self.root / 'plan.json').write_text(json.dumps(plan))
        self.workspace = self.root / 'workspace'
        review_corpus.init_workspace(self.root / 'plan.json', self.workspace)
        _, _, ledger = review_corpus.load_workspace(self.workspace)
        self.proposals = self.root / 'proposals.json'
        self.drafts = {'schema_version': 'lfg.ai-label-drafts.v1', 'taxonomy_sha256': taxonomy_sha,
                       'guide_sha256': draft.digest(self.guide.read_bytes()), 'author': 'test-ai',
                       'items': [{'id': '0', 'annotation_sha256': ledger['0']['annotation_sha256'],
                                  'labels': {'intents': ['build'], 'capabilities': ['write'], 'risks': []},
                                  'note': 'test-only suggestion'}]}
        self.proposals.write_text(json.dumps(self.drafts))
        self.output = self.root / 'draft-output'

    def test_drafts_preserve_pending_decisions_and_original_workspace(self):
        before = {path.name: path.read_bytes() for path in self.workspace.iterdir() if path.is_file()}
        result = draft.prepare(self.workspace, self.proposals, self.guide, self.output)
        self.assertEqual((result['drafts'], result['pending'], result['approved']), (1, 2, 0))
        self.assertFalse(result['promotion_eligible'])
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.workspace.iterdir() if path.is_file()})
        plan, _, ledger = review_corpus.load_workspace(self.output / 'workspace')
        _, rows, index, _ = review_corpus.find_row(self.output / 'workspace', plan, '0')
        self.assertEqual(rows[index]['intents'], ['build'])
        self.assertFalse(rows[index]['verified'])
        self.assertEqual(rows[index]['ai_label_draft']['author'], 'test-ai')
        self.assertIn('not reviewed', rows[index]['review_note'])
        self.assertIn('original note', rows[index]['review_note'])
        self.assertEqual({row['decision'] for row in ledger.values()}, {'pending'})
        _, rows, index, _ = review_corpus.find_row(self.output / 'workspace', plan, '1')
        self.assertTrue(rows[index]['labels_pending'])

    def test_rejects_stale_drafts_and_guide_changes_without_output(self):
        self.drafts['items'][0]['annotation_sha256'] = '0' * 64
        self.proposals.write_text(json.dumps(self.drafts))
        with self.assertRaisesRegex(ValueError, 'stale'):
            draft.prepare(self.workspace, self.proposals, self.guide, self.output)
        self.assertFalse(self.output.exists())
        self.drafts['items'][0]['annotation_sha256'] = review_corpus.load_workspace(self.workspace)[2]['0']['annotation_sha256']
        self.drafts['guide_sha256'] = '0' * 64
        self.proposals.write_text(json.dumps(self.drafts))
        with self.assertRaisesRegex(ValueError, 'guide changed'):
            draft.prepare(self.workspace, self.proposals, self.guide, self.output)

    def test_test_split_is_rejected_before_opening_workspace_records(self):
        path = self.workspace / 'plan.json'
        plan = json.loads(path.read_bytes())
        plan['sources'][0]['split'] = 'test'
        path.write_text(json.dumps(plan))
        with patch.object(review_corpus, 'load_workspace', side_effect=AssertionError('records opened')):
            with self.assertRaisesRegex(ValueError, 'training sources'):
                draft.prepare(self.workspace, self.proposals, self.guide, self.output)
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
