import json
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import corpus_audit
import review_corpus as review


class ReviewWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        taxonomy = {'schema_version': 'lfg.encoder-evaluation-taxonomy.v1',
                    'intents': ['build', 'explain'], 'capabilities': ['write'], 'risks': ['sensitive']}
        (self.root / 'taxonomy.json').write_text(json.dumps(taxonomy), encoding='utf-8')
        self.rows = []
        sources = []
        for index, split in enumerate(('train', 'validation', 'test')):
            project = f'project-{index}'
            row = {'id': f'id-{index}', 'text': f'fixture request {index}', 'family': f'family-{index}',
                   'intents': ['build'], 'capabilities': ['write'], 'risks': [], 'verified': False,
                   'source': {'project': project}}
            self.rows.append(row)
            path = f'input-{index}.jsonl'
            (self.root / path).write_text(json.dumps(row) + '\n', encoding='utf-8')
            sources.append({'path': path, 'project': project, 'split': split})
        plan = {'schema_version': corpus_audit.PLAN_SCHEMA, 'taxonomy': 'taxonomy.json',
                'review_ledger': None, 'sources': sources}
        self.plan_path = self.root / 'input-plan.json'
        self.plan_path.write_text(json.dumps(plan), encoding='utf-8')
        self.workspace = self.root / 'workspace'
        review.init_workspace(self.plan_path, self.workspace)

    def digest(self, identifier):
        plan, _, ledger = review.load_workspace(self.workspace)
        row_path, rows, index, source = review.find_row(self.workspace, plan, identifier)
        del row_path
        return corpus_audit.annotation_digest(rows[index], source['project'], source['split'],
                                              review.evaluate.load_taxonomy(self.workspace / 'taxonomy.json')[1])

    def test_initialization_is_pending_private_and_does_not_overwrite(self):
        result = review.status(self.workspace)
        self.assertEqual(result['decisions'], {'pending': 3, 'approved': 0, 'rejected': 0})
        self.assertNotIn('fixture request', json.dumps(result))
        with self.assertRaises(FileExistsError):
            review.init_workspace(self.plan_path, self.workspace)
        self.assertFalse(json.loads((self.root / 'input-0.jsonl').read_text())['verified'])

    def test_approval_corrections_and_export_exclude_pending_and_rejected(self):
        review.record_decision(self.workspace, 'id-0', 'fixture reviewer', 'approved', self.digest('id-0'),
                               {'intents': ['explain'], 'capabilities': [], 'risks': []})
        review.record_decision(self.workspace, 'id-1', 'fixture reviewer', 'rejected', self.digest('id-1'))
        result = review.status(self.workspace)
        self.assertEqual(result['decisions'], {'pending': 1, 'approved': 1, 'rejected': 1})
        self.assertEqual(result['counts']['train']['ready_for_reviewed_subset'], 1)
        output = self.root / 'export'
        result = review.export_approved(self.workspace, output)
        self.assertEqual(result['exported'], 1)
        row = json.loads((output / 'source-0000.jsonl').read_text())
        self.assertTrue(row['verified'])
        self.assertEqual(row['intents'], ['explain'])
        with self.assertRaisesRegex(ValueError, 'nonempty validation'):
            corpus_audit.assemble_training(output / 'plan.json')

    def test_approved_project_splits_feed_existing_audit_without_test_leakage(self):
        for index in range(3):
            identifier = f'id-{index}'
            review.record_decision(self.workspace, identifier, 'fixture reviewer', 'approved', self.digest(identifier))
        output = self.root / 'complete-export'
        result = review.export_approved(self.workspace, output)
        self.assertEqual(result['exported'], 3)
        payload, _ = corpus_audit.assemble_training(output / 'plan.json')
        training = [json.loads(line) for line in payload.splitlines()]
        _, validation = corpus_audit.load_reviewed_validation(training, output / 'plan.json')
        self.assertEqual([row['id'] for row in training], ['id-0'])
        self.assertEqual([row['id'] for row in validation], ['id-1'])

    def test_invalid_labels_or_reviewer_never_approve(self):
        digest = self.digest('id-0')
        for reviewer, labels in (('', {'intents': ['build'], 'capabilities': [], 'risks': []}),
                                 ('person', {'intents': ['wrong'], 'capabilities': [], 'risks': []}),
                                 ('person', {'intents': [], 'capabilities': [], 'risks': []})):
            with self.subTest(reviewer=reviewer, labels=labels), self.assertRaises(ValueError):
                review.record_decision(self.workspace, 'id-0', reviewer, 'approved', digest, labels)
        self.assertEqual(review.status(self.workspace)['decisions']['approved'], 0)

    def test_stale_display_and_changed_pending_record_are_rejected(self):
        digest = self.digest('id-0')
        path = self.workspace / 'source-0000.jsonl'
        row = json.loads(path.read_text())
        row['text'] = 'modified after display'
        path.write_text(json.dumps(row) + '\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'changed after display'):
            review.record_decision(self.workspace, 'id-0', 'person', 'approved', digest)
        with self.assertRaisesRegex(ValueError, 'pending record changed'):
            review.record_decision(self.workspace, 'id-0', 'person', 'approved', self.digest('id-0'))

    def test_stale_approval_cannot_be_exported(self):
        review.record_decision(self.workspace, 'id-0', 'person', 'approved', self.digest('id-0'))
        path = self.workspace / 'source-0000.jsonl'
        row = json.loads(path.read_text())
        row['text'] = 'modified after approval'
        path.write_text(json.dumps(row) + '\n', encoding='utf-8')
        self.assertEqual(review.status(self.workspace)['counts']['train']['ready_for_reviewed_subset'], 0)
        with self.assertRaisesRegex(ValueError, 'no current approvals'):
            review.export_approved(self.workspace, self.root / 'export')

    def test_interrupted_source_write_requires_a_new_explicit_decision(self):
        original = review.atomic_replace
        def fail_ledger(path, content):
            if path.name == 'reviews.jsonl':
                raise OSError('simulated interruption')
            original(path, content)
        with patch.object(review, 'atomic_replace', side_effect=fail_ledger), self.assertRaises(OSError):
            review.record_decision(self.workspace, 'id-0', 'person', 'approved', self.digest('id-0'),
                                   {'intents': ['explain'], 'capabilities': [], 'risks': []})
        self.assertEqual(review.status(self.workspace)['counts']['train']['ready_for_reviewed_subset'], 0)
        self.assertEqual(review.status(self.workspace)['decisions']['pending'], 3)
        review.record_decision(self.workspace, 'id-0', 'person', 'approved', self.digest('id-0'))
        self.assertEqual(review.status(self.workspace)['counts']['train']['ready_for_reviewed_subset'], 1)

    def test_interactive_approval_requires_the_exact_confirmation(self):
        with patch('builtins.input', side_effect=['a', 'no', 'a', 'APPROVE', 'q']), contextlib.redirect_stdout(io.StringIO()):
            review.interactive_review(self.workspace, 'fixture reviewer')
        self.assertEqual(review.status(self.workspace)['decisions'], {'pending': 2, 'approved': 1, 'rejected': 0})

    def test_workspace_paths_cannot_escape_and_terminal_controls_are_escaped(self):
        self.assertEqual(review.visible('name\x1b[2J\n'), 'name\\u001b[2J\\u000a')
        plan_path = self.workspace / 'plan.json'
        plan = json.loads(plan_path.read_text())
        plan['sources'][0]['path'] = '../input-0.jsonl'
        plan_path.write_text(json.dumps(plan), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'stay inside'):
            review.status(self.workspace)


if __name__ == '__main__':
    unittest.main()
