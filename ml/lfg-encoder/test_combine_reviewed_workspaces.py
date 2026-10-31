import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import combine_reviewed_workspaces as combine
import corpus_audit
import evaluate_reviewed_native
import review_corpus
import train


class CombineReviewedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.taxonomy = {'schema_version': 'lfg.encoder-evaluation-taxonomy.v1',
                         'intents': ['build'], 'capabilities': ['write'], 'risks': ['sensitive']}
        self.workspaces = [self.make_workspace(split, index) for index, split in enumerate(corpus_audit.SPLITS)]

    def make_workspace(self, split, index, text=None, group=None, family=None, extra_pending=False):
        root = self.root / f'input-{index}'
        root.mkdir()
        project = f'project-{index}'
        (root / 'taxonomy.json').write_text(json.dumps(self.taxonomy), encoding='utf-8')
        row = {'id': f'id-{index}', 'text': text or f'example {index}', 'family': family or f'family-{index}',
               'verified': False, 'intents': ['build'], 'capabilities': [], 'risks': [], 'source': {'project': project}}
        records = [row]
        if extra_pending:
            records.append({**row, 'id': f'pending-{index}', 'text': f'pending example {index}'})
        (root / 'source.jsonl').write_text(''.join(json.dumps(value) + '\n' for value in records), encoding='utf-8')
        plan = {'schema_version': corpus_audit.PLAN_SCHEMA, 'taxonomy': 'taxonomy.json', 'review_ledger': None,
                'sources': [{'path': 'source.jsonl', 'project': project, 'project_group': group or project, 'split': split}]}
        (root / 'plan.json').write_text(json.dumps(plan), encoding='utf-8')
        workspace = root / 'workspace'
        review_corpus.init_workspace(root / 'plan.json', workspace)
        return workspace

    def approve(self, workspace):
        _, _, ledger = review_corpus.load_workspace(workspace)
        identifier, item = next(iter(ledger.items()))
        review_corpus.record_decision(workspace, identifier, 'test-reviewer', 'approved', item['annotation_sha256'])

    def test_status_and_pending_inputs_do_not_create_output(self):
        result = combine.run(self.workspaces, None)
        self.assertEqual(result['decisions'], {'pending': 3, 'approved': 0, 'rejected': 0})
        policy = json.loads(train.POLICY.read_text())
        policy['minimum_human_verified_examples'] = 7
        path = self.root / 'custom-policy.json'
        path.write_text(json.dumps(policy))
        custom = combine.run(self.workspaces, None, path)
        self.assertEqual(custom['sample_gate']['minimum_train_examples'], 7)
        self.assertEqual(custom['sample_gate']['approved_train_shortfall'], 7)
        self.assertEqual(custom['sample_gate']['candidate_train_shortfall'], 6)
        output = self.root / 'combined'
        with self.assertRaisesRegex(ValueError, 'each split needs'):
            combine.run(self.workspaces, output)
        self.assertFalse(output.exists())
        self.assertEqual(combine.run(self.workspaces, None)['decisions']['approved'], 0)

    def test_combines_only_reviewed_rows_and_feeds_existing_training_gate(self):
        for workspace in self.workspaces:
            self.approve(workspace)
        output = self.root / 'combined'
        result = combine.run(self.workspaces, output)
        self.assertFalse(result['promotion_eligible'])
        rows, _ = train.load_dataset(output / 'training.jsonl')
        self.assertEqual([row['id'] for row in rows], ['id-0'])
        _, validation = corpus_audit.load_reviewed_validation(rows, output / 'plan.json')
        self.assertEqual([row['id'] for row in validation], ['id-1'])
        manifest = json.loads((output / 'assembly-manifest.json').read_text())
        self.assertEqual(manifest['files_sha256']['training.jsonl'], hashlib.sha256((output / 'training.jsonl').read_bytes()).hexdigest())
        with self.assertRaises(FileExistsError):
            combine.run(self.workspaces, output)

    def test_pending_records_are_not_exported_with_approved_records(self):
        extra = self.make_workspace('train', 3, extra_pending=True)
        selected = [extra, *self.workspaces[1:]]
        for workspace in selected:
            self.approve(workspace)
        output = self.root / 'combined'
        combine.run(selected, output)
        rows, _ = train.load_dataset(output / 'training.jsonl')
        self.assertEqual([row['id'] for row in rows], ['id-3'])
        ledger, _ = corpus_audit.read_ledger(output / 'reviews.jsonl')
        self.assertNotIn('pending-3', ledger)
        self.assertEqual(review_corpus.status(extra)['decisions']['pending'], 1)

    def test_rejects_cross_split_duplicate_and_related_group(self):
        for mode in ('duplicate', 'group', 'family'):
            with self.subTest(mode=mode):
                third = self.make_workspace('test', {'duplicate': 3, 'group': 4, 'family': 5}[mode],
                    text='example 0' if mode == 'duplicate' else None,
                    group='project-0' if mode == 'group' else None,
                    family='family-0' if mode == 'family' else None)
                for workspace in (*self.workspaces[:2], third):
                    if review_corpus.status(workspace)['decisions']['pending']:
                        self.approve(workspace)
                output = self.root / f'failed-{mode}'
                with self.assertRaises(ValueError):
                    combine.run([*self.workspaces[:2], third], output)
                self.assertFalse(output.exists())

    def test_rejects_duplicate_workspace_and_stale_approval(self):
        with self.assertRaisesRegex(ValueError, 'distinct'):
            combine.run([self.workspaces[0], self.workspaces[0]], None)
        for workspace in self.workspaces:
            self.approve(workspace)
        source = self.workspaces[0] / 'source-0000.jsonl'
        row = json.loads(source.read_text())
        row['text'] = 'changed after approval'
        source.write_text(json.dumps(row) + '\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'each split needs'):
            combine.run(self.workspaces, self.root / 'combined')
        self.assertFalse((self.root / 'combined').exists())

    def test_development_assembly_excludes_test_and_does_not_open_a_test_workspace(self):
        for workspace in self.workspaces[:2]:
            self.approve(workspace)
        with patch.object(review_corpus, 'load_workspace', side_effect=AssertionError('source opened')):
            with self.assertRaisesRegex(ValueError, 'only train and validation'):
                combine.run(self.workspaces, None, scope='development')
        output = self.root / 'development'
        result = combine.run(self.workspaces[:2], output, scope='development')
        self.assertEqual(result['scope'], 'development')
        self.assertEqual(result['holdout_assessment'], 'not_performed')
        plan = json.loads((output / 'plan.json').read_bytes())
        self.assertEqual(plan['schema_version'], corpus_audit.DEVELOPMENT_PLAN_SCHEMA)
        self.assertEqual({source['split'] for source in plan['sources']}, {'train', 'validation'})
        records, _ = train.load_dataset(output / 'training.jsonl')
        report, validation = corpus_audit.load_reviewed_validation(records, output / 'plan.json')
        self.assertEqual(report['scope'], 'development')
        self.assertEqual([row['id'] for row in validation], ['id-1'])
        with self.assertRaisesRegex(ValueError, 'separate controlled holdout evaluator'):
            evaluate_reviewed_native.audited_test(output / 'plan.json', output / 'training.jsonl', {})


if __name__ == '__main__':
    unittest.main()
