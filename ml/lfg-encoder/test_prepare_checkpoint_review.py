import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import corpus_audit
import inventory_checkpoint_prompts as inventory
import prepare_checkpoint_review as prepare
import review_corpus


class CheckpointPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.revision = 'f' * 40
        messages = [('a', 'inspected text', 'one'), ('b', 'same session', 'one'),
                    ('c', 'reference text', 'two'), ('d', 'new candidate', 'three'),
                    ('e', 'NEW  CANDIDATE', 'four'), ('1', 'inspected text', 'five'),
                    ('2', 'another candidate', 'six')]
        chunks = []
        for oid, text, session in messages:
            message = f'checkpoint\n\nUser prompt:\n\n{text}\n\nReplit-Commit-Author: Assistant\nReplit-Commit-Session-Id: {session}\n'
            chunks.append(f'{oid * 40}\0date\0{message}\0\n')
        self.history = ''.join(chunks).encode()
        _, self.inventory = inventory.parse_history(self.history, 'project', self.revision)
        self.inventory_path = self.root / 'inventory.jsonl'
        self.inventory_path.write_bytes(review_corpus.encoded_rows(self.inventory))
        self.exclusions = self.root / 'prior.jsonl'
        self.exclusions.write_text(json.dumps({'source': {'project': 'project', 'commit': 'a' * 40}}) + '\n')
        self.taxonomy = self.root / 'taxonomy.json'
        self.taxonomy.write_text(json.dumps({'schema_version': 'lfg.encoder-evaluation-taxonomy.v1',
                                              'intents': ['build'], 'capabilities': ['write'], 'risks': ['sensitive']}))
        reference = {'id': 'ref', 'text': 'reference text', 'family': 'ref', 'verified': False,
                     'intents': ['build'], 'capabilities': [], 'risks': [], 'source': {'project': 'reference'}}
        (self.root / 'reference.jsonl').write_text(json.dumps(reference) + '\n')
        self.plan = self.root / 'reference-plan.json'
        self.plan.write_text(json.dumps({'schema_version': corpus_audit.PLAN_SCHEMA, 'taxonomy': 'taxonomy.json',
                                         'review_ledger': None, 'sources': [{'path': 'reference.jsonl',
                                         'project': 'reference', 'project_group': 'reference-group', 'split': 'train'}]}))

    def run_prepare(self, project_group='project-group'):
        with patch.object(inventory, 'read_history', return_value=self.history):
            return prepare.prepare(self.root, self.revision, 'project', project_group, self.inventory_path,
                                   self.exclusions, [self.plan], self.taxonomy, self.root / 'output', 10000)

    def test_excludes_prior_session_overlap_and_duplicates_without_approvals(self):
        result = self.run_prepare()
        self.assertEqual(result['excluded_inspected_sessions'], 2)
        self.assertEqual(result['excluded_text_from_inspected_sessions'], 1)
        self.assertEqual(result['excluded_reference_overlap'], 1)
        self.assertEqual(result['excluded_normalized_duplicates'], 1)
        self.assertEqual(result['pending_review'], 2)
        workspace = self.root / 'output' / 'workspace'
        status = review_corpus.status(workspace)
        self.assertEqual(status['decisions'], {'pending': 2, 'approved': 0, 'rejected': 0})
        plan, _, _ = review_corpus.load_workspace(workspace)
        self.assertEqual(plan['sources'][0]['split'], 'test')
        rows = [json.loads(line) for line in (workspace / plan['sources'][0]['path']).read_text().splitlines()]
        self.assertTrue(all(row['labels_pending'] and not row['verified'] for row in rows))
        manifest = json.loads((workspace.parent / 'preparation-manifest.json').read_text())
        self.assertEqual(manifest['summary'], result)
        self.assertEqual(manifest['workspace_sha256']['source-0000.jsonl'],
                         hashlib.sha256((workspace / 'source-0000.jsonl').read_bytes()).hexdigest())
        self.assertEqual(len(manifest['reference_audit_sha256']), 1)

    def test_rejects_changed_inventory_missing_exclusion_and_shared_group(self):
        self.inventory[0]['text'] = 'changed'
        self.inventory_path.write_bytes(review_corpus.encoded_rows(self.inventory))
        with self.assertRaisesRegex(ValueError, 'inventory does not match'):
            self.run_prepare()
        _, self.inventory = inventory.parse_history(self.history, 'project', self.revision)
        self.inventory_path.write_bytes(review_corpus.encoded_rows(self.inventory))
        self.exclusions.write_text(json.dumps({'source': {'project': 'project', 'commit': '9' * 40}}) + '\n')
        with self.assertRaisesRegex(ValueError, 'absent'):
            self.run_prepare()
        self.exclusions.write_text(json.dumps({'source': {'project': 'project', 'commit': 'a' * 40}}) + '\n')
        with self.assertRaisesRegex(ValueError, 'absent from the reference'):
            self.run_prepare('reference-group')

    def test_checks_every_supplied_reference_plan(self):
        row = {'id': 'prior', 'text': 'another candidate', 'family': 'prior', 'verified': False,
               'intents': ['build'], 'capabilities': [], 'risks': [], 'source': {'project': 'prior'}}
        (self.root / 'prior-source.jsonl').write_text(json.dumps(row) + '\n')
        plan = json.loads(self.plan.read_text())
        plan['sources'] = [{'path': 'prior-source.jsonl', 'project': 'prior', 'split': 'validation'}]
        second = self.root / 'prior-plan.json'
        second.write_text(json.dumps(plan))
        with patch.object(inventory, 'read_history', return_value=self.history):
            result = prepare.prepare(self.root, self.revision, 'project', 'project-group', self.inventory_path,
                                     self.exclusions, [self.plan, second], self.taxonomy, self.root / 'output', 10000)
        self.assertEqual(result['excluded_reference_overlap'], 2)
        self.assertEqual(result['pending_review'], 1)


if __name__ == '__main__':
    unittest.main()
