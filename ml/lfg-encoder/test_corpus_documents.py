"""Fixture-only tests of local screened multi-document preparation."""
import json
from pathlib import Path
import random
import tempfile
import unittest

import corpus_documents as subject


class CorpusDocumentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.policy = Path(subject.__file__).with_name('private-corpus-screen.example.json')
        self.first = self.root / 'first.txt'
        self.second = self.root / 'second.txt'
        self.first.write_text('A' * 20 + ' Bearer fixtureSecret0123456789 ' + 'A' * 20)
        self.second.write_text('B' * 80)
        def source(path, project):
            return {'path': str(path), 'project': project, 'project_group': project,
                    'split': 'train', 'sha256': subject.digest(path.read_bytes())}
        self.plan_value = {'schema_version': 'lfg.document-development-plan.v1',
                           'sources': [source(self.first, 'one'), source(self.second, 'two'),
                                       {'path': str(self.root / 'must-not-open'), 'project': 'validation',
                                        'project_group': 'validation', 'split': 'validation', 'sha256': 'f' * 64}]}
        self.plan = self.root / 'plan.json'
        self.write_plan()

    def write_plan(self):
        self.plan.write_text(json.dumps(self.plan_value), encoding='utf-8')

    def load(self):
        return subject.load(self.plan, self.policy, 10000, 8)

    def test_redaction_and_validation_exclusion(self):
        result = self.load()
        self.assertEqual(len(result['documents']), 2)
        self.assertNotIn(b'fixtureSecret0123456789', result['documents'][0])
        self.assertEqual(result['identity']['redaction_counts']['bearer-credential'], 1)
        self.assertFalse(result['identity']['redaction_complete'])
        self.assertFalse(result['identity']['validation_sources_opened'])
        self.assertNotIn(str(self.first), json.dumps(result['identity']))
        subject.verify(result)

    def test_sampling_does_not_cross_document_boundaries(self):
        result = self.load()
        inputs, targets = subject.batch(result, random.Random(4), 100, 8)
        for left, right in zip(inputs, targets, strict=True):
            combined = bytes(left + right[-1:])
            self.assertTrue(any(combined in document for document in result['documents']))

    def test_group_uniform_sampling_balances_unequal_groups(self):
        self.plan_value['sampling'] = 'group_uniform'
        self.first.write_text('A' * 2000)
        self.plan_value['sources'][0]['sha256'] = subject.digest(self.first.read_bytes())
        self.write_plan()
        result = self.load()
        inputs, _ = subject.batch(result, random.Random(4), 1000, 8)
        second = sum(bytes(row) == b'B' * 8 for row in inputs)
        self.assertGreater(second, 400)
        self.assertLess(second, 600)
        self.assertEqual(result['identity']['sampling'], 'group_uniform')

    def test_test_split_rejected_before_source_open(self):
        self.plan_value['sources'][0]['path'] = str(self.root / 'missing')
        self.plan_value['sources'][-1]['split'] = 'test'
        self.write_plan()
        with self.assertRaisesRegex(ValueError, 'held-out split'):
            self.load()

    def test_declared_cross_split_identity_fails(self):
        self.plan_value['sources'][-1]['sha256'] = self.plan_value['sources'][0]['sha256']
        self.write_plan()
        with self.assertRaisesRegex(ValueError, 'crosses split'):
            self.load()

    def test_source_changes_are_rejected(self):
        result = self.load()
        self.first.write_text('changed fixture')
        with self.assertRaisesRegex(ValueError, 'changed'):
            subject.verify(result)
        with self.assertRaisesRegex(ValueError, 'differs'):
            self.load()


if __name__ == '__main__':
    unittest.main()
