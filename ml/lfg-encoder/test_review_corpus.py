import json
import contextlib
import io
import hashlib
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import corpus_audit
import review_corpus as review
import train


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

    def partial_binding(self, identifier='id-0', guide=None):
        state = review.partial_review_state(self.workspace, identifier, 65536, guide)
        return state['binding']

    def test_partial_review_keeps_unknown_and_proposal_rejection_out_of_legacy_labels(self):
        plan, _, _ = review.load_workspace(self.workspace)
        source_path, _, _, _ = review.find_row(self.workspace, plan, 'id-0')
        source_before = source_path.read_bytes()
        ledger_before = (self.workspace / 'reviews.jsonl').read_bytes()
        binding = self.partial_binding()
        with train.run_lock(self.workspace / 'review'):
            review.record_partial_decision(self.workspace, 'id-0', 'fixture reviewer', binding,
                {'intents': {'build': 'positive'}}, {'intents': {'build': 'rejected'}}, None, [], 65536,
                'REVIEW PARTIAL LABELS')
        state = review.partial_review_state(self.workspace, 'id-0', 65536)
        self.assertEqual(state['state'], 'current')
        self.assertEqual(state['event']['judgments'], {'intents': {'build': 'positive'}})
        self.assertEqual(state['event']['proposal_actions'], {'intents': {'build': 'rejected'}})
        self.assertEqual(source_path.read_bytes(), source_before)
        self.assertEqual((self.workspace / 'reviews.jsonl').read_bytes(), ledger_before)
        self.assertEqual(review.status(self.workspace)['decisions']['pending'], 3)
        self.assertEqual(review.status(self.workspace)['counts']['train']['ready_for_reviewed_subset'], 0)
        self.assertFalse(state['event']['judgments'].get('capabilities'))
        stale = review.partial_review_state(self.workspace, 'id-0', 65536, 'f' * 64)
        self.assertEqual(stale['state'], 'stale')
        self.assertIsNone(stale['event'])
        if os.name != 'nt':
            self.assertEqual((self.workspace / review.PARTIAL_REVIEW_FILE).stat().st_mode & 0o777, 0o600)
        with self.assertRaisesRegex(ValueError, 'already exists'):
            with train.run_lock(self.workspace / 'review'):
                review.record_partial_decision(self.workspace, 'id-0', 'fixture reviewer', binding,
                    {'intents': {'explain': 'negative'}}, {}, None, [], 65536, 'REVIEW PARTIAL LABELS')

    def test_rejecting_proposal_without_judgment_remains_unknown(self):
        binding = self.partial_binding()
        with train.run_lock(self.workspace / 'review'):
            review.record_partial_decision(self.workspace, 'id-0', 'fixture reviewer', binding,
                {}, {'intents': {'build': 'rejected'}}, None, [], 65536, 'REVIEW PARTIAL LABELS')
        event = review.partial_review_state(self.workspace, 'id-0', 65536)['event']
        self.assertEqual(event['judgments'], {})
        self.assertEqual(event['proposal_actions']['intents']['build'], 'rejected')
        binding = self.partial_binding('id-1')
        with train.run_lock(self.workspace / 'review'):
            review.record_partial_decision(self.workspace, 'id-1', 'fixture reviewer', binding,
                {'intents': {'explain': 'negative'}}, {}, None, [], 65536, 'REVIEW PARTIAL LABELS')
        self.assertEqual(review.partial_review_state(self.workspace, 'id-1', 65536)['event']['judgments'],
                         {'intents': {'explain': 'negative'}})

    def test_partial_review_requires_lock_and_rejects_stale_or_unsupported_data(self):
        binding = self.partial_binding()
        with self.assertRaisesRegex(ValueError, 'active workspace review lock'):
            review.record_partial_decision(self.workspace, 'id-0', 'reviewer', binding,
                {'intents': {'build': 'negative'}}, {}, None, [], 65536, 'REVIEW PARTIAL LABELS')
        with train.run_lock(self.workspace / 'review'):
            with self.assertRaisesRegex(ValueError, 'changed'):
                review.record_partial_decision(self.workspace, 'id-0', 'reviewer', {**binding, 'guide_sha256': 'f' * 64},
                    {'intents': {'build': 'negative'}}, {}, None, [], 65536, 'REVIEW PARTIAL LABELS')
            with self.assertRaisesRegex(ValueError, 'invalid intents judgment'):
                review.record_partial_decision(self.workspace, 'id-0', 'reviewer', {**binding, 'guide_sha256': None},
                    {'intents': {'not-in-taxonomy': 'negative'}}, {}, None, [], 65536, 'REVIEW PARTIAL LABELS')
        changed = self.workspace / 'source-0000.jsonl'
        rows = [json.loads(line) for line in changed.read_text(encoding='utf-8').splitlines()]
        rows[0]['review_note'] = 'changed after display'
        changed.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'changed|reinitialize'):
            review.partial_review_state(self.workspace, 'id-0', 65536)

    def test_partial_review_rejects_competing_lock_and_interrupted_atomic_write(self):
        binding = self.partial_binding()
        sidecar = self.workspace / review.PARTIAL_REVIEW_FILE
        with train.run_lock(self.workspace / 'review'):
            lock = self.workspace / 'review.lock'
            original = lock.read_text(encoding='utf-8')
            lock.write_text('pid=99999999\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'active workspace review lock'):
                review.record_partial_decision(self.workspace, 'id-0', 'reviewer', binding,
                    {'intents': {'build': 'negative'}}, {}, None, [], 65536, 'REVIEW PARTIAL LABELS')
            lock.write_text(original, encoding='utf-8')
            with patch('review_corpus.os.replace', side_effect=OSError('simulated interruption')):
                with self.assertRaisesRegex(OSError, 'simulated interruption'):
                    review.record_partial_decision(self.workspace, 'id-0', 'reviewer', binding,
                        {'intents': {'build': 'negative'}}, {}, None, [], 65536, 'REVIEW PARTIAL LABELS')
        self.assertFalse(sidecar.exists())

    def test_partial_review_is_separate_from_approved_export(self):
        binding = self.partial_binding()
        with train.run_lock(self.workspace / 'review'):
            review.record_partial_decision(self.workspace, 'id-0', 'reviewer', binding,
                {'intents': {'build': 'positive'}}, {}, None, [], 65536, 'REVIEW PARTIAL LABELS')
        with self.assertRaisesRegex(ValueError, 'no current approvals'):
            review.export_approved(self.workspace, self.root / 'partial-export')

    def test_truncated_partial_sidecar_fails_closed(self):
        sidecar = self.workspace / review.PARTIAL_REVIEW_FILE
        sidecar.write_bytes(b'{"schema_version":"lfg.partial-label-review.v1"}')
        with self.assertRaisesRegex(ValueError, 'truncated'):
            review.partial_review_state(self.workspace, 'id-0', 65536)

    def test_partial_sidecar_must_be_ignored_when_workspace_is_inside_git(self):
        subprocess.run(['git', 'init', str(self.root)], capture_output=True, check=True)
        with self.assertRaisesRegex(ValueError, 'ignored by Git'):
            review.partial_review_state(self.workspace, 'id-0', 65536)
        (self.root / '.gitignore').write_text('workspace/partial-reviews.v1.jsonl\n', encoding='utf-8')
        self.assertEqual(review.partial_review_state(self.workspace, 'id-0', 65536)['state'], 'not_reviewed')

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

    def test_redacted_approval_preserves_provenance_and_exports_only_reviewed_text(self):
        original = self.rows[0]['text']
        start = original.index('request')
        operations = [{'start': start, 'end': start + len('request'), 'replacement': '[removed]'}]
        review.record_decision(self.workspace, 'id-0', 'person', 'approved', self.digest('id-0'),
                               {'intents': ['build'], 'capabilities': ['write'], 'risks': []}, operations)
        output = self.root / 'redacted-export'
        review.export_approved(self.workspace, output)
        exported = json.loads((output / 'source-0000.jsonl').read_text())
        self.assertEqual(exported['text'], 'fixture [removed] 0')
        transform = exported['review_transforms'][0]
        self.assertEqual(transform['before_sha256'], hashlib.sha256(original.encode()).hexdigest())
        self.assertEqual(transform['after_sha256'], hashlib.sha256(exported['text'].encode()).hexdigest())
        self.assertEqual(transform['operations'], operations)
        self.assertEqual(json.loads((self.root / 'input-0.jsonl').read_text())['text'], original)

    def test_redaction_rejects_invalid_ranges_duplicates_and_rejections(self):
        digest = self.digest('id-0')
        labels = {'intents': ['build'], 'capabilities': [], 'risks': []}
        for operations in ([{'start': True, 'end': 1, 'replacement': 'x'}],
                           [{'start': 0, 'end': 0, 'replacement': 'x'}],
                           [{'start': 0, 'end': 1, 'replacement': 'f'}],
                           [{'start': 0, 'end': 1000, 'replacement': 'x'}]):
            with self.subTest(operations=operations), self.assertRaises(ValueError):
                review.record_decision(self.workspace, 'id-0', 'person', 'approved', digest, labels, operations)
        with self.assertRaisesRegex(ValueError, 'require an approved'):
            review.record_decision(self.workspace, 'id-0', 'person', 'rejected', digest, None,
                                   [{'start': 0, 'end': 1, 'replacement': 'x'}])
        original = self.rows[0]['text']
        with self.assertRaisesRegex(ValueError, 'duplicates'):
            review.record_decision(self.workspace, 'id-0', 'person', 'approved', digest, labels,
                                   [{'start': len(original) - 1, 'end': len(original), 'replacement': '1'}])
        with self.assertRaisesRegex(ValueError, 'display limit'):
            review.record_decision(self.workspace, 'id-0', 'person', 'approved', digest, labels,
                                   [{'start': 0, 'end': 1, 'replacement': 'more'}], 2)
        self.assertEqual(review.status(self.workspace)['decisions']['approved'], 0)

    def test_redaction_offsets_use_unicode_codepoints(self):
        text = 'one \U0001f680 private data'
        operations = [{'start': 6, 'end': 13, 'replacement': '[x]'},
                      {'start': 0, 'end': 3, 'replacement': '1'}]
        self.assertEqual(review.apply_redactions(text, operations), '1 \U0001f680 [x] data')
        self.assertEqual(review.apply_redactions('one \U0001f680\r\nprivate data',
                         [{'start': 7, 'end': 14, 'replacement': '[x]'}]),
                         'one \U0001f680\r\n[x] data')

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

    def test_interrupted_redaction_remains_pending_and_can_be_reviewed_again(self):
        original_replace = review.atomic_replace
        def fail_ledger(path, content):
            if path.name == 'reviews.jsonl':
                raise OSError('simulated interruption')
            original_replace(path, content)
        operations = [{'start': 8, 'end': 15, 'replacement': '[x]'}]
        with patch.object(review, 'atomic_replace', side_effect=fail_ledger), self.assertRaises(OSError):
            review.record_decision(self.workspace, 'id-0', 'person', 'approved', self.digest('id-0'),
                                   {'intents': ['build'], 'capabilities': [], 'risks': []}, operations)
        self.assertEqual(review.status(self.workspace)['decisions']['approved'], 0)
        with self.assertRaisesRegex(ValueError, 'no current approvals'):
            review.export_approved(self.workspace, self.root / 'not-yet')
        review.record_decision(self.workspace, 'id-0', 'person', 'approved', self.digest('id-0'))
        row = json.loads((self.workspace / 'source-0000.jsonl').read_text(encoding='utf-8'))
        self.assertEqual(len(row['review_transforms']), 1)
        self.assertEqual(review.status(self.workspace)['decisions']['approved'], 1)

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
