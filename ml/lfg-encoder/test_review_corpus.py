import json
import contextlib
import io
import hashlib
import os
import subprocess
import threading
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
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
        self.assertEqual(stale['event']['judgments'], {'intents': {'build': 'positive'}})
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

    def test_partial_review_rejects_stale_or_unsupported_data(self):
        binding = self.partial_binding()
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
            with self.assertRaisesRegex(ValueError, 'workspace review lock'):
                review.record_partial_decision(self.workspace, 'id-0', 'reviewer', binding,
                    {'intents': {'build': 'negative'}}, {}, None, [], 65536, 'REVIEW PARTIAL LABELS')
            lock.write_text(original, encoding='utf-8')
            with patch('review_corpus.os.replace', side_effect=OSError('simulated interruption')):
                with self.assertRaisesRegex(OSError, 'simulated interruption'):
                    review.record_partial_decision(self.workspace, 'id-0', 'reviewer', binding,
                        {'intents': {'build': 'negative'}}, {}, None, [], 65536, 'REVIEW PARTIAL LABELS')
        self.assertFalse(sidecar.exists())

    def test_direct_partial_write_acquires_process_lock(self):
        binding = self.partial_binding()
        review.record_partial_decision(self.workspace, 'id-0', 'reviewer', binding,
            {'intents': {'build': 'negative'}}, {}, None, [], 65536, 'REVIEW PARTIAL LABELS')
        self.assertEqual(review.partial_review_state(self.workspace, 'id-0', 65536)['state'], 'current')

    def test_full_approval_rejects_a_competing_process_lock(self):
        marker = self.workspace / 'review.lock'
        marker.write_text('pid=99999999\n', encoding='utf-8')
        try:
            with self.assertRaisesRegex(ValueError, 'another process holds'):
                review.record_decision(self.workspace, 'id-0', 'reviewer', 'approved', self.digest('id-0'))
        finally:
            marker.unlink()
        self.assertEqual(review.status(self.workspace)['decisions']['pending'], 3)

    def test_partial_review_revisions_retractions_and_stale_revision_rejection(self):
        binding = self.partial_binding()
        with train.run_lock(self.workspace / 'review'):
            review.record_partial_decision(self.workspace, 'id-0', 'reviewer', binding,
                {'intents': {'build': 'positive'}}, {}, None, [], 65536, 'REVIEW PARTIAL LABELS')
            first_state = review.partial_review_state(self.workspace, 'id-0', 65536)
            first = first_state['revision']
            review.record_partial_decision(self.workspace, 'id-0', 'reviewer', binding,
                {'intents': {'build': 'negative'}}, {}, None, [], 65536, 'REVISE PARTIAL REVIEW',
                'revise', first['event_id'], first['event_sha256'])
            second_state = review.partial_review_state(self.workspace, 'id-0', 65536)
            second = second_state['revision']
            self.assertEqual(second_state['state'], 'current')
            self.assertEqual(second_state['event']['judgments'], {'intents': {'build': 'negative'}})
            self.assertEqual(len(second_state['history']), 2)
            with self.assertRaisesRegex(ValueError, 'stale or conflicts'):
                review.record_partial_decision(self.workspace, 'id-0', 'reviewer', binding,
                    {'intents': {'build': 'positive'}}, {}, None, [], 65536, 'REVISE PARTIAL REVIEW',
                    'revise', first['event_id'], first['event_sha256'])
            review.record_partial_decision(self.workspace, 'id-0', 'reviewer', binding,
                {}, {}, None, [], 65536, 'RETRACT PARTIAL REVIEW',
                'retract', second['event_id'], second['event_sha256'])
        retracted = review.partial_review_state(self.workspace, 'id-0', 65536)
        self.assertEqual(retracted['state'], 'retracted')
        self.assertEqual(len(retracted['history']), 3)
        self.assertEqual(retracted['event']['judgments'], {})

    def test_partial_review_can_be_revised_after_guide_change_and_checks_guide_at_commit(self):
        guide = self.root / 'guide.json'
        guide.write_text('old guidance', encoding='utf-8')
        old_sha = hashlib.sha256(guide.read_bytes()).hexdigest()
        binding = self.partial_binding(guide=old_sha)
        with train.run_lock(self.workspace / 'review'):
            review.record_partial_decision(self.workspace, 'id-0', 'reviewer', binding,
                {'intents': {'build': 'positive'}}, {}, old_sha, [], 65536, 'REVIEW PARTIAL LABELS',
                guide_path=guide)
        first = review.partial_review_state(self.workspace, 'id-0', 65536, old_sha)['revision']
        guide.write_text('new reviewed guidance', encoding='utf-8')
        new_sha = hashlib.sha256(guide.read_bytes()).hexdigest()
        stale = review.partial_review_state(self.workspace, 'id-0', 65536, new_sha)
        self.assertEqual(stale['state'], 'stale')
        with train.run_lock(self.workspace / 'review'):
            review.record_partial_decision(self.workspace, 'id-0', 'reviewer', stale['binding'],
                {'intents': {'build': 'negative'}}, {}, new_sha, [], 65536, 'REVISE PARTIAL REVIEW',
                'revise', first['event_id'], first['event_sha256'], guide)
        self.assertEqual(review.partial_review_state(self.workspace, 'id-0', 65536, new_sha)['state'], 'current')
        second = review.partial_review_state(self.workspace, 'id-0', 65536, new_sha)['revision']
        guide.write_text('changed while saving', encoding='utf-8')
        changed_binding = review.partial_review_state(self.workspace, 'id-0', 65536, new_sha)['binding']
        with train.run_lock(self.workspace / 'review'):
            with self.assertRaisesRegex(ValueError, 'guide changed'):
                review.record_partial_decision(self.workspace, 'id-0', 'reviewer', changed_binding,
                    {'intents': {'build': 'positive'}}, {}, new_sha, [], 65536, 'REVISE PARTIAL REVIEW',
                    'revise', second['event_id'], second['event_sha256'], guide)

    def test_partial_writers_serialize_same_process_threads_without_lost_updates(self):
        barrier = threading.Barrier(2)

        def save(identifier, label):
            binding = self.partial_binding(identifier)
            barrier.wait(timeout=5)
            try:
                review.record_partial_decision(self.workspace, identifier, 'reviewer', binding,
                    {'intents': {label: 'positive'}}, {}, None, [], 65536, 'REVIEW PARTIAL LABELS')
                return 'saved'
            except ValueError as error:
                return str(error)

        with train.run_lock(self.workspace / 'review'):
            with ThreadPoolExecutor(max_workers=2) as pool:
                different = list(pool.map(lambda value: save(*value), [('id-0', 'build'), ('id-1', 'explain')]))
        self.assertEqual(different, ['saved', 'saved'])
        self.assertEqual(len(review._read_partial_events(self.workspace / review.PARTIAL_REVIEW_FILE, 65536)), 2)

        barrier = threading.Barrier(2)
        with train.run_lock(self.workspace / 'review'):
            with ThreadPoolExecutor(max_workers=2) as pool:
                same = list(pool.map(lambda _: save('id-2', 'build'), range(2)))
        self.assertEqual(sum(result == 'saved' for result in same), 1)
        self.assertEqual(sum('already exists' in result for result in same), 1)
        events = review._read_partial_events(self.workspace / review.PARTIAL_REVIEW_FILE, 65536)
        self.assertEqual(len(events), 3)

    def test_partial_review_is_separate_from_approved_export(self):
        binding = self.partial_binding()
        with train.run_lock(self.workspace / 'review'):
            review.record_partial_decision(self.workspace, 'id-0', 'reviewer', binding,
                {'intents': {'build': 'positive'}}, {}, None, [], 65536, 'REVIEW PARTIAL LABELS')
        with self.assertRaisesRegex(ValueError, 'no current approvals'):
            review.export_approved(self.workspace, self.root / 'partial-export')

    def test_library_approval_requires_and_persists_partial_reconciliation(self):
        binding = self.partial_binding()
        with train.run_lock(self.workspace / 'review'):
            review.record_partial_decision(self.workspace, 'id-0', 'reviewer', binding,
                {'intents': {'build': 'positive'}}, {}, None, [], 65536, 'REVIEW PARTIAL LABELS')
        partial = review.partial_review_state(self.workspace, 'id-0', 65536)
        revision = partial['revision']
        with self.assertRaisesRegex(ValueError, 'explicitly reconcile'):
            review.record_decision(self.workspace, 'id-0', 'reviewer', 'approved', self.digest('id-0'),
                {'intents': ['build'], 'capabilities': ['write'], 'risks': []},
                max_partial_review_bytes=65536)
        self.assertEqual(review.status(self.workspace)['decisions']['pending'], 3)
        reconciliation = {'event_id': revision['event_id'], 'event_sha256': revision['event_sha256'],
                           'confirmation': 'RECONCILE PARTIAL REVIEW'}
        review.record_decision(self.workspace, 'id-0', 'reviewer', 'approved', self.digest('id-0'),
            {'intents': ['build'], 'capabilities': ['write'], 'risks': []},
            partial_reconciliation=reconciliation, max_partial_review_bytes=65536)
        _, _, ledger = review.load_workspace(self.workspace)
        self.assertEqual(ledger['id-0']['partial_reconciliation'], {
            'event_id': revision['event_id'], 'event_sha256': revision['event_sha256'],
            'event_type': 'decision', 'revision': 1, 'acknowledgment': 'RECONCILE PARTIAL REVIEW'})

    def test_interactive_cli_cannot_approve_past_partial_review(self):
        binding = self.partial_binding()
        with train.run_lock(self.workspace / 'review'):
            review.record_partial_decision(self.workspace, 'id-0', 'reviewer', binding,
                {'intents': {'build': 'positive'}}, {}, None, [], 65536, 'REVIEW PARTIAL LABELS')
        with patch('builtins.input', side_effect=['a', 'APPROVE']), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'configure --max-partial-review-bytes'):
                review.interactive_review(self.workspace, 'reviewer')
        self.assertEqual(review.status(self.workspace)['decisions']['pending'], 3)

    def test_interactive_cli_reconciles_only_after_showing_partial_history(self):
        binding = self.partial_binding()
        with train.run_lock(self.workspace / 'review'):
            review.record_partial_decision(self.workspace, 'id-0', 'reviewer', binding,
                {'intents': {'build': 'positive'}}, {}, None, [], 65536, 'REVIEW PARTIAL LABELS')
        output = io.StringIO()
        with patch('builtins.input', side_effect=['a', 'APPROVE', 'RECONCILE PARTIAL REVIEW', 'q']), \
                contextlib.redirect_stdout(output):
            review.interactive_review(self.workspace, 'reviewer', 65536)
        self.assertIn('SAVED PARTIAL REVIEW HISTORY', output.getvalue())
        _, _, ledger = review.load_workspace(self.workspace)
        self.assertEqual(ledger['id-0']['partial_reconciliation']['acknowledgment'],
                         'RECONCILE PARTIAL REVIEW')

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
        binding = self.partial_binding()
        with train.run_lock(self.workspace / 'review'):
            with self.assertRaisesRegex(ValueError, 'ignored by Git'):
                review.record_partial_decision(self.workspace, 'id-0', 'reviewer', binding,
                    {'intents': {'build': 'positive'}}, {}, None, [], 65536, 'REVIEW PARTIAL LABELS')
        self.assertFalse((self.workspace / review.PARTIAL_REVIEW_FILE).exists())
        self.assertFalse(list(self.workspace.glob('.partial-review-*')))
        (self.root / '.gitignore').write_text(
            'workspace/partial-reviews.v1.jsonl\nworkspace/.partial-review-*\n', encoding='utf-8')
        with train.run_lock(self.workspace / 'review'):
            review.record_partial_decision(self.workspace, 'id-0', 'reviewer', binding,
                {'intents': {'build': 'positive'}}, {}, None, [], 65536, 'REVIEW PARTIAL LABELS')

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

    def test_interrupted_approval_with_partial_history_can_be_reconciled_on_recovery(self):
        binding = self.partial_binding()
        with train.run_lock(self.workspace / 'review'):
            review.record_partial_decision(self.workspace, 'id-0', 'reviewer', binding,
                {'intents': {'build': 'positive'}}, {}, None, [], 65536, 'REVIEW PARTIAL LABELS')
        state = review.partial_review_state(self.workspace, 'id-0', 65536)
        revision = state['revision']
        reconciliation = {'event_id': revision['event_id'], 'event_sha256': revision['event_sha256'],
                           'confirmation': 'RECONCILE PARTIAL REVIEW'}
        original = review.atomic_replace

        def fail_ledger(path, content):
            if path.name == 'reviews.jsonl':
                raise OSError('simulated interruption')
            original(path, content)

        with patch.object(review, 'atomic_replace', side_effect=fail_ledger), self.assertRaises(OSError):
            review.record_decision(self.workspace, 'id-0', 'reviewer', 'approved', self.digest('id-0'),
                {'intents': ['explain'], 'capabilities': [], 'risks': []},
                [{'start': 8, 'end': 15, 'replacement': '[request]'}],
                partial_reconciliation=reconciliation, max_partial_review_bytes=65536)
        row = json.loads((self.workspace / 'source-0000.jsonl').read_text(encoding='utf-8'))
        self.assertTrue(row['verified'])
        self.assertEqual(row['intents'], ['explain'])
        self.assertEqual(row['text'], 'fixture [request] 0')
        self.assertEqual(review.status(self.workspace)['decisions']['pending'], 3)
        recovered = review.partial_review_state(self.workspace, 'id-0', 65536)
        self.assertEqual(recovered['state'], 'stale')
        self.assertEqual(recovered['event']['judgments'], {'intents': {'build': 'positive'}})
        review.record_decision(self.workspace, 'id-0', 'reviewer', 'approved', self.digest('id-0'),
            partial_reconciliation=reconciliation, max_partial_review_bytes=65536)
        _, _, ledger = review.load_workspace(self.workspace)
        self.assertEqual(ledger['id-0']['decision'], 'approved')
        self.assertEqual(ledger['id-0']['partial_reconciliation']['event_id'], revision['event_id'])

    def test_interrupted_approval_rejects_externally_changed_verified_content(self):
        binding = self.partial_binding()
        with train.run_lock(self.workspace / 'review'):
            review.record_partial_decision(self.workspace, 'id-0', 'reviewer', binding,
                {'intents': {'build': 'positive'}}, {}, None, [], 65536, 'REVIEW PARTIAL LABELS')
        revision = review.partial_review_state(self.workspace, 'id-0', 65536)['revision']
        reconciliation = {'event_id': revision['event_id'], 'event_sha256': revision['event_sha256'],
                           'confirmation': 'RECONCILE PARTIAL REVIEW'}
        original = review.atomic_replace

        def fail_ledger(path, content):
            if path.name == 'reviews.jsonl':
                raise OSError('simulated interruption')
            original(path, content)

        with patch.object(review, 'atomic_replace', side_effect=fail_ledger), self.assertRaises(OSError):
            review.record_decision(self.workspace, 'id-0', 'reviewer', 'approved', self.digest('id-0'),
                {'intents': ['explain'], 'capabilities': ['write'], 'risks': []},
                partial_reconciliation=reconciliation, max_partial_review_bytes=65536)
        path = self.workspace / 'source-0000.jsonl'
        row = json.loads(path.read_text(encoding='utf-8'))
        row['text'] = 'unrelated external modification'
        path.write_text(json.dumps(row) + '\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'interrupted review source content changed'):
            review.record_decision(self.workspace, 'id-0', 'reviewer', 'approved', self.digest('id-0'),
                partial_reconciliation=reconciliation, max_partial_review_bytes=65536)
        self.assertEqual(review.status(self.workspace)['decisions']['pending'], 3)

    def test_interrupted_write_accepts_previous_bound_marker_shape(self):
        original = review.atomic_replace

        def fail_ledger(path, content):
            if path.name == 'reviews.jsonl':
                raise OSError('simulated interruption')
            original(path, content)

        with patch.object(review, 'atomic_replace', side_effect=fail_ledger), self.assertRaises(OSError):
            review.record_decision(self.workspace, 'id-0', 'reviewer', 'approved', self.digest('id-0'),
                                   {'intents': ['explain'], 'capabilities': [], 'risks': []})
        path = self.workspace / 'source-0000.jsonl'
        row = json.loads(path.read_text(encoding='utf-8'))
        marker = row['review_write_provenance']
        marker.pop('recovery_mode')
        marker.pop('legacy_recovery_acknowledgment')
        path.write_text(json.dumps(row) + '\n', encoding='utf-8')
        review.record_decision(self.workspace, 'id-0', 'reviewer', 'approved', self.digest('id-0'))
        _, _, ledger = review.load_workspace(self.workspace)
        self.assertEqual(ledger['id-0']['decision'], 'approved')

    def test_legacy_interrupted_approval_requires_fresh_attested_rereview(self):
        path = self.workspace / 'source-0000.jsonl'
        row = json.loads(path.read_text(encoding='utf-8'))
        row['verified'] = True
        row['text'] = 'legacy interrupted content to re-review'
        row['intents'] = ['explain']
        path.write_text(json.dumps(row) + '\n', encoding='utf-8')
        digest = self.digest('id-0')
        with self.assertRaisesRegex(ValueError, 'requires explicit re-review'):
            review.record_decision(self.workspace, 'id-0', 'reviewer', 'approved', digest)
        review.record_decision(self.workspace, 'id-0', 'reviewer', 'approved', digest,
            interrupted_recovery_ack='REVIEW CURRENT INTERRUPTED RECORD')
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        marker = rows[0]['review_write_provenance']
        self.assertEqual(marker['recovery_mode'], 'legacy_explicit_rereview')
        self.assertEqual(marker['legacy_recovery_acknowledgment'], 'REVIEW CURRENT INTERRUPTED RECORD')
        _, _, ledger = review.load_workspace(self.workspace)
        self.assertEqual(ledger['id-0']['decision'], 'approved')

    def test_interactive_cli_supports_explicit_legacy_interrupted_rereview(self):
        path = self.workspace / 'source-0000.jsonl'
        row = json.loads(path.read_text(encoding='utf-8'))
        row['verified'] = True
        row['text'] = 'legacy interrupted content to re-review'
        path.write_text(json.dumps(row) + '\n', encoding='utf-8')
        output = io.StringIO()
        with patch('builtins.input', side_effect=['a', 'APPROVE', 'REVIEW CURRENT INTERRUPTED RECORD', 'q']), \
                contextlib.redirect_stdout(output):
            review.interactive_review(self.workspace, 'reviewer')
        _, _, ledger = review.load_workspace(self.workspace)
        self.assertEqual(ledger['id-0']['decision'], 'approved')
        saved_row = json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual(saved_row['review_write_provenance']['recovery_mode'], 'legacy_explicit_rereview')

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
