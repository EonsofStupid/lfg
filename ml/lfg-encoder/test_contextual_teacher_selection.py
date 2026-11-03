import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import contextual_teacher_selection as selection
import inventory_codex_sessions as inventory
import screen_private_corpus as screen


class ContextualSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        project = self.root / 'project'
        project.mkdir()
        self.session = self.root / 'session.jsonl'
        def message(text):
            return {'type': 'response_item', 'payload': {'type': 'message', 'role': 'user',
                    'content': [{'type': 'input_text', 'text': text}],
                    'internal_chat_message_metadata_passthrough': {'content_item_kinds': ['user.text']}}}
        records = [{'type': 'session_meta', 'payload': {'cwd': str(project)}},
                   message('Check password=supersecretvalue in the project.'), message('please continue')]
        self.session.write_text(''.join(json.dumps(row) + '\n' for row in records), encoding='utf-8')
        status, rows = inventory.scan_file(self.session, {'example': inventory.normalize(str(project))}, 100000, 100000)
        self.assertEqual(status, 'scanned')
        source = {**rows[-1], 'kind': 'codex-session-user-text'}
        original = {'id': 'request-1', 'text': 'please continue', 'family': 'family-1', 'source': source}
        source_raw = (json.dumps(original, sort_keys=True) + '\n').encode()
        source_path = self.root / 'source.jsonl'
        source_path.write_bytes(source_raw)
        plan = {'schema_version': 'lfg.encoder-development-corpus-plan.v2', 'sources': [
            {'path': str(source_path), 'project': 'example', 'project_group': 'group', 'split': 'train'},
            {'path': str(self.root / 'must-not-open'), 'project': 'heldout', 'project_group': 'heldout', 'split': 'test'}]}
        self.plan = self.root / 'plan.json'
        self.plan.write_text(json.dumps(plan), encoding='utf-8')
        self.policy = Path(__file__).with_name('private-corpus-screen.example.json')
        _, _, policy_hash = screen.load_policy(self.policy, 1048576)
        text = original['text']
        row = {'schema_version': screen.OUTPUT_VERSION, 'id': original['id'], 'text': text,
               'family': original['family'], 'project': 'example', 'project_group': 'group',
               'split': 'train', 'review_status': 'unreviewed', 'screened_text_sha256': selection.sha(text.encode()),
               'source_text_sha256': selection.sha(text.encode()), 'source_record_sha256': selection.sha(source_raw.rstrip(b'\n')),
               'candidate_file_sha256': selection.sha(source_raw), 'screen_policy_sha256': policy_hash}
        self.dataset = self.root / 'dataset'
        self.dataset.mkdir()
        train = selection.encoded(row)
        (self.dataset / 'train.jsonl').write_bytes(train)
        (self.dataset / 'manifest.json').write_text(json.dumps({'schema_version': 'lfg.private-corpus-screen-manifest.v1',
            'plan_sha256': selection.sha(self.plan.read_bytes()), 'policy_sha256': policy_hash,
            'outputs': {'train': selection.sha(train)}}), encoding='utf-8')
        self.config = self.root / 'config.json'
        config = {'schema_version': 'lfg.contextual-teacher-selection.v1', 'seed': 'test',
                  'max_manifest_bytes': 1048576, 'max_train_bytes': 1048576, 'max_source_bytes': 1048576,
                  'max_session_bytes': 1048576, 'max_line_bytes': 1048576, 'max_prior_messages': 3,
                  'max_prior_bytes': 4096, 'max_ancestry_records': 1000, 'min_followup_bytes': 1, 'max_followup_bytes': 80,
                  'max_request_bytes': 2048, 'max_candidates': 100, 'max_selected': 10, 'max_per_project': 10,
                  'prior_prefix': 'Previous user context:\n', 'separator': '\n', 'current_prefix': '\nCurrent request:\n'}
        self.config.write_text(json.dumps(config), encoding='utf-8')

    def test_verified_context_is_screened_and_test_source_is_never_opened(self):
        output = self.root / 'selected'
        report = selection.select(self.plan, self.dataset, self.policy, self.config, output)
        row = json.loads((output / 'requests.jsonl').read_text(encoding='utf-8'))
        self.assertEqual(report['requests'], 1)
        self.assertFalse(report['test_opened'])
        self.assertFalse(report['validation_opened'])
        self.assertIn('please continue', row['text'])
        self.assertIn('REDACTED', row['text'])
        self.assertNotIn('supersecretvalue', row['text'])
        self.assertEqual(row['context_message_count'], 1)
        self.assertEqual(row['outcome'], 'unknown')

    def test_changed_original_prefix_cannot_produce_selection(self):
        self.session.write_text(self.session.read_text(encoding='utf-8').replace('Check password', 'Changed password'), encoding='utf-8')
        output = self.root / 'selected'
        with self.assertRaisesRegex(ValueError, 'no eligible contextual requests'):
            selection.select(self.plan, self.dataset, self.policy, self.config, output)
        self.assertFalse(output.exists())

    def test_excluded_request_does_not_reenter_training(self):
        cfg = json.loads(self.config.read_text())
        cfg['exclude_ids'] = ['request-1']
        self.config.write_text(json.dumps(cfg), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'no eligible contextual requests'):
            selection.select(self.plan, self.dataset, self.policy, self.config, self.root / 'selected')

    @unittest.skipUnless(shutil.which('git'), 'Git is required for output privacy checks')
    def test_git_output_must_be_ignored_and_not_tracked(self):
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True)
        output = self.root / 'private' / 'selection'
        with self.assertRaisesRegex(ValueError, 'ignored and untracked'):
            selection.private_directory(output)
        (self.root / '.gitignore').write_text('private/\n', encoding='utf-8')
        selection.private_directory(output)
        output.mkdir(parents=True)
        (output / 'requests.jsonl').write_text('fixture\n', encoding='utf-8')
        subprocess.run(['git', '-C', str(self.root), 'add', '-f', 'private/selection/requests.jsonl'], check=True)
        with self.assertRaisesRegex(ValueError, 'ignored and untracked'):
            selection.private_directory(output)


if __name__ == '__main__':
    unittest.main()
