import json
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

AVAILABLE = all(importlib.util.find_spec(name) for name in ('torch', 'safetensors'))
if AVAILABLE:
    import stage_owned_source_corpus as subject


@unittest.skipUnless(AVAILABLE, 'torch and safetensors required')
class OwnedSourceCorpusTests(unittest.TestCase):
    def test_stages_only_selected_tracked_files_with_redaction_and_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            repo = root / 'repo'; repo.mkdir()
            subprocess.run(['git', 'init', '-q', str(repo)], check=True)
            (repo / 'crates').mkdir()
            (repo / 'crates' / 'main.rs').write_text('fn main() { /* token fixture-secret-123 */ }\n')
            (repo / 'crates' / 'generated.rs').write_text('generated content is excluded')
            subprocess.run(['git', '-C', str(repo), 'add', 'crates'], check=True)
            subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Test', '-c', 'user.email=test@example.test',
                            'commit', '-qm', 'fixture'], check=True)
            (repo / 'crates' / 'untracked.rs').write_text('untracked source must not be selected')
            policy = root / 'policy.json'
            policy.write_text(json.dumps({'schema_version': 'lfg.private-corpus-screen.v1', 'revision': 'fixture',
                'limits': {'max_policy_bytes': 65536, 'max_plan_bytes': 65536, 'max_source_bytes': 10000,
                           'max_total_source_bytes': 100000, 'max_total_output_bytes': 100000, 'max_records': 100,
                           'max_text_bytes': 10000, 'max_matches_per_record': 100},
                'rules': [{'id': 'fixture-secret', 'pattern': r'fixture-secret-[0-9]+', 'replacement': '[REDACTED]'}]}))
            config = root / 'config.json'
            config.write_text(json.dumps({'schema_version': 'lfg.owned-source-selection.v1', 'sampling': 'group_uniform',
                'min_document_bytes': 1, 'max_total_bytes': 100000, 'max_files': 10,
                'projects': [{'id': 'project', 'group': 'group', 'root': str(repo),
                    'include_paths': ['crates'], 'extensions': ['.rs'], 'exclude_substrings': ['generated']}]}))
            out = root / 'private-output'
            result = subject.stage(config, policy, out)
            self.assertEqual(result['documents'], 1)
            self.assertEqual(result['redaction_counts'], {'fixture-secret': 1})
            self.assertEqual(result['sources'][0]['source_path'], 'crates/main.rs')
            self.assertNotIn('fixture-secret-123', (out / 'document-000001.utf8').read_text())
            self.assertFalse(result['test_opened'])


if __name__ == '__main__':
    unittest.main()
