"""Small software fixtures for resumable distillation; not model-quality data."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(all(importlib.util.find_spec(x) for x in ('torch', 'safetensors')), 'torch and safetensors required')
class ResumableDistillationTests(unittest.TestCase):
    def setUp(self):
        import owned_student
        import train_owned_distill_resumable as subject
        from safetensors.torch import save_file
        self.subject = subject
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sha = lambda raw: hashlib.sha256(raw).hexdigest()
        self.student = json.loads(Path(subject.base.__file__).with_name('owned-student.example.json').read_text())
        self.student.update(steps=3, batch_size=1, cpu_threads=1, max_parameters=100000)
        self.student['architecture'].update(width=8, layers=2, heads=2, context=8, attention_every=2)
        self.student_path = self.root / 'student.json'
        self.distill = json.loads(Path(subject.distill.__file__).with_name('owned-distillation.example.json').read_text())
        self.distill.update(top_k=4, max_teacher_parameters=100000)
        self.distill_path = self.root / 'distill.json'
        self.runtime = json.loads(Path(subject.__file__).with_name('owned-distillation-runtime.example.json').read_text())
        self.runtime.update(checkpoint_every=1)
        self.runtime_path = self.root / 'runtime.json'
        self.write_configs()
        self.teacher_dir = self.root / 'teacher'
        self.teacher_dir.mkdir()
        self.teacher_weights = self.teacher_dir / 'weights.safetensors'
        save_file(owned_student.OwnedByteStudent(self.student['architecture']).state_dict(), str(self.teacher_weights))
        self.teacher_manifest = self.teacher_dir / 'manifest.json'
        self.teacher_manifest.write_text(json.dumps({'schema_version': 'lfg.owned-student-experiment-result.v1',
            'architecture': self.student['architecture'], 'weights_sha256': self.sha(self.teacher_weights.read_bytes()),
            'code_sha256': {'owned_student.py': self.sha(Path(owned_student.__file__).read_bytes())}}))
        self.source = self.root / 'document.txt'
        self.source.write_text('Fixture source text for exercising a bounded distillation workflow. ' * 3)
        self.plan = {'schema_version': 'lfg.document-development-plan.v1', 'sampling': 'group_uniform',
                     'sources': [{'path': self.source.name, 'project': 'fixture', 'project_group': 'fixture',
                                  'split': 'train', 'sha256': self.sha(self.source.read_bytes())},
                                 {'path': 'must-not-open', 'project': 'other', 'project_group': 'other',
                                  'split': 'validation', 'sha256': 'b' * 64}]}
        self.plan_path = self.root / 'plan.json'
        self.write_plan()
        self.policy = Path(subject.base.__file__).with_name('private-corpus-screen.example.json')

    def write_configs(self):
        for path, value in ((self.student_path, self.student), (self.distill_path, self.distill), (self.runtime_path, self.runtime)):
            path.write_text(json.dumps(value))

    def write_plan(self):
        self.plan_path.write_text(json.dumps(self.plan))

    def run_experiment(self, output, **kwargs):
        return self.subject.experiment(self.student_path, self.distill_path, self.runtime_path, self.plan_path,
               self.policy, self.teacher_manifest, self.teacher_weights, 'owned-byte', None, output, **kwargs)

    def test_resume_matches_uninterrupted_and_never_opens_validation(self):
        full = self.run_experiment(self.root / 'full')
        partial = self.run_experiment(self.root / 'partial', max_steps_this_run=1)
        resumed = self.run_experiment(self.root / 'resumed', resume=Path(partial['latest_checkpoint']),
                                      resume_state_sha256=partial['latest_checkpoint_state_sha256'])
        self.assertEqual(full['weights_sha256'], resumed['weights_sha256'])
        self.assertEqual(full['losses'], resumed['losses'])
        self.assertEqual(full['tokens_processed'], 3 * 8)
        self.assertTrue(full['teacher_distillation_performed'])
        self.assertFalse(full['validation_performed'])
        self.assertFalse(full['test_opened'])
        self.assertFalse(full['assessment_training_performed'])
        self.assertFalse(full['promotion_eligible'])

    def test_heldout_source_and_missing_cuda_are_rejected_without_output(self):
        self.plan['sources'][1]['split'] = 'test'
        self.write_plan()
        with self.assertRaisesRegex(ValueError, 'held-out'):
            self.run_experiment(self.root / 'bad')
        self.plan['sources'][1]['split'] = 'validation'
        self.write_plan()
        self.runtime['device'] = 'cuda'
        self.write_configs()
        with patch('torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(ValueError, 'CUDA device is unavailable'):
                self.run_experiment(self.root / 'bad')
        self.assertFalse((self.root / 'bad').exists())

    def test_final_checkpoint_can_publish_after_interruption(self):
        output = self.root / 'full'
        full = self.run_experiment(output)
        pointer = json.loads((output / 'latest.json').read_text())
        recovered = self.run_experiment(self.root / 'finalized',
            resume=output / 'checkpoints' / pointer['checkpoint'], resume_state_sha256=pointer['state_sha256'])
        self.assertEqual(recovered['weights_sha256'], full['weights_sha256'])
        self.assertEqual(recovered['losses'], full['losses'])
        self.assertEqual(recovered['resumed_from_step'], self.student['steps'])

    def test_resume_rejects_changed_teacher_and_objective(self):
        partial = self.run_experiment(self.root / 'partial', max_steps_this_run=1)
        self.distill['temperature'] = 3.0
        self.write_configs()
        with self.assertRaisesRegex(ValueError, 'checkpoint identity'):
            self.run_experiment(self.root / 'bad', resume=Path(partial['latest_checkpoint']),
                                resume_state_sha256=partial['latest_checkpoint_state_sha256'])
        self.distill['temperature'] = 2.0
        self.write_configs()
        self.teacher_weights.write_bytes(self.teacher_weights.read_bytes() + b'tampered')
        with self.assertRaisesRegex(ValueError, 'teacher identity'):
            self.run_experiment(self.root / 'bad', resume=Path(partial['latest_checkpoint']),
                                resume_state_sha256=partial['latest_checkpoint_state_sha256'])
        self.assertFalse((self.root / 'bad').exists())

    def test_stages_only_screened_training_documents_with_provenance(self):
        import stage_owned_documents
        fixture_secret = 'fixture-password-123456789'
        self.source.write_text('Fixture document text ' * 8 + '\npassword=' + fixture_secret + '\n')
        self.plan['sources'][0]['sha256'] = self.sha(self.source.read_bytes())
        self.write_plan()
        output = self.root / 'stage'
        manifest = stage_owned_documents.stage(self.plan_path, self.policy, self.student_path, output)
        staged_plan = json.loads((output / 'plan.json').read_bytes())
        self.assertEqual(len(staged_plan['sources']), 1)
        self.assertEqual(staged_plan['sources'][0]['split'], 'train')
        self.assertNotIn(fixture_secret, (output / staged_plan['sources'][0]['path']).read_text())
        self.assertFalse(manifest['validation_sources_opened'])
        self.assertFalse(manifest['test_sources_opened'])
        self.assertFalse(manifest['redaction_complete'])
        self.assertFalse(manifest['human_reviewed'])
        with self.assertRaisesRegex(ValueError, 'already exists'):
            stage_owned_documents.stage(self.plan_path, self.policy, self.student_path, output)


if __name__ == '__main__':
    unittest.main()
