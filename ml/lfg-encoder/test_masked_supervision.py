from __future__ import annotations

import copy
from types import SimpleNamespace
import tempfile
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate
import masked_supervision as masked
import review_corpus


def export_config() -> dict:
    return {
        'schema_version': masked.CONFIG_SCHEMA,
        'workspace': 'C:/private/workspace', 'output': 'C:/private/export',
        'screening_policy': 'C:/private/policy.json', 'guide': None,
        'selected_labels': {field: ['label'] for field in masked.FIELDS},
        'allow_unreviewed_proposals': True,
        'unreviewed_complete_array_fields': [],
        'rejected_record_policy': 'exclude', 'stale_retracted_event_policy': 'omit_record',
        'weights': {'default': 1.0, 'per_label': {}},
        'limits': {key: 100 for key in masked.LIMIT_FIELDS},
        'near_duplicate': {'similarity': 0.98}, 'extension_inert': {'unrecognized': True},
    }


def target_row():
    return {'intents': ['positive'], 'capabilities': [], 'risks': []}


class MaskedLossTests(unittest.TestCase):
    def test_weighted_loss_matches_known_cell_reference_and_masks_gradients(self):
        logits = torch.tensor([[0.2, -0.7, 1.2]], requires_grad=True)
        targets = torch.tensor([[1.0, 0.0, 0.0]])
        known = torch.tensor([[True, True, False]])
        weights = torch.tensor([[2.0, 1.0, 0.0]])
        expected = (torch.nn.functional.binary_cross_entropy_with_logits(logits[0, :2], targets[0, :2], reduction='none') *
                    torch.tensor([2.0, 1.0])).sum() / 3.0
        loss = masked.masked_bce_with_logits(logits, targets, known, weights)
        self.assertTrue(torch.allclose(loss, expected))
        loss.backward()
        self.assertNotEqual(float(logits.grad[0, 0]), 0.0)
        self.assertNotEqual(float(logits.grad[0, 1]), 0.0)
        self.assertEqual(float(logits.grad[0, 2]), 0.0)

    def test_all_unknown_batch_has_zero_loss_and_zero_gradient(self):
        logits = torch.tensor([[3.0, -2.0]], requires_grad=True)
        targets = torch.tensor([[float('nan'), float('nan')]])
        loss = masked.masked_bce_with_logits(logits, targets,
                                             torch.zeros_like(logits, dtype=torch.bool),
                                             torch.zeros_like(logits))
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    def test_unknown_nonzero_weight_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unknown target cells'):
            masked.masked_bce_with_logits(torch.zeros((1, 1)), torch.zeros((1, 1)),
                                          torch.zeros((1, 1), dtype=torch.bool), torch.ones((1, 1)))

    def test_known_weight_that_underflows_float32_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'not representable'):
            masked.masked_bce_with_logits(torch.zeros((1, 1)), torch.zeros((1, 1)),
                                          torch.ones((1, 1), dtype=torch.bool), torch.tensor([[1e-100]], dtype=torch.float64))

    def test_subunit_total_weight_keeps_nonzero_normalized_loss_and_gradient(self):
        logits = torch.tensor([[0.0]], requires_grad=True)
        loss = masked.masked_bce_with_logits(logits, torch.tensor([[1.0]]),
                                             torch.tensor([[True]]), torch.tensor([[0.5]]))
        self.assertAlmostEqual(float(loss.detach()), 0.693147, places=5)
        loss.backward()
        self.assertAlmostEqual(float(logits.grad[0, 0]), -0.5, places=6)

    def test_validation_batch_scaling_reduces_to_global_weighted_cell_loss(self):
        first = masked.masked_eval_batch_loss(torch.tensor(1.0), batch_size=2,
                                              total_examples=3, total_known_weight=5.0)
        second = masked.masked_eval_batch_loss(torch.tensor(2.0), batch_size=1,
                                               total_examples=3, total_known_weight=5.0)
        hf_example_weighted_mean = (first * 2 + second) / 3
        self.assertAlmostEqual(float(hf_example_weighted_mean), 3.0 / 5.0, places=6)

    def test_metrics_count_only_known_cells_and_leave_unsupported_null(self):
        result = masked.masked_metrics([[1, 0], [0, 0]], [[1, 1], [1, 0]],
                                       [[True, False], [True, False]])
        self.assertEqual(result['known_cells'], 2)
        self.assertEqual(result['unknown_cells'], 2)
        self.assertEqual(result['per_label']['0']['fp'], 1)
        self.assertIsNone(result['per_label']['1']['precision'])
        self.assertIsNone(result['per_label']['1']['recall'])

    def test_trainer_rejects_accumulation_and_distributed_execution(self):
        class Base: pass
        with self.assertRaisesRegex(ValueError, 'gradient_accumulation_steps=1'):
            masked.make_masked_trainer_class(Base, 2, 1)
        with self.assertRaisesRegex(ValueError, 'distributed'):
            masked.make_masked_trainer_class(Base, 1, 2)

    def test_collator_keeps_mask_and_weights_out_of_model_input(self):
        class Tokenizer:
            def pad(self, features, **kwargs):
                return {'input_ids': torch.tensor([item['input_ids'] for item in features]),
                        'attention_mask': torch.tensor([item['attention_mask'] for item in features])}
        collated = masked.MaskedDataCollator(Tokenizer(), pad_to_multiple_of=None)([
            {'input_ids': [1, 2], 'attention_mask': [1, 1], 'labels': [1.0, 0.0],
             'known_mask': [True, False], 'cell_weights': [1.0, 0.0]},
        ])
        self.assertEqual(set(collated), {'input_ids', 'attention_mask', 'labels', 'known_mask', 'cell_weights'})
        self.assertEqual(collated['known_mask'].dtype, torch.bool)
        self.assertEqual(float(collated['cell_weights'][0, 1]), 0.0)

    def test_trainer_compute_loss_drops_mask_metadata_before_model_call(self):
        class Base: pass
        trainer = masked.make_masked_trainer_class(Base)()
        logits = torch.tensor([[0.5, -1.0]], requires_grad=True)
        class Model:
            training = True
            def __call__(self, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(logits=logits)
        model = Model()
        loss = trainer.compute_loss(model, {
            'input_ids': torch.tensor([[1, 2]]),
            'labels': torch.tensor([[1.0, 0.0]]),
            'known_mask': torch.tensor([[True, False]]),
            'cell_weights': torch.tensor([[1.0, 0.0]]),
        })
        self.assertEqual(set(model.kwargs), {'input_ids'})
        loss.backward()
        self.assertEqual(float(logits.grad[0, 1]), 0.0)

    def test_eval_compute_loss_aggregates_unequal_masks_and_weights_globally(self):
        class Base: pass
        trainer = masked.make_masked_trainer_class(Base)()
        trainer.masked_validation_examples = 3
        trainer.masked_validation_weight_total = 5.0
        class Model:
            training = False
            def __init__(self, logits): self.logits = logits
            def __call__(self, **kwargs): return SimpleNamespace(logits=self.logits)
        first_logits = torch.zeros((2, 1), requires_grad=True)
        second_logits = torch.zeros((1, 1), requires_grad=True)
        first, _ = trainer.compute_loss(Model(first_logits), {
            'input_ids': torch.tensor([[1], [2]]), 'labels': torch.tensor([[1.0], [0.0]]),
            'known_mask': torch.tensor([[True], [False]]), 'cell_weights': torch.tensor([[2.0], [0.0]]),
        }, return_outputs=True)
        second = trainer.compute_loss(Model(second_logits), {
            'input_ids': torch.tensor([[3]]), 'labels': torch.tensor([[0.0]]),
            'known_mask': torch.tensor([[True]]), 'cell_weights': torch.tensor([[3.0]]),
        })
        aggregate = (first.detach() * 2 + second.detach()) / 3
        self.assertAlmostEqual(float(aggregate), 0.693147, places=5)


class TargetPolicyTests(unittest.TestCase):
    def test_unreviewed_positive_is_known_but_absent_label_stays_unknown(self):
        config = export_config()
        columns = [{'field': 'intents', 'label': 'positive'}, {'field': 'intents', 'label': 'other'}]
        values, known, weights, origins, _ = masked._record_targets(
            target_row(), columns, config, None, False, None)
        self.assertEqual(values, [1, None])
        self.assertEqual(known, [True, False])
        self.assertEqual(weights, [1.0, 0.0])
        self.assertEqual(origins, ['owner_authorized_unreviewed_positive', 'unspecified_unknown'])

    def test_only_explicitly_configured_complete_arrays_emit_negative(self):
        config = export_config()
        columns = [{'field': 'intents', 'label': 'absent'}]
        row = target_row()
        values, known, _, origins, _ = masked._record_targets(row, columns, config, None, False, None)
        self.assertEqual((values, known), ([None], [False]))
        config['unreviewed_complete_array_fields'] = ['intents']
        values, known, _, origins, _ = masked._record_targets(row, columns, config, None, False, None)
        self.assertEqual((values, known), ([0], [True]))
        self.assertEqual(origins, ['operator_assumed_complete_array_negative'])

    def test_partial_unknown_and_rejected_proposal_override_draft(self):
        config = export_config()
        columns = [{'field': 'intents', 'label': 'positive'}]
        event = {'judgments': {'intents': {'positive': 'unknown'}},
                 'proposal_actions': {'intents': {'positive': 'rejected'}}}
        values, known, weights, origins, _ = masked._record_targets(
            target_row(), columns, config, event, False, None)
        self.assertEqual((values, known, weights), ([None], [False], [0.0]))
        self.assertEqual(origins, ['partial_explicit_unknown'])
        event['judgments']['intents'].clear()
        values, known, weights, origins, _ = masked._record_targets(
            target_row(), columns, config, event, False, None)
        self.assertEqual((values, known, weights), ([None], [False], [0.0]))
        self.assertEqual(origins, ['rejected_proposal_unknown'])

    def test_partial_positive_negative_and_source_completeness_are_explicit(self):
        config = export_config()
        columns = [{'field': 'intents', 'label': 'positive'}, {'field': 'intents', 'label': 'negative'}]
        event = {'judgments': {'intents': {'positive': 'positive', 'negative': 'negative'}},
                 'proposal_actions': {}}
        values, known, _, origins, _ = masked._record_targets(target_row(), columns, config, event, False, None)
        self.assertEqual((values, known), ([1, 0], [True, True]))
        self.assertEqual(origins, ['partial_explicit_positive', 'partial_explicit_negative'])
        event['judgments']['intents'].clear()
        row = {**target_row(), 'target_completeness': {'intents': True}}
        values, known, _, origins, _ = masked._record_targets(row, columns, config, event, False, None)
        self.assertEqual((values, known), ([1, 0], [True, True]))
        self.assertEqual(origins[1], 'source_attested_complete_array_negative')

    def test_partial_positive_requires_separate_opt_in_at_training(self):
        row = {'targets': [1, None], 'known_mask': [True, False], 'weights': [1.0, 0.0],
               'target_origins': ['partial_explicit_positive', 'unspecified_unknown'],
               'target_reviewer_claims': ['reviewer', None],
               'columns': [{'field': 'intents', 'label': 'positive'}, {'field': 'intents', 'label': 'negative'}]}
        config = {'allow_partial_review_targets': False, 'allow_unreviewed_targets': False}
        with self.assertRaisesRegex(ValueError, 'no known train target'):
            masked._apply_training_target_policy([row], config)
        config['allow_partial_review_targets'] = True
        selected, support = masked._apply_training_target_policy([row], config)
        self.assertEqual(selected[0]['targets'], [1, None])
        self.assertEqual(support['known_target_cells'], 1)

    def test_unknown_validation_rows_are_retained_for_coverage(self):
        rows = [
            {'targets': [1], 'known_mask': [True], 'weights': [1.0],
             'target_origins': ['owner_authorized_unreviewed_positive'],
             'target_reviewer_claims': [None], 'columns': [{'field': 'intents', 'label': 'positive'}]},
            {'targets': [None], 'known_mask': [False], 'weights': [0.0],
             'target_origins': ['unspecified_unknown'],
             'target_reviewer_claims': [None], 'columns': [{'field': 'intents', 'label': 'positive'}]},
        ]
        selected, support = masked._apply_training_target_policy(rows, {
            'allow_partial_review_targets': False, 'allow_unreviewed_targets': True,
        }, require_known=False)
        self.assertEqual(len(selected), 2)
        self.assertEqual(support['rows_with_known_targets'], 1)
        self.assertEqual(support['all_unknown_rows_retained'], 1)
        self.assertEqual(support['per_target_support']['intents:positive']['unknown'], 1)

    def test_retraction_and_stale_binding_are_not_current(self):
        binding = {'source_sha256': 'a', 'annotation_sha256': 'b', 'taxonomy_sha256': 'c',
                   'text_sha256': masked._sha(b'screened'), 'proposal_sha256': 'd', 'guide_sha256': None}
        event = {'event_type': 'decision', 'binding': binding, 'judgments': {},
                 'proposal_actions': {}}
        taxonomy = {field: {'label'} for field in masked.FIELDS}
        self.assertEqual(masked._validate_event(event, binding, taxonomy, 'screened'), (True, 'current'))
        self.assertEqual(masked._validate_event(event, binding, taxonomy, 'changed'), (False, 'stale_screened_text_binding'))
        event['event_type'] = 'retraction'
        self.assertEqual(masked._validate_event(event, binding, taxonomy, 'screened'), (False, 'retracted'))

    def test_config_accepts_extensions_inertly_and_validates_dynamic_limits(self):
        config = export_config()
        config['limits']['max_records'] = 1_000_000
        self.assertIs(masked.validate_config(config), config)
        broken = copy.deepcopy(config)
        broken['weights']['default'] = float('nan')
        with self.assertRaises(ValueError):
            masked.validate_config(broken)

    def test_training_config_requires_pinned_export_digest(self):
        config = {
            'schema_version': masked.TRAINING_CONFIG_SCHEMA,
            'export_dir': 'C:/private/export',
            'export_manifest_sha256': 'a' * 64,
            'output': 'C:/private/run', 'base_model': 'owner/model',
            'model_revision': 'b' * 40, 'model_manifest': None,
            'seed': 1, 'epochs': 1.0, 'max_length': 32, 'batch_size': 2,
            'checkpoint_steps': 1, 'device': 'auto', 'gradient_accumulation_steps': 1,
            'allow_mismatched_sizes': False, 'allow_unreviewed_targets': True,
            'allow_partial_review_targets': False, 'threshold': 0.5, 'resume': False,
            'extension': 'accepted but inert',
        }
        self.assertIs(masked.validate_training_config(config), config)
        del config['export_manifest_sha256']
        with self.assertRaisesRegex(ValueError, 'unsupported masked-training'):
            masked.validate_training_config(config)

    def test_pinned_export_rejects_manifest_drift_during_load(self):
        config_raw = masked._json_bytes(export_config())
        before, after = b'{"version":1}', b'{"version":2}'
        with patch.object(masked, '_read_bounded', side_effect=[config_raw, before, after]), \
                patch.object(masked, 'load_export', return_value=({}, {'train': []})):
            with self.assertRaisesRegex(ValueError, 'changed while loading'):
                masked._verify_pinned_export(Path('C:/private/export'), masked._sha(before))


class HeldoutGuardTests(unittest.TestCase):
    def test_heldout_plan_rejected_before_any_source_open(self):
        config = export_config()
        workspace = Path(config['workspace'])
        plan = {'schema_version': 'lfg.encoder-corpus-plan.v1',
                'sources': [{'path': 'DO_NOT_OPEN.jsonl', 'split': 'test'}]}
        with patch.object(masked, '_verify_owner_private_access'), \
                patch.object(masked, '_read_bounded', return_value=masked._json_bytes(plan)) as read:
            with self.assertRaisesRegex(ValueError, 'no source opened'):
                masked._hash_declared_workspace(workspace, config['limits'])
        self.assertEqual(read.call_count, 1)
        self.assertEqual(read.call_args.args[0], workspace / 'plan.json')


class PrivateOutputTests(unittest.TestCase):
    def test_current_process_liveness_probe(self):
        import os
        self.assertTrue(masked._process_is_active(os.getpid()))

    def test_foreign_posix_owner_and_windows_sid_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'not owned by the current user'):
            masked._verify_posix_owner(1001, 1002)
        with self.assertRaisesRegex(ValueError, 'owner is not the current user'):
            masked._verify_owner_sid('S-1-5-21-1', 'S-1-5-21-2')
        self.assertTrue(masked._is_allowed_owner_principal(
            r'NT AUTHORITY\LogonSessionId_0_12', r'DABO\jessa', r'NT AUTHORITY\LogonSessionId_0_12'))
        self.assertFalse(masked._is_allowed_owner_principal(
            r'NT AUTHORITY\LogonSessionId_0_11', r'DABO\jessa', r'NT AUTHORITY\LogonSessionId_0_12'))

    def test_private_output_directory_creation_is_exclusive_and_owner_only(self):
        with tempfile.TemporaryDirectory(prefix='lfg-masked-fixture-') as directory:
            output = Path(directory) / 'masked-export'
            self.assertEqual(masked._verify_private_output(output), output.resolve())
            masked._new_private_directory(output)
            masked._verify_owner_private_access(output, is_directory=True)
            with self.assertRaises(FileExistsError):
                masked._new_private_directory(output)

    def test_sidecar_uses_current_owner_check_before_read(self):
        import os
        with tempfile.TemporaryDirectory(prefix='lfg-sidecar-fixture-') as directory:
            sidecar = Path(directory) / 'partial-reviews.v1.jsonl'
            sidecar.write_bytes(b'')
            if sys.platform == 'win32':
                owner_check = patch.object(masked, '_verify_owner_private_access',
                                           side_effect=ValueError('private workspace input is not owned by the current user'))
            else:
                owner_check = patch.object(masked, '_verify_posix_owner',
                                           side_effect=ValueError('private workspace input is not owned by the current user'))
            with owner_check as check:
                with self.assertRaisesRegex(ValueError, 'not owned by the current user'):
                    masked._read_owner_private_sidecar(sidecar, 1024)
            if sys.platform == 'win32':
                check.assert_called_once_with(sidecar, is_directory=False)
            else:
                check.assert_called_once_with(os.stat(sidecar).st_uid, os.geteuid())

    def test_dangling_sidecar_symlink_is_not_treated_as_absent(self):
        with tempfile.TemporaryDirectory(prefix='lfg-sidecar-fixture-') as directory:
            sidecar = Path(directory) / 'partial-reviews.v1.jsonl'
            target = Path(directory) / 'missing-target'
            try:
                sidecar.symlink_to(target)
            except (OSError, NotImplementedError):
                self.skipTest('symlink creation is unavailable')
            self.assertFalse(sidecar.exists())
            self.assertTrue(sidecar.is_symlink())
            with self.assertRaisesRegex(ValueError, 'cannot be a symlink'):
                masked._read_owner_private_sidecar(sidecar, 1024)

    def test_sidecar_replacement_after_open_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix='lfg-sidecar-fixture-') as directory:
            sidecar = Path(directory) / 'partial-reviews.v1.jsonl'
            sidecar.write_bytes(b'')
            if sys.platform == 'win32':
                with patch.object(masked, '_verify_owner_private_access'), \
                        patch.object(masked, '_same_file_identity', side_effect=[True, False]):
                    with self.assertRaisesRegex(ValueError, 'changed during owner verification'):
                        masked._read_owner_private_sidecar(sidecar, 1024)
            else:
                sidecar.chmod(0o600)
                def replace_after_open(_uid, _expected):
                    sidecar.unlink()
                    sidecar.write_bytes(b'not-the-opened-file\n')
                    sidecar.chmod(0o600)
                with patch.object(masked, '_verify_posix_owner', side_effect=replace_after_open):
                    with self.assertRaisesRegex(ValueError, 'changed during secure read'):
                        masked._read_owner_private_sidecar(sidecar, 1024)

    def test_sidecar_parser_receives_exact_descriptor_bytes(self):
        with tempfile.TemporaryDirectory(prefix='lfg-sidecar-fixture-') as directory:
            sidecar = Path(directory) / 'partial-reviews.v1.jsonl'
            sidecar.write_bytes(b'')
            if sys.platform == 'win32':
                masked._secure_private_path(sidecar, is_directory=False)
            else:
                sidecar.chmod(0o600)
            with patch.object(review_corpus, '_parse_partial_events', return_value=[]) as parse:
                raw, events, present = masked._read_owner_private_sidecar(sidecar, 1024)
            self.assertEqual(raw, b'')
            self.assertEqual(events, [])
            self.assertTrue(present)
            parse.assert_called_once_with(b'', 1024)


if __name__ == '__main__':
    unittest.main()
