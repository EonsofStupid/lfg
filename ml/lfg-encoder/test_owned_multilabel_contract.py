from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch
from unittest.mock import call

sys.path.insert(0, str(Path(__file__).resolve().parent))
import masked_supervision as masked
import owned_multilabel_contract as contract
import corpus_audit


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class ResponseContractTests(unittest.TestCase):
    columns = [{'field': 'intents', 'label': 'build.software'},
               {'field': 'risks', 'label': 'sensitive-data'}]

    def test_known_decisions_and_omissions_are_not_negative(self):
        response = contract.validate_response({'schema_version': contract.CONTRACT_SCHEMA,
            'taxonomy_sha256': 'a' * 64,
            'decisions': [{'field': 'intents', 'label': 'build.software', 'value': 'positive'}]}, self.columns)
        self.assertEqual(len(response['decisions']), 1)
        self.assertNotIn(('risks', 'sensitive-data'), {(d['field'], d['label']) for d in response['decisions']})

    def test_guide_identity_requires_null_or_sha256_and_is_preserved(self):
        self.assertIsNone(contract._validate_guide_identity({'label_guide_sha256': None}))
        self.assertEqual(contract._validate_guide_identity({'label_guide_sha256': 'a' * 64}), 'a' * 64)
        with self.assertRaisesRegex(ValueError, 'missing'):
            contract._validate_guide_identity({})
        with self.assertRaisesRegex(ValueError, 'malformed'):
            contract._validate_guide_identity({'label_guide_sha256': 'A' * 64})

    def test_full_review_origin_requires_verified_full_provenance(self):
        reviewer = 'a' * 64
        full = {'reviewer_claim_sha256': reviewer, 'reviewed_at': '2026-11-07T12:00:00Z',
                'reconciled_partial_review': None}
        source = {'targets': [0, 1], 'known_mask': [True, True], 'weights': [0.5, 1.0],
                  'target_origins': ['full_reviewed_legacy', 'full_reviewed_legacy'],
                  'target_reviewer_claims': [reviewer, reviewer], 'verified_source_claim': True,
                  'partial_review_provenance': None, 'full_review_provenance': full}
        contract._validate_source_target_semantics(source)
        contract._validate_review_claims(source, 2)
        output = {'decision': 'negative', 'target_origin': 'full_reviewed_legacy',
                  'verified_source_claim': True, 'target_reviewer_claim_sha256': reviewer,
                  'partial_review_provenance': None, 'full_review_provenance': full}
        contract._validate_training_target_semantics(output)
        contract._validate_review_claims({'target_origin': [output['target_origin']],
            'target_reviewer_claim_sha256': [reviewer], 'verified_source_claim': True,
            'partial_review_provenance': None, 'full_review_provenance': full}, 1)
        unverified = {**source, 'verified_source_claim': False, 'full_review_provenance': None,
                      'target_reviewer_claims': [None, None]}
        with self.assertRaisesRegex(ValueError, 'verified source claim'):
            contract._validate_source_target_semantics(unverified)
        with self.assertRaisesRegex(ValueError, 'full-review provenance'):
            contract._validate_review_claims(unverified, 2)

    def test_bounded_unknown_top_level_fields_are_retained_inertly(self):
        response = contract.validate_response({'schema_version': contract.CONTRACT_SCHEMA,
            'taxonomy_sha256': 'a' * 64, 'decisions': [], 'future_extension': {'opaque': True}}, self.columns)
        self.assertEqual(response['future_extension'], {'opaque': True})
        with self.assertRaisesRegex(ValueError, 'bounded JSON'):
            contract.validate_response({'schema_version': contract.CONTRACT_SCHEMA,
                'taxonomy_sha256': 'a' * 64, 'decisions': [], 'future_extension': 'x' * (contract.MAX_EXTENSION_BYTES + 1)}, self.columns)

    def test_cli_emits_only_digest_and_aggregate_counts(self):
        output = StringIO()
        result = {'manifest_sha256': 'a' * 64, 'rows': 2,
                  'target_support': {'positive': 1, 'negative': 0, 'unknown': 4},
                  'weight_totals': {'positive': 1.0, 'negative': 0.0},
                  'output': 'C:/private/credential-like-path',
                  'per_label': {'private-label': {'positive': 1}}}
        with patch.object(contract, 'build_dataset', return_value=result), \
                patch.object(sys, 'argv', ['owned_multilabel_contract.py', 'build', '--config', 'C:/private/config']), \
                redirect_stdout(output):
            self.assertEqual(contract.main(), 0)
        printed = output.getvalue()
        self.assertNotIn('credential-like-path', printed)
        self.assertNotIn('private-label', printed)
        self.assertNotIn('C:/private/config', printed)
        self.assertEqual(set(json.loads(printed)), {'manifest_sha256', 'rows', 'target_support', 'weight_totals'})

    def test_unknown_and_full_abstention_are_valid(self):
        self.assertEqual(contract.validate_response({'schema_version': contract.CONTRACT_SCHEMA,
            'taxonomy_sha256': 'a' * 64,
            'decisions': [{'field': 'intents', 'label': 'build.software', 'value': 'unknown'}]}, self.columns)['decisions'][0]['value'], 'unknown')
        result = contract.validate_response({'schema_version': contract.CONTRACT_SCHEMA,
            'taxonomy_sha256': 'a' * 64, 'decisions': [], 'abstain': True}, self.columns)
        self.assertTrue(result['abstain'])

    def test_duplicate_contradiction_invalid_label_and_assertive_abstention_rejected(self):
        base = {'schema_version': contract.CONTRACT_SCHEMA, 'taxonomy_sha256': 'a' * 64, 'decisions': [
            {'field': 'intents', 'label': 'build.software', 'value': 'positive'}]}
        for decisions, abstain in [
            (base['decisions'] * 2, False),
            (base['decisions'] + [{'field': 'intents', 'label': 'build.software', 'value': 'negative'}], False),
            ([{'field': 'intents', 'label': 'invented', 'value': 'positive'}], False),
            (base['decisions'], True),
        ]:
            with self.subTest(decisions=decisions, abstain=abstain), self.assertRaises(ValueError):
                contract.validate_response({**base, 'decisions': decisions, 'abstain': abstain}, self.columns)

    def test_bad_contract_version_values_extensions_and_hashes_rejected(self):
        for value in [
            {'schema_version': 'v2', 'decisions': []},
            {'schema_version': contract.CONTRACT_SCHEMA, 'taxonomy_sha256': 'a' * 64, 'decisions': [{'field': 'intents', 'label': 'build.software', 'value': 'maybe'}]},
            {'schema_version': contract.CONTRACT_SCHEMA, 'taxonomy_sha256': 'a' * 64, 'decisions': [], 'extensions': []},
            {'schema_version': contract.CONTRACT_SCHEMA, 'taxonomy_sha256': 'a' * 64, 'decisions': [{'field': [], 'label': 'x', 'value': 'unknown'}]},
            {'schema_version': contract.CONTRACT_SCHEMA, 'taxonomy_sha256': 'bad', 'decisions': []},
        ]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                contract.validate_response(value, self.columns)

    def test_config_accepts_inert_extensions_but_rejects_duplicate_or_bad_weight(self):
        cfg = {'schema_version': contract.CONFIG_SCHEMA, 'export_dir': 'C:/e',
            'export_manifest_sha256': 'a' * 64, 'taxonomy_file': 'C:/t', 'taxonomy_sha256': 'b' * 64,
            'selected_columns': [{'field': 'intents', 'label': 'build.software'}],
            'instruction_template': '{field} {label}', 'output': 'C:/o', 'max_rows': 10,
            'max_output_bytes': 10000, 'default_weight': 1.0, 'per_label_weights': {}, 'extension': {'x': 1}}
        self.assertIs(contract.validate_config(cfg), cfg)
        dup = copy.deepcopy(cfg); dup['selected_columns'].append(dup['selected_columns'][0])
        with self.assertRaisesRegex(ValueError, 'duplicates'):
            contract.validate_config(dup)
        bad = copy.deepcopy(cfg); bad['default_weight'] = float('nan')
        with self.assertRaises(ValueError):
            contract.validate_config(bad)
        oversized = copy.deepcopy(cfg); oversized['extension'] = 'x' * (contract.MAX_EXTENSION_BYTES + 1)
        with self.assertRaisesRegex(ValueError, 'bounded JSON'):
            contract.validate_config(oversized)
        oversized_output = copy.deepcopy(cfg); oversized_output['max_output_bytes'] = contract.MAX_DATASET_BYTES + 1
        with self.assertRaisesRegex(ValueError, 'output limits'):
            contract.validate_config(oversized_output)


class AdapterTests(unittest.TestCase):
    def test_writer_reader_preserves_known_cells_lineage_and_unknowns(self):
        with tempfile.TemporaryDirectory(prefix='lfg-owned-multilabel-') as temp:
            root = Path(temp)
            export = root / 'export'; export.mkdir()
            taxonomy = {'schema_version': 'lfg.encoder-evaluation-taxonomy.v1', 'intents': ['build.software', 'other', 'negative']}
            taxonomy_raw = (json.dumps(taxonomy, sort_keys=True) + '\n').encode()
            taxonomy_path = root / 'taxonomy.json'; taxonomy_path.write_bytes(taxonomy_raw)
            train_raw = b'{"fixture":true}\n'
            guide_sha = sha(b'fixture guide')
            export_manifest = {'schema_version': masked.EXPORT_SCHEMA, 'config_sha256': 'c' * 64,
                'taxonomy_sha256': sha(taxonomy_raw), 'label_guide_sha256': guide_sha,
                'selected_columns': [{'field': 'intents', 'label': 'build.software'},
                                     {'field': 'intents', 'label': 'other'},
                                     {'field': 'intents', 'label': 'negative'}],
                'source_artifacts': [{'path': 'project-a.jsonl', 'split': 'train', 'sha256': 'd' * 64,
                    'project_claim_sha256': 'e' * 64, 'project_group_claim_sha256': 'f' * 64}],
                'screening_policy_sha256': '3' * 64,
                'output_files': {'train.jsonl': {'sha256': sha(train_raw), 'bytes': len(train_raw)}}}
            manifest_raw = masked._json_bytes(export_manifest)
            (export / 'manifest.json').write_bytes(manifest_raw)
            manifest_path = export / 'manifest.json'
            source = {'split': 'train', 'text': 'screened sentinel request',
                'text_sha256': sha(b'screened sentinel request'),
                'original_text_sha256': sha(b'original sentinel request'),
                'family': '4' * 64,
                'columns': export_manifest['selected_columns'], 'targets': [1, None, 0],
                'known_mask': [True, False, True], 'weights': [1.25, 0.0, 1.5],
                'target_origins': ['owner_authorized_unreviewed_positive', 'unspecified_unknown',
                                   'source_attested_complete_array_negative'],
                'target_reviewer_claims': [None, None,
                    masked._claim_digest('reviewer', 'source_target_completeness_claim')],
                'verified_source_claim': False,
                'source_identity': {'artifact_path': 'project-a.jsonl', 'artifact_sha256': 'd' * 64,
                    'project_claim_sha256': 'e' * 64, 'project_group_claim_sha256': 'f' * 64,
                    'record_id_sha256': '1' * 64, 'source_claim_sha256': '2' * 64},
                'screening': {'policy_sha256': '3' * 64, 'redaction_counts': {}, 'text_changed': True},
                'partial_review_provenance': None, 'full_review_provenance': None}
            cfg = {'schema_version': contract.CONFIG_SCHEMA, 'export_dir': str(export),
                'export_manifest_sha256': sha(manifest_raw), 'taxonomy_file': str(taxonomy_path),
                'taxonomy_sha256': sha(taxonomy_raw),
                'selected_columns': [{'field': 'intents', 'label': 'build.software'},
                                     {'field': 'intents', 'label': 'other'},
                                     {'field': 'intents', 'label': 'negative'}],
                'instruction_template': 'For {field}, decide {label}.', 'output': str(root / 'out'),
                'max_rows': 10, 'max_output_bytes': 100000, 'default_weight': 2.0,
                'per_label_weights': {}, 'extension_inert': {'retained': True}}
            cfg_path = root / 'config.json'; cfg_path.write_bytes(masked._json_bytes(cfg))
            # This test focuses on adapter transformations; platform ACL behavior has dedicated tests.
            with patch.object(masked, '_verify_owner_private_access'), \
                 patch.object(masked, 'load_export', return_value=(export_manifest, {'train': [source]})):
                result = contract.build_dataset(cfg_path)
            output = Path(cfg['output'])
            out_manifest, rows = contract.load_training_dataset(output, result['manifest_sha256'])
            self.assertEqual(set(result), {'manifest_sha256', 'rows', 'target_support', 'weight_totals'})
            self.assertNotIn('per_label', result['target_support'])
            self.assertEqual(len(rows), 2)
            self.assertEqual([row['decision'] for row in rows], ['positive', 'negative'])
            self.assertEqual(json.loads(rows[0]['target_response'])['schema_version'], contract.CONTRACT_SCHEMA)
            self.assertEqual(json.loads(rows[0]['target_response'])['decisions'][0]['value'], 'positive')
            self.assertEqual(json.loads(rows[0]['target_response'])['guide_sha256'], guide_sha)
            self.assertEqual(out_manifest['guide_sha256'], guide_sha)
            self.assertEqual(rows[0]['weight'], 2.5)
            self.assertEqual(rows[0]['configured_weight'], 2.0)
            self.assertEqual(rows[0]['source_weight'], 1.25)
            self.assertEqual(rows[0]['original_text_sha256'], sha(b'original sentinel request'))
            self.assertEqual(rows[0]['source_identity']['project_claim_sha256'], 'e' * 64)
            self.assertEqual(out_manifest['target_support']['unknown'], 1)
            self.assertEqual(out_manifest['target_support']['positive'], 1)
            self.assertEqual(out_manifest['target_support']['negative'], 1)
            self.assertEqual(out_manifest['model_training_performed'], False)
            self.assertNotIn('screened sentinel request', json.dumps(result))
            tiny_rows = copy.deepcopy(rows)
            tiny_rows[0]['source_weight'] = 1e-20
            tiny_rows[0]['weight'] = 2e-20
            tiny_raw = b''.join(masked._json_bytes(row) for row in tiny_rows)
            (output / 'train.jsonl').write_bytes(tiny_raw)
            out_manifest['train_file'] = {'sha256': sha(tiny_raw), 'bytes': len(tiny_raw)}
            tiny_manifest_raw = masked._json_bytes(out_manifest)
            (output / 'manifest.json').write_bytes(tiny_manifest_raw)
            with patch.object(masked, '_verify_owner_private_access'):
                self.assertEqual(len(contract.load_training_dataset(output, sha(tiny_manifest_raw))[1]), 2)
            mismatch_manifest = copy.deepcopy(export_manifest)
            mismatch_manifest['source_artifacts'][0]['split'] = 'validation'
            bad_config = copy.deepcopy(cfg); bad_config['output'] = str(root / 'bad-split-output')
            bad_config_path = root / 'bad-split-config.json'; bad_config_path.write_bytes(masked._json_bytes(bad_config))
            with patch.object(masked, '_verify_owner_private_access'), \
                    patch.object(masked, 'load_export', return_value=(mismatch_manifest, {'train': [source]})):
                with self.assertRaisesRegex(ValueError, 'source identity does not match'):
                    contract.build_dataset(bad_config_path)
            original_rows_raw = b''.join(masked._json_bytes(row) for row in rows)
            (output / 'train.jsonl').write_bytes(original_rows_raw)
            out_manifest['train_file'] = {'sha256': sha(original_rows_raw), 'bytes': len(original_rows_raw)}
            original_manifest_raw = masked._json_bytes(out_manifest)
            (output / 'manifest.json').write_bytes(original_manifest_raw)
            with patch.object(masked, '_verify_owner_private_access',
                              side_effect=ValueError('private path ACL denied')):
                with self.assertRaisesRegex(ValueError, 'ACL denied'):
                    contract.load_training_dataset(output, result['manifest_sha256'])
            (output / 'train.jsonl').write_text('{}\n')
            with patch.object(masked, '_verify_owner_private_access'):
                with self.assertRaisesRegex(ValueError, 'digest does not match'):
                    contract.load_training_dataset(output, result['manifest_sha256'])

            valid_target = json.loads(rows[0]['target_response'])
            for changes, error in [
                ({'decisions': []}, 'exactly one decision'),
                ({'taxonomy_sha256': 'a' * 64}, 'identity mismatch'),
                ({'guide_sha256': 'b' * 64}, 'identity mismatch'),
            ]:
                changed_rows = copy.deepcopy(rows)
                changed_rows[0]['target_response'] = json.dumps({**valid_target, **changes})
                malformed_raw = b''.join(masked._json_bytes(row) for row in changed_rows)
                (output / 'train.jsonl').write_bytes(malformed_raw)
                out_manifest['train_file'] = {'sha256': sha(malformed_raw), 'bytes': len(malformed_raw)}
                (output / 'manifest.json').write_bytes(masked._json_bytes(out_manifest))
                with patch.object(masked, '_verify_owner_private_access'):
                    with self.assertRaisesRegex(ValueError, error):
                        altered_manifest_sha = sha((output / 'manifest.json').read_bytes())
                        contract.load_training_dataset(output, altered_manifest_sha)
            corruption_cases = [
                ({'source_weight': 1e-20, 'configured_weight': 2.0, 'weight': 1e-12},
                    'private training row is invalid'),
                ({'decision': 'negative'}, 'training target decision does not match its origin'),
                ({'target_origin': 'source_attested_complete_array_negative'},
                    'training target decision does not match its origin'),
                ({'instruction': 'wrong instruction'}, 'private training row is invalid'),
                ({'weight': 99.0}, 'private training row is invalid'),
                ({'source_identity': {}}, 'private training row is invalid'),
                ({'source_identity': {'artifact_path': 'missing.jsonl', 'artifact_sha256': 'd' * 64,
                    'project_claim_sha256': 'e' * 64, 'project_group_claim_sha256': 'f' * 64,
                    'record_id_sha256': '1' * 64, 'source_claim_sha256': '2' * 64}},
                    'does not match source manifest'),
                ({'family_sha256': 'z' * 64}, 'private training row is invalid'),
                ({'screening': {}}, 'screening provenance is malformed'),
                ({'target_reviewer_claim_sha256': 'not-a-digest'}, 'target reviewer provenance is malformed'),
                ({'verified_source_claim': True, 'full_review_provenance': []}, 'full-review provenance'),
                ({'verified_source_claim': False, 'full_review_provenance': []}, 'unverified source claim'),
                ({'screening': {'policy_sha256': '3' * 64, 'redaction_counts': {}, 'text_changed': False}},
                    'screening provenance is malformed'),
            ]
            for changes, error in corruption_cases:
                changed_rows = copy.deepcopy(rows)
                changed_rows[0].update(changes)
                changed_raw = b''.join(masked._json_bytes(row) for row in changed_rows)
                (output / 'train.jsonl').write_bytes(changed_raw)
                out_manifest['train_file'] = {'sha256': sha(changed_raw), 'bytes': len(changed_raw)}
                manifest_raw = masked._json_bytes(out_manifest)
                (output / 'manifest.json').write_bytes(manifest_raw)
                with patch.object(masked, '_verify_owner_private_access'):
                    with self.assertRaisesRegex(ValueError, error):
                        contract.load_training_dataset(output, sha(manifest_raw))

            split_manifest = copy.deepcopy(out_manifest)
            split_manifest['source_artifacts'][0]['split'] = 'validation'
            split_raw = masked._json_bytes(split_manifest)
            (output / 'manifest.json').write_bytes(split_raw)
            with patch.object(masked, '_verify_owner_private_access'):
                with self.assertRaisesRegex(ValueError, 'does not match source manifest'):
                    contract.load_training_dataset(output, sha(split_raw))

            valid_rows_raw = b''.join(masked._json_bytes(row) for row in rows)
            (output / 'train.jsonl').write_bytes(valid_rows_raw)
            out_manifest['train_file'] = {'sha256': sha(valid_rows_raw), 'bytes': len(valid_rows_raw)}
            valid_manifest = masked._json_bytes(out_manifest)
            (output / 'manifest.json').write_bytes(valid_manifest)
            tampered_manifest = valid_manifest.replace(b'"rows":2', b'"rows":3')
            (output / 'manifest.json').write_bytes(tampered_manifest)
            with patch.object(masked, '_verify_owner_private_access'):
                with self.assertRaisesRegex(ValueError, 'caller-pinned digest'):
                    contract.load_training_dataset(output, sha(valid_manifest))

    def test_train_only_loader_never_opens_validation_path(self):
        with tempfile.TemporaryDirectory(prefix='lfg-train-only-') as temp:
            root = Path(temp); export = root / 'export'; export.mkdir()
            columns = [{'field': 'intents', 'label': 'build.software'}]
            config = {'schema_version': masked.CONFIG_SCHEMA, 'workspace': str(root / 'workspace'),
                'output': str(root / 'new-export'), 'screening_policy': str(root / 'policy.json'),
                'guide': None, 'selected_labels': {field: (['build.software'] if field == 'intents' else [])
                                                  for field in masked.FIELDS},
                'allow_unreviewed_proposals': True, 'unreviewed_complete_array_fields': [],
                'rejected_record_policy': 'exclude', 'stale_retracted_event_policy': 'omit_record',
                'weights': {'default': 1.0, 'per_label': {}}, 'limits': {key: 100000 for key in masked.LIMIT_FIELDS},
                'near_duplicate': {'similarity': 0.98}}
            config_raw = masked._json_bytes(config); (export / 'config.json').write_bytes(config_raw)
            text = 'train sentinel'; raw_text = text.encode(); rid = '1' * 64
            source = {'schema_version': masked.ROW_SCHEMA, 'id': rid, 'text': text, 'family': '2' * 64,
                'split': 'train', 'source_identity': {'artifact_path': 'source.jsonl', 'artifact_sha256': '3' * 64,
                    'project_claim_sha256': '4' * 64, 'project_group_claim_sha256': '5' * 64,
                    'record_id_sha256': rid, 'source_claim_sha256': '6' * 64},
                'text_sha256': sha(raw_text), 'columns': columns, 'targets': [1], 'known_mask': [True],
                'weights': [1.0], 'target_origins': ['owner_authorized_unreviewed_positive'],
                'target_reviewer_claims': [None]}
            train_raw = masked._json_bytes(source); (export / 'train.jsonl').write_bytes(train_raw)
            manifest = {'schema_version': masked.EXPORT_SCHEMA, 'config_sha256': sha(config_raw),
                'selected_columns': columns, 'output_files': {
                    'train.jsonl': {'sha256': sha(train_raw), 'bytes': len(train_raw)},
                    'validation.jsonl': {'sha256': 'a' * 64, 'bytes': 128}}}
            (export / 'manifest.json').write_bytes(masked._json_bytes(manifest))
            actual_read = masked._read_bounded
            opened = []
            def tracked(path, *args, **kwargs):
                opened.append(Path(path).name)
                return actual_read(path, *args, **kwargs)
            with patch.object(masked, '_verify_owner_private_access'), patch.object(masked, '_read_bounded', side_effect=tracked):
                loaded, splits = masked.load_export(export, train_only=True)
            self.assertEqual([row['id'] for row in splits['train']], [rid])
            self.assertNotIn('validation.jsonl', opened)

    def test_real_export_helpers_preserve_partial_review_and_completeness_claims(self):
        with tempfile.TemporaryDirectory(prefix='lfg-completeness-claim-') as temp:
            root = Path(temp); export = root / 'export'; export.mkdir()
            guide_sha, source_sha = 'a' * 64, 'c' * 64
            columns = [{'field': 'intents', 'label': 'known_positive'},
                       {'field': 'intents', 'label': 'source_negative'},
                       {'field': 'intents', 'label': 'unknown_label'},
                       {'field': 'capabilities', 'label': 'operator_negative'}]
            taxonomy = {'schema_version': 'lfg.encoder-evaluation-taxonomy.v1',
                        'intents': ['known_positive', 'source_negative', 'unknown_label'],
                        'capabilities': ['operator_negative']}
            taxonomy_raw = masked._json_bytes(taxonomy)
            taxonomy_sha = sha(taxonomy_raw)
            taxonomy_path = root / 'taxonomy.json'; taxonomy_path.write_bytes(taxonomy_raw)
            record_id = 'synthetic-record'
            record = {'schema_version': 'fixture', 'id': record_id, 'text': 'screened fixture request',
                      'family': 'synthetic-family', 'intents': ['known_positive'], 'capabilities': [],
                      'risks': [], 'target_completeness': {'intents': True}, 'verified': False,
                      'source': {'fixture': True}}
            project, group = 'fixture-project', 'fixture-group'
            metadata = {'split': 'train', 'project': project, 'project_group': group,
                        'path': 'fixture/project.jsonl', 'sha256': source_sha}
            annotation_sha = corpus_audit.annotation_digest(record, project, 'train', taxonomy_sha)
            event_binding = masked._event_binding(record, source_sha, annotation_sha, taxonomy_sha, guide_sha)
            event = {'event_id': '1' * 32, 'revision': 1, 'event_type': 'decision',
                     'reviewer_claim': 'synthetic-reviewer', 'reviewed_at': '2026-11-07T12:00:00+00:00',
                     'binding': event_binding,
                     'judgments': {'intents': {'known_positive': 'positive', 'unknown_label': 'unknown'}},
                     'proposal_actions': {}}
            source_record = {**record, '_masked_source': metadata}
            snapshot = {'rows': [source_record], 'columns': columns, 'events': {record_id: event},
                        'event_count': 1, 'ledger': {record_id: {'decision': 'pending',
                            'annotation_sha256': annotation_sha}}, 'taxonomy_sha256': taxonomy_sha,
                        'taxonomy': {'intents': {column['label'] for column in columns}},
                        'screen_policy_sha256': 'd' * 64, 'source_manifest': {'guide_sha256': guide_sha}}
            exporter_config = {'rejected_record_policy': 'exclude', 'stale_retracted_event_policy': 'omit_record',
                'allow_unreviewed_proposals': True, 'unreviewed_complete_array_fields': ['capabilities'],
                'weights': {'default': 1.0, 'per_label': {}},
                'limits': {'max_near_duplicate_pairs': 10, 'max_near_duplicate_character_product': 100000},
                'near_duplicate': {'similarity': 0.98}}
            with patch.object(masked, '_validate_event', return_value=(True, 'current')), \
                    patch.object(masked.screen_private_corpus, '_screen', return_value=(record['text'], {})):
                rows_by_split, _ = masked._make_export_rows(snapshot, exporter_config, [], {'max_text_bytes': 1000})
            exported = rows_by_split['train'][0]
            self.assertEqual(exported['targets'], [1, 0, None, 0])
            self.assertEqual(exported['target_origins'], [
                'partial_explicit_positive', 'source_attested_complete_array_negative',
                'partial_explicit_unknown', 'operator_assumed_complete_array_negative'])
            self.assertEqual(exported['target_reviewer_claims'], [
                masked._claim_digest('reviewer', event['reviewer_claim']),
                masked._claim_digest('reviewer', 'source_target_completeness_claim'),
                masked._claim_digest('reviewer', event['reviewer_claim']),
                masked._claim_digest('reviewer', 'operator_configured_complete_array_assumption')])
            for contradictory in [
                {'targets': [1, 1, None, 0]},
                {'targets': [0, 0, None, 0], 'target_origins': [
                    'owner_authorized_unreviewed_positive', *exported['target_origins'][1:]]},
                {'known_mask': [False, True, False, True]},
            ]:
                with self.subTest(contradictory=contradictory), self.assertRaises(ValueError):
                    contract._validate_source_target_semantics({**exported, **contradictory})

            source_artifact = {'path': metadata['path'], 'split': 'train', 'sha256': source_sha,
                'project_claim_sha256': masked._claim_digest('project', project),
                'project_group_claim_sha256': masked._claim_digest('project-group', group)}
            train_raw = masked._json_bytes(exported)
            export_manifest = {'schema_version': masked.EXPORT_SCHEMA, 'config_sha256': 'e' * 64,
                'taxonomy_sha256': taxonomy_sha, 'label_guide_sha256': guide_sha,
                'screening_policy_sha256': 'd' * 64, 'selected_columns': columns,
                'source_artifacts': [source_artifact], 'output_files': {
                    'train.jsonl': {'sha256': sha(train_raw), 'bytes': len(train_raw)}}}
            manifest_raw = masked._json_bytes(export_manifest)
            (export / 'manifest.json').write_bytes(manifest_raw)
            cfg = {'schema_version': contract.CONFIG_SCHEMA, 'export_dir': str(export),
                'export_manifest_sha256': sha(manifest_raw), 'taxonomy_file': str(taxonomy_path),
                'taxonomy_sha256': sha(taxonomy_raw), 'selected_columns': columns,
                'instruction_template': 'Decide {label} for {field}.', 'output': str(root / 'adapter-output'),
                'max_rows': 10, 'max_output_bytes': 100000, 'default_weight': 1.0,
                'per_label_weights': {}}
            cfg_path = root / 'config.json'; cfg_path.write_bytes(masked._json_bytes(cfg))
            with patch.object(masked, '_verify_owner_private_access'), \
                    patch.object(masked, 'load_export', return_value=(export_manifest, {'train': [exported]})):
                result = contract.build_dataset(cfg_path)
            out_manifest, rows = contract.load_training_dataset(Path(cfg['output']), result['manifest_sha256'])
            self.assertEqual([row['decision'] for row in rows], ['positive', 'negative', 'negative'])
            self.assertEqual(out_manifest['target_support']['unknown'], 1)
            self.assertEqual(rows[1]['target_origin'], 'source_attested_complete_array_negative')
            self.assertEqual(rows[1]['target_reviewer_claim_sha256'],
                             masked._claim_digest('reviewer', 'source_target_completeness_claim'))
            self.assertEqual(rows[2]['target_origin'], 'operator_assumed_complete_array_negative')
            self.assertEqual(rows[2]['target_reviewer_claim_sha256'],
                             masked._claim_digest('reviewer', 'operator_configured_complete_array_assumption'))
            malformed = dict(exported)
            malformed_claims = list(malformed['target_reviewer_claims'])
            malformed_claims[1] = 'z' * 64
            malformed['target_reviewer_claims'] = malformed_claims
            with self.assertRaisesRegex(ValueError, 'target reviewer provenance'):
                contract._validate_review_claims(malformed, len(columns), taxonomy_sha256=taxonomy_sha,
                    guide_sha256=guide_sha, source_sha256=source_sha, text_sha256=exported['text_sha256'])
            mismatched_claims = list(exported['target_reviewer_claims'])
            mismatched_claims[1] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'completeness target claim'):
                contract._validate_review_claims({**exported, 'target_reviewer_claims': mismatched_claims},
                    len(columns), taxonomy_sha256=taxonomy_sha, guide_sha256=guide_sha,
                    source_sha256=source_sha, text_sha256=exported['text_sha256'])

            operator_config = {'weights': {'default': 1.0, 'per_label': {}},
                               'allow_unreviewed_proposals': True,
                               'unreviewed_complete_array_fields': ['intents']}
            values, known, _, origins, claims = masked._record_targets(
                {'intents': [], 'capabilities': [], 'risks': []},
                [{'field': 'intents', 'label': 'operator_negative'}], operator_config, None, False, None)
            self.assertEqual((values, known, origins, claims),
                ([0], [True], ['operator_assumed_complete_array_negative'],
                 ['operator_configured_complete_array_assumption']))
            operator_claim_row = {'target_reviewer_claims': [masked._claim_digest('reviewer', claims[0])],
                'target_origins': origins, 'verified_source_claim': False,
                'partial_review_provenance': None, 'full_review_provenance': None}
            contract._validate_review_claims(operator_claim_row, 1)

if __name__ == '__main__':
    unittest.main()
