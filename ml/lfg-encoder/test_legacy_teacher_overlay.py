from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import legacy_teacher_overlay as overlay
import masked_supervision as masked
import owned_multilabel_contract as contract


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class LegacyTeacherOverlayTests(unittest.TestCase):
    columns = [{'field': 'intents', 'label': 'build.software'},
               {'field': 'intents', 'label': 'known.software'},
               {'field': 'risks', 'label': 'sensitive-data'}]

    def make_source(self, row_id: str, targets: list[int | None], known: list[bool]) -> dict:
        return {'id': row_id, 'split': 'train', 'family': 'a' * 64,
            'text': 'screened synthetic request', 'text_sha256': 'b' * 64,
            'original_text_sha256': 'c' * 64,
            'screening': {'policy_sha256': 'd' * 64, 'redaction_counts': {}, 'text_changed': True},
            'source_identity': {'artifact_path': 'projeté/fixture.jsonl', 'artifact_sha256': 'e' * 64,
                'project_claim_sha256': 'f' * 64, 'project_group_claim_sha256': '1' * 64,
                'record_id_sha256': row_id, 'source_claim_sha256': '2' * 64},
            'columns': self.columns, 'targets': targets, 'known_mask': known,
            'weights': [1.0 if value else 0.0 for value in known],
            'target_origins': ['owner_authorized_unreviewed_positive' if known[i] and targets[i] == 1
                else 'source_attested_complete_array_negative' if known[i]
                else 'unspecified_unknown' for i in range(len(known))],
            'target_reviewer_claims': [None] * len(known), 'verified_source_claim': False}

    def make_lineage(self, columns, taxonomy_sha, export_config_sha, screening_sha, terminal_verified=False):
        artifact_path = 'projeté/fixture.jsonl'
        artifacts = [{'path': artifact_path, 'split': 'train', 'sha256': 'e' * 64,
            'project_claim_sha256': 'f' * 64, 'project_group_claim_sha256': '1' * 64}]
        hashes = {artifact_path: 'e' * 64}
        inventory_bytes = json.dumps(hashes, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=terminal_verified).encode()
        return {'source_artifacts': artifacts, 'source_artifacts_sha256': hashes,
            'source_artifact_inventory_sha256': sha(inventory_bytes), 'screening_policy_sha256': screening_sha,
            'screening_code_sha256': '2' * 64, 'taxonomy_sha256': taxonomy_sha,
            'taxonomy_loader_sha256': '3' * 64, 'export_config_sha256': export_config_sha,
            'workspace_plan_sha256': '4' * 64, 'review_ledger_sha256': '5' * 64,
            'partial_sidecar_sha256': '6' * 64, 'label_guide_sha256': '7' * 64,
            'target_policy': {'mode': 'explicit'}, 'selected_columns': columns,
            'counts': {'train_rows': 2}, 'promotion_eligible': False}

    def make_package(self, report_status='partial_aborted_ambiguous', terminal_verified=None):
        if terminal_verified is None:
            terminal_verified = report_status == 'complete'
        source = self.make_source('3' * 64, [None, 1, 0], [False, True, True])
        truncated = self.make_source('4' * 64, [None, None, None], [False, False, False])
        labels = [f"{item['field']}:{item['label']}" for item in self.columns]
        proposals = [
            {'row_id': source['id'], 'family_sha256': source['family'],
             'text_sha256': source['text_sha256'], 'original_text_sha256': source['original_text_sha256'],
             'screening': source['screening'], 'source_identity': source['source_identity'],
             'columns': self.columns, 'source_targets': source['targets'],
             'source_known_mask': source['known_mask'], 'source_target_origins': source['target_origins'],
             'source_reviewer_claims': source['target_reviewer_claims'],
             'verified_source_claim': source['verified_source_claim'], 'status': 'answered',
             'proposal_states': {labels[0]: 'proposed_positive', labels[1]: 'proposed_positive',
                                 labels[2]: 'proposed_positive'},
             'conflict_labels': [labels[2]], 'provenance': 'unreviewed_local_teacher_claim'},
            {'row_id': truncated['id'], 'family_sha256': truncated['family'],
             'text_sha256': truncated['text_sha256'], 'original_text_sha256': truncated['original_text_sha256'],
             'screening': truncated['screening'], 'source_identity': truncated['source_identity'],
             'columns': self.columns, 'source_targets': truncated['targets'],
             'source_known_mask': truncated['known_mask'], 'source_target_origins': truncated['target_origins'],
             'source_reviewer_claims': truncated['target_reviewer_claims'],
             'verified_source_claim': truncated['verified_source_claim'], 'status': 'truncated',
             'proposal_states': {label: 'unknown' for label in labels},
             'conflict_labels': [], 'provenance': 'unreviewed_local_teacher_claim'}]
        proposals_raw = b''.join((json.dumps(row, sort_keys=True, ensure_ascii=False,
            separators=(',', ':')) + '\n').encode() for row in proposals)
        digest_names = ('proposal_config_sha256', 'proposal_config_origin_sha256',
            'proposal_config_identity_sha256', 'comparison_config_sha256', 'adapter_code_sha256',
            'comparator_code_sha256', 'worker_config_sha256', 'payload_sha256',
            'export_manifest_sha256', 'export_config_sha256', 'train_file_sha256', 'taxonomy_sha256',
            'selected_row_ids_sha256', 'selected_text_digests_sha256', 'target_mask_sha256',
            'proposal_schema_sha256', 'prompt_contract_sha256', 'progress_store_id_sha256',
            'progress_store_events_sha256', 'model_manifest_sha256', 'server_binary_sha256')
        provenance = {key: '5' * 64 for key in digest_names}
        lineage = self.make_lineage(self.columns, '9' * 64, '7' * 64, 'd' * 64,
                                    terminal_verified=terminal_verified)
        schema = {'type': 'object', 'additionalProperties': False,
            'properties': {'labels': {'type': 'array', 'items': {'type': 'string', 'enum': labels},
                                      'uniqueItems': True}, 'abstain': {'type': 'boolean'}},
            'required': ['labels', 'abstain']}
        provenance.update({'export_manifest_sha256': '6' * 64, 'export_config_sha256': '7' * 64,
            'train_file_sha256': '8' * 64, 'taxonomy_sha256': '9' * 64,
            'selected_row_ids_sha256': sha(json.dumps([source['id'], truncated['id']], separators=(',', ':')).encode()),
            'selected_text_digests_sha256': sha(json.dumps([source['text_sha256'], truncated['text_sha256']], separators=(',', ':')).encode()),
            'target_mask_sha256': sha(json.dumps([source['known_mask'], truncated['known_mask']], separators=(',', ':')).encode()),
            'proposal_schema_sha256': sha(json.dumps(schema, sort_keys=True, separators=(',', ':')).encode()),
            'source_lineage': lineage,
            'selected_row_count': 2, 'model_name': 'synthetic teacher',
            'terminal_worker_result_verified': terminal_verified, 'progress_resumed': True})
        if not terminal_verified:
            provenance['worker_config_sha256'] = None
            provenance['payload_sha256'] = None
        statuses = {'answered': 1, 'truncated': 1}
        manifest = {'schema_version': overlay.REPORT_SCHEMA, 'status': report_status,
            'promotion_eligible': False,
            'claims': {'teacher_suggestions_are_unreviewed': True, 'heldout_accessed': False,
                       'labels_independently_verified': False,
                       'screening_independently_verified': False,
                       'promotion_eligible': False},
            'counts': {'selected_rows': 2, 'statuses': statuses, 'answered_rows': 1,
                'abstentions': 0, 'failures': 1, 'not_attempted_after_abort': 0},
            'uncertainty': {'source_and_review_claims_independently_verified': False,
                'teacher_labels_are_claims': True, 'omitted_labels_are_negative': False,
                'selection_claim': 'configured train-only subset; no heldout data opened'},
            'provenance': provenance}
        manifest.update({'proposals_sha256': sha(proposals_raw), 'proposals_bytes': len(proposals_raw),
                         'rows_written': len(proposals_raw.splitlines())})
        manifest['publication_id'] = sha((json.dumps({'proposal_sha256': sha(proposals_raw),
            'progress_events_sha256': provenance['progress_store_events_sha256'],
            'status': manifest['status']}, sort_keys=True, ensure_ascii=False,
            separators=(',', ':'))).encode())
        manifest_raw = json.dumps(manifest).encode()
        config = {'schema_version': overlay.CONFIG_SCHEMA, 'report_manifest': 'C:/private/report.json',
            'report_manifest_sha256': sha(manifest_raw), 'proposals_file': 'C:/private/proposals.jsonl',
            'proposals_sha256': sha(proposals_raw), 'default_weight': 1.5,
            'per_label_weights': {labels[0]: 2.0},
            'conflict_policy': 'retain_source_and_skip_conflicting_teacher_positive',
            'max_manifest_bytes': 100000, 'max_proposals_bytes': 100000}
        return [source, truncated], config, manifest_raw, proposals_raw, lineage

    def test_only_unknown_positive_becomes_candidate(self):
        sources, config, manifest_raw, proposals_raw, lineage = self.make_package()
        with patch.object(overlay, '_read_private', side_effect=[manifest_raw, proposals_raw]):
            result = overlay.load_and_join(config, sources, source_manifest_sha256='6' * 64,
                source_config_sha256='7' * 64, source_train_sha256='8' * 64,
                taxonomy_sha256='9' * 64, selected_columns=self.columns,
                expected_source_lineage=lineage, max_rows=10)
        self.assertEqual(len(result['candidates']), 1)
        self.assertEqual(result['candidates'][0]['label_name'], 'intents:build.software')
        self.assertEqual(result['counts']['source_positive_agreement'], 1)
        self.assertEqual(result['counts']['source_negative_conflict'], 1)
        self.assertEqual(result['counts']['nonanswered'], 1)
        self.assertEqual(result['counts']['unknown'], 3)
        self.assertFalse(result['provenance']['terminal_worker_result_verified'])

    def test_complete_producer_schema_uses_nested_source_provenance(self):
        sources, config, manifest_raw, proposals_raw, lineage = self.make_package('complete')
        report = masked._load_json_bytes(manifest_raw, 'synthetic report')
        self.assertNotIn('selected_columns', report)
        self.assertNotIn('taxonomy_sha256', report)
        self.assertNotIn('source_lineage', report)
        self.assertEqual(report['provenance']['source_lineage'], lineage)
        with patch.object(overlay, '_read_private', side_effect=[manifest_raw, proposals_raw]):
            result = overlay.load_and_join(config, sources, source_manifest_sha256='6' * 64,
                source_config_sha256='7' * 64, source_train_sha256='8' * 64,
                taxonomy_sha256='9' * 64, selected_columns=self.columns,
                expected_source_lineage=lineage, max_rows=10)
        self.assertEqual(len(result['candidates']), 1)

    def test_partial_terminal_verified_report_uses_completed_serializer(self):
        sources, config, manifest_raw, proposals_raw, lineage = self.make_package(
            'partial_aborted_ambiguous', terminal_verified=True)
        report = json.loads(manifest_raw)
        self.assertTrue(report['provenance']['terminal_worker_result_verified'])
        with patch.object(overlay, '_read_private', side_effect=[manifest_raw, proposals_raw]):
            result = overlay.load_and_join(config, sources, source_manifest_sha256='6' * 64,
                source_config_sha256='7' * 64, source_train_sha256='8' * 64,
                taxonomy_sha256='9' * 64, selected_columns=self.columns,
                expected_source_lineage=lineage, max_rows=10)
        self.assertEqual(len(result['candidates']), 1)

    def test_never_interprets_negative_or_missing_states(self):
        sources, config, manifest_raw, proposals_raw, lineage = self.make_package()
        rows = [json.loads(line) for line in proposals_raw.splitlines()]
        rows[0]['proposal_states']['risks:sensitive-data'] = 'proposed_negative'
        proposals_raw = b''.join((json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n').encode() for row in rows)
        config['proposals_sha256'] = sha(proposals_raw)
        manifest = json.loads(manifest_raw)
        manifest.update({'proposals_sha256': sha(proposals_raw), 'proposals_bytes': len(proposals_raw),
                         'rows_written': len(proposals_raw.splitlines())})
        manifest['publication_id'] = sha(json.dumps({'proposal_sha256': sha(proposals_raw),
            'progress_events_sha256': manifest['provenance']['progress_store_events_sha256'],
            'status': manifest['status']}, sort_keys=True, ensure_ascii=False,
            separators=(',', ':')).encode())
        manifest_raw = json.dumps(manifest).encode()
        config['report_manifest_sha256'] = sha(manifest_raw)
        with patch.object(overlay, '_read_private', side_effect=[manifest_raw, proposals_raw]), \
             self.assertRaisesRegex(ValueError, 'non-exhaustive teacher report'):
            overlay.load_and_join(config, sources, source_manifest_sha256='6' * 64,
                source_config_sha256='7' * 64, source_train_sha256='8' * 64,
                taxonomy_sha256='9' * 64, selected_columns=self.columns,
                expected_source_lineage=lineage, max_rows=10)

    def test_invalid_conflict_strategy_and_weight_are_rejected(self):
        _, config, _, _, _ = self.make_package()
        config['conflict_policy'] = 'teacher_wins'
        with self.assertRaisesRegex(ValueError, 'conflict policy'):
            overlay.validate_config(config, self.columns)
        _, config, _, _, _ = self.make_package()
        config['default_weight'] = float('inf')
        with self.assertRaisesRegex(ValueError, 'finite and positive'):
            overlay.validate_config(config, self.columns)

    def test_unknown_overlay_extension_is_retained_but_inert(self):
        sources, config, manifest_raw, proposals_raw, lineage = self.make_package()
        config['future_extension'] = {'teacher_weight': 999999, 'claim': 'opaque'}
        with patch.object(overlay, '_read_private', side_effect=[manifest_raw, proposals_raw]):
            result = overlay.load_and_join(config, sources, source_manifest_sha256='6' * 64,
                source_config_sha256='7' * 64, source_train_sha256='8' * 64,
                taxonomy_sha256='9' * 64, selected_columns=self.columns,
                expected_source_lineage=lineage, max_rows=10)
        self.assertEqual(result['weight_policy']['default_weight'], 1.5)
        self.assertEqual(result['weight_policy']['per_label_weights'], {'intents:build.software': 2.0})

    def test_missing_or_exhaustive_mode_evidence_fails_closed(self):
        sources, config, manifest_raw, proposals_raw, lineage = self.make_package()
        report = masked._load_json_bytes(manifest_raw, 'synthetic report')
        report['uncertainty']['omitted_labels_are_negative'] = True
        report_raw = json.dumps(report).encode()
        config['report_manifest_sha256'] = sha(report_raw)
        with patch.object(overlay, '_read_private', side_effect=[report_raw, proposals_raw]), \
             self.assertRaisesRegex(ValueError, 'non-exhaustive uncertainty'):
            overlay.load_and_join(config, sources, source_manifest_sha256='6' * 64,
                source_config_sha256='7' * 64, source_train_sha256='8' * 64,
                taxonomy_sha256='9' * 64, selected_columns=self.columns,
                expected_source_lineage=lineage, max_rows=10)

        _, config, manifest_raw, proposals_raw, lineage = self.make_package()
        report = masked._load_json_bytes(manifest_raw, 'synthetic report')
        del report['uncertainty']['omitted_labels_are_negative']
        report_raw = json.dumps(report).encode()
        config['report_manifest_sha256'] = sha(report_raw)
        with patch.object(overlay, '_read_private', side_effect=[report_raw, proposals_raw]), \
             self.assertRaisesRegex(ValueError, 'non-exhaustive uncertainty'):
            overlay.load_and_join(config, sources, source_manifest_sha256='6' * 64,
                source_config_sha256='7' * 64, source_train_sha256='8' * 64,
                taxonomy_sha256='9' * 64, selected_columns=self.columns,
                expected_source_lineage=lineage, max_rows=10)

    def test_report_parser_rejects_duplicate_keys_and_nonfinite_numbers_safely(self):
        sources, config, _, proposals_raw, lineage = self.make_package()
        base = json.dumps({'schema_version': overlay.REPORT_SCHEMA}).encode()[:-1]
        for suffix in (b',"extension":{"x":1,"x":2}}', b',"extension":Infinity}'):
            report_raw = base + suffix
            config['report_manifest_sha256'] = sha(report_raw)
            with patch.object(overlay, '_read_private', side_effect=[report_raw, proposals_raw]), \
                 self.assertRaisesRegex(ValueError, 'manifest is invalid JSON') as caught:
                overlay.load_and_join(config, sources, source_manifest_sha256='6' * 64,
                    source_config_sha256='7' * 64, source_train_sha256='8' * 64,
                    taxonomy_sha256='9' * 64, selected_columns=self.columns,
                    expected_source_lineage=lineage, max_rows=10)
            self.assertNotIn('C:/private', str(caught.exception))
            self.assertNotIn('proposals.jsonl', str(caught.exception))

    def test_proposal_parser_rejects_duplicate_nested_keys_and_nan(self):
        sources, config, manifest_raw, proposals_raw, lineage = self.make_package()
        malformed_rows = [
            proposals_raw.replace(b'"proposal_states":{', b'"proposal_states":{"x":1,"x":2,', 1),
            proposals_raw.replace(b'"family_sha256":"', b'"elapsed_ms":NaN,"family_sha256":"', 1),
        ]
        for bad_raw in malformed_rows:
            manifest = json.loads(manifest_raw)
            manifest.update({'proposals_sha256': sha(bad_raw), 'proposals_bytes': len(bad_raw),
                             'rows_written': len(bad_raw.splitlines())})
            manifest['publication_id'] = sha(json.dumps({'proposal_sha256': sha(bad_raw),
                'progress_events_sha256': manifest['provenance']['progress_store_events_sha256'],
                'status': manifest['status']}, sort_keys=True, ensure_ascii=False,
                separators=(',', ':')).encode())
            bound_manifest = json.dumps(manifest).encode()
            config['report_manifest_sha256'] = sha(bound_manifest)
            config['proposals_sha256'] = sha(bad_raw)
            with patch.object(overlay, '_read_private', side_effect=[bound_manifest, bad_raw]), \
                 self.assertRaisesRegex(ValueError, 'proposal row is invalid') as caught:
                overlay.load_and_join(config, sources, source_manifest_sha256='6' * 64,
                    source_config_sha256='7' * 64, source_train_sha256='8' * 64,
                    taxonomy_sha256='9' * 64, selected_columns=self.columns,
                    expected_source_lineage=lineage, max_rows=10)
            self.assertNotIn('screened synthetic request', str(caught.exception))

    def test_terminal_verified_unicode_inventory_uses_producer_ascii_serialization(self):
        sources, config, manifest_raw, proposals_raw, lineage = self.make_package('complete')
        report = json.loads(manifest_raw)
        hashes = lineage['source_artifacts_sha256']
        wrong_unicode_hash = sha(json.dumps(hashes, sort_keys=True, ensure_ascii=False,
            separators=(',', ':')).encode())
        self.assertNotEqual(wrong_unicode_hash, lineage['source_artifact_inventory_sha256'])
        report['provenance']['source_lineage']['source_artifact_inventory_sha256'] = wrong_unicode_hash
        report_raw = json.dumps(report).encode()
        config['report_manifest_sha256'] = sha(report_raw)
        with patch.object(overlay, '_read_private', side_effect=[report_raw, proposals_raw]), \
             self.assertRaisesRegex(ValueError, 'source-lineage claims differ'):
            overlay.load_and_join(config, sources, source_manifest_sha256='6' * 64,
                source_config_sha256='7' * 64, source_train_sha256='8' * 64,
                taxonomy_sha256='9' * 64, selected_columns=self.columns,
                expected_source_lineage=lineage, max_rows=10)

    def test_posix_report_parent_and_file_permissions(self):
        if os.name != 'posix':
            self.skipTest('POSIX permissions are platform-specific')
        with tempfile.TemporaryDirectory(prefix='lfg-teacher-private-') as temp:
            root = Path(temp); root.chmod(0o700)
            private = root / 'report'; private.mkdir(mode=0o700)
            report = private / 'manifest.json'; report.write_bytes(b'{"synthetic":true}')
            report.chmod(0o600)
            self.assertTrue(overlay._verify_report_directory(private))
            self.assertEqual(overlay._read_private(report, 1024, 'synthetic report'), b'{"synthetic":true}')
            private.chmod(0o755)
            self.assertFalse(overlay._verify_report_directory(private))
            private.chmod(0o700)
            public_file = private / 'public.json'; public_file.write_bytes(b'{}'); public_file.chmod(0o644)
            with self.assertRaises(ValueError):
                overlay._read_private(public_file, 1024, 'synthetic report')
            file_link = private / 'link.json'
            file_link.symlink_to(public_file)
            with self.assertRaises(ValueError):
                overlay._read_private(file_link, 1024, 'synthetic report')
            symlink = root / 'report-link'
            symlink.symlink_to(private, target_is_directory=True)
            self.assertFalse(overlay._verify_report_directory(symlink))

    @unittest.skipUnless(os.name == 'nt', 'Windows report ACL policy is platform-specific')
    def test_windows_producer_acl_parent_and_owner_private_files(self):
        import compare_local_12b as producer
        with tempfile.TemporaryDirectory(prefix='lfg-teacher-report-acl-') as temp:
            private = Path(temp) / 'report'
            producer._private_report_directory(private)
            report = private / 'manifest.json'
            masked._write_private_file(report, b'{"synthetic":true}')
            file_acl_before = subprocess.run(['icacls', str(report)], check=True, capture_output=True,
                                             text=True, shell=False).stdout
            acl_before = subprocess.run(['icacls', str(private)], check=True, capture_output=True,
                                        text=True, shell=False).stdout
            self.assertTrue(overlay._verify_report_directory(private))
            self.assertEqual(overlay._read_private(report, 1024, 'synthetic report'), b'{"synthetic":true}')
            acl_after = subprocess.run(['icacls', str(private)], check=True, capture_output=True,
                                       text=True, shell=False).stdout
            file_acl_after = subprocess.run(['icacls', str(report)], check=True, capture_output=True,
                                            text=True, shell=False).stdout
            self.assertEqual(acl_before, acl_after)
            self.assertEqual(file_acl_before, file_acl_after)
            subprocess.run(['icacls', str(private), '/grant', 'Everyone:(F)'], check=True,
                           capture_output=True, text=True, shell=False)
            self.assertFalse(overlay._verify_report_directory(private))

    @unittest.skipUnless(os.name == 'nt', 'Windows reparse-point policy is platform-specific')
    def test_windows_reparse_report_parent_is_rejected(self):
        import compare_local_12b as producer
        with tempfile.TemporaryDirectory(prefix='lfg-teacher-report-reparse-') as temp:
            root = Path(temp)
            private = root / 'report'
            producer._private_report_directory(private)
            link = root / 'report-link'
            try:
                os.symlink(private, link, target_is_directory=True)
            except OSError:
                self.skipTest('host cannot create a synthetic Windows reparse point')
            self.assertFalse(overlay._verify_report_directory(link))

    def test_builder_output_is_accepted_by_the_real_owned_loader(self):
        columns = [{'field': 'intents', 'label': 'build.software'}]
        row_id = 'a' * 64
        taxonomy = {'schema_version': 'lfg.encoder-evaluation-taxonomy.v1',
                    'intents': ['build.software']}
        taxonomy_raw = masked._json_bytes(taxonomy)
        taxonomy_sha = sha(taxonomy_raw)
        train_raw = b'{"synthetic":true}\n'
        export_manifest = {'schema_version': masked.EXPORT_SCHEMA, 'config_sha256': 'b' * 64,
            'taxonomy_sha256': taxonomy_sha, 'label_guide_sha256': None,
            'selected_columns': columns,
            'source_artifacts': [{'path': 'prøject/fixture.jsonl', 'split': 'train', 'sha256': 'c' * 64,
                'project_claim_sha256': 'd' * 64, 'project_group_claim_sha256': 'e' * 64}],
            'screening_policy_sha256': 'f' * 64,
            'output_files': {'train.jsonl': {'sha256': sha(train_raw), 'bytes': len(train_raw)}}}
        export_manifest_raw = masked._json_bytes(export_manifest)
        export_manifest_sha = sha(export_manifest_raw)
        source = {'id': row_id, 'split': 'train', 'text': 'synthetic screened request',
            'text_sha256': sha(b'synthetic screened request'), 'original_text_sha256': sha(b'original request'),
            'family': '1' * 64, 'columns': columns, 'targets': [None], 'known_mask': [False],
            'weights': [0.0], 'target_origins': ['unspecified_unknown'], 'target_reviewer_claims': [None],
            'verified_source_claim': False, 'partial_review_provenance': None, 'full_review_provenance': None,
            'source_identity': {'artifact_path': 'prøject/fixture.jsonl', 'artifact_sha256': 'c' * 64,
                'project_claim_sha256': 'd' * 64, 'project_group_claim_sha256': 'e' * 64,
                'record_id_sha256': row_id, 'source_claim_sha256': '2' * 64},
            'screening': {'policy_sha256': 'f' * 64, 'redaction_counts': {}, 'text_changed': True}}
        proposal = {'row_id': row_id, 'family_sha256': source['family'],
            'text_sha256': source['text_sha256'], 'original_text_sha256': source['original_text_sha256'],
            'screening': source['screening'], 'source_identity': source['source_identity'],
            'columns': columns, 'source_targets': source['targets'], 'source_known_mask': source['known_mask'],
            'source_target_origins': source['target_origins'], 'source_reviewer_claims': source['target_reviewer_claims'],
            'verified_source_claim': False, 'status': 'answered',
            'proposal_states': {'intents:build.software': 'proposed_positive'}, 'conflict_labels': [],
            'provenance': 'unreviewed_local_teacher_claim'}
        proposals_raw = (json.dumps(proposal, sort_keys=True, separators=(',', ':')) + '\n').encode()
        provenance = {key: '3' * 64 for key in ('proposal_config_sha256', 'proposal_config_origin_sha256',
            'proposal_config_identity_sha256', 'comparison_config_sha256', 'adapter_code_sha256',
            'comparator_code_sha256', 'worker_config_sha256', 'payload_sha256', 'export_manifest_sha256',
            'export_config_sha256', 'train_file_sha256', 'taxonomy_sha256', 'selected_row_ids_sha256',
            'selected_text_digests_sha256', 'target_mask_sha256', 'proposal_schema_sha256',
            'prompt_contract_sha256', 'progress_store_id_sha256', 'progress_store_events_sha256',
            'model_manifest_sha256', 'server_binary_sha256')}
        schema = {'type': 'object', 'additionalProperties': False,
            'properties': {'labels': {'type': 'array', 'items': {'type': 'string',
                'enum': ['intents:build.software']}, 'uniqueItems': True}, 'abstain': {'type': 'boolean'}},
            'required': ['labels', 'abstain']}
        lineage_hashes = {'prøject/fixture.jsonl': 'c' * 64}
        source_lineage = {'source_artifacts': export_manifest['source_artifacts'],
            'source_artifacts_sha256': lineage_hashes,
            'source_artifact_inventory_sha256': sha(json.dumps(lineage_hashes,
                sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()),
            'screening_policy_sha256': 'f' * 64, 'screening_code_sha256': None,
            'taxonomy_sha256': taxonomy_sha, 'taxonomy_loader_sha256': None,
            'export_config_sha256': 'b' * 64, 'workspace_plan_sha256': None,
            'review_ledger_sha256': None, 'partial_sidecar_sha256': None,
            'label_guide_sha256': None, 'target_policy': None, 'selected_columns': columns,
            'counts': None, 'promotion_eligible': False}
        provenance.update({'export_manifest_sha256': export_manifest_sha, 'export_config_sha256': 'b' * 64,
            'train_file_sha256': sha(train_raw), 'taxonomy_sha256': taxonomy_sha,
            'selected_row_ids_sha256': sha(json.dumps([row_id], separators=(',', ':')).encode()),
            'selected_text_digests_sha256': sha(json.dumps([source['text_sha256']], separators=(',', ':')).encode()),
            'target_mask_sha256': sha(json.dumps([source['known_mask']], separators=(',', ':')).encode()),
            'proposal_schema_sha256': sha(json.dumps(schema, sort_keys=True, separators=(',', ':')).encode()),
            'source_lineage': source_lineage, 'selected_row_count': 1, 'model_name': 'synthetic-model',
            'terminal_worker_result_verified': False, 'progress_resumed': True})
        provenance['worker_config_sha256'] = None
        provenance['payload_sha256'] = None
        report = {'schema_version': overlay.REPORT_SCHEMA, 'status': 'partial_aborted_ambiguous',
            'promotion_eligible': False,
            'claims': {'teacher_suggestions_are_unreviewed': True, 'heldout_accessed': False,
                       'labels_independently_verified': False, 'screening_independently_verified': False,
                       'promotion_eligible': False},
            'counts': {'selected_rows': 1, 'statuses': {'answered': 1}, 'answered_rows': 1,
                'abstentions': 0, 'failures': 0, 'not_attempted_after_abort': 0},
            'uncertainty': {'source_and_review_claims_independently_verified': False,
                'teacher_labels_are_claims': True, 'omitted_labels_are_negative': False,
                'selection_claim': 'configured train-only subset; no heldout data opened'},
            'provenance': provenance}
        report.update({'proposals_sha256': sha(proposals_raw), 'proposals_bytes': len(proposals_raw),
                       'rows_written': len(proposals_raw.splitlines())})
        report['publication_id'] = sha((json.dumps({'proposal_sha256': sha(proposals_raw),
            'progress_events_sha256': provenance['progress_store_events_sha256'],
            'status': report['status']}, sort_keys=True, ensure_ascii=False,
            separators=(',', ':'))).encode())
        report_raw = json.dumps(report).encode()
        with tempfile.TemporaryDirectory(prefix='lfg-teacher-overlay-e2e-') as temp:
            root = Path(temp)
            export_dir = root / 'export'; export_dir.mkdir()
            (export_dir / 'manifest.json').write_bytes(export_manifest_raw)
            taxonomy_path = root / 'taxonomy.json'; taxonomy_path.write_bytes(taxonomy_raw)
            config = {'schema_version': contract.CONFIG_SCHEMA, 'export_dir': str(export_dir),
                'export_manifest_sha256': export_manifest_sha, 'taxonomy_file': str(taxonomy_path),
                'taxonomy_sha256': taxonomy_sha, 'selected_columns': columns,
                'instruction_template': 'Classify {field}:{label}', 'output': str(root / 'out'),
                'max_rows': 10, 'max_output_bytes': 100000, 'default_weight': 2.0,
                'per_label_weights': {},
                'legacy_teacher_overlay': {'schema_version': overlay.CONFIG_SCHEMA,
                    'report_manifest': 'C:/private/report.json', 'report_manifest_sha256': sha(report_raw),
                    'proposals_file': 'C:/private/proposals.jsonl', 'proposals_sha256': sha(proposals_raw),
                    'default_weight': 1.5, 'per_label_weights': {},
                    'conflict_policy': 'retain_source_and_skip_conflicting_teacher_positive',
                    'max_manifest_bytes': 100000, 'max_proposals_bytes': 100000}}
            config_path = root / 'config.json'; config_path.write_bytes(masked._json_bytes(config))
            with patch.object(masked, '_verify_owner_private_access'), \
                 patch.object(masked, 'load_export', return_value=(export_manifest, {'train': [source]})), \
                 patch.object(overlay, '_read_private', side_effect=[report_raw, proposals_raw,
                                                                      report_raw, proposals_raw]):
                result = contract.build_dataset(config_path)
            with patch.object(masked, '_verify_owner_private_access'):
                manifest, rows = contract.load_training_dataset(Path(config['output']), result['manifest_sha256'])
                import train_owned_assessment_sft as trainer
                example = json.loads(Path(trainer.__file__).with_name('owned-assessment-sft.example.json').read_text())
                records, _, _, coverage = trainer.read_multilabel_training(Path(config['output']), {
                    'expected_manifest_sha256': result['manifest_sha256'],
                    'max_manifest_bytes': 1_000_000, 'max_dataset_bytes': 1_000_000,
                    'max_record_bytes': 100_000, 'max_examples': 10,
                    'framing': example['framing'], 'multilabel_system_prompt': 'Synthetic system.',
                    'context': 1000, 'allow_truncated_prompt': False})
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['decision'], 'positive')
            self.assertEqual(rows[0]['target_origin'], overlay.ORIGIN)
            self.assertEqual(rows[0]['source_target_origin'], 'unspecified_unknown')
            self.assertEqual(rows[0]['weight'], 3.0)
            self.assertEqual(records[0]['weight'], 3.0)
            self.assertEqual(coverage['selected'], 1)
            self.assertEqual(coverage['target_support_claim']['unknown'], 0)
            self.assertEqual(manifest['target_support']['unknown'], 0)
            self.assertEqual(manifest['target_support']['positive'], 1)
            self.assertEqual(rows[0]['teacher_evidence']['prompt_contract_sha256'], '3' * 64)

            complete_report = json.loads(report_raw)
            complete_report['status'] = 'complete'
            complete_report['provenance']['terminal_worker_result_verified'] = True
            complete_report['provenance']['worker_config_sha256'] = '4' * 64
            complete_report['provenance']['payload_sha256'] = '5' * 64
            complete_report['provenance']['source_lineage']['source_artifact_inventory_sha256'] = sha(
                json.dumps(lineage_hashes, sort_keys=True, separators=(',', ':')).encode())
            complete_report['publication_id'] = sha(json.dumps({
                'proposal_sha256': sha(proposals_raw),
                'progress_events_sha256': complete_report['provenance']['progress_store_events_sha256'],
                'status': 'complete'}, sort_keys=True, ensure_ascii=False,
                separators=(',', ':')).encode())
            complete_report_raw = json.dumps(complete_report).encode()
            complete_config = json.loads(json.dumps(config))
            complete_config['output'] = str(root / 'complete-out')
            complete_config['legacy_teacher_overlay']['report_manifest_sha256'] = sha(complete_report_raw)
            complete_config_path = root / 'complete-config.json'
            complete_config_path.write_bytes(masked._json_bytes(complete_config))
            with patch.object(masked, '_verify_owner_private_access'), \
                 patch.object(masked, 'load_export', return_value=(export_manifest, {'train': [source]})), \
                 patch.object(overlay, '_read_private', side_effect=[complete_report_raw, proposals_raw,
                                                                      complete_report_raw, proposals_raw]):
                complete_result = contract.build_dataset(complete_config_path)
            with patch.object(masked, '_verify_owner_private_access'):
                complete_manifest, complete_rows = contract.load_training_dataset(
                    Path(complete_config['output']), complete_result['manifest_sha256'])
                complete_records, _, _, complete_coverage = trainer.read_multilabel_training(
                    Path(complete_config['output']), {
                        'expected_manifest_sha256': complete_result['manifest_sha256'],
                        'max_manifest_bytes': 1_000_000, 'max_dataset_bytes': 1_000_000,
                        'max_record_bytes': 100_000, 'max_examples': 10,
                        'framing': example['framing'], 'multilabel_system_prompt': 'Synthetic system.',
                        'context': 1000, 'allow_truncated_prompt': False})
            self.assertEqual(complete_manifest['legacy_teacher_overlay']['report_status'], 'complete')
            self.assertEqual(len(complete_rows), 1)
            self.assertEqual(complete_records[0]['weight'], 3.0)
            self.assertEqual(complete_coverage['target_support_claim']['unknown'], 0)

            # A non-teacher row cannot use an inert extra field to replace the
            # source-review origin that the legacy loader is required to validate.
            spoof = json.loads(json.dumps(rows[0]))
            spoof['target_origin'] = 'partial_explicit_positive'
            spoof['source_target_origin'] = 'owner_authorized_unreviewed_positive'
            spoof.pop('teacher_evidence')
            spoof['source_weight'] = 1.0
            spoof['configured_weight'] = 2.0
            spoof['weight'] = 2.0
            source_only = json.loads(json.dumps(manifest))
            source_only.pop('legacy_teacher_overlay')
            source_only['rows'] = 1
            spoof_raw = masked._json_bytes(spoof)
            source_only['train_file'] = {'sha256': sha(spoof_raw), 'bytes': len(spoof_raw)}
            source_only_raw = masked._json_bytes(source_only)
            spoof_dir = root / 'spoofed-source-only'; spoof_dir.mkdir()
            (spoof_dir / 'train.jsonl').write_bytes(spoof_raw)
            (spoof_dir / 'manifest.json').write_bytes(source_only_raw)
            with patch.object(masked, '_verify_owner_private_access'), \
                 self.assertRaisesRegex(ValueError, 'partial-review reviewer claim'):
                contract.load_training_dataset(spoof_dir, sha(source_only_raw))


if __name__ == '__main__':
    unittest.main()
