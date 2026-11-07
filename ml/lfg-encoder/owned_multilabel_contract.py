#!/usr/bin/env python3
"""Private, train-only atomic supervision and explicit multilabel response contract.

This adapter prepares examples; it does not train or qualify a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any

import masked_supervision as masked

CONTRACT_SCHEMA = 'lfg.owned-multilabel-response.v1'
CONFIG_SCHEMA = 'lfg.owned-multilabel-training-config.v1'
DATASET_SCHEMA = 'lfg.owned-multilabel-training-row.v1'
MANIFEST_SCHEMA = 'lfg.owned-multilabel-training-manifest.v1'
MAX_CONFIG_BYTES = 1024 * 1024
MAX_TEMPLATE_BYTES = 64 * 1024
MAX_EXTENSION_BYTES = 64 * 1024
MAX_MANIFEST_BYTES = 64 * 1024 * 1024
MAX_DATASET_BYTES = 2 * 1024 * 1024 * 1024
SHA256_RE = __import__('re').compile(r'^[0-9a-f]{64}$')
HEX32_RE = __import__('re').compile(r'^[0-9a-f]{32}$')


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':'),
                       allow_nan=False) + '\n').encode('utf-8')


def _read_json(path: Path, maximum: int, what: str, *, owner_private: bool = True) -> tuple[Any, bytes]:
    raw = masked._read_bounded(path, maximum, what)
    if owner_private:
        masked._verify_owner_private_access(path, is_directory=False)
    return masked._load_json_bytes(raw, what), raw


def validate_response(value: Any, columns: list[dict[str, str]], *, allow_abstention: bool = True) -> dict[str, Any]:
    """Validate decisions. Omitted decisions stay unknown and never mean negative."""
    if not isinstance(value, dict) or value.get('schema_version') != CONTRACT_SCHEMA:
        raise ValueError('unsupported multilabel response contract')
    allowed = {'schema_version', 'taxonomy_sha256', 'guide_sha256', 'decisions', 'abstain', 'extensions'}
    inert_fields = {key: item for key, item in value.items() if key not in allowed}
    try:
        if len(_json_bytes(inert_fields)) > MAX_EXTENSION_BYTES:
            raise ValueError('response extension fields exceed the configured bound')
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError('response extension fields are not bounded JSON data') from error
    decisions = value.get('decisions')
    if not isinstance(decisions, list):
        raise ValueError('decisions must be an array')
    expected = {(item['field'], item['label']) for item in columns}
    seen: set[tuple[str, str]] = set()
    normalized = []
    for decision in decisions:
        if not isinstance(decision, dict) or set(decision) != {'field', 'label', 'value'}:
            raise ValueError('decision must contain exactly field, label and value')
        if not isinstance(decision['field'], str) or not isinstance(decision['label'], str):
            raise ValueError('decision identity must use string field and label values')
        identity = (decision['field'], decision['label'])
        if identity not in expected:
            raise ValueError('decision is outside the selected taxonomy')
        if identity in seen:
            raise ValueError('duplicate or contradictory decision')
        if decision['value'] not in ('positive', 'negative', 'unknown'):
            raise ValueError('invalid decision value')
        seen.add(identity)
        normalized.append(dict(zip(('field', 'label', 'value'), identity + (decision['value'],))))
    abstain = value.get('abstain', False)
    if type(abstain) is not bool or (abstain and not allow_abstention):
        raise ValueError('abstention is disabled or invalid')
    if abstain and any(item['value'] in ('positive', 'negative') for item in normalized):
        raise ValueError('abstention cannot assert positive or negative labels')
    extensions = value.get('extensions', {})
    if not isinstance(extensions, dict):
        raise ValueError('extensions must be an object')
    try:
        if len(_json_bytes(extensions)) > MAX_EXTENSION_BYTES:
            raise ValueError('response extensions exceed the configured bound')
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError('response extensions are not bounded JSON data') from error
    for name in ('taxonomy_sha256', 'guide_sha256'):
        digest = value.get(name)
        if name == 'taxonomy_sha256' and digest is None:
            raise ValueError('taxonomy_sha256 is required')
        if digest is not None and (not isinstance(digest, str) or len(digest) != 64 or
                                   any(c not in '0123456789abcdef' for c in digest)):
            raise ValueError(f'{name} is invalid')
    return {**value, 'decisions': normalized}


def validate_config(config: Any) -> dict[str, Any]:
    required = {'schema_version', 'export_dir', 'export_manifest_sha256', 'taxonomy_file',
                'taxonomy_sha256', 'selected_columns', 'instruction_template', 'output',
                'max_rows', 'max_output_bytes', 'default_weight', 'per_label_weights'}
    if not isinstance(config, dict) or not required <= set(config) or config.get('schema_version') != CONFIG_SCHEMA:
        raise ValueError('unsupported owned multilabel training configuration')
    try:
        inert_extensions = {key: value for key, value in config.items() if key not in required}
        if len(_json_bytes(inert_extensions)) > MAX_EXTENSION_BYTES:
            raise ValueError('configuration extensions exceed the configured bound')
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError('configuration extensions are not bounded JSON data') from error
    for field in ('export_dir', 'taxonomy_file', 'output'):
        if not isinstance(config[field], str) or not Path(config[field]).is_absolute():
            raise ValueError(f'{field} must be an absolute path')
    for digest in ('export_manifest_sha256', 'taxonomy_sha256'):
        value = config[digest]
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError(f'{digest} must be a SHA-256')
    cols = config['selected_columns']
    if not isinstance(cols, list) or not cols or any(not isinstance(c, dict) or set(c) != {'field', 'label'} or
            not isinstance(c['field'], str) or not isinstance(c['label'], str) or not c['label'].strip() for c in cols):
        raise ValueError('selected_columns must contain field/label identities')
    keys = [(c['field'], c['label']) for c in cols]
    if len(set(keys)) != len(keys):
        raise ValueError('selected_columns contains duplicates')
    template = config['instruction_template']
    if not isinstance(template, str) or not template.strip() or len(template.encode()) > MAX_TEMPLATE_BYTES or \
            '{field}' not in template or '{label}' not in template:
        raise ValueError('instruction_template must be bounded and bind field and label')
    if type(config['max_rows']) is not int or config['max_rows'] < 1 or \
            type(config['max_output_bytes']) is not int or not 1 <= config['max_output_bytes'] <= MAX_DATASET_BYTES:
        raise ValueError('output limits must be positive integers')
    weight = config['default_weight']
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
        raise ValueError('default_weight must be finite and positive')
    overrides = config['per_label_weights']
    if not isinstance(overrides, dict):
        raise ValueError('per_label_weights must be an object')
    valid_keys = {f"{c['field']}:{c['label']}" for c in cols}
    if set(overrides) - valid_keys or any(isinstance(v, bool) or not isinstance(v, (int, float)) or
                                          not math.isfinite(v) or v <= 0 for v in overrides.values()):
        raise ValueError('per-label weights must be finite, positive and selected')
    return config


def _validate_taxonomy(taxonomy: Any, columns: list[dict[str, str]]) -> None:
    if not isinstance(taxonomy, dict) or taxonomy.get('schema_version') != 'lfg.encoder-evaluation-taxonomy.v1':
        raise ValueError('unsupported taxonomy')
    for column in columns:
        labels = taxonomy.get(column['field'])
        if not isinstance(labels, list) or column['label'] not in labels:
            raise ValueError('selected label is outside the pinned taxonomy')


def _valid_digest(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _validate_guide_identity(manifest: dict[str, Any]) -> str | None:
    if 'label_guide_sha256' not in manifest:
        raise ValueError('source manifest is missing label_guide_sha256')
    value = manifest['label_guide_sha256']
    if value is not None and not _valid_digest(value):
        raise ValueError('source manifest label_guide_sha256 is malformed')
    return value


def _validate_review_claims(row: dict[str, Any], column_count: int, *, taxonomy_sha256: str | None = None,
                            guide_sha256: str | None = None, source_sha256: str | None = None,
                            text_sha256: str | None = None) -> None:
    reviewers = row.get('target_reviewer_claim_sha256', row.get('target_reviewer_claims'))
    if not isinstance(reviewers, list) or len(reviewers) != column_count or any(
            value is not None and not _valid_digest(value) for value in reviewers):
        raise ValueError('target reviewer provenance is malformed')
    origins = row.get('target_origin', row.get('target_origins'))
    if not isinstance(origins, list) or len(origins) != column_count or any(
            not isinstance(value, str) or not value for value in origins):
        raise ValueError('target origin provenance is malformed')
    verified = row.get('verified_source_claim')
    partial = row.get('partial_review_provenance')
    full = row.get('full_review_provenance')
    if type(verified) is not bool:
        raise ValueError('verified source claim is malformed')
    if verified:
        if partial is not None or not isinstance(full, dict) or \
                set(full) != {'reviewer_claim_sha256', 'reviewed_at', 'reconciled_partial_review'}:
            raise ValueError('verified source claim lacks consistent full-review provenance')
        full_reviewer = full.get('reviewer_claim_sha256')
        if not _valid_digest(full_reviewer) or not isinstance(full.get('reviewed_at'), str) or not full['reviewed_at']:
            raise ValueError('full-review provenance is malformed')
        reconciliation = full.get('reconciled_partial_review')
        if reconciliation is not None and (not isinstance(reconciliation, dict) or
                set(reconciliation) != {'event_id', 'event_sha256', 'event_type', 'revision', 'acknowledgment'} or
                not isinstance(reconciliation.get('event_id'), str) or not HEX32_RE.fullmatch(reconciliation['event_id']) or
                not _valid_digest(reconciliation.get('event_sha256')) or
                reconciliation.get('event_type') not in ('decision', 'retraction') or
                type(reconciliation.get('revision')) is not int or reconciliation['revision'] < 1 or
                reconciliation.get('acknowledgment') != 'RECONCILE PARTIAL REVIEW'):
            raise ValueError('full-review reconciliation provenance is malformed')
        if any(value != full_reviewer for value in reviewers) or any(origin != 'full_reviewed_legacy' for origin in origins):
            raise ValueError('full-review reviewer claim does not match target reviewer claims')
    else:
        if any(origin == 'full_reviewed_legacy' for origin in origins):
            raise ValueError('full-reviewed target origin requires full-review provenance')
        if full is not None:
            raise ValueError('unverified source claim has full-review provenance')
        if partial is not None:
            required = {'event_id', 'event_sha256', 'revision', 'event_type',
                        'reviewer_claim_sha256', 'reviewed_at', 'binding'}
            if not isinstance(partial, dict) or set(partial) != required or \
                    not isinstance(partial.get('event_id'), str) or not HEX32_RE.fullmatch(partial['event_id']) or \
                    not _valid_digest(partial.get('event_sha256')) or \
                    type(partial.get('revision')) is not int or partial['revision'] < 1 or \
                    partial.get('event_type') != 'decision' or \
                    not _valid_digest(partial.get('reviewer_claim_sha256')) or \
                    not isinstance(partial.get('reviewed_at'), str) or not partial['reviewed_at']:
                raise ValueError('partial-review provenance is malformed')
            binding = partial.get('binding')
            bind_keys = {'source_sha256', 'annotation_sha256', 'taxonomy_sha256', 'text_sha256',
                         'proposal_sha256', 'guide_sha256'}
            if not isinstance(binding, dict) or set(binding) != bind_keys or any(
                    not _valid_digest(binding.get(key)) for key in bind_keys - {'guide_sha256'}) or \
                    (binding['guide_sha256'] is not None and not _valid_digest(binding['guide_sha256'])) or \
                    binding.get('taxonomy_sha256') != taxonomy_sha256 or \
                    binding.get('guide_sha256') != guide_sha256 or \
                    binding.get('source_sha256') != source_sha256 or binding.get('text_sha256') != text_sha256:
                raise ValueError('partial-review binding is malformed')
        completeness_claims = {
            'source_attested_complete_array_negative': 'source_target_completeness_claim',
            'operator_assumed_complete_array_negative': 'operator_configured_complete_array_assumption',
        }
        for origin, claim in zip(origins, reviewers, strict=True):
            if origin in completeness_claims:
                expected_claim = masked._claim_digest('reviewer', completeness_claims[origin])
                if claim != expected_claim:
                    raise ValueError('completeness target claim does not match its origin')
            elif origin.startswith(('partial_explicit_', 'rejected_proposal_unknown')):
                if partial is None or claim != partial['reviewer_claim_sha256']:
                    raise ValueError('partial-review reviewer claim does not match target origin')
            elif claim is not None:
                raise ValueError('target reviewer claim lacks matching review origin')


def _validate_source_target_semantics(row: dict[str, Any]) -> None:
    targets, known, weights, origins = (row.get('targets'), row.get('known_mask'),
                                        row.get('weights'), row.get('target_origins'))
    if not isinstance(targets, list) or not isinstance(known, list) or not isinstance(weights, list) or \
            not isinstance(origins, list) or not targets or any(
                len(value) != len(targets) for value in (known, weights, origins)):
        raise ValueError('source target arrays are malformed')
    positive_origins = {'partial_explicit_positive', 'owner_authorized_unreviewed_positive'}
    negative_origins = {'partial_explicit_negative', 'source_attested_complete_array_negative',
                        'operator_assumed_complete_array_negative'}
    unknown_origins = {'partial_explicit_unknown', 'rejected_proposal_unknown', 'unspecified_unknown'}
    if 'full_reviewed_legacy' in origins and row.get('verified_source_claim') is not True:
        raise ValueError('full-reviewed target origin requires a verified source claim')
    for target, is_known, weight, origin in zip(targets, known, weights, origins, strict=True):
        if type(is_known) is not bool or not isinstance(origin, str):
            raise ValueError('source target mask or origin is malformed')
        if is_known:
            if type(target) is not int or target not in (0, 1) or isinstance(weight, bool) or \
                    not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
                raise ValueError('known source target or weight is invalid')
            if origin in positive_origins:
                expected = 1
            elif origin in negative_origins:
                expected = 0
            elif origin == 'full_reviewed_legacy':
                expected = target
            else:
                raise ValueError('known source target origin is unsupported')
            if target != expected:
                raise ValueError('source target value does not match its origin')
        else:
            if target is not None or isinstance(weight, bool) or not isinstance(weight, (int, float)) or \
                    not math.isfinite(weight) or weight != 0 or origin not in unknown_origins:
                raise ValueError('unknown source target mask, weight or origin is inconsistent')


def _validate_training_target_semantics(row: dict[str, Any]) -> None:
    decision, origin = row.get('decision'), row.get('target_origin')
    if origin == 'full_reviewed_legacy' and row.get('verified_source_claim') is not True:
        raise ValueError('full-reviewed target origin requires a verified source claim')
    if origin in ('partial_explicit_positive', 'owner_authorized_unreviewed_positive'):
        expected = 'positive'
    elif origin in ('partial_explicit_negative', 'source_attested_complete_array_negative',
                    'operator_assumed_complete_array_negative', 'full_reviewed_legacy'):
        expected = 'negative' if origin != 'full_reviewed_legacy' else decision
    else:
        raise ValueError('training target origin is unsupported')
    if decision not in ('positive', 'negative') or decision != expected:
        raise ValueError('training target decision does not match its origin')


def _private_output(output: Path) -> Path:
    output = output.resolve(strict=False)
    if output.exists() or output.is_symlink():
        raise ValueError('training artifact already exists')
    probe = output.parent
    while not probe.exists():
        probe = probe.parent
    import subprocess
    result = subprocess.run(['git', '-C', str(probe), 'rev-parse', '--show-toplevel'],
                            capture_output=True, text=True, timeout=10)
    if result.returncode == 0 and output.is_relative_to(Path(result.stdout.strip()).resolve()):
        raise ValueError('training output must be outside Git worktrees')
    return output


def build_dataset(config_path: Path) -> dict[str, Any]:
    config_obj, config_raw = _read_json(config_path, MAX_CONFIG_BYTES, 'training configuration')
    config = validate_config(config_obj)
    export_dir = Path(config['export_dir'])
    manifest_raw = masked._read_bounded(export_dir / 'manifest.json', 64 * 1024 * 1024, 'export manifest')
    if _sha(manifest_raw) != config['export_manifest_sha256']:
        raise ValueError('export manifest differs from pinned digest')
    manifest, splits = masked.load_export(export_dir, train_only=True)
    if _sha(masked._read_bounded(export_dir / 'manifest.json', 64 * 1024 * 1024, 'export manifest')) != config['export_manifest_sha256']:
        raise ValueError('export manifest changed during load')
    taxonomy, taxonomy_raw = _read_json(Path(config['taxonomy_file']), 2 * 1024 * 1024,
                                        'taxonomy', owner_private=False)
    # Text anchors use LF-normalized bytes so Windows checkout endings do not
    # invalidate the pinned taxonomy identity.
    if _sha(taxonomy_raw.replace(b'\r\n', b'\n')) != config['taxonomy_sha256'] or \
            manifest.get('taxonomy_sha256') != config['taxonomy_sha256']:
        raise ValueError('taxonomy identity drift')
    guide_sha = _validate_guide_identity(manifest)
    screening_policy_sha = manifest.get('screening_policy_sha256')
    if not _valid_digest(screening_policy_sha):
        raise ValueError('source manifest screening policy digest is malformed')
    columns = config['selected_columns']
    _validate_taxonomy(taxonomy, columns)
    if any(column not in manifest['selected_columns'] for column in columns):
        raise ValueError('selected columns are not present in export')
    template = config['instruction_template']
    instruction_sha = _sha(template.encode('utf-8'))
    rows = []
    state = {'positive': 0, 'negative': 0, 'unknown': 0}
    weight_totals = {'positive': 0.0, 'negative': 0.0}
    by_label: dict[str, dict[str, int]] = {}
    source_artifacts = manifest.get('source_artifacts')
    if not isinstance(source_artifacts, list):
        raise ValueError('source artifact manifest is missing')
    artifact_index = {}
    for artifact in source_artifacts:
        if not isinstance(artifact, dict) or not isinstance(artifact.get('path'), str) or \
                artifact.get('split') not in ('train', 'validation') or \
                not SHA256_RE.fullmatch(str(artifact.get('sha256', ''))) or \
                not SHA256_RE.fullmatch(str(artifact.get('project_claim_sha256', ''))) or \
                not SHA256_RE.fullmatch(str(artifact.get('project_group_claim_sha256', ''))) or \
                artifact['path'] in artifact_index:
            raise ValueError('source artifact manifest is malformed')
        artifact_index[artifact['path']] = artifact
    for source in splits['train']:
        if source.get('split') != 'train':
            raise ValueError('non-train row encountered in train-only input')
        if not isinstance(source.get('screening'), dict) or not isinstance(source.get('source_identity'), dict):
            raise ValueError('source lineage is incomplete')
        if source.get('text_sha256') != _sha(source['text'].encode('utf-8')):
            raise ValueError('screened source text digest mismatch')
        source_identity = source['source_identity']
        artifact = artifact_index.get(source_identity.get('artifact_path'))
        if artifact is None or artifact.get('split') != 'train' or any(source_identity.get(identity) != artifact.get(source_field)
                                   for identity, source_field in (
                                       ('artifact_sha256', 'sha256'),
                                       ('project_claim_sha256', 'project_claim_sha256'),
                                       ('project_group_claim_sha256', 'project_group_claim_sha256'))):
            raise ValueError('row source identity does not match source manifest')
        if not isinstance(source.get('original_text_sha256'), str) or \
                not SHA256_RE.fullmatch(source['original_text_sha256']):
            raise ValueError('original source text identity is invalid')
        _validate_source_target_semantics(source)
        screening = source['screening']
        if screening.get('policy_sha256') != screening_policy_sha or \
                type(screening.get('text_changed')) is not bool or \
                screening['text_changed'] != (source['original_text_sha256'] != source['text_sha256']) or \
                not isinstance(screening.get('redaction_counts'), dict) or any(
                    not isinstance(key, str) or type(value) is not int or value < 0
                    for key, value in screening['redaction_counts'].items()):
            raise ValueError('source screening provenance does not match its manifest')
        _validate_review_claims(source, len(source['columns']), taxonomy_sha256=config['taxonomy_sha256'],
                                 guide_sha256=guide_sha, source_sha256=source_identity['artifact_sha256'],
                                 text_sha256=source['text_sha256'])
        for column in columns:
            index = source['columns'].index(column)
            known, target, source_weight = source['known_mask'][index], source['targets'][index], source['weights'][index]
            key = f"{column['field']}:{column['label']}"
            support = by_label.setdefault(key, {'positive': 0, 'negative': 0, 'unknown': 0})
            if not known:
                state['unknown'] += 1
                support['unknown'] += 1
                continue
            decision = 'positive' if target == 1 else 'negative' if target == 0 else None
            if decision is None or isinstance(source_weight, bool) or not math.isfinite(source_weight) or source_weight <= 0:
                raise ValueError('known source target or weight is invalid')
            weight = float(config['per_label_weights'].get(key, config['default_weight'])) * float(source_weight)
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError('effective source weight is invalid')
            state[decision] += 1
            support[decision] += 1
            weight_totals[decision] += weight
            prompt = template.format(field=column['field'], label=column['label'])
            target_response = {'schema_version': CONTRACT_SCHEMA, 'taxonomy_sha256': config['taxonomy_sha256'],
                               'guide_sha256': guide_sha,
                               'decisions': [{'field': column['field'], 'label': column['label'], 'value': decision}]}
            rows.append({'schema_version': DATASET_SCHEMA, 'text': source['text'], 'instruction': prompt,
                         'field': column['field'], 'label': column['label'], 'decision': decision,
                         'target_response': _json_bytes(target_response).decode('utf-8').rstrip('\n'),
                         'weight': weight, 'source_weight': float(source_weight),
                         'configured_weight': float(config['per_label_weights'].get(key, config['default_weight'])),
                         'family_sha256': source['family'], 'source_identity': source['source_identity'],
                         'text_sha256': source['text_sha256'],
                         'original_text_sha256': source['original_text_sha256'],
                         'target_origin': source['target_origins'][index],
                         'target_reviewer_claim_sha256': source['target_reviewer_claims'][index],
                         'verified_source_claim': source['verified_source_claim'],
                         'partial_review_provenance': source['partial_review_provenance'],
                         'full_review_provenance': source['full_review_provenance'],
                         'screening': source['screening']})
            if len(rows) > config['max_rows']:
                raise ValueError('configured maximum training row count exceeded')
    if not rows:
        raise ValueError('train export has no known selected target cells')
    data_raw = b''.join(_json_bytes(row) for row in rows)
    manifest_out = {'schema_version': MANIFEST_SCHEMA, 'config_sha256': _sha(config_raw),
                    'source_export_manifest_sha256': config['export_manifest_sha256'],
                    'source_export_config_sha256': manifest['config_sha256'],
                    'taxonomy_sha256': config['taxonomy_sha256'], 'guide_sha256': guide_sha,
                    'selected_columns': columns, 'response_contract_schema': CONTRACT_SCHEMA,
                    'response_contract_code_sha256': _sha(Path(__file__).read_bytes()),
                    'instruction_template_sha256': instruction_sha,
                    'instruction_template': template, 'adapter_code_sha256': _sha(Path(__file__).read_bytes()),
                    'default_weight': float(config['default_weight']),
                    'per_label_weights': {key: float(value) for key, value in config['per_label_weights'].items()},
                    'max_output_bytes': config['max_output_bytes'],
                    'screening_policy_sha256': screening_policy_sha,
                    'source_code_digests': {key: manifest.get(key) for key in
                        ('screening_policy_sha256', 'screening_code_sha256', 'taxonomy_loader_sha256',
                         'partial_review_code_sha256', 'exporter_code_sha256',
                         'comparison_semantics_code_sha256')},
                    'masked_supervision_loader_sha256': _sha(Path(masked.__file__).read_bytes()),
                    'train_split_sha256': manifest['output_files']['train.jsonl']['sha256'],
                    'source_artifacts': manifest.get('source_artifacts'),
                    'source_copy_claim': manifest.get('source_copy_claim'),
                    'schema_version_source': manifest.get('schema_version'), 'rows': len(rows),
                    'target_support': {'positive': state['positive'], 'negative': state['negative'],
                                       'unknown': state['unknown'], 'per_label': by_label},
                    'weight_totals': weight_totals,
                    'claims_notice': 'Lineage, redaction, reviewer and verification fields preserve source claims; digests do not independently verify them.',
                    'training_only': True, 'model_training_performed': False, 'promotion_eligible': False}
    manifest_out['train_file'] = {'sha256': _sha(data_raw), 'bytes': len(data_raw)}
    manifest_out_raw = _json_bytes(manifest_out)
    if len(manifest_out_raw) > MAX_MANIFEST_BYTES or len(data_raw) + len(manifest_out_raw) > config['max_output_bytes']:
        raise ValueError('private training artifact exceeds configured output byte limit')
    output = _private_output(Path(config['output']))
    output.mkdir(parents=False, mode=0o700)
    masked._secure_private_path(output, is_directory=True)
    try:
        masked._write_private_file(output / 'train.jsonl', data_raw)
        masked._write_private_file(output / 'manifest.json', manifest_out_raw)
    except Exception:
        # Keep partial private output for inspection; never delete operator data implicitly.
        raise
    return {'manifest_sha256': _sha(manifest_out_raw), 'rows': len(rows),
            'target_support': {name: state[name] for name in ('positive', 'negative', 'unknown')},
            'weight_totals': weight_totals}


def load_training_dataset(directory: Path, expected_manifest_sha256: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read and verify a private dataset emitted by this adapter."""
    directory = directory.resolve(strict=True)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError('training dataset path must be a private directory')
    masked._verify_owner_private_access(directory, is_directory=True)
    manifest, manifest_raw = _read_json(directory / 'manifest.json', MAX_MANIFEST_BYTES, 'training manifest')
    if not isinstance(expected_manifest_sha256, str) or not SHA256_RE.fullmatch(expected_manifest_sha256) or \
            _sha(manifest_raw) != expected_manifest_sha256:
        raise ValueError('private training manifest differs from caller-pinned digest')
    if not isinstance(manifest, dict) or manifest.get('schema_version') != MANIFEST_SCHEMA:
        raise ValueError('unsupported private training manifest')
    info = manifest.get('train_file')
    if not isinstance(info, dict) or set(info) != {'sha256', 'bytes'} or type(info['bytes']) is not int or info['bytes'] < 1:
        raise ValueError('training file manifest is malformed')
    if type(manifest.get('max_output_bytes')) is not int or not 1 <= manifest['max_output_bytes'] <= MAX_DATASET_BYTES or \
            info['bytes'] + len(manifest_raw) > manifest['max_output_bytes']:
        raise ValueError('private training dataset exceeds its pinned byte bound')
    masked._verify_owner_private_access(directory / 'train.jsonl', is_directory=False)
    raw = masked._read_bounded(directory / 'train.jsonl', info['bytes'], 'private training rows')
    if len(raw) != info['bytes'] or _sha(raw) != info['sha256']:
        raise ValueError('private training rows digest does not match manifest')
    rows = masked._parse_jsonl(raw, 'private training rows')
    if len(rows) != manifest.get('rows') or not rows:
        raise ValueError('private training row count does not match manifest')
    template = manifest.get('instruction_template')
    if not isinstance(template, str) or _sha(template.encode('utf-8')) != manifest.get('instruction_template_sha256'):
        raise ValueError('private training instruction template digest mismatch')
    expected = set(map(tuple, ((c['field'], c['label']) for c in manifest.get('selected_columns', []))))
    artifacts = manifest.get('source_artifacts')
    if not isinstance(artifacts, list):
        raise ValueError('private training source manifest is malformed')
    artifact_index = {a.get('path'): a for a in artifacts if isinstance(a, dict) and isinstance(a.get('path'), str)}
    if len(artifact_index) != len(artifacts) or any(
            a.get('split') not in ('train', 'validation') or not _valid_digest(a.get('sha256')) or
            not _valid_digest(a.get('project_claim_sha256')) or
            not _valid_digest(a.get('project_group_claim_sha256')) for a in artifacts):
        raise ValueError('private training source manifest contains invalid paths')
    if 'guide_sha256' not in manifest:
        raise ValueError('private training manifest is missing guide_sha256')
    guide_sha = manifest['guide_sha256']
    if guide_sha is not None and not _valid_digest(guide_sha):
        raise ValueError('private training guide digest is malformed')
    screening_policy_sha = manifest.get('screening_policy_sha256')
    if not _valid_digest(screening_policy_sha):
        raise ValueError('private training screening policy digest is malformed')
    weights = manifest.get('per_label_weights')
    default_weight = manifest.get('default_weight')
    if not isinstance(weights, dict) or isinstance(default_weight, bool) or \
            not isinstance(default_weight, (int, float)) or not math.isfinite(default_weight) or default_weight <= 0:
        raise ValueError('private training weight policy is malformed')
    if any(not isinstance(key, str) or isinstance(value, bool) or not isinstance(value, (int, float)) or
           not math.isfinite(value) or value <= 0 for key, value in weights.items()):
        raise ValueError('private training per-label weights are malformed')
    for row in rows:
        if not isinstance(row.get('field'), str) or not isinstance(row.get('label'), str):
            raise ValueError('private training row identity is invalid')
        key = f"{row.get('field')}:{row.get('label')}"
        configured_weight = weights.get(key, default_weight)
        source_weight = row.get('source_weight')
        if row.get('schema_version') != DATASET_SCHEMA or \
                (row.get('field'), row.get('label')) not in expected or \
                row.get('decision') not in ('positive', 'negative') or \
                not isinstance(row.get('text'), str) or not row['text'].strip() or \
                row.get('text_sha256') != _sha(row['text'].encode('utf-8')) or \
                row.get('instruction') != template.format(field=row['field'], label=row['label']) or \
                isinstance(row.get('weight'), bool) or not isinstance(row.get('weight'), (int, float)) or \
                not math.isfinite(row['weight']) or row['weight'] <= 0 or \
                isinstance(source_weight, bool) or not isinstance(source_weight, (int, float)) or \
                not math.isfinite(source_weight) or source_weight <= 0 or \
                isinstance(row.get('configured_weight'), bool) or row.get('configured_weight') != configured_weight or \
                row['weight'] != source_weight * configured_weight or \
                not isinstance(row.get('source_identity'), dict) or \
                not SHA256_RE.fullmatch(str(row.get('family_sha256', ''))) or \
                not SHA256_RE.fullmatch(str(row.get('original_text_sha256', ''))) or \
                not SHA256_RE.fullmatch(str(row.get('text_sha256', ''))) or \
                not SHA256_RE.fullmatch(str(row.get('source_identity', {}).get('record_id_sha256', ''))) or \
                not SHA256_RE.fullmatch(str(row.get('source_identity', {}).get('source_claim_sha256', ''))) or \
                not isinstance(row.get('target_origin'), str) or not row['target_origin'] or \
                type(row.get('verified_source_claim')) is not bool or \
                not isinstance(row.get('screening'), dict):
            raise ValueError('private training row is invalid')
        identity = row['source_identity']
        _validate_training_target_semantics(row)
        if not isinstance(identity.get('artifact_path'), str) or Path(identity['artifact_path']).is_absolute() or \
                '..' in Path(identity['artifact_path']).parts or \
                any(not SHA256_RE.fullmatch(str(identity.get(name, ''))) for name in
                    ('artifact_sha256', 'project_claim_sha256', 'project_group_claim_sha256',
                     'record_id_sha256', 'source_claim_sha256')):
            raise ValueError('private training source identity is malformed')
        artifact = artifact_index.get(identity.get('artifact_path'))
        if artifact is None or identity.get('artifact_sha256') != artifact.get('sha256') or \
                identity.get('project_claim_sha256') != artifact.get('project_claim_sha256') or \
                identity.get('project_group_claim_sha256') != artifact.get('project_group_claim_sha256') or \
                artifact.get('split') != 'train':
            raise ValueError('private training row does not match source manifest')
        screening = row['screening']
        if screening.get('policy_sha256') != screening_policy_sha or \
                type(screening.get('text_changed')) is not bool or \
                screening['text_changed'] != (row['original_text_sha256'] != row['text_sha256']) or \
                not isinstance(screening.get('redaction_counts'), dict) or any(
                    not isinstance(k, str) or type(v) is not int or v < 0
                    for k, v in screening['redaction_counts'].items()):
            raise ValueError('private training screening provenance is malformed')
        _validate_review_claims({'target_reviewer_claim_sha256': [row.get('target_reviewer_claim_sha256')],
                                 'target_origin': [row['target_origin']],
                                 'verified_source_claim': row['verified_source_claim'],
                                 'partial_review_provenance': row.get('partial_review_provenance'),
                                 'full_review_provenance': row.get('full_review_provenance')}, 1,
                                 taxonomy_sha256=manifest.get('taxonomy_sha256'), guide_sha256=guide_sha,
                                 source_sha256=identity['artifact_sha256'], text_sha256=row['text_sha256'])
        try:
            target = json.loads(row['target_response'])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError('private training response target is invalid') from error
        validate_response(target, [{'field': row['field'], 'label': row['label']}], allow_abstention=False)
        if len(target['decisions']) != 1:
            raise ValueError('private training response target must contain exactly one decision')
        if target['decisions'][0].get('field') != row['field'] or \
                target['decisions'][0].get('label') != row['label'] or \
                target.get('taxonomy_sha256') != manifest.get('taxonomy_sha256') or \
                target.get('guide_sha256') != manifest.get('guide_sha256') or \
                target['decisions'][0]['value'] != row['decision']:
            raise ValueError('private training response target identity mismatch')
    return manifest, rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('build', choices=['build'])
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    result = build_dataset(args.config)
    print(json.dumps({'manifest_sha256': result['manifest_sha256'], 'rows': result['rows'],
                      'target_support': result['target_support'],
                      'weight_totals': result['weight_totals']}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
