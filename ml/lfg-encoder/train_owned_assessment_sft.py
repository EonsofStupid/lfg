#!/usr/bin/env python3
"""Fine-tune an owned byte student on explicitly prepared local teacher traces."""
from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import time

FRAMING_FIELDS = ('system_prefix', 'system_suffix', 'user_prefix', 'user_suffix',
                 'assistant_prefix', 'assistant_suffix')
USER_REQUEST_DATASET_SCHEMA = 'lfg.user-requested-sft-dataset.v2'
USER_REQUEST_CONFIG_SCHEMA = 'lfg.owned-user-request-sft-config.v1'
USER_REQUEST_TASK = 'user_requested_model_effort'
USER_REQUEST_OUTPUT_SCHEMA = 'lfg.user-requested-routing-output.v2'

# Keep the existing public helper available when training extras are installed;
# request-dataset validation itself remains importable without those extras.
try:
    import owned_sft_checkpoint as checkpoints
except ImportError:
    checkpoints = None


def framing_sha256(framing: dict) -> str:
    """Hash the exact delimiter strings used to render every SFT example."""
    canonical = json.dumps({key: framing[key] for key in FRAMING_FIELDS}, sort_keys=True,
                           ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(canonical).hexdigest()


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_bounded(path: Path, limit: int) -> bytes:
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError('input exceeds configured limit')
    return raw


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
            + '\n').encode('utf-8')


def validate_bounded_inert_extensions(value: object, max_bytes: int) -> None:
    nodes = 0

    def visit(item: object, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > 4096 or depth > 16:
            raise ValueError('inert extension bounds exceeded')
        if isinstance(item, str):
            if len(item.encode('utf-8')) > 4096:
                raise ValueError('inert extension string exceeds limit')
        elif isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str) or not key or len(key.encode('utf-8')) > 4096:
                    raise ValueError('invalid inert extension key')
                visit(key, depth + 1)
                visit(child, depth + 1)
        elif isinstance(item, list):
            for child in item:
                visit(child, depth + 1)
        elif item is None or type(item) in (bool, int):
            return
        else:
            raise ValueError('unsupported inert extension value')

    if not isinstance(value, dict):
        raise ValueError('inert extensions must be objects')
    visit(value, 0)
    raw = json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')
    if len(raw) > max_bytes:
        raise ValueError('inert extensions exceed configured byte limit')


def strict_json(raw: bytes | str):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result

    def bad_constant(_value):
        raise ValueError('invalid JSON numeric constant')

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad_constant)


def configuration(path: Path, max_bytes: int = 65536) -> tuple[dict, bytes]:
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError('max config bytes must be positive')
    raw = read_bounded(path, max_bytes)
    config = json.loads(raw)
    integers = ('seed', 'cpu_threads', 'batch_size', 'steps', 'context', 'max_parameters',
                'max_attention_elements', 'max_dataset_bytes', 'max_manifest_bytes',
                'max_checkpoint_bytes', 'max_optimizer_bytes', 'max_examples', 'max_record_bytes', 'checkpoint_every')
    framing_keys = ('system_prefix', 'system_suffix', 'user_prefix', 'user_suffix',
                    'assistant_prefix', 'assistant_suffix')
    weights = {'review_status_weights': ('reviewed', 'unreviewed'),
               'outcome_weights': ('success', 'unknown', 'failure')}
    config_schema = config.get('schema_version') if isinstance(config, dict) else None
    user_request_mode = config_schema == USER_REQUEST_CONFIG_SCHEMA
    if (not isinstance(config, dict)
            or config_schema not in ('lfg.owned-assessment-sft-config.v1', USER_REQUEST_CONFIG_SCHEMA)
            or (user_request_mode and config.get('task') != USER_REQUEST_TASK)
            or (user_request_mode and (not isinstance(config.get('revision'), str) or not config['revision'].strip()))
            or any(type(config.get(k)) is not int or config[k] < (0 if k == 'seed' else 1) for k in integers)
            or config.get('device') not in ('cpu', 'cuda')
            or type(config.get('cuda_device_index')) is not int or config['cuda_device_index'] < 0
            or type(config.get('cuda_memory_fraction')) not in (int, float)
            or not math.isfinite(config['cuda_memory_fraction']) or not 0 < config['cuda_memory_fraction'] <= 1
            or type(config.get('deterministic_algorithms')) is not bool
            or type(config.get('allow_tf32')) is not bool
            or type(config.get('allow_truncated_prompt')) is not bool
            or any(type(config.get(k)) not in (int, float) or not math.isfinite(config[k])
                   for k in ('learning_rate', 'weight_decay'))
            or config['learning_rate'] <= 0 or config['weight_decay'] < 0
            or ('allow_unlinked_legacy_sft_checkpoint' in config
                and type(config['allow_unlinked_legacy_sft_checkpoint']) is not bool)
            or (config.get('allow_unlinked_legacy_sft_checkpoint', False) and not user_request_mode)
            or not isinstance(config.get('framing'), dict)
            or any(not isinstance(config['framing'].get(k), str) for k in framing_keys)):
        raise ValueError('invalid supervised training configuration')
    for key, names in weights.items():
        value = config.get(key)
        if (not isinstance(value, dict) or any(type(value.get(n)) not in (int, float)
                or not math.isfinite(value[n]) or value[n] < 0 for n in names)):
            raise ValueError('invalid example weights')
    if user_request_mode and config['review_status_weights']['unreviewed'] <= 0:
        raise ValueError('unreviewed user-request examples must remain trainable')
    return config, raw


def read_training(dataset: Path, config: dict) -> tuple[list[dict], bytes, bytes, dict]:
    if dataset.is_file():
        return read_user_request_training(dataset, config)
    if config.get('schema_version') == USER_REQUEST_CONFIG_SCHEMA:
        raise ValueError('user-request SFT requires its distinct single-file dataset schema')
    manifest_raw = read_bounded(dataset / 'manifest.json', config['max_manifest_bytes'])
    manifest = json.loads(manifest_raw)
    files = manifest.get('files') if isinstance(manifest, dict) else None
    sources = manifest.get('sources') if isinstance(manifest, dict) else None
    if (not isinstance(manifest, dict) or manifest.get('schema_version') != 'lfg.assessment-distillation-dataset.v1'
            or not isinstance(files, dict) or not isinstance(files.get('train.jsonl'), str)
            or any(not isinstance(name, str) or re.match(r'(?i)^(test|held[-_]?out)([._/-]|$)',
                   name.replace('\\', '/').rsplit('/', 1)[-1]) for name in files)
            or not isinstance(sources, list) or not sources
            or any(not isinstance(item, dict) or item.get('split') not in ('train', 'validation')
                   or item.get('origin') not in ('owner_chat', 'owner_project')
                   or any(not isinstance(item.get(k), str) or not item[k].strip()
                          for k in ('project', 'project_group', 'sha256'))
                   or not re.fullmatch(r'[0-9a-f]{64}', item['sha256']) for item in sources)):
        raise ValueError('unsupported development dataset or held-out file declared')
    train_sources = {(item['sha256'], item['project'], item['project_group'], item['origin'])
                     for item in sources if item['split'] == 'train'}
    projects, groups, source_splits = {}, {}, {}
    for item in sources:
        project, group, split, source_sha = (item['project'], item['project_group'], item['split'], item['sha256'])
        if ((project in projects and projects[project] != (group, split))
                or (group in groups and groups[group] != split)
                or (source_sha in source_splits and source_splits[source_sha] != split)):
            raise ValueError('declared training and validation sources overlap')
        projects[project], groups[group], source_splits[source_sha] = (group, split), split, split
    # Only the training file is opened. A declared validation file is not used.
    train_raw = read_bounded(dataset / 'train.jsonl', config['max_dataset_bytes'])
    if sha(train_raw) != files['train.jsonl']:
        raise ValueError('training dataset digest mismatch')
    records, identifiers, counts = [], set(), Counter()
    system_sha = None
    for line in train_raw.splitlines():
        if not line.strip() or len(line) > config['max_record_bytes']:
            raise ValueError('invalid or oversized training record')
        row = json.loads(line)
        messages = row.get('messages') if isinstance(row, dict) else None
        metadata = row.get('metadata') if isinstance(row, dict) else None
        if (not isinstance(row, dict) or not isinstance(row.get('id'), str) or not row['id'].strip() or row['id'] in identifiers
                or not isinstance(messages, list) or len(messages) != 3
                or any(not isinstance(message, dict) or message.get('role') != role
                       or not isinstance(message.get('content'), str) or not message['content']
                       for message, role in zip(messages, ('system', 'user', 'assistant'), strict=True))
                or not isinstance(metadata, dict)
                or metadata.get('review_status_claim') not in ('reviewed', 'unreviewed')
                or metadata.get('outcome_claim') not in ('success', 'unknown', 'failure')
                or not isinstance(metadata.get('teacher'), str) or not metadata['teacher'].strip()
                or not isinstance(metadata.get('original_source_ref_claim'), str) or not metadata['original_source_ref_claim'].strip()
                or not isinstance(metadata.get('original_source_sha256_claim'), str)
                or not re.fullmatch(r'[0-9a-f]{64}', metadata['original_source_sha256_claim'])
                or metadata.get('task') != manifest.get('task')
                or tuple(metadata.get(k) for k in ('trace_file_sha256', 'project', 'project_group', 'origin')) not in train_sources):
            raise ValueError('invalid training trace')
        identifiers.add(row['id'])
        if len(identifiers) > config['max_examples']:
            raise ValueError('training example limit exceeded')
        status, outcome = metadata['review_status_claim'], metadata['outcome_claim']
        weight = config['review_status_weights'][status] * config['outcome_weights'][outcome]
        if not math.isfinite(weight):
            raise ValueError('combined example weight is not finite')
        counts[f'review_status:{status}'] += 1
        counts[f'outcome:{outcome}'] += 1
        if weight == 0:
            continue
        system, user, assistant = (message['content'] for message in messages)
        current_system_sha = sha(system.encode('utf-8'))
        if system_sha is not None and current_system_sha != system_sha:
            raise ValueError('training traces use different system prompts')
        system_sha = current_system_sha
        fmt = config['framing']
        prefix = (fmt['system_prefix'] + system + fmt['system_suffix'] + fmt['user_prefix'] + user
                  + fmt['user_suffix'] + fmt['assistant_prefix']).encode('utf-8')
        response = (assistant + fmt['assistant_suffix']).encode('utf-8')
        if not prefix or not response:
            raise ValueError('a training trace needs a prefix and response')
        if len(prefix) + len(response) - 1 > config['context'] and not config['allow_truncated_prompt']:
            raise ValueError('full prompt and response exceed context; choose an explicit truncation policy or larger context')
        records.append({'bytes': prefix + response, 'response_start': len(prefix), 'weight': weight})
    if not records or not math.isfinite(sum(item['weight'] for item in records)):
        raise ValueError('no positively weighted training traces')
    return records, manifest_raw, train_raw, {'selected': len(records), 'claims': dict(sorted(counts.items())),
        'system_sha256': system_sha,
        'framing_sha256': framing_sha256(config['framing']),
        'initial_prompt_exceeds_context': sum(item['response_start'] > config['context'] for item in records),
        'complete_example_exceeds_context': sum(len(item['bytes']) > config['context'] + 1 for item in records)}


def read_user_request_training(dataset: Path, config: dict) -> tuple[list[dict], bytes, bytes, dict]:
    """Read the distinct, teacherless sparse user-request SFT contract."""
    if (config.get('schema_version') != USER_REQUEST_CONFIG_SCHEMA
            or config.get('task') != USER_REQUEST_TASK or os.name != 'posix'):
        raise ValueError('user-request SFT requires its explicit task config on POSIX')
    dataset_raw = read_bounded(dataset, config['max_dataset_bytes'])
    envelope = strict_json(dataset_raw)
    if (not isinstance(envelope, dict) or set(envelope) != {'schema_version', 'manifest', 'train_records'}
            or envelope.get('schema_version') != USER_REQUEST_DATASET_SCHEMA
            or not isinstance(envelope.get('manifest'), dict)
            or not isinstance(envelope.get('train_records'), list)):
        raise ValueError('invalid user-request SFT dataset envelope')
    manifest = envelope['manifest']
    manifest_fields = {'schema_version', 'task', 'task_schema_version', 'revision', 'config_sha256', 'builder_sha256',
        'observation_config_sha256', 'system_prompt_sha256', 'framing_sha256', 'training_assignment',
        'split_policy', 'heldout_separation', 'source_split_observation', 'review_status', 'outcome',
        'provider_entitlement', 'provider_listing_is_entitlement', 'execution', 'total_agent_count',
        'promotion_eligible', 'records', 'records_sha256', 'record_id', 'source_evidence',
        'provider_cache_sha256', 'relations_sha256', 'target_record_sha256', 'screen_policy_sha256',
        'screened_request_sha256', 'redaction_counts', 'user_text_redaction_counts', 'validation_opened', 'test_opened',
        'assistant_teacher_claimed', 'user_text_utf8_bytes', 'source_representation',
        'caller_extensions', 'sft_config_extensions'}
    projection_fields = {'projection_sha256', 'byte_exact_to_screened', 'windows_origin_claims'}
    if manifest.get('source_representation') == 'verified_posix_single_message_projection':
        manifest_fields |= projection_fields
    sha_fields = ('config_sha256', 'builder_sha256', 'observation_config_sha256', 'system_prompt_sha256',
        'records_sha256', 'provider_cache_sha256', 'relations_sha256', 'target_record_sha256',
        'screen_policy_sha256', 'screened_request_sha256')
    if (set(manifest) != manifest_fields
            or manifest.get('schema_version') != USER_REQUEST_DATASET_SCHEMA
            or manifest.get('task') != USER_REQUEST_TASK
            or manifest.get('task_schema_version') != USER_REQUEST_OUTPUT_SCHEMA
            or manifest.get('revision') != config.get('revision')
            or any(not isinstance(manifest.get(key), str) or not re.fullmatch(r'[0-9a-f]{64}', manifest[key])
                   for key in sha_fields)
            or manifest.get('framing_sha256') != framing_sha256(config['framing'])
            or manifest.get('training_assignment') != 'train'
            or manifest.get('split_policy') != 'exploratory_train'
            or manifest.get('heldout_separation') != 'unknown'
            or manifest.get('source_split_observation') != {'assignment': 'unknown', 'heldout_separation': 'unknown'}
            or manifest.get('review_status') != 'unreviewed' or manifest.get('outcome') != 'unknown'
            or manifest.get('provider_entitlement') != 'unknown'
            or manifest.get('provider_listing_is_entitlement') is not False
            or manifest.get('execution') != 'unknown' or manifest.get('total_agent_count') is not None
            or manifest.get('promotion_eligible') is not False
            or manifest.get('validation_opened') is not False or manifest.get('test_opened') is not False
            or manifest.get('assistant_teacher_claimed') is not False
            or manifest.get('source_representation') not in
               ('verified_posix_source', 'verified_posix_single_message_projection')
            or type(manifest.get('records')) is not int or not 1 <= manifest['records'] <= config['max_examples']
            or not isinstance(manifest.get('record_id'), str)
            or type(manifest.get('user_text_utf8_bytes')) is not int or manifest['user_text_utf8_bytes'] < 1
            or not isinstance(manifest.get('source_evidence'), dict)
            or not isinstance(manifest.get('redaction_counts'), dict)
            or not isinstance(manifest.get('caller_extensions'), dict)
            or not isinstance(manifest.get('sft_config_extensions'), dict)):
        raise ValueError('user-request SFT manifest violates explicit unknown/task policy')
    validate_bounded_inert_extensions(manifest['caller_extensions'], config['max_dataset_bytes'])
    validate_bounded_inert_extensions(manifest['sft_config_extensions'], config['max_dataset_bytes'])
    if manifest['source_representation'] == 'verified_posix_single_message_projection':
        claims = manifest.get('windows_origin_claims')
        claim_fields = {'schema_version', 'origin_platform', 'origin_verification',
                        'line_sha256_claim', 'prefix_sha256_claim', 'user_text_sha256_claim'}
        if (manifest.get('byte_exact_to_screened') is not True
                or not isinstance(manifest.get('projection_sha256'), str)
                or not re.fullmatch(r'[0-9a-f]{64}', manifest['projection_sha256'])
                or not isinstance(claims, dict) or not claim_fields <= set(claims)
                or claims.get('schema_version') != 'lfg.windows-origin-projection-claims.v1'
                or claims.get('origin_platform') != 'windows'
                or claims.get('origin_verification') != 'unverified_claim'
                or any(not isinstance(claims.get(key), str) or not re.fullmatch(r'[0-9a-f]{64}', claims[key])
                       for key in ('line_sha256_claim', 'prefix_sha256_claim', 'user_text_sha256_claim'))
                or claims.get('user_text_sha256_claim') != manifest.get('projection_sha256')):
            raise ValueError('invalid screened POSIX projection provenance')
        optional_origin = {'origin_locator_sha256_claim', 'origin_line_number_claim', 'origin_prefix_bytes_claim'}
        optional_present = optional_origin & set(claims)
        if (set(claims) not in (claim_fields, claim_fields | optional_origin)
                or (optional_present and optional_present != optional_origin)
                or (optional_present and (not re.fullmatch(r'[0-9a-f]{64}', claims['origin_locator_sha256_claim'])
                    or type(claims['origin_line_number_claim']) is not int
                    or not 1 <= claims['origin_line_number_claim'] <= 10_000_000
                    or type(claims['origin_prefix_bytes_claim']) is not int
                    or not 1 <= claims['origin_prefix_bytes_claim'] <= 512 * 1024 * 1024))):
            raise ValueError('invalid optional Windows-origin locator claims')
    elif manifest.get('source_representation') != 'verified_posix_source':
        raise ValueError('unknown user-request source representation')
    source_fields = {'source_file_sha256_at_inventory', 'source_prefix_sha256', 'source_line_sha256',
                     'user_text_sha256', 'source_group_sha256'}
    if (set(manifest['source_evidence']) != source_fields
            or any(not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value)
                   for value in manifest['source_evidence'].values())):
        raise ValueError('invalid user-request source evidence')
    rows = envelope['train_records']
    if (len(rows) != manifest['records']
            or sha(b''.join(canonical(row) for row in rows)) != manifest['records_sha256']):
        raise ValueError('user-request SFT record digest mismatch')
    result, identifiers, counts = [], set(), Counter()
    for row in rows:
        seen_spans, seen_intervals = set(), []

        def register_span(span_id: str, start: int, end: int) -> None:
            if span_id in seen_spans or any(start < prior_end and prior_start < end
                                            for prior_start, prior_end in seen_intervals):
                raise ValueError('duplicate or overlapping request span interval')
            seen_spans.add(span_id)
            seen_intervals.append((start, end))

        messages = row.get('messages') if isinstance(row, dict) else None
        metadata = row.get('metadata') if isinstance(row, dict) else None
        if (not isinstance(row, dict) or set(row) != {'id', 'messages', 'metadata'}
                or not isinstance(row.get('id'), str) or not re.fullmatch(r'[0-9a-f]{64}', row['id'])
                or row['id'] in identifiers or not isinstance(messages, list) or len(messages) != 3
                or any(not isinstance(message, dict) or set(message) != {'role', 'content'}
                       or message.get('role') != role or not isinstance(message.get('content'), str)
                       or not message['content'] for message, role in zip(messages, ('system', 'user', 'assistant'), strict=True))
                or not isinstance(metadata, dict)):
            raise ValueError('invalid user-request SFT record')
        identifiers.add(row['id'])
        metadata_fields = {'task', 'task_schema_version', 'review_status', 'outcome',
            'provider_entitlement', 'execution', 'total_agent_count', 'promotion_eligible',
            'training_assignment', 'split_policy', 'heldout_separation', *source_fields,
            'provider_cache_sha256', 'relations_sha256', 'target_record_sha256', 'screen_policy_sha256',
            'screened_request_sha256', 'target_span_refs', 'unknown_span_refs', 'redaction_counts', 'user_text_redaction_counts',
            'user_text_utf8_bytes', 'caller_extensions'}
        metadata_fields.add('source_representation')
        if manifest['source_representation'] == 'verified_posix_single_message_projection':
            metadata_fields |= projection_fields
        digest_fields = (*source_fields, 'provider_cache_sha256', 'relations_sha256',
                         'target_record_sha256', 'screen_policy_sha256', 'screened_request_sha256')
        if (set(metadata) != metadata_fields or metadata.get('task') != USER_REQUEST_TASK
                or metadata.get('task_schema_version') != USER_REQUEST_OUTPUT_SCHEMA
                or metadata.get('review_status') != 'unreviewed' or metadata.get('outcome') != 'unknown'
                or metadata.get('provider_entitlement') != 'unknown' or metadata.get('execution') != 'unknown'
                or metadata.get('total_agent_count') is not None or metadata.get('promotion_eligible') is not False
                or metadata.get('training_assignment') != 'train' or metadata.get('split_policy') != 'exploratory_train'
                or metadata.get('heldout_separation') != 'unknown'
                or metadata.get('source_representation') != manifest['source_representation']
                or type(metadata.get('user_text_utf8_bytes')) is not int
                or metadata['user_text_utf8_bytes'] != manifest['user_text_utf8_bytes']
                or any(not isinstance(metadata.get(key), str) or not re.fullmatch(r'[0-9a-f]{64}', metadata[key])
                       for key in digest_fields)
                or not isinstance(metadata.get('target_span_refs'), list)
                or not isinstance(metadata.get('unknown_span_refs'), list)
                or not isinstance(metadata.get('caller_extensions'), dict)
                or metadata.get('caller_extensions') != manifest['caller_extensions']
                or metadata.get('redaction_counts') != manifest['redaction_counts']
                or metadata.get('user_text_redaction_counts') != manifest['redaction_counts']
                or any(metadata[key] != manifest['source_evidence'][key] for key in source_fields)
                or metadata['provider_cache_sha256'] != manifest['provider_cache_sha256']
                or metadata['relations_sha256'] != manifest['relations_sha256']
                or metadata['target_record_sha256'] != manifest['target_record_sha256']
                or metadata['screen_policy_sha256'] != manifest['screen_policy_sha256']
                or metadata['screened_request_sha256'] != manifest['screened_request_sha256']):
            raise ValueError('user-request SFT record contains unsupported claims')
        validate_bounded_inert_extensions(metadata['caller_extensions'], config['max_record_bytes'])
        if manifest['source_representation'] == 'verified_posix_single_message_projection':
            if (metadata.get('projection_sha256') != manifest['projection_sha256']
                    or metadata.get('byte_exact_to_screened') is not True
                    or metadata.get('windows_origin_claims') != manifest['windows_origin_claims']
                    or metadata.get('projection_sha256') != metadata.get('user_text_sha256')
                    or metadata['windows_origin_claims']['user_text_sha256_claim']
                       != metadata.get('projection_sha256')):
                raise ValueError('record projection evidence differs from the manifest')
        system, user, assistant = (message['content'] for message in messages)
        if (sha(system.encode('utf-8')) != manifest['system_prompt_sha256']
                or sha(user.encode('utf-8')) != metadata['screened_request_sha256']):
            raise ValueError('user-request SFT prompt digest mismatch')
        user_input = strict_json(user)
        if (not isinstance(user_input, dict) or set(user_input) != {'user_request', 'provider_candidates'}
                or not isinstance(user_input.get('user_request'), str) or not user_input['user_request'].strip()
                or not isinstance(user_input.get('provider_candidates'), list)
                ):
            raise ValueError('user-request SFT input must carry the screened request and current candidate projection')
        request_bytes = user_input['user_request'].encode('utf-8')
        if (len(request_bytes) != metadata['user_text_utf8_bytes']
                or sha(request_bytes) != metadata['user_text_sha256']
                or (manifest['source_representation'] == 'verified_posix_single_message_projection'
                    and sha(request_bytes) != manifest['projection_sha256'])):
            raise ValueError('request text does not match byte-exact source evidence')
        candidate_by_id = {}
        for candidate in user_input['provider_candidates']:
            if (not isinstance(candidate, dict) or set(candidate) != {'model_id', 'display_name', 'supported_efforts'}
                    or not isinstance(candidate.get('model_id'), str) or not candidate['model_id'].strip()
                    or candidate['model_id'] in candidate_by_id
                    or not isinstance(candidate.get('display_name'), str) or not candidate['display_name'].strip()
                    or not isinstance(candidate.get('supported_efforts'), list)
                    or any(not isinstance(item, str) or not item.strip() for item in candidate['supported_efforts'])):
                raise ValueError('invalid or duplicate provider candidate projection')
            candidate_by_id[candidate['model_id']] = candidate
        try:
            target = strict_json(assistant)
        except (json.JSONDecodeError, ValueError):
            raise ValueError('invalid sparse user-request target JSON') from None
        if (not isinstance(target, dict) or set(target) != {'schema_version', 'task', 'requested', 'unknown'}
                or target.get('schema_version') != USER_REQUEST_OUTPUT_SCHEMA
                or target.get('task') != USER_REQUEST_TASK or not isinstance(target.get('requested'), list)
                or not isinstance(target.get('unknown'), list)
                or (not target['requested'] and not target['unknown'])
                or len(target['requested']) != len(metadata['target_span_refs'])
                or len(target['unknown']) != len(metadata['unknown_span_refs'])):
            raise ValueError('invalid sparse user-request target output')
        refs = metadata['target_span_refs']
        for requested, ref in zip(target['requested'], refs, strict=True):
            if (not isinstance(requested, dict)
                    or set(requested) not in ({'model_id', 'polarity'},
                                              {'model_id', 'polarity', 'effort', 'effort_polarity'})
                    or not isinstance(requested.get('model_id'), str) or not requested['model_id'].strip()
                    or requested['model_id'] not in candidate_by_id
                    or requested.get('polarity') not in ('affirmed', 'negated')
                    or ('effort' in requested and (not isinstance(requested['effort'], str)
                        or not requested['effort'].strip()
                        or requested.get('effort_polarity') not in ('affirmed', 'negated')))
                    or (requested.get('polarity') == 'negated' and 'effort' in requested)
                    or not isinstance(ref, dict)
                    or set(ref) != {'model_span_id', 'model_span_sha256', 'model_span_start_byte',
                                    'model_span_end_byte', 'model_polarity', 'effort_span_id',
                                    'effort_span_sha256', 'effort_span_start_byte', 'effort_span_end_byte',
                                    'effort_polarity'}
                    or ref.get('model_polarity') != requested['polarity']
                    or not isinstance(ref.get('model_span_id'), str) or not ref['model_span_id']
                    or not isinstance(ref.get('model_span_sha256'), str)
                    or not re.fullmatch(r'[0-9a-f]{64}', ref['model_span_sha256'])
                    or type(ref.get('model_span_start_byte')) is not int or ref['model_span_start_byte'] < 0
                    or type(ref.get('model_span_end_byte')) is not int
                    or ref['model_span_end_byte'] <= ref['model_span_start_byte']
                    or ref['model_span_end_byte'] > metadata['user_text_utf8_bytes']):
                raise ValueError('sparse user-request target is not bound to a unique explicit source span')
            register_span(ref['model_span_id'], ref['model_span_start_byte'], ref['model_span_end_byte'])
            model_slice = request_bytes[ref['model_span_start_byte']:ref['model_span_end_byte']]
            try:
                model_slice.decode('utf-8')
            except UnicodeDecodeError:
                raise ValueError('model span splits a UTF-8 request character') from None
            if sha(model_slice) != ref['model_span_sha256']:
                raise ValueError('model target span digest does not match actual request bytes')
            if ref['effort_span_id'] is not None and (not isinstance(ref['effort_span_id'], str)
                    or not ref['effort_span_id'] or not isinstance(ref['effort_span_sha256'], str)
                    or not re.fullmatch(r'[0-9a-f]{64}', ref['effort_span_sha256'])
                    or type(ref['effort_span_start_byte']) is not int or ref['effort_span_start_byte'] < 0
                    or type(ref['effort_span_end_byte']) is not int
                    or ref['effort_span_end_byte'] <= ref['effort_span_start_byte']
                    or ref['effort_span_end_byte'] > metadata['user_text_utf8_bytes']
                    or ref['effort_polarity'] not in ('affirmed', 'negated', 'quoted', 'uncertain')):
                raise ValueError('invalid effort span reference')
            if ref['effort_span_id'] is not None:
                register_span(ref['effort_span_id'], ref['effort_span_start_byte'], ref['effort_span_end_byte'])
                effort_slice = request_bytes[ref['effort_span_start_byte']:ref['effort_span_end_byte']]
                try:
                    effort_slice.decode('utf-8')
                except UnicodeDecodeError:
                    raise ValueError('effort span splits a UTF-8 request character') from None
                if sha(effort_slice) != ref['effort_span_sha256']:
                    raise ValueError('effort target span digest does not match actual request bytes')
            if (ref['effort_span_id'] is None
                    and any(ref[key] is not None for key in ('effort_span_sha256', 'effort_polarity',
                        'effort_span_start_byte', 'effort_span_end_byte'))):
                raise ValueError('unsupplied effort cannot carry source evidence')
            if 'effort' in requested and requested['effort_polarity'] != ref['effort_polarity']:
                raise ValueError('effort output is not bound to its explicit source span')
            if ('effort' in requested
                    and requested['effort'] not in candidate_by_id[requested['model_id']]['supported_efforts']):
                raise ValueError('effort output is absent from the pinned provider candidate projection')
        allowed_unknown_statuses = {'unmatched', 'ambiguous', 'requires_redaction', 'quoted', 'uncertain',
                                    'model_unresolved', 'model_not_affirmed', 'model_polarity_unknown'}
        for item, ref in zip(target['unknown'], metadata['unknown_span_refs'], strict=True):
            if (not isinstance(item, dict) or set(item) != {'field', 'mention', 'status'}
                    or item.get('field') not in ('model', 'effort')
                    or item.get('status') not in allowed_unknown_statuses
                    or not isinstance(item.get('mention'), str)
                    or not isinstance(ref, dict) or set(ref) != {'field', 'span_id', 'sha256', 'start_byte', 'end_byte', 'status'}
                    or ref.get('field') != item['field'] or ref.get('status') != item['status']
                    or not isinstance(ref.get('span_id'), str) or not ref['span_id']
                    or not isinstance(ref.get('sha256'), str) or not re.fullmatch(r'[0-9a-f]{64}', ref['sha256'])
                    or type(ref.get('start_byte')) is not int or type(ref.get('end_byte')) is not int
                    or ref['start_byte'] < 0 or ref['end_byte'] <= ref['start_byte']
                    or ref['end_byte'] > len(request_bytes)):
                raise ValueError('unknown request output is not bound to a unique source span')
            register_span(ref['span_id'], ref['start_byte'], ref['end_byte'])
            unknown_slice = request_bytes[ref['start_byte']:ref['end_byte']]
            try:
                mention = unknown_slice.decode('utf-8')
            except UnicodeDecodeError:
                raise ValueError('unknown span splits a UTF-8 request character') from None
            if sha(unknown_slice) != ref['sha256'] or mention != item['mention']:
                raise ValueError('unknown request span does not match actual request bytes')
        fmt = config['framing']
        prefix = (fmt['system_prefix'] + system + fmt['system_suffix'] + fmt['user_prefix'] + user
                  + fmt['user_suffix'] + fmt['assistant_prefix']).encode('utf-8')
        response = (assistant + fmt['assistant_suffix']).encode('utf-8')
        if len(prefix) + len(response) - 1 > config['context'] and not config['allow_truncated_prompt']:
            raise ValueError('full user-request prompt and sparse target exceed context')
        result.append({'bytes': prefix + response, 'response_start': len(prefix),
                       'weight': config['review_status_weights']['unreviewed']})
        counts['explicit_user_request_records'] += 1
        counts['unreviewed'] += 1
        counts['outcome_unknown'] += 1
    if not result or manifest['record_id'] not in identifiers:
        raise ValueError('user-request dataset contains no selected training records')
    train_raw = b''.join(canonical(row) for row in rows)
    return result, dataset_raw, train_raw, {
        'training_task': USER_REQUEST_TASK, 'selected': len(result), 'claims': dict(sorted(counts.items())),
        'system_sha256': manifest['system_prompt_sha256'], 'framing_sha256': framing_sha256(config['framing']),
        'initial_prompt_exceeds_context': sum(item['response_start'] > config['context'] for item in result),
        'complete_example_exceeds_context': sum(len(item['bytes']) > config['context'] + 1 for item in result),
    }


def validate_completed_sft_checkpoint(manifest: object, weights: bytes, current_code: dict, *,
                                       allow_unlinked_legacy: bool = False) -> dict:
    """Validate an SFT checkpoint; optionally preserve missing legacy linkage as unknown."""
    required_digests = {'weights_sha256', 'source_checkpoint_manifest_sha256',
        'source_checkpoint_weights_sha256', 'dataset_manifest_sha256', 'training_file_sha256',
        'config_sha256'}
    if (not isinstance(manifest, dict) or manifest.get('completed') is not True
            or manifest.get('schema_version') not in ('lfg.owned-assessment-sft-result.v1',
                                                       'lfg.owned-user-request-sft-result.v1')
            or any(not isinstance(manifest.get(name), str)
                   or not re.fullmatch(r'[0-9a-f]{64}', manifest[name]) for name in required_digests)
            or manifest['weights_sha256'] != sha(weights)
            or not isinstance(manifest.get('architecture'), dict)
            or not isinstance(manifest.get('coverage'), dict)
            or not isinstance(manifest.get('code_sha256'), dict)):
        raise ValueError('completed SFT checkpoint provenance is incomplete or inconsistent')
    links = {}
    for name in ('system_sha256', 'framing_sha256'):
        top_present = name in manifest
        coverage_present = name in manifest['coverage']
        if not top_present:
            if not allow_unlinked_legacy or coverage_present:
                raise ValueError('completed SFT checkpoint provenance is incomplete or inconsistent')
            links[name] = 'unknown_unlinked'
            continue
        value = manifest[name]
        if (not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value)
                or not coverage_present or manifest['coverage'][name] != value):
            raise ValueError('completed SFT checkpoint provenance is incomplete or inconsistent')
        links[name] = 'verified_linked'
    if len(set(links.values())) != 1:
        raise ValueError('completed SFT checkpoint provenance is incomplete or inconsistent')
    expected_code_names = {'train_owned_assessment_sft.py', 'train_owned_student.py',
                           'owned_student.py', 'owned_sft_checkpoint.py'}
    code = manifest['code_sha256']
    if (not expected_code_names <= set(code)
            or any(not isinstance(code.get(name), str) or not re.fullmatch(r'[0-9a-f]{64}', code[name])
                   for name in code)
            or code.get('owned_student.py') != current_code.get('owned_student.py')):
        raise ValueError('completed SFT checkpoint code provenance is invalid')
    for name in ('steps', 'parameters', 'supervised_tokens'):
        if type(manifest.get(name)) is not int or manifest[name] < 1:
            raise ValueError('completed SFT checkpoint training provenance is invalid')
    return links


def _cli_is_user_request_mode(args: argparse.Namespace) -> bool:
    try:
        value = strict_json(read_bounded(args.config, args.max_config_bytes))
        if isinstance(value, dict) and value.get('schema_version') == USER_REQUEST_CONFIG_SCHEMA:
            return True
    except (OSError, ValueError, TypeError, UnicodeError):
        pass
    try:
        value = strict_json(read_bounded(args.dataset, 1024 * 1024))
        return isinstance(value, dict) and value.get('schema_version') == USER_REQUEST_DATASET_SCHEMA
    except (OSError, ValueError, TypeError, UnicodeError):
        return False


def sample_batch(records: list[dict], rng: random.Random, batch_size: int, context: int):
    inputs, labels, corrections = [], [], []
    for item in rng.choices(records, weights=[entry['weight'] for entry in records], k=batch_size):
        end = rng.randrange(item['response_start'], len(item['bytes']))
        start = max(0, end - context)
        segment = item['bytes'][start:end + 1]
        inputs.append(list(segment[:-1]))
        labels.append([byte if start + index + 1 >= item['response_start'] else -100
                       for index, byte in enumerate(segment[1:])])
        # A target near the response end appears in fewer uniformly sampled
        # windows. Its inverse inclusion count gives an unbiased per-example
        # mean loss when summed over the sampled window.
        corrections.append([1.0 / min(context, len(item['bytes']) - (start + index + 1))
                            if start + index + 1 >= item['response_start'] else 0.0
                            for index in range(len(segment) - 1)])
    width = max(map(len, inputs))
    for source, target, correction in zip(inputs, labels, corrections, strict=True):
        padding = width - len(source)
        source.extend([0] * padding)
        target.extend([-100] * padding)
        correction.extend([0.0] * padding)
    return inputs, labels, corrections


def experiment(dataset: Path, checkpoint: Path, config_path: Path, output: Path, *,
               resume: Path | None = None, resume_state_sha256: str | None = None,
               max_steps_this_run: int | None = None, max_config_bytes: int = 65536) -> dict:
    config, config_raw = configuration(config_path, max_config_bytes)
    if (resume is None) != (resume_state_sha256 is None):
        raise ValueError('resume and resume state digest must be supplied together')
    if max_steps_this_run is not None and (type(max_steps_this_run) is not int or max_steps_this_run < 1):
        raise ValueError('max_steps_this_run must be positive')
    user_request_mode = config['schema_version'] == USER_REQUEST_CONFIG_SCHEMA
    if user_request_mode and os.name != 'posix':
        raise ValueError('user-request SFT training output is disabled on Windows until private path safety is verified')
    try:
        import safetensors
        from safetensors.torch import load, save
        import torch
        from torch.nn import functional as F
        import owned_student
        import owned_sft_checkpoint as checkpoints
        import train_owned_student as base
    except ImportError as error:
        raise RuntimeError('owned SFT training requires torch and safetensors') from error
    records, dataset_manifest, train_raw, coverage = read_training(dataset, config)
    checkpoint_schema = ('lfg.owned-user-request-sft-checkpoint.v1' if user_request_mode
                         else 'lfg.owned-assessment-sft-checkpoint.v1')
    checkpoint_manifest = base.read_bounded(checkpoint / 'manifest.json', config['max_manifest_bytes'])
    initial = json.loads(checkpoint_manifest)
    initial_weights = base.read_bounded(checkpoint / 'weights.safetensors', config['max_checkpoint_bytes'])
    code_paths = (Path(__file__), Path(base.__file__), Path(owned_student.__file__), Path(checkpoints.__file__))
    code = {path.name: base.sha(path.read_bytes()) for path in code_paths}
    allowed_checkpoint_schemas = ('lfg.owned-student-experiment-result.v1', 'lfg.owned-distillation-result.v1',
        *(('lfg.owned-user-request-sft-result.v1', 'lfg.owned-assessment-sft-result.v1')
          if user_request_mode else ('lfg.owned-assessment-sft-result.v1',)))
    if (not isinstance(initial, dict) or initial.get('schema_version') not in allowed_checkpoint_schemas
            or initial.get('weights_sha256') != base.sha(initial_weights)
            or not isinstance(initial.get('code_sha256'), dict)
            or initial['code_sha256'].get('owned_student.py') != code['owned_student.py']):
        raise ValueError('owned checkpoint identity does not match')
    source_checkpoint_links = {'system_sha256': 'not_applicable', 'framing_sha256': 'not_applicable'}
    if initial['schema_version'] in ('lfg.owned-user-request-sft-result.v1',
                                     'lfg.owned-assessment-sft-result.v1'):
        allow_unlinked = (user_request_mode and config.get('allow_unlinked_legacy_sft_checkpoint', False)
                          and initial['schema_version'] == 'lfg.owned-assessment-sft-result.v1')
        source_checkpoint_links = validate_completed_sft_checkpoint(
            initial, initial_weights, code, allow_unlinked_legacy=allow_unlinked)
    architecture = dict(initial['architecture'])
    owned_student.validate_architecture(architecture)
    old_context = architecture['context']
    if config['context'] < old_context:
        raise ValueError('context cannot shrink the source checkpoint')
    architecture['context'] = config['context']
    with torch.device('meta'):
        template = owned_student.OwnedByteStudent(architecture)
    parameters = sum(item.numel() for item in template.parameters())
    del template
    attention_layers = architecture['layers'] // architecture['attention_every']
    estimate = config['batch_size'] * architecture['heads'] * config['context'] ** 2 * attention_layers
    if parameters > config['max_parameters'] or estimate > config['max_attention_elements']:
        raise ValueError('configured model or attention estimate exceeds limits')
    if output.exists():
        raise ValueError('output already exists')
    checkpoints.private_directory(output)
    if config['device'] == 'cuda' and not torch.cuda.is_available():
        raise ValueError('requested CUDA device is unavailable')
    if config['device'] == 'cuda' and config['cuda_device_index'] >= torch.cuda.device_count():
        raise ValueError('requested CUDA device index is unavailable')
    torch.set_num_threads(config['cpu_threads'])
    torch.use_deterministic_algorithms(config['deterministic_algorithms'])
    torch.backends.cuda.matmul.allow_tf32 = config['allow_tf32']
    torch.backends.cudnn.allow_tf32 = config['allow_tf32']
    torch.manual_seed(config['seed'])
    device = torch.device('cpu')
    if config['device'] == 'cuda':
        device = torch.device('cuda', config['cuda_device_index'])
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(config['cuda_memory_fraction'], device=device)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.manual_seed(config['seed'])
    model = owned_student.OwnedByteStudent(architecture)
    state = load(initial_weights)
    if config['context'] > old_context:
        positions = model.position.weight.detach().clone()
        if state['position.weight'].shape != (old_context, architecture['width']):
            raise ValueError('source position embedding shape mismatch')
        positions[:old_context] = state['position.weight']
        state['position.weight'] = positions
    model.load_state_dict(state, strict=True)
    if any(not bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
        raise ValueError('source checkpoint has non-finite weights')
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    rng, losses, supervised = random.Random(config['seed']), [], 0
    identity = {'dataset_manifest_sha256': base.sha(dataset_manifest), 'training_file_sha256': base.sha(train_raw),
                'source_checkpoint_manifest_sha256': base.sha(checkpoint_manifest),
                'source_checkpoint_weights_sha256': base.sha(initial_weights),
                'source_checkpoint_schema': initial['schema_version'],
                'source_checkpoint_system_linkage': source_checkpoint_links['system_sha256'],
                'source_checkpoint_framing_linkage': source_checkpoint_links['framing_sha256'],
                'code_sha256': code,
                'continuation_config_sha256': checkpoints.continuation_config(config), 'architecture': architecture,
                'python': platform.python_version(), 'torch': str(torch.__version__), 'safetensors': safetensors.__version__,
                'cuda_version': torch.version.cuda if config['device'] == 'cuda' else None,
                'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG') if config['device'] == 'cuda' else None,
                'cuda_devices': ([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                                 if config['device'] == 'cuda' else [])}
    initial_step = 0
    if resume is not None:
        progress = checkpoints.restore(resume, resume_state_sha256, model, optimizer, rng, identity, config,
                                       schema=checkpoint_schema)
        initial_step, losses, supervised = progress['step'], progress['losses'], progress['supervised_tokens']
        if initial_step > config['steps']:
            raise ValueError('checkpoint already reached the requested total steps')
    end_step = min(config['steps'], initial_step + max_steps_this_run) if max_steps_this_run else config['steps']
    output.mkdir(mode=0o700)
    start_time = time.perf_counter()
    latest = None
    for step in range(initial_step + 1, end_step + 1):
        input_rows, target_rows, correction_rows = sample_batch(records, rng, config['batch_size'], config['context'])
        inputs = torch.tensor(input_rows, dtype=torch.long, device=device)
        targets = torch.tensor(target_rows, dtype=torch.long, device=device)
        correction = torch.tensor(correction_rows, dtype=torch.float32, device=device)
        supervised += int((targets != -100).sum().item())
        optimizer.zero_grad(set_to_none=True)
        token_loss = F.cross_entropy(model(inputs).reshape(-1, 256), targets.reshape(-1),
                                     ignore_index=-100, reduction='none')
        loss = (token_loss * correction.reshape(-1)).sum() / config['batch_size']
        if not bool(torch.isfinite(loss)):
            raise ValueError('non-finite supervised loss')
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if step % config['checkpoint_every'] == 0 or step == end_step:
            latest = checkpoints.snapshot(output, model, optimizer, rng,
                    {'step': step, 'losses': losses, 'supervised_tokens': supervised}, identity, config,
                    schema=checkpoint_schema)
    elapsed = time.perf_counter() - start_time
    if dataset.is_file():
        dataset_stable = base.read_bounded(dataset, config['max_dataset_bytes']) == dataset_manifest
    else:
        dataset_stable = (base.read_bounded(dataset / 'manifest.json', config['max_manifest_bytes']) == dataset_manifest
                          and base.read_bounded(dataset / 'train.jsonl', config['max_dataset_bytes']) == train_raw)
    if (configuration(config_path, max_config_bytes)[1] != config_raw
            or not dataset_stable
            or base.read_bounded(checkpoint / 'manifest.json', config['max_manifest_bytes']) != checkpoint_manifest
            or base.read_bounded(checkpoint / 'weights.safetensors', config['max_checkpoint_bytes']) != initial_weights
            or any(base.sha(path.read_bytes()) != code[path.name] for path in code_paths)):
        raise ValueError('training inputs changed during execution')
    if any(not bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
        raise ValueError('non-finite trained weights')
    if end_step < config['steps']:
        partial_schema = ('lfg.owned-user-request-sft-partial.v1' if user_request_mode
                          else 'lfg.owned-assessment-sft-partial.v1')
        partial = {'schema_version': partial_schema, 'completed': False,
                'steps': end_step, 'parameters': parameters, 'supervised_tokens': supervised,
                'source_checkpoint_schema': initial['schema_version'],
                'source_checkpoint_system_linkage': source_checkpoint_links['system_sha256'],
                'source_checkpoint_framing_linkage': source_checkpoint_links['framing_sha256'],
                'coverage': coverage, 'latest_checkpoint': str(latest),
                'latest_checkpoint_state_sha256': base.sha(base.read_bounded(latest / 'state.json', config['max_manifest_bytes'])),
                'promotion_eligible': False}
        if user_request_mode:
            partial.update({'quality_evaluated': False, 'runtime_compatible': False})
        else:
            partial['assessment_quality_evaluated'] = False
        return partial
    trained = save({name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()})
    result_schema = ('lfg.owned-user-request-sft-result.v1' if user_request_mode
                     else 'lfg.owned-assessment-sft-result.v1')
    manifest = {'schema_version': result_schema, 'architecture': architecture,
                'completed': True, 'resumed_from_step': initial_step,
                'source_checkpoint_schema': initial['schema_version'],
                'source_checkpoint_system_linkage': source_checkpoint_links['system_sha256'],
                'source_checkpoint_framing_linkage': source_checkpoint_links['framing_sha256'],
                'resume_checkpoint_state_sha256': resume_state_sha256,
                'source_checkpoint_manifest_sha256': base.sha(checkpoint_manifest),
                'source_checkpoint_weights_sha256': base.sha(initial_weights),
                'dataset_manifest_sha256': base.sha(dataset_manifest), 'training_file_sha256': base.sha(train_raw),
                'system_sha256': coverage['system_sha256'], 'framing_sha256': coverage['framing_sha256'],
                'config_sha256': base.sha(config_raw), 'code_sha256': code, 'coverage': coverage,
                'position_rows_reused': old_context, 'position_rows_initialized': config['context'] - old_context,
                'context_clipping_possible': coverage['complete_example_exceeds_context'] > 0,
                'steps': config['steps'], 'batch_size': config['batch_size'], 'supervised_tokens': supervised,
                'losses': losses, 'elapsed_seconds': elapsed, 'parameters': parameters,
                'estimated_attention_elements': estimate, 'weights_sha256': base.sha(trained),
                'device': config['device'], 'python': platform.python_version(), 'torch': str(torch.__version__),
                'safetensors': safetensors.__version__, 'platform': platform.platform(),
                'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                'allow_tf32': config['allow_tf32'],
                'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                'teacher_claims_verified': False, 'validation_opened': False, 'test_opened': False,
                'assessment_quality_evaluated': False, 'promotion_eligible': False}
    if user_request_mode:
        manifest.pop('assessment_quality_evaluated')
        manifest.update({'training_task': USER_REQUEST_TASK, 'assistant_teacher_claimed': False,
                         'review_status': 'unreviewed', 'outcome_claim': 'unknown',
                         'provider_entitlement': 'unknown', 'execution': 'unknown',
                         'total_agent_count': None, 'heldout_separation': 'unknown',
                         'runtime_compatible': False, 'deployment_ready': False,
                         'user_request_quality_evaluated': False})
    if config['device'] == 'cuda':
        manifest['cuda_device'] = torch.cuda.get_device_name(device)
        manifest['cuda_device_index'] = config['cuda_device_index']
        manifest['cuda_memory_fraction'] = config['cuda_memory_fraction']
        manifest['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
        manifest['cuda_peak_reserved_bytes'] = torch.cuda.max_memory_reserved(device)
        manifest['cuda_version'] = torch.version.cuda
    for name, raw in (('weights.safetensors', trained),
                      ('manifest.json', (json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + '\n').encode())):
        descriptor = os.open(output / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(raw)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'checkpoint', 'config', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--resume', type=Path, help='completed private checkpoint step directory')
    parser.add_argument('--resume-state-sha256', help='expected checkpoint state digest from latest.json')
    parser.add_argument('--max-steps-this-run', type=int, help='optional bound on steps in this invocation')
    parser.add_argument('--max-config-bytes', type=int, default=65536, help='maximum configuration file size')
    args = parser.parse_args()
    try:
        result = experiment(args.dataset, args.checkpoint, args.config, args.output,
                            resume=args.resume, resume_state_sha256=args.resume_state_sha256,
                            max_steps_this_run=args.max_steps_this_run, max_config_bytes=args.max_config_bytes)
        summary_fields = ('completed', 'latest_checkpoint', 'latest_checkpoint_state_sha256', 'weights_sha256',
                          'parameters', 'steps', 'supervised_tokens', 'coverage',
                          'assessment_quality_evaluated', 'promotion_eligible')
        if result.get('schema_version', '').startswith('lfg.owned-user-request-sft-'):
            summary_fields += ('schema_version', 'training_task', 'review_status', 'outcome_claim',
                               'heldout_separation', 'runtime_compatible', 'user_request_quality_evaluated')
        print(json.dumps({key: result.get(key) for key in summary_fields}))
    except (OSError, ValueError, TypeError, KeyError, UnicodeError, OverflowError, RuntimeError) as error:
        detail = str(error)
        if isinstance(error, OSError) and _cli_is_user_request_mode(args):
            detail = f'filesystem operation failed (errno={error.errno})'
        raise SystemExit(f'Owned assessment SFT failed ({type(error).__name__}): {detail}') from None
