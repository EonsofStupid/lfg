"""Build a private teacherless SFT record for explicit user routing requests.

The adapter revalidates caller-linked spans and emits only uniquely resolved,
explicit model requests. It never creates an assessment, teacher claim, or
outcome label.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import screen_private_corpus
import source_bound_snapshot
import user_requested_observation
import user_request_targets


CONFIG_SCHEMA = 'lfg.user-requested-sft-config.v1'
DATASET_SCHEMA = 'lfg.user-requested-sft-dataset.v1'
CALLER_INPUT_SCHEMA = 'lfg.user-requested-sft-caller-input.v1'
TASK = 'user_requested_model_effort'
OUTPUT_SCHEMA = 'lfg.user-requested-routing-output.v1'
CONFIG_FIELDS = {'schema_version', 'revision', 'system_prompt', 'max_config_bytes',
                 'max_dataset_bytes', 'max_record_bytes', 'max_request_bytes',
                 'max_target_bytes', 'max_examples', 'framing'}
FRAMING_FIELDS = ('system_prefix', 'system_suffix', 'user_prefix', 'user_suffix',
                 'assistant_prefix', 'assistant_suffix')


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
            + '\n').encode('utf-8')


def _framing_sha(value: dict) -> str:
    raw = json.dumps({key: value[key] for key in FRAMING_FIELDS}, sort_keys=True,
                     ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')
    return _sha(raw)


def _strict_json(raw: bytes) -> object:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result

    def bad_constant(_value):
        raise ValueError('invalid JSON number')

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad_constant)


def _read_json(path: Path, limit: int) -> tuple[object, bytes]:
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError('input exceeds configured byte limit')
    return _strict_json(raw), raw


def load_config(path: Path, max_bytes: int = 65536) -> tuple[dict, bytes]:
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError('max config bytes must be positive')
    value, raw = _read_json(path, max_bytes)
    if (not isinstance(value, dict) or set(value) != CONFIG_FIELDS
            or value.get('schema_version') != CONFIG_SCHEMA
            or not isinstance(value.get('revision'), str) or not value['revision'].strip()
            or not isinstance(value.get('system_prompt'), str) or not value['system_prompt'].strip()
            or any(type(value.get(name)) is not int or value[name] < 1 for name in
                   ('max_config_bytes', 'max_dataset_bytes', 'max_record_bytes', 'max_request_bytes',
                    'max_target_bytes', 'max_examples'))
            or not isinstance(value.get('framing'), dict)
            or set(value['framing']) != set(FRAMING_FIELDS)
            or any(not isinstance(value['framing'].get(name), str) for name in FRAMING_FIELDS)):
        raise ValueError('invalid user-request SFT configuration')
    if len(value['system_prompt'].encode('utf-8')) > value['max_config_bytes']:
        raise ValueError('system prompt exceeds configured byte limit')
    return value, raw


def _screen(text: str, screen_policy_path: Path, max_policy_bytes: int) -> tuple[str, dict[str, int], str]:
    baseline_path = source_bound_snapshot._default_screen_policy_path()
    baseline, baseline_rules, _ = screen_private_corpus.load_policy(baseline_path, max_policy_bytes)
    policy, rules, policy_sha = screen_private_corpus.load_policy(screen_policy_path, max_policy_bytes)
    counts = {}
    for current, compiled in ((baseline, baseline_rules), (policy, rules)):
        text, found = screen_private_corpus._screen(text, compiled, current['limits'])
        for name, count in found.items():
            counts[name] = counts.get(name, 0) + count
    return text, dict(sorted(counts.items())), policy_sha


def _sparse_target(target: dict, provider_models: list[dict], observation: dict) -> tuple[dict, list[dict], list[dict]]:
    """Keep only explicit resolved model polarity and explicitly related effort."""
    requests, refs, candidates = [], [], []
    provider_by_id = {row['model_id']: row for row in provider_models}
    span_by_id = {span['id']: span for span in observation['source']['spans']}
    for row in target['targets']:
        if (row['model_match_status'] != 'matched'
                or row['requested_model_target'] not in (0, 1)):
            continue
        polarity = 'affirmed' if row['requested_model_target'] == 1 else 'negated'
        request = {'model_id': row['model_id'], 'polarity': polarity}
        candidate = provider_by_id.get(row['model_id'])
        if candidate is None:
            raise ValueError('resolved request ID is absent from pinned provider observation')
        if row['model_id'] not in {item['model_id'] for item in candidates}:
            candidates.append({'model_id': candidate['model_id'],
                               'display_name': candidate.get('display_name') or candidate['model_id'],
                               'supported_efforts': candidate['provider_supported_efforts']})
        if (polarity == 'affirmed' and row['effort_match_status'] == 'matched'
                and row['requested_effort_target'] in (0, 1)):
            request['effort'] = row['effort']
            request['effort_polarity'] = ('affirmed' if row['requested_effort_target'] == 1
                                          else 'negated')
        requests.append(request)
        model_source_span = span_by_id[row['model_span']['id']]
        effort_source_span = (None if row['effort_span'] is None
                              else span_by_id[row['effort_span']['id']])
        refs.append({'model_span_id': row['model_span']['id'],
                     'model_span_sha256': row['model_span']['sha256'],
                     'model_span_start_byte': model_source_span['start_byte'],
                     'model_span_end_byte': model_source_span['end_byte'],
                     'model_polarity': row['model_span']['polarity'],
                     'effort_span_id': None if row['effort_span'] is None else row['effort_span']['id'],
                     'effort_span_sha256': None if row['effort_span'] is None else row['effort_span']['sha256'],
                     'effort_span_start_byte': None if effort_source_span is None else effort_source_span['start_byte'],
                     'effort_span_end_byte': None if effort_source_span is None else effort_source_span['end_byte'],
                     'effort_polarity': None if row['effort_span'] is None else row['effort_span']['polarity']})
    if not requests:
        return {}, [], []
    return {'schema_version': OUTPUT_SCHEMA, 'task': TASK, 'requested': requests}, refs, candidates


def _validate_config_text(text: str, policies: list[tuple[dict, list]]) -> None:
    for policy, rules in policies:
        screened, _ = source_bound_snapshot._screen_text_policies(text, [(policy, rules)])
        if screened != text:
            raise ValueError('task prompt or target requires redaction')


def _observation_from_caller_input(snapshot: dict, input_path: Path, config: dict,
                                   screen_policy_path: Path, max_bytes: int) -> tuple[dict, dict, bytes]:
    caller_input, raw = _read_json(input_path, max_bytes)
    if (not isinstance(caller_input, dict)
            or set(caller_input) != {'schema_version', 'spans', 'interpretation', 'relations'}
            or caller_input.get('schema_version') != CALLER_INPUT_SCHEMA
            or not isinstance(caller_input.get('spans'), list)
            or not isinstance(caller_input.get('interpretation'), dict)
            or not isinstance(caller_input.get('relations'), dict)):
        raise ValueError('invalid private caller-interpretation input')
    source_bound_snapshot.revalidate_snapshot(snapshot)
    split = snapshot['split']
    reserved = None
    manifest_sha = split.get('known_reserved_manifest_sha256')
    locator = snapshot.get('reserved_session_manifest_locator')
    if manifest_sha is not None or locator is not None:
        if not isinstance(manifest_sha, str) or not isinstance(locator, str):
            raise ValueError('incomplete pinned reserved-session evidence')
        reserved = {'_locator': locator, 'manifest_sha256': manifest_sha,
                    'groups': split.get('known_reserved_group_sha256', []),
                    'identities': split.get('known_reserved_source_identities', []),
                    'coverage': split.get('known_reserved_groups_coverage', 'unknown')}
    observation = user_requested_observation.build_observation(
        snapshot['source']['inventory_identity'], caller_input['spans'], caller_input['interpretation'],
        config, screen_policy_path, reserved=reserved,
        reserved_project_labels=split.get('known_reserved_project_labels', []))
    return observation, caller_input['relations'], raw


def build_dataset(observation_path: Path | None, provider_snapshot_path: Path, relations_path: Path | None,
                  observation_config_path: Path, sft_config_path: Path, screen_policy_path: Path,
                  output_path: Path, *, caller_input_path: Path | None = None,
                  max_input_bytes: int = 8 * 1024 * 1024) -> dict:
    """Revalidate source-linked request evidence and write one private dataset file.

    The destination parent must already be an owner-only POSIX directory and
    ignored/untracked by Git. Windows private output is intentionally disabled.
    """
    if type(max_input_bytes) is not int or max_input_bytes < 1:
        raise ValueError('max input bytes must be positive')
    if os.name == 'nt':
        raise ValueError('user-request SFT output is disabled on Windows until private path safety is verified')
    parent_fd, output_name, actual_parent = source_bound_snapshot._open_private_parent(output_path)
    try:
        source_bound_snapshot._require_ignored_untracked(Path(actual_parent) / output_name)
    finally:
        os.close(parent_fd)
    obs_config, obs_config_sha = user_requested_observation.load_config(observation_config_path)
    config, config_raw = load_config(sft_config_path)
    snapshot, snapshot_raw = _read_json(provider_snapshot_path, max_input_bytes)
    if (caller_input_path is None) == (observation_path is None):
        raise ValueError('supply exactly one observation or caller-input file')
    pinned_observation_raw = pinned_relations_raw = pinned_caller_raw = None
    if caller_input_path is not None:
        observation, relations, pinned_caller_raw = _observation_from_caller_input(
            snapshot, caller_input_path, obs_config, screen_policy_path, max_input_bytes)
    else:
        observation, pinned_observation_raw = _read_json(
            observation_path, min(max_input_bytes, obs_config['max_observation_bytes']))
        if relations_path is None:
            raise ValueError('relations file is required with an existing observation')
        relations, pinned_relations_raw = _read_json(relations_path, max_input_bytes)
    target = user_request_targets.import_request_targets(
        observation, snapshot, relations, obs_config, screen_policy_path)
    user_request_targets.validate_target_record(target)
    source_bound_snapshot.revalidate_snapshot(snapshot)
    source_record = snapshot['source']['inventory_identity']
    raw_request = source_bound_snapshot.verify_selected_source(
        source_record, max_session_bytes=obs_config['max_source_prefix_bytes'],
        max_line_bytes=obs_config['max_line_bytes'])
    request_bytes = raw_request.encode('utf-8')
    if (len(request_bytes) != target['source']['user_text_utf8_bytes']
            or _sha(request_bytes) != target['source']['user_text_sha256']):
        raise ValueError('source request changed after target revalidation')
    relation_digest = _sha(_canonical({
        'schema_version': relations['schema_version'], 'origin': relations['origin'],
        'review_status': relations['review_status'],
        'items': [{'model_span_id': item['model_span_id'],
                  'effort_span_id': item.get('effort_span_id')} for item in relations['items']]}))
    if relation_digest != target['relations_sha256']:
        raise ValueError('caller relations changed during target revalidation')
    screened_request, redaction_counts, policy_sha = _screen(
        raw_request, screen_policy_path, min(config['max_config_bytes'], 1024 * 1024))
    baseline, baseline_rules, _ = screen_private_corpus.load_policy(
        source_bound_snapshot._default_screen_policy_path(), min(config['max_config_bytes'], 1024 * 1024))
    policy, policy_rules, loaded_policy_sha = screen_private_corpus.load_policy(
        screen_policy_path, min(config['max_config_bytes'], 1024 * 1024))
    if policy_sha != loaded_policy_sha or policy_sha != target['provider_observation']['screen_policy_sha256']:
        raise ValueError('SFT screening policy does not match source-bound request evidence')
    _validate_config_text(config['system_prompt'], [(baseline, baseline_rules), (policy, policy_rules)])
    sparse, span_refs, provider_candidates = _sparse_target(
        target, snapshot['provider_list']['models'], observation)
    if not sparse:
        raise ValueError('no uniquely resolved explicit request targets for SFT')
    response = json.dumps(sparse, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    response_bytes = response.encode('utf-8')
    _validate_config_text(response, [(baseline, baseline_rules), (policy, policy_rules)])
    if len(response_bytes) > config['max_target_bytes']:
        raise ValueError('sparse target exceeds configured byte limit')
    model_input = {'user_request': screened_request, 'provider_candidates': provider_candidates}
    screened_input, input_redactions, _ = _screen(
        json.dumps(model_input, ensure_ascii=False, separators=(',', ':'), allow_nan=False),
        screen_policy_path, min(config['max_config_bytes'], 1024 * 1024))
    if input_redactions:
        raise ValueError('provider candidate input requires redaction')
    if len(screened_input.encode('utf-8')) > config['max_request_bytes']:
        raise ValueError('screened request and provider candidates exceed configured byte limit')
    record_core = {
        'messages': [{'role': 'system', 'content': config['system_prompt']},
                     {'role': 'user', 'content': screened_input},
                     {'role': 'assistant', 'content': response}],
    }
    record = {
        'id': _sha(_canonical({'source_line_sha256': target['source']['source_line_sha256'],
                               'relations_sha256': target['relations_sha256'],
                               'target_record_sha256': target['target_record_sha256']})),
        **record_core,
        'metadata': {
            'task': TASK, 'task_schema_version': OUTPUT_SCHEMA,
            'review_status': 'unreviewed', 'outcome': 'unknown',
            'provider_entitlement': 'unknown', 'execution': 'unknown',
            'total_agent_count': None, 'promotion_eligible': False,
            'training_assignment': 'train', 'split_policy': 'exploratory_train',
            'heldout_separation': 'unknown',
            'source_file_sha256_at_inventory': target['source']['source_file_sha256_at_inventory'],
            'source_prefix_sha256': target['source']['source_prefix_sha256'],
            'source_line_sha256': target['source']['source_line_sha256'],
            'user_text_sha256': target['source']['user_text_sha256'],
            'user_text_utf8_bytes': target['source']['user_text_utf8_bytes'],
            'source_group_sha256': target['source']['source_group_sha256'],
            'provider_cache_sha256': target['provider_observation']['cache_sha256'],
            'relations_sha256': target['relations_sha256'],
            'target_record_sha256': target['target_record_sha256'],
            'screen_policy_sha256': policy_sha,
            'screened_request_sha256': _sha(screened_input.encode('utf-8')),
            'user_text_redaction_counts': redaction_counts,
            'target_span_refs': span_refs,
            'redaction_counts': redaction_counts,
        },
    }
    record_raw = _canonical(record)
    if len(record_raw) > config['max_record_bytes']:
        raise ValueError('user-request SFT record exceeds configured limit')
    records_raw = record_raw
    if len(records_raw) > config['max_dataset_bytes']:
        raise ValueError('user-request SFT dataset exceeds configured limit')
    manifest = {
        'schema_version': DATASET_SCHEMA, 'task': TASK, 'task_schema_version': OUTPUT_SCHEMA,
        'revision': config['revision'], 'config_sha256': _sha(config_raw),
        'builder_sha256': _sha(Path(__file__).read_bytes()),
        'observation_config_sha256': obs_config_sha,
        'system_prompt_sha256': _sha(config['system_prompt'].encode('utf-8')),
        'framing_sha256': _framing_sha(config['framing']),
        'training_assignment': 'train', 'split_policy': 'exploratory_train',
        'heldout_separation': 'unknown', 'source_split_observation': target['source_split'],
        'review_status': 'unreviewed', 'outcome': 'unknown', 'provider_entitlement': 'unknown',
        'provider_listing_is_entitlement': False, 'execution': 'unknown', 'total_agent_count': None,
        'promotion_eligible': False, 'records': 1, 'records_sha256': _sha(records_raw),
        'record_id': record['id'], 'source_evidence': {key: target['source'][key] for key in
            ('source_file_sha256_at_inventory', 'source_prefix_sha256', 'source_line_sha256',
             'user_text_sha256', 'source_group_sha256')},
        'user_text_utf8_bytes': target['source']['user_text_utf8_bytes'],
        'provider_cache_sha256': target['provider_observation']['cache_sha256'],
        'relations_sha256': target['relations_sha256'],
        'target_record_sha256': target['target_record_sha256'],
        'screen_policy_sha256': policy_sha, 'screened_request_sha256': record['metadata']['screened_request_sha256'],
        'redaction_counts': redaction_counts,
        'user_text_redaction_counts': redaction_counts,
        'validation_opened': False, 'test_opened': False,
        'assistant_teacher_claimed': False,
    }
    envelope = {'schema_version': DATASET_SCHEMA, 'manifest': manifest, 'train_records': [record]}
    envelope_raw = _canonical(envelope)
    if len(envelope_raw) > config['max_dataset_bytes']:
        raise ValueError('serialized user-request dataset exceeds configured limit')
    current_snapshot_raw = _read_json(provider_snapshot_path, max_input_bytes)[1]
    _current_config, current_config_raw = load_config(sft_config_path)
    _current_obs_config, current_obs_config_sha = user_requested_observation.load_config(observation_config_path)
    _current_policy, _current_rules, current_policy_sha = screen_private_corpus.load_policy(
        screen_policy_path, min(config['max_config_bytes'], 1024 * 1024))
    current_request = source_bound_snapshot.verify_selected_source(
        source_record, max_session_bytes=obs_config['max_source_prefix_bytes'],
        max_line_bytes=obs_config['max_line_bytes'])
    if (current_snapshot_raw != snapshot_raw or current_config_raw != config_raw
            or current_obs_config_sha != obs_config_sha or current_policy_sha != policy_sha
            or _sha(current_request.encode('utf-8')) != target['source']['user_text_sha256']):
        raise ValueError('source-bound SFT input changed during preparation')
    if caller_input_path is not None:
        if _read_json(caller_input_path, max_input_bytes)[1] != pinned_caller_raw:
            raise ValueError('caller interpretation changed during preparation')
    elif (_read_json(observation_path, min(max_input_bytes, obs_config['max_observation_bytes']))[1]
          != pinned_observation_raw
          or _read_json(relations_path, max_input_bytes)[1] != pinned_relations_raw):
        raise ValueError('observation or relation input changed during preparation')
    source_bound_snapshot.revalidate_snapshot(snapshot)
    source_bound_snapshot.write_private_bytes(output_path, envelope_raw)
    return {'schema_version': 'lfg.user-requested-sft-build-result.v1', 'records': 1,
            'dataset_sha256': _sha(envelope_raw), 'promotion_eligible': False,
            'heldout_separation': 'unknown'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observation', type=Path)
    parser.add_argument('--caller-input', type=Path,
                        help='private spans/interpretation/relations JSON; materialize observation in memory')
    for name in ('provider-snapshot', 'observation-config', 'sft-config', 'screen-policy', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--relations', type=Path,
                        help='private relation JSON when --observation is supplied')
    parser.add_argument('--max-input-bytes', type=int, default=8 * 1024 * 1024)
    args = parser.parse_args()
    try:
        result = build_dataset(args.observation, args.provider_snapshot, args.relations,
                               args.observation_config, args.sft_config, args.screen_policy,
                               args.output, caller_input_path=args.caller_input,
                               max_input_bytes=args.max_input_bytes)
    except (OSError, ValueError, TypeError, KeyError, UnicodeError) as error:
        raise SystemExit(f'User-request SFT preparation failed ({type(error).__name__}): {error}') from None
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
