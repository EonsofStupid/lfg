"""Project unreviewed, source-linked user model/effort requests to inert targets.

This creates explicit request targets only. It does not create recommendation,
entitlement, outcome, review, training-split, or generative-response labels.
"""
from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path

import screen_private_corpus
import source_bound_snapshot
import user_requested_observation


SCHEMA = 'lfg.user-requested-model-effort-targets.v1'
RELATIONS_SCHEMA = 'lfg.user-requested-model-effort-relations.v1'
MODEL_POLARITIES = {'affirmed': 1, 'negated': 0}
MODEL_MATCH_STATUSES = {'matched', 'unmatched', 'ambiguous', 'requires_redaction'}
EFFORT_MATCH_STATUSES = {'not_supplied', 'matched', 'unmatched', 'ambiguous', 'requires_redaction', 'model_unresolved'}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
            + '\n').encode('utf-8')


def _normal(value: str) -> str:
    return ' '.join(unicodedata.normalize('NFKC', value).casefold().split())


def _validate_relations(value: object, observation: dict, config: dict) -> list[dict]:
    if (not isinstance(value, dict) or set(value) != {'schema_version', 'origin', 'review_status', 'items'}
            or value.get('schema_version') != RELATIONS_SCHEMA
            or value.get('origin') != 'caller_supplied_interpretation'
            or value.get('review_status') != 'unreviewed'
            or not isinstance(value.get('items'), list)
            or len(value['items']) > config['max_claims_per_field']):
        raise ValueError('invalid unreviewed user-request relations')
    interpretation = observation['interpretation']
    models = {item['evidence_span_id']: item for item in interpretation['model_claims'].get('items', [])}
    efforts = {item['evidence_span_id']: item for item in interpretation['effort_claims'].get('items', [])}
    used_models, used_efforts = set(), set()
    normalized = []
    for item in value['items']:
        if (not isinstance(item, dict) or set(item) - {'model_span_id', 'effort_span_id'}
                or 'model_span_id' not in item or not isinstance(item['model_span_id'], str)
                or item['model_span_id'] not in models
                or ('effort_span_id' in item and (not isinstance(item['effort_span_id'], str)
                                                  or item['effort_span_id'] not in efforts))):
            raise ValueError('relation must reference existing caller-interpreted model and effort spans')
        model_span_id = item['model_span_id']
        effort_span_id = item.get('effort_span_id')
        if model_span_id in used_models or (effort_span_id is not None and effort_span_id in used_efforts):
            raise ValueError('a source span cannot be assigned to multiple model-effort relations')
        used_models.add(model_span_id)
        if effort_span_id is not None:
            used_efforts.add(effort_span_id)
        normalized.append({'model_span_id': model_span_id, 'effort_span_id': effort_span_id})
    return normalized


def _claim_polarity(observation: dict, field: str, span_id: str) -> str:
    claims = observation['interpretation'][field].get('items', [])
    claim = next((item for item in claims if item['evidence_span_id'] == span_id), None)
    if claim is None:
        raise ValueError('relation references a span without an explicit claim')
    return claim['polarity']


def _span_text(raw_text: str, span: dict) -> str:
    raw = raw_text.encode('utf-8')
    start, end = span['start_byte'], span['end_byte']
    try:
        text = raw[start:end].decode('utf-8')
    except UnicodeDecodeError:
        raise ValueError('observation span splits a UTF-8 code point') from None
    if _sha(raw[start:end]) != span['sha256']:
        raise ValueError('observation span changed')
    return text


def _unique_model(text: str, rows: list[dict]) -> tuple[str, str | None]:
    key = _normal(text)
    matches = [row for row in rows if key in {_normal(row['model_id']), _normal(row['display_name'])}]
    if len(matches) == 1:
        return 'matched', matches[0]['model_id']
    return ('unmatched' if not matches else 'ambiguous'), None


def _unique_effort(text: str, model: dict) -> tuple[str, str | None]:
    key = _normal(text)
    matches = [value for value in model['provider_supported_efforts'] if _normal(value) == key]
    if len(matches) == 1:
        return 'matched', matches[0]
    return ('unmatched' if not matches else 'ambiguous'), None


def validate_target_record(value: object) -> None:
    """Validate output semantics without reopening source or provider files."""
    fields = {'schema_version', 'kind', 'target_semantics', 'origin', 'review_status', 'source',
              'provider_observation', 'relations_sha256', 'source_split', 'state', 'targets',
              'target_record_sha256'}
    if (not isinstance(value, dict) or set(value) != fields or value.get('schema_version') != SCHEMA
            or value.get('kind') != 'user_requested_model_effort'
            or value.get('target_semantics') != 'explicit_request_for_this_prompt'
            or value.get('origin') != 'caller_supplied_interpretation'
            or value.get('review_status') != 'unreviewed'):
        raise ValueError('invalid user-request target record')
    source = value['source']
    source_digests = ('source_file_sha256_at_inventory', 'source_prefix_sha256', 'source_line_sha256',
                      'user_text_sha256', 'source_group_sha256', 'project_label_sha256')
    if (not isinstance(source, dict)
            or set(source) != {*source_digests, 'source_prefix_bytes', 'source_line', 'user_text_utf8_bytes'}
            or any(not isinstance(source.get(key), str) or len(source[key]) != 64
                   or any(char not in '0123456789abcdef' for char in source[key]) for key in source_digests)
            or any(type(source.get(key)) is not int or source[key] < 1
                   for key in ('source_prefix_bytes', 'source_line', 'user_text_utf8_bytes'))):
        raise ValueError('invalid user-request target source identity')
    provider = value['provider_observation']
    if (not isinstance(provider, dict)
            or set(provider) != {'cache_sha256', 'screen_policy_sha256', 'model_rows_observed', 'listing_is_entitlement'}
            or any(not isinstance(provider.get(key), str) or len(provider[key]) != 64
                   or any(char not in '0123456789abcdef' for char in provider[key])
                   for key in ('cache_sha256', 'screen_policy_sha256'))
            or type(provider.get('model_rows_observed')) is not int or provider['model_rows_observed'] < 1
            or provider.get('listing_is_entitlement') is not False):
        raise ValueError('invalid provider observation provenance')
    relation_sha = value.get('relations_sha256')
    if (not isinstance(relation_sha, str) or len(relation_sha) != 64
            or any(char not in '0123456789abcdef' for char in relation_sha)):
        raise ValueError('invalid relation digest')
    if value['source_split'] != {'assignment': 'unknown', 'heldout_separation': 'unknown'}:
        raise ValueError('user-request target split must remain unknown')
    expected_state = {'provider_entitlement': 'unknown', 'execution': 'unknown', 'outcome': 'unknown',
                      'total_agent_count': None, 'promotion_eligible': False}
    if value['state'] != expected_state:
        raise ValueError('user-request target state must remain unknown and unpromoted')
    rows = value['targets']
    if not isinstance(rows, list) or len(rows) > 64:
        raise ValueError('invalid user-request target rows')
    for row in rows:
        if (not isinstance(row, dict) or set(row) != {'model_span', 'model_match_status', 'model_id',
                    'requested_model_target', 'effort_span', 'effort_match_status', 'effort',
                    'requested_effort_target'}):
            raise ValueError('invalid user-request target row')
        model_span = row['model_span']
        if (not isinstance(model_span, dict) or set(model_span) != {'id', 'sha256', 'polarity'}
                or not isinstance(model_span['id'], str) or not model_span['id']
                or not isinstance(model_span['sha256'], str) or len(model_span['sha256']) != 64
                or any(char not in '0123456789abcdef' for char in model_span['sha256'])
                or model_span['polarity'] not in {'affirmed', 'negated', 'quoted', 'uncertain'}
                or row['model_match_status'] not in MODEL_MATCH_STATUSES):
            raise ValueError('invalid model evidence')
        model_resolved = row['model_match_status'] == 'matched'
        if ((model_resolved and (not isinstance(row['model_id'], str) or not row['model_id']))
                or (not model_resolved and row['model_id'] is not None)):
            raise ValueError('model candidate resolution does not match its status')
        expected_model_target = (MODEL_POLARITIES.get(model_span['polarity']) if model_resolved else None)
        if (row['requested_model_target'] != expected_model_target
                or (row['requested_model_target'] is not None
                    and type(row['requested_model_target']) is not int)):
            raise ValueError('model target does not match explicit polarity and resolution')
        effort_span = row['effort_span']
        if row['effort_match_status'] not in EFFORT_MATCH_STATUSES:
            raise ValueError('invalid effort match status')
        if effort_span is None:
            if (row['effort_match_status'] != 'not_supplied' or row['effort'] is not None
                    or row['requested_effort_target'] is not None):
                raise ValueError('unsupplied effort must remain unknown')
            continue
        if (not isinstance(effort_span, dict) or set(effort_span) != {'id', 'sha256', 'polarity'}
                or not isinstance(effort_span['id'], str) or not effort_span['id']
                or not isinstance(effort_span['sha256'], str) or len(effort_span['sha256']) != 64
                or any(char not in '0123456789abcdef' for char in effort_span['sha256'])
                or effort_span['polarity'] not in {'affirmed', 'negated', 'quoted', 'uncertain'}):
            raise ValueError('invalid effort evidence')
        effort_resolved = row['effort_match_status'] == 'matched'
        if ((effort_resolved and not isinstance(row['effort'], str))
                or (not effort_resolved and row['effort'] is not None)):
            raise ValueError('effort resolution does not match its status')
        expected_effort_target = (MODEL_POLARITIES.get(effort_span['polarity'])
                                  if effort_resolved and model_resolved
                                  and model_span['polarity'] == 'affirmed' else None)
        if (row['requested_effort_target'] != expected_effort_target
                or (row['requested_effort_target'] is not None
                    and type(row['requested_effort_target']) is not int)):
            raise ValueError('effort target does not match explicit source relation')
    digest_payload = {key: item for key, item in value.items() if key != 'target_record_sha256'}
    if value.get('target_record_sha256') != _sha(_canonical(digest_payload)):
        raise ValueError('user-request target record digest mismatch')


def import_request_targets(observation: dict, provider_snapshot: dict, relations: dict,
                           config: dict, screen_policy_path: Path) -> dict:
    """Revalidate one source and provider snapshot, then produce text-free partial targets.

    `relations` is explicitly caller supplied and remains unreviewed. Source split
    assignment stays unknown here; an independent, explicit plan must assign it.
    """
    if not isinstance(provider_snapshot, dict) or provider_snapshot.get('schema_version') != source_bound_snapshot.SCHEMA:
        raise ValueError('provider evidence must be a source-bound provider snapshot')
    split = provider_snapshot.get('split')
    if (not isinstance(split, dict) or split.get('policy') != 'exploratory_train'
            or split.get('assignment') != 'train'):
        raise ValueError('only an explicitly selected exploratory train source may be opened')
    user_requested_observation.validate_observation(observation, config, screen_policy_path)
    normalized_relations = _validate_relations(relations, observation, config)
    source_bound_snapshot.revalidate_snapshot(provider_snapshot)
    selected_source = provider_snapshot['source']
    record = selected_source['inventory_identity']
    observed_source = observation['source']
    identity_fields = {
        'source_file_sha256_at_inventory': record['source_file_sha256'],
        'source_prefix_bytes': record['source_prefix_bytes'],
        'source_prefix_sha256': record['source_prefix_sha256'],
        'source_line': record['source_line'],
        'source_line_sha256': record['source_line_sha256'],
        'user_text_sha256': record['text_sha256'],
        'source_group_sha256': source_bound_snapshot.source_group_id(record),
        'project_label_sha256': _sha(record['project'].encode('utf-8')),
    }
    if any(observed_source.get(key) != expected for key, expected in identity_fields.items()):
        raise ValueError('observation and provider snapshot refer to different source identities')
    provider = provider_snapshot['provider_list']
    if observation['screen_policy_sha256'] != provider['screen_policy_sha256']:
        raise ValueError('observation and provider snapshot use different screening policies')
    raw_text = source_bound_snapshot.verify_selected_source(
        record, max_session_bytes=512 * 1024 * 1024, max_line_bytes=16 * 1024 * 1024)
    raw_bytes = raw_text.encode('utf-8')
    if (len(raw_bytes) != observed_source['user_text_utf8_bytes']
            or _sha(raw_bytes) != observed_source['user_text_sha256']):
        raise ValueError('selected source changed after observation')

    span_map = {item['id']: item for item in observed_source['spans']}
    span_texts = {span_id: _span_text(raw_text, span) for span_id, span in span_map.items()}
    baseline, baseline_rules, _ = screen_private_corpus.load_policy(
        source_bound_snapshot._default_screen_policy_path(), 1024 * 1024)
    policy, policy_rules, _policy_sha = screen_private_corpus.load_policy(screen_policy_path, 1024 * 1024)
    policies = [(baseline, baseline_rules), (policy, policy_rules)]
    rows = provider.get('models')
    if not isinstance(rows, list) or not rows:
        raise ValueError('pinned provider observation has no listed models')
    clean_rows = []
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get('model_id'), str)
                or not isinstance(row.get('display_name'), str)
                or not isinstance(row.get('provider_supported_efforts'), list)
                or any(not isinstance(effort, str) for effort in row['provider_supported_efforts'])):
            raise ValueError('invalid pinned provider observation')
        clean_rows.append(row)

    target_rows = []
    for relation in normalized_relations:
        model_span_id = relation['model_span_id']
        effort_span_id = relation['effort_span_id']
        model_span = span_map.get(model_span_id)
        if model_span is None or model_span['field'] != 'model':
            raise ValueError('model relation does not reference a model source span')
        model_text = span_texts[model_span_id]
        model_polarity = _claim_polarity(observation, 'model_claims', model_span_id)
        screened_model, _ = source_bound_snapshot._screen_text_policies(model_text, policies)
        if screened_model != model_text:
            model_status, model_id = 'requires_redaction', None
        else:
            model_status, model_id = _unique_model(model_text, clean_rows)
        model_record = next((row for row in clean_rows if row['model_id'] == model_id), None)
        model_label = MODEL_POLARITIES.get(model_polarity) if model_status == 'matched' else None

        effort_status, effort_value, effort_label = 'not_supplied', None, None
        effort_polarity = None
        effort_span_digest = None
        if effort_span_id is not None:
            effort_span = span_map.get(effort_span_id)
            if effort_span is None or effort_span['field'] != 'effort':
                raise ValueError('effort relation does not reference an effort source span')
            effort_text = span_texts[effort_span_id]
            effort_polarity = _claim_polarity(observation, 'effort_claims', effort_span_id)
            effort_span_digest = effort_span['sha256']
            screened_effort, _ = source_bound_snapshot._screen_text_policies(effort_text, policies)
            if screened_effort != effort_text:
                effort_status = 'requires_redaction'
            elif model_record is None:
                effort_status = 'model_unresolved'
            else:
                effort_status, effort_value = _unique_effort(effort_text, model_record)
                if model_polarity == 'affirmed' and effort_status == 'matched':
                    effort_label = MODEL_POLARITIES.get(effort_polarity)
        target_rows.append({
            'model_span': {'id': model_span_id, 'sha256': model_span['sha256'], 'polarity': model_polarity},
            'model_match_status': model_status,
            'model_id': model_id,
            'requested_model_target': model_label,
            'effort_span': (None if effort_span_id is None else {
                'id': effort_span_id, 'sha256': effort_span_digest, 'polarity': effort_polarity}),
            'effort_match_status': effort_status,
            'effort': effort_value,
            'requested_effort_target': effort_label,
        })

    if observation['state']['provider_entitlement'] != 'unknown':
        raise ValueError('source observation must preserve unknown provider entitlement')
    result = {
        'schema_version': SCHEMA,
        'kind': 'user_requested_model_effort',
        'target_semantics': 'explicit_request_for_this_prompt',
        'origin': 'caller_supplied_interpretation',
        'review_status': 'unreviewed',
        'source': {**identity_fields, 'user_text_utf8_bytes': len(raw_bytes)},
        'provider_observation': {
            'cache_sha256': provider['cache_sha256'],
            'screen_policy_sha256': provider['screen_policy_sha256'],
            'model_rows_observed': len(clean_rows),
            'listing_is_entitlement': False,
        },
        'relations_sha256': _sha(_canonical({'schema_version': relations['schema_version'],
                                              'origin': relations['origin'],
                                              'review_status': relations['review_status'],
                                              'items': normalized_relations})),
        'source_split': {'assignment': 'unknown', 'heldout_separation': 'unknown'},
        'state': {
            'provider_entitlement': 'unknown',
            'execution': 'unknown',
            'outcome': 'unknown',
            'total_agent_count': None,
            'promotion_eligible': False,
        },
        'targets': target_rows,
    }
    result['target_record_sha256'] = _sha(_canonical(result))
    validate_target_record(result)
    return result
