"""Configurable, explicitly unknown assessment targets for human review."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

HEX = re.compile(r'^[0-9a-f]{64}$')
FIELDS = ('complexity', 'models', 'agent_count')
COVERAGE_FIELDS = (*FIELDS, 'effort')


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and '\n' not in value and '\r' not in value


def digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def validate_project(profile: Any) -> None:
    if (not isinstance(profile, dict) or set(profile) != {'project', 'complexities', 'models', 'agent_range'}
            or not _text(profile['project']) or not isinstance(profile['complexities'], list)
            or not isinstance(profile['models'], list)):
        raise ValueError('invalid assessment project profile')
    for field in ('complexities', 'models'):
        ids = set()
        for option in profile[field]:
            fields = {'id', 'label'} if field == 'complexities' else {
                'id', 'label', 'provider', 'model', 'subscription_id', 'availability', 'effort_levels'}
            if (not isinstance(option, dict) or set(option) != fields
                    or any(not _text(option[k]) for k in fields - {'effort_levels', 'subscription_id'})):
                raise ValueError('invalid assessment option')
            if option['id'] in ids:
                raise ValueError('duplicate assessment option ID')
            ids.add(option['id'])
            if field == 'models':
                levels = option['effort_levels']
                if (option['availability'] not in ('available', 'unavailable', 'unknown')
                        or (option['subscription_id'] is not None and not _text(option['subscription_id']))
                        or not isinstance(levels, list) or any(not _text(level) for level in levels)
                        or len(set(levels)) != len(levels)):
                    raise ValueError('invalid model availability or effort levels')
    bounds = profile['agent_range']
    if bounds is not None and (not isinstance(bounds, dict) or set(bounds) != {'min', 'max'}
                               or type(bounds['min']) is not int or type(bounds['max']) is not int
                               or bounds['min'] < 0 or bounds['max'] < bounds['min']
                               or bounds['max'] > 2**53 - 1):
        raise ValueError('invalid agent count range')


def load_config(path: Path, max_bytes: int | None = None) -> tuple[dict[str, Any], str]:
    if max_bytes is not None and (type(max_bytes) is not int or max_bytes < 1):
        raise ValueError('invalid assessment profile byte limit')
    def read() -> bytes:
        with path.open('rb') as stream:
            raw = stream.read() if max_bytes is None else stream.read(max_bytes + 1)
        if max_bytes is not None and len(raw) > max_bytes:
            raise ValueError('assessment profile exceeds configured byte limit')
        return raw
    raw = read()
    config = json.loads(raw)
    if (not isinstance(config, dict) or set(config) != {'schema_version', 'revision', 'projects'}
            or config['schema_version'] != 'lfg.assessment-review-config.v1'
            or not _text(config['revision']) or not isinstance(config['projects'], list)):
        raise ValueError('invalid assessment review configuration')
    projects = set()
    for profile in config['projects']:
        validate_project(profile)
        if profile['project'] in projects:
            raise ValueError('duplicate assessment project')
        projects.add(profile['project'])
    if read() != raw:
        raise ValueError('assessment configuration changed while loading')
    return config, hashlib.sha256(raw).hexdigest()


def project_profile(config: dict[str, Any], project: str) -> dict[str, Any] | None:
    return next((value for value in config['projects'] if value['project'] == project), None)


def make_record(values: Any, profile: dict[str, Any], revision: str, config_sha256: str) -> dict[str, Any]:
    record = {'schema_version': 'lfg.review-assessment-targets.v1', 'config_revision': revision,
              'config_sha256': config_sha256, 'profile_sha256': digest(profile), 'project_profile': profile,
              'values': values}
    validate_stored(record, profile['project'])
    return record


def validate_stored(record: Any, project: str) -> None:
    if (not isinstance(record, dict) or set(record) != {'schema_version', 'config_revision', 'config_sha256',
            'profile_sha256', 'project_profile', 'values'}
            or record['schema_version'] != 'lfg.review-assessment-targets.v1'
            or not _text(record['config_revision'])
            or any(not isinstance(record[key], str) or not HEX.fullmatch(record[key]) for key in ('config_sha256', 'profile_sha256'))):
        raise ValueError('invalid stored assessment targets')
    profile, values = record['project_profile'], record['values']
    validate_project(profile)
    if profile['project'] != project or digest(profile) != record['profile_sha256']:
        raise ValueError('assessment profile does not match its project or digest')
    if not isinstance(values, dict) or set(values) != set(FIELDS):
        raise ValueError('assessment values must include all fields; null means unknown')
    if values['complexity'] is not None and (not _text(values['complexity']) or
                                             values['complexity'] not in {x['id'] for x in profile['complexities']}):
        raise ValueError('complexity is not in the configured profile')
    models = values['models']
    if models is not None:
        if not isinstance(models, list) or not models:
            raise ValueError('models must be null or a nonempty list')
        catalog = {x['id']: x for x in profile['models']}
        seen = set()
        for model in models:
            if (not isinstance(model, dict) or set(model) != {'id', 'effort'} or not _text(model['id'])
                    or model['id'] in seen or model['id'] not in catalog):
                raise ValueError('invalid or duplicate model selection')
            item = catalog[model['id']]
            if item['availability'] != 'available':
                raise ValueError('selected model is not declared available in this project')
            if model['effort'] is not None and (not _text(model['effort']) or model['effort'] not in item['effort_levels']):
                raise ValueError('model effort is not in the configured profile')
            seen.add(model['id'])
    count, bounds = values['agent_count'], profile['agent_range']
    if count is not None and (type(count) is not int or bounds is None or not bounds['min'] <= count <= bounds['max']):
        raise ValueError('agent count is outside the configured profile')


def coverage(values: dict[str, Any] | None) -> dict[str, str]:
    """Summarize known targets on a single already-validated reviewed record."""
    result = {field: 'known' if values is not None and values[field] is not None else 'unknown' for field in FIELDS}
    models = None if values is None else values['models']
    if models is None:
        result['effort'] = 'unknown'
    else:
        known = sum(model['effort'] is not None for model in models)
        result['effort'] = 'known' if known == len(models) else 'partial' if known else 'unknown'
    return result
