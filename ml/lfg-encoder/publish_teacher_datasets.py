#!/usr/bin/env python3
"""Prepare completed private teacher collections and deliver a train-only dataset."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

import assessment_distill
import train


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def encoded(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False) + '\n').encode()


def read(path: Path, limit: int) -> bytes:
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError('input exceeds configured limit')
    return raw


def config_file(path: Path, max_config_bytes: int = 1048576) -> tuple[dict, bytes]:
    if type(max_config_bytes) is not int or max_config_bytes < 1:
        raise ValueError('publisher config byte limit must be positive')
    raw = read(path, max_config_bytes)
    value = json.loads(raw)
    if (not isinstance(value, dict) or value.get('schema_version') != 'lfg.teacher-dataset-publisher.v1'
            or any(not isinstance(value.get(k), str) or not Path(value[k]).is_absolute()
                   for k in ('collection_root', 'staging_root'))
            or not isinstance(value.get('collector_configs'), list) or not value['collector_configs']
            or any(not isinstance(x, str) or not Path(x).is_absolute() for x in value['collector_configs'])
            or not isinstance(value.get('plan'), dict) or not isinstance(value.get('delivery'), dict)
            or any(type(value.get(k)) is not int or value[k] < 1
                   for k in ('max_manifest_bytes', 'max_dataset_bytes', 'max_transfer_bytes'))
            or type(value.get('resend_interval_seconds', 0)) is not int or value.get('resend_interval_seconds', 0) < 0):
        raise ValueError('invalid publisher configuration')
    return value, raw


def prepare(value: dict, config_raw: bytes) -> Path:
    root = Path(value['collection_root']).resolve(strict=True)
    staging = Path(value['staging_root'])
    approved = {}
    for name in value['collector_configs']:
        raw = read(Path(name), value['max_manifest_bytes'])
        item = json.loads(raw)
        if (item.get('schema_version') != 'lfg.local-teacher-collection.v1'
                or item.get('system_prompt') != value['plan'].get('system_prompt')
                or item.get('source_origin') not in ('owner_chat', 'owner_project')):
            raise ValueError('collector config does not match the selected assessment task')
        approved[sha(raw)] = item['source_origin']
    sources, collections = [], []
    for path in sorted(root.rglob('manifest.json')):
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError('collection manifest is not a private regular file')
        raw = read(path, value['max_manifest_bytes'])
        item = json.loads(raw)
        if not isinstance(item, dict) or item.get('schema_version') != 'lfg.local-teacher-collection-result.v1':
            continue
        origin = approved.get(item.get('config_sha256'))
        if (origin is None or item.get('test_opened') is not False or item.get('validation_opened') is not False
                or item.get('review_status') != 'unreviewed' or item.get('outcome') != 'unknown'
                or not isinstance(item.get('files'), list)):
            raise ValueError('collection identity or policy is not approved')
        collections.append({'manifest_sha256': sha(raw), 'path': str(path)})
        for entry in item['files']:
            if (not isinstance(entry, dict) or entry.get('split') != 'train' or entry.get('origin') != origin
                    or any(not isinstance(entry.get(k), str) or not entry[k]
                           for k in ('path', 'sha256', 'project', 'project_group'))):
                raise ValueError('invalid collection source')
            relative = Path(entry['path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('collection source path escapes its directory')
            file = path.parent / relative
            if file.is_symlink() or not file.is_file() or not file.resolve().is_relative_to(path.parent.resolve()):
                raise ValueError('collection source is not a regular local file')
            sources.append({'path': str(file.resolve()), 'sha256': entry['sha256'], 'project': entry['project'],
                            'project_group': entry['project_group'], 'split': 'train', 'origin': origin})
    if not sources:
        raise ValueError('no completed teacher collections')
    plan = dict(value['plan'])
    plan['sources'] = sources
    fingerprint = sha(encoded({'config_sha256': sha(config_raw), 'collections': collections, 'plan': plan,
                               'publisher_sha256': sha(Path(__file__).read_bytes()),
                               'builder_sha256': sha(Path(assessment_distill.__file__).read_bytes())}))
    assessment_distill.private_output(staging)
    staging.mkdir(mode=0o700, parents=True, exist_ok=True)
    final = staging / fingerprint
    if final.exists():
        if final.is_symlink() or not (final / 'dataset' / 'manifest.json').is_file():
            raise ValueError('existing prepared dataset is incomplete')
        return final / 'dataset'
    temporary = Path(tempfile.mkdtemp(prefix='.preparing-', dir=staging))
    plan_path = temporary / 'plan.json'
    plan_path.write_bytes(encoded(plan))
    assessment_distill.build(plan_path, temporary / 'dataset')
    os.rename(temporary, final)
    return final / 'dataset'


def deliver(dataset: Path, value: dict) -> dict:
    manifest = read(dataset / 'manifest.json', value['max_manifest_bytes'])
    training = read(dataset / 'train.jsonl', value['max_dataset_bytes'])
    if len(manifest) + len(training) > value['max_dataset_bytes']:
        raise ValueError('prepared dataset exceeds configured size')
    payload = encoded({'schema_version': 'lfg.teacher-dataset-transfer.v1',
                       'manifest_base64': base64.b64encode(manifest).decode(),
                       'train_base64': base64.b64encode(training).decode()})
    if len(payload) > value['max_transfer_bytes']:
        raise ValueError('transfer exceeds configured size')
    delivery = value['delivery']
    if (not isinstance(delivery.get('endpoint'), str) or not isinstance(delivery.get('allowed_hosts'), list)
            or not delivery['allowed_hosts'] or any(not isinstance(x, str) or not x for x in delivery['allowed_hosts'])
            or not isinstance(delivery.get('token_file'), str) or not Path(delivery['token_file']).is_absolute()
            or type(delivery.get('timeout_seconds')) is not int or delivery['timeout_seconds'] < 1):
        raise ValueError('invalid HTTPS delivery configuration')
    url = urllib.parse.urlsplit(delivery['endpoint'])
    if (url.scheme != 'https' or url.hostname not in delivery['allowed_hosts'] or url.username is not None
            or url.password is not None or url.path != '/api/datasets' or url.query or url.fragment):
        raise ValueError('delivery endpoint is not an approved HTTPS dataset endpoint')
    token = read(Path(delivery['token_file']), 512)
    if not 32 <= len(token) <= 512 or b'\n' in token or b'\r' in token:
        raise ValueError('invalid delivery credential')
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None
    request = urllib.request.Request(delivery['endpoint'], data=payload, method='POST',
        headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + token.decode('ascii')})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        with opener.open(request, timeout=delivery['timeout_seconds']) as stream:
            result = stream.read(value['max_manifest_bytes'] + 1)
    except urllib.error.URLError:
        raise ValueError('teacher dataset delivery failed') from None
    if len(result) > value['max_manifest_bytes']:
        raise ValueError('delivery confirmation exceeds configured limit')
    response = json.loads(result)
    if (not isinstance(response, dict) or response.get('dataset_id') != sha(manifest + b'\0' + training)
            or type(response.get('records')) is not int or response['records'] < 1):
        raise ValueError('teacher dataset delivery confirmation mismatch')
    return response


def run(config_path: Path, *, max_config_bytes: int = 1048576) -> dict:
    value, raw = config_file(config_path, max_config_bytes)
    lock_target = Path(value['staging_root']) / 'publisher'
    with train.training_run_lock(lock_target):
        dataset = prepare(value, raw)
        if config_file(config_path, max_config_bytes)[1] != raw:
            raise ValueError('publisher config changed')
        manifest = read(dataset / 'manifest.json', value['max_manifest_bytes'])
        training = read(dataset / 'train.jsonl', value['max_dataset_bytes'])
        dataset_id = sha(manifest + b'\0' + training)
        delivery = value['delivery']
        if not isinstance(delivery.get('token_file'), str) or not Path(delivery['token_file']).is_absolute():
            raise ValueError('delivery token path must be absolute')
        delivery_id = sha(encoded(delivery) + b'\0' + sha(read(Path(delivery['token_file']), 512)).encode())
        marker = Path(value['staging_root']) / 'last-delivery.json'
        interval = value.get('resend_interval_seconds', 0)
        if interval and marker.exists():
            previous = json.loads(read(marker, value['max_manifest_bytes']))
            if (previous.get('dataset_id') == dataset_id and previous.get('delivery_sha256') == delivery_id
                    and type(previous.get('sent_unix')) in (int, float)
                    and 0 <= time.time() - previous['sent_unix'] < interval):
                return {'dataset_id': dataset_id, 'records': previous['records'], 'recently_delivered': True}
        result = deliver(dataset, value)
        fd, temporary = tempfile.mkstemp(prefix='.delivery-', dir=marker.parent)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(encoded({'dataset_id': dataset_id, 'delivery_sha256': delivery_id,
                                  'records': result['records'], 'sent_unix': time.time()}))
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, marker)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--max-config-bytes', type=int, default=1048576)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.config, max_config_bytes=args.max_config_bytes)), flush=True)
    except (OSError, ValueError, KeyError, TypeError, UnicodeError, json.JSONDecodeError):
        raise SystemExit('teacher dataset publication failed; inspect private configuration and inputs') from None


if __name__ == '__main__':
    main()
