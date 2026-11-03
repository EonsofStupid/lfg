#!/usr/bin/env python3
"""Collect private, unreviewed assessments from a loopback teacher service."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import tempfile
import urllib.error
import urllib.parse
import urllib.request

import screen_private_corpus as screen
import train
from owned_sft_checkpoint import private_directory


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def encoded(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n').encode()


def write_new(path: Path, raw: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def config_file(path: Path) -> tuple[dict, bytes]:
    raw = screen._read(path, 65536, 'teacher_config_unavailable')
    value = screen._json(raw, 'invalid_teacher_config')
    ints = ('max_requests', 'max_request_bytes', 'max_http_response_bytes', 'max_response_bytes',
            'max_tokens', 'timeout_seconds', 'max_manifest_bytes')
    if (not isinstance(value, dict) or value.get('schema_version') != 'lfg.local-teacher-collection.v1'
            or any(type(value.get(k)) is not int or value[k] < 1 for k in ints)
            or any(not isinstance(value.get(k), str) or not value[k].strip()
                   for k in ('endpoint', 'model', 'teacher_identity', 'system_prompt', 'api_key_env'))
            or value.get('source_origin') not in ('owner_chat', 'owner_project')
            or not isinstance(value.get('required_fields'), list) or not value['required_fields']
            or any(not isinstance(k, str) or not k for k in value['required_fields'])
            or len(set(value['required_fields'])) != len(value['required_fields'])
            or type(value.get('temperature')) not in (float, int) or not math.isfinite(value['temperature'])
            or not 0 <= value['temperature'] <= 2 or type(value.get('json_mode')) is not bool):
        raise ValueError('invalid teacher collection configuration')
    url = urllib.parse.urlsplit(value['endpoint'])
    try:
        local = ipaddress.ip_address(url.hostname or '').is_loopback
        port = url.port
    except ValueError:
        raise ValueError('teacher endpoint must use a loopback IP') from None
    if (url.scheme != 'http' or not local or port is None or url.username is not None or url.password is not None
            or url.path != '/v1/chat/completions' or url.query or url.fragment):
        raise ValueError('teacher endpoint must be a loopback chat-completions URL')
    extensions = value.get('request_extensions', {})
    if (not isinstance(extensions, dict) or any(not isinstance(k, str) or not k for k in extensions)
            or set(extensions) & {'model', 'messages', 'max_tokens', 'temperature', 'response_format', 'stream'}):
        raise ValueError('invalid or reserved teacher request extension')
    if value.get('on_incomplete', 'fail') not in ('fail', 'skip'):
        raise ValueError('invalid incomplete-response policy')
    try:
        json.dumps(extensions, allow_nan=False)
    except (TypeError, ValueError):
        raise ValueError('teacher request extensions must be JSON values') from None
    return value, raw


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class TeacherHTTPError(ValueError):
    def __init__(self, status: int):
        self.status = status
        super().__init__('teacher HTTP request failed')


def ask(config: dict, text: str, key: str) -> str:
    payload = {'model': config['model'], 'messages': [{'role': 'system', 'content': config['system_prompt']},
                {'role': 'user', 'content': text}], 'max_tokens': config['max_tokens'],
               'temperature': config['temperature']}
    if config['json_mode']:
        payload['response_format'] = {'type': 'json_object'}
    payload.update(config.get('request_extensions', {}))
    request = urllib.request.Request(config['endpoint'], data=encoded(payload), method='POST',
                                     headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        with opener.open(request, timeout=config['timeout_seconds']) as response:
            raw = response.read(config['max_http_response_bytes'] + 1)
    except urllib.error.HTTPError as error:
        raise TeacherHTTPError(error.code) from None
    if len(raw) > config['max_http_response_bytes']:
        raise ValueError('teacher HTTP response exceeds configured limit')
    result = screen._json(raw, 'invalid_teacher_response')
    choices = result.get('choices') if isinstance(result, dict) else None
    if (not isinstance(result, dict) or result.get('model') != config['model'] or not isinstance(choices, list) or len(choices) != 1
            or not isinstance(choices[0], dict) or choices[0].get('finish_reason') != 'stop'
            or not isinstance(choices[0].get('message'), dict)
            or not isinstance(choices[0]['message'].get('content'), str)):
        raise ValueError('teacher response is incomplete or identifies another model')
    content = choices[0]['message']['content']
    if not content.strip() or len(content.encode()) > config['max_response_bytes']:
        raise ValueError('teacher assessment exceeds configured limit or is empty')
    return content


def clean_response(value: object, rules, limits: dict) -> object:
    if isinstance(value, str):
        return screen._screen(value, rules, limits)[0] if value.strip() else value
    if isinstance(value, list):
        return [clean_response(item, rules, limits) for item in value]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError('invalid assessment field')
        result = {}
        for key, item in value.items():
            clean_key = screen._screen(key, rules, limits)[0] if key.strip() else key
            if clean_key in result:
                raise ValueError('redaction made assessment keys ambiguous')
            result[clean_key] = clean_response(item, rules, limits)
        return result
    if value is None or type(value) in (bool, int, float):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError('non-finite assessment')
        return value
    raise ValueError('invalid assessment value')


def collect(selection: Path, config_path: Path, policy_path: Path, output: Path, *, resume: bool = False) -> dict:
    config, config_raw = config_file(config_path)
    key = os.environ.get(config['api_key_env'])
    if not key or '\n' in key or '\r' in key:
        raise ValueError('teacher API key is unavailable')
    policy, rules, policy_sha = screen.load_policy(policy_path, config['max_manifest_bytes'])
    selection_raw = screen._read(selection / 'manifest.json', config['max_manifest_bytes'], 'selection_manifest_unavailable')
    manifest = screen._json(selection_raw, 'invalid_selection_manifest')
    source_raw = screen._read(selection / 'requests.jsonl', config['max_request_bytes'] * config['max_requests'], 'requests_unavailable')
    if (not isinstance(manifest, dict) or manifest.get('schema_version') != 'lfg.teacher-request-selection-result.v1'
            or manifest.get('requests_sha256') != sha(source_raw) or manifest.get('test_opened') is not False
            or manifest.get('validation_opened') is not False):
        raise ValueError('invalid selected request identity')
    rows = [screen._json(line, 'invalid_selected_request') for line in source_raw.splitlines()]
    if not rows or len(rows) > config['max_requests'] or len(rows) != manifest.get('requests'):
        raise ValueError('invalid selected request count')
    ids = set()
    for row in rows:
        if (not isinstance(row, dict) or row.get('split') != 'train' or row.get('review_status') != 'unreviewed'
                or row.get('outcome') != 'unknown' or any(not isinstance(row.get(k), str) or not row[k].strip()
                   for k in ('id', 'text', 'project', 'project_group', 'screened_text_sha256', 'source_text_sha256_claim'))
                or row['id'] in ids or sha(row['text'].encode()) != row['screened_text_sha256']
                or len(row['text'].encode()) > config['max_request_bytes']
                or len(row['source_text_sha256_claim']) != 64
                or any(c not in '0123456789abcdef' for c in row['source_text_sha256_claim'])):
            raise ValueError('invalid selected train request')
        if screen._screen(row['text'], rules, policy['limits'])[0] != row['text']:
            raise ValueError('request needs additional screening')
        ids.add(row['id'])
    private_directory(output)
    identity = {'schema_version': 'lfg.local-teacher-run.v1', 'config_sha256': sha(config_raw),
                'selection_manifest_sha256': sha(selection_raw), 'requests_sha256': sha(source_raw),
                'policy_sha256': policy_sha, 'collector_sha256': sha(Path(__file__).read_bytes()),
                'screen_code_sha256': sha(Path(screen.__file__).read_bytes())}
    # The OS lock is held across all requests; the marker is not removed after exit.
    with train.training_run_lock(output):
        if not output.exists():
            if resume:
                raise ValueError('no run exists to resume')
            output.mkdir(mode=0o700)
            write_new(output / 'run.json', encoded(identity))
            (output / 'records').mkdir(mode=0o700)
        elif not resume or not output.is_dir() or output.is_symlink():
            raise ValueError('existing output requires explicit resume')
        if screen._read(output / 'run.json', config['max_manifest_bytes'], 'run_identity_unavailable') != encoded(identity):
            raise ValueError('run identity changed')
        records_dir = output / 'records'
        if not records_dir.is_dir() or records_dir.is_symlink():
            raise ValueError('invalid record directory')
        failure_dir = output / 'failures'
        if config.get('on_incomplete', 'fail') == 'skip':
            failure_dir.mkdir(mode=0o700, exist_ok=True)
            if failure_dir.is_symlink() or not failure_dir.is_dir():
                raise ValueError('invalid failure directory')
        traces, failures = [], []
        for index, row in enumerate(rows):
            record_path = records_dir / f'{index:06d}.json'
            failure_path = failure_dir / f'{index:06d}.json'
            if record_path.is_symlink():
                raise ValueError('record must not be a symbolic link')
            if failure_path.is_symlink() or (record_path.exists() and failure_path.exists()):
                raise ValueError('ambiguous or symbolic failure record')
            failed = {'schema_version': 'lfg.local-teacher-failure.v1', 'id': row['id'],
                      'request_sha256': row['screened_text_sha256'], 'reason': 'incomplete_or_invalid_response'}
            if failure_path.exists():
                if config.get('on_incomplete', 'fail') != 'skip' or screen._read(
                        failure_path, config['max_manifest_bytes'], 'failure_unavailable') != encoded(failed):
                    raise ValueError('failure record does not match request')
                failures.append(failed)
                continue
            if record_path.exists():
                envelope = screen._json(screen._read(record_path, config['max_http_response_bytes'], 'record_unavailable'), 'invalid_record')
                trace = envelope.get('trace') if isinstance(envelope, dict) else None
                if not isinstance(trace, dict) or envelope.get('trace_sha256') != sha(encoded(trace)):
                    raise ValueError('record digest mismatch')
            else:
                try:
                    response = ask(config, row['text'], key)
                    parsed = screen._json(response.encode(), 'invalid_assessment_json')
                    if not isinstance(parsed, dict) or any(field not in parsed for field in config['required_fields']):
                        raise ValueError('teacher response lacks required fields')
                    parsed = clean_response(parsed, rules, policy['limits'])
                    if any(field not in parsed for field in config['required_fields']):
                        raise ValueError('screening removed a required assessment field')
                    clean = encoded(parsed).decode().rstrip('\n')
                    if len(clean.encode()) > config['max_response_bytes']:
                        raise ValueError('screened teacher assessment exceeds configured limit')
                except TeacherHTTPError:
                    raise
                except ValueError:
                    if config.get('on_incomplete', 'fail') != 'skip':
                        raise
                    fd, temporary = tempfile.mkstemp(prefix='.writing-', dir=failure_dir)
                    with os.fdopen(fd, 'wb') as stream:
                        stream.write(encoded(failed)); stream.flush(); os.fsync(stream.fileno())
                    os.replace(temporary, failure_path)
                    failures.append(failed)
                    continue
                trace = {'id': row['id'], 'request': row['text'], 'response': clean,
                         'teacher': config['teacher_identity'], 'review_status': 'unreviewed', 'outcome': 'unknown',
                         'source_ref': row['id'], 'source_sha256': row['source_text_sha256_claim']}
                envelope = {'schema_version': 'lfg.local-teacher-record.v1', 'trace': trace,
                            'trace_sha256': sha(encoded(trace)),
                            'source_claims': {k: row.get(k) for k in ('project', 'project_group', 'family',
                                              'candidate_file_sha256_claim', 'screened_text_sha256')}}
                fd, temporary = tempfile.mkstemp(prefix='.writing-', dir=records_dir)
                with os.fdopen(fd, 'wb') as stream:
                    stream.write(encoded(envelope)); stream.flush(); os.fsync(stream.fileno())
                os.replace(temporary, record_path)
            if (trace.get('id') != row['id'] or trace.get('request') != row['text']
                    or trace.get('teacher') != config['teacher_identity'] or trace.get('review_status') != 'unreviewed'
                    or trace.get('outcome') != 'unknown' or trace.get('source_ref') != row['id']
                    or trace.get('source_sha256') != row['source_text_sha256_claim']
                    or not isinstance(trace.get('response'), str)
                    or envelope.get('schema_version') != 'lfg.local-teacher-record.v1'
                    or envelope.get('source_claims') != {k: row.get(k) for k in ('project', 'project_group', 'family',
                                                       'candidate_file_sha256_claim', 'screened_text_sha256')}):
                raise ValueError('saved trace does not match selected request')
            saved = screen._json(trace.get('response', '').encode(), 'invalid_saved_assessment')
            if (not isinstance(saved, dict) or any(field not in saved for field in config['required_fields'])
                    or clean_response(saved, rules, policy['limits']) != saved):
                raise ValueError('saved assessment is invalid or needs screening')
            traces.append((row, trace))
        if (config_file(config_path)[1] != config_raw or screen.load_policy(policy_path, config['max_manifest_bytes'])[2] != policy_sha
                or screen._read(selection / 'manifest.json', config['max_manifest_bytes'], 'selection_changed') != selection_raw
                or screen._read(selection / 'requests.jsonl', config['max_request_bytes'] * config['max_requests'], 'requests_changed') != source_raw
                or sha(Path(__file__).read_bytes()) != identity['collector_sha256']
                or sha(Path(screen.__file__).read_bytes()) != identity['screen_code_sha256']):
            raise ValueError('collection inputs changed')
        # Per-project trace files are consumable by assessment_distill.build.
        groups = {}
        for row, trace in traces:
            groups.setdefault((row['project'], row['project_group']), []).append(trace)
        files = []
        for index, ((project, group), items) in enumerate(sorted(groups.items())):
            name = f'trace-{index:06d}.jsonl'; raw = b''.join(encoded(item) for item in items)
            path = output / name
            if path.exists():
                if path.is_symlink() or screen._read(path, config['max_http_response_bytes'] * config['max_requests'], 'trace_unavailable') != raw:
                    raise ValueError('existing trace output changed')
            else:
                write_new(path, raw)
            files.append({'path': name, 'sha256': sha(raw), 'project': project, 'project_group': group,
                          'split': 'train', 'origin': config['source_origin'], 'records': len(items)})
        report = {**identity, 'schema_version': 'lfg.local-teacher-collection-result.v1', 'files': files,
                  'requests': len(traces), 'projects': dict(sorted(Counter(row['project'] for row, _ in traces).items())),
                  'requested_requests': len(rows), 'failed_requests': len(failures),
                  'failure_records_sha256': sha(b''.join(encoded(item) for item in failures)),
                  'teacher_identity': config['teacher_identity'], 'review_status': 'unreviewed', 'outcome': 'unknown',
                  'teacher_identity_verified': False, 'redaction_complete': False, 'validation_opened': False,
                  'test_opened': False, 'assessment_quality_evaluated': False, 'promotion_eligible': False}
        final = output / 'manifest.json'
        if final.exists():
            if final.is_symlink() or screen._read(final, config['max_manifest_bytes'], 'manifest_unavailable') != encoded(report):
                raise ValueError('existing collection manifest changed')
        else:
            write_new(final, encoded(report))
        return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('selection', 'config', 'policy', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    try:
        result = collect(args.selection, args.config, args.policy, args.output, resume=args.resume)
        print(json.dumps({k: result[k] for k in ('requested_requests', 'requests', 'failed_requests',
                                                 'projects', 'review_status', 'outcome', 'test_opened')}))
    except TeacherHTTPError as error:
        raise SystemExit(f'Local teacher returned HTTP {error.status}; see service diagnostics') from None
    except (OSError, ValueError, TypeError, KeyError, UnicodeError) as error:
        raise SystemExit(f'Local teacher collection failed ({type(error).__name__})') from None
