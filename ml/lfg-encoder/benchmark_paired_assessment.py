#!/usr/bin/env python3
"""Compare deterministic and experimental owned LFG HTTP assessment paths."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import http.client
import ipaddress
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
import screen_private_corpus as screening


SCHEMA = 'lfg.paired-http-benchmark-config.v1'
REPORT_SCHEMA = 'lfg.paired-http-assessment-benchmark.v1'
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FIELDS = {
    'schema_version', 'target_base_url', 'owned_token_env', 'owned_token_file',
    'dataset_path', 'dataset_split', 'dataset_source_ref', 'selection_seed', 'screening_policy_path',
    'max_records', 'max_source_records', 'max_dataset_bytes', 'max_input_bytes',
    'max_request_bytes', 'max_response_bytes', 'max_report_bytes',
    'client_timeout_seconds', 'server_timeout_overhead_ms', 'warmup_rounds',
    'measured_rounds', 'declared_subscriptions', 'first_order',
}
MAX_CONFIG_BYTES = 64 * 1024
MAX_SCREEN_POLICY_BYTES = 1024 * 1024
ID_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$')
ENV_RE = re.compile(r'^[A-Z][A-Z0-9_]{0,63}$')
SHA_RE = re.compile(r'^[0-9a-f]{64}$')
KNOWN_OWNED_STATUSES = {
    'ok', 'truncated', 'timeout', 'busy', 'invalid_input', 'context_limit',
    'incomplete_output', 'malformed_output', 'inference_error', 'unavailable',
}
ENGINE_ID_FIELDS = (
    'id', 'version', 'sha256', 'backend', 'modelSha256', 'modelManifestSha256',
    'trainingManifestSha256', 'framingSha256', 'runtimeConfigSha256',
    'serviceConfigSha256', 'cachedManifestSha256',
)


class BenchmarkError(Exception):
    """An error carrying only a stable, non-sensitive public code."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class ClientTimeout(BenchmarkError):
    def __init__(self, code: str, elapsed_ns: int):
        super().__init__(code)
        self.elapsed_ns = elapsed_ns


@dataclass
class HttpResult:
    status: int | None
    payload: dict | None
    elapsed_ns: int
    response_code: str | None = None


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_bounded(path: Path, limit: int, error_code: str) -> bytes:
    try:
        with path.open('rb') as source:
            raw = source.read(limit + 1)
    except OSError:
        raise BenchmarkError(error_code) from None
    if len(raw) > limit:
        raise BenchmarkError(error_code)
    return raw


def is_int(value: object, minimum: int = 0, maximum: int | None = None) -> bool:
    return type(value) is int and value >= minimum and (maximum is None or value <= maximum)


def inside(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def _resolve_outside_repo(value: str, code: str) -> Path:
    path = Path(value).expanduser()
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        raise BenchmarkError(code) from None
    if inside(resolved, REPO_ROOT):
        raise BenchmarkError('private_input_inside_repository')
    return resolved


def validate_loopback_url(value: object) -> str:
    if not isinstance(value, str) or len(value) > 512:
        raise BenchmarkError('invalid_loopback_target')
    try:
        parsed = urllib.parse.urlsplit(value)
        address = ipaddress.ip_address(parsed.hostname or '')
        port = parsed.port or 80
    except (ValueError, TypeError):
        raise BenchmarkError('invalid_loopback_target') from None
    if (parsed.scheme != 'http' or not address.is_loopback or parsed.username is not None
            or parsed.password is not None or parsed.query or parsed.fragment
            or parsed.path not in ('', '/') or not 1 <= port <= 65535):
        raise BenchmarkError('invalid_loopback_target')
    return f'http://{parsed.netloc}'


def load_config(path: Path) -> tuple[dict, bytes, str]:
    path = _resolve_outside_repo(str(path), 'config_unavailable')
    try:
        with path.open('rb') as source:
            raw = source.read(MAX_CONFIG_BYTES + 1)
        if len(raw) > MAX_CONFIG_BYTES:
            raise BenchmarkError('config_size_limit')
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise BenchmarkError('invalid_config') from None
    if not isinstance(value, dict) or not CONFIG_FIELDS.issubset(value) or value.get('schema_version') != SCHEMA:
        raise BenchmarkError('unsupported_config')
    for name, minimum, maximum in (
        ('selection_seed', 0, sys.maxsize), ('max_records', 1, sys.maxsize),
        ('max_source_records', 1, sys.maxsize), ('max_dataset_bytes', 1, sys.maxsize - 1),
        ('max_input_bytes', 1, sys.maxsize - 1), ('max_request_bytes', 1, sys.maxsize - 1),
        ('max_response_bytes', 1, sys.maxsize - 1), ('max_report_bytes', 1, sys.maxsize),
        ('server_timeout_overhead_ms', 1, sys.maxsize), ('warmup_rounds', 0, sys.maxsize),
        ('measured_rounds', 1, sys.maxsize),
    ):
        if not is_int(value.get(name), minimum, maximum):
            raise BenchmarkError('invalid_config')
    timeout = value.get('client_timeout_seconds')
    if type(timeout) not in (int, float) or timeout <= 0:
        raise BenchmarkError('invalid_config')
    try:
        if not math.isfinite(float(timeout)) or timeout > threading.TIMEOUT_MAX:
            raise BenchmarkError('invalid_config')
    except (OverflowError, ValueError):
        raise BenchmarkError('invalid_config') from None
    if not is_int(value.get('max_source_records'), value.get('max_records')):
        raise BenchmarkError('invalid_config')
    if value['warmup_rounds'] + value['measured_rounds'] > sys.maxsize:
        raise BenchmarkError('invalid_config')
    if value.get('dataset_split') not in ('train', 'training', 'dev', 'development') \
            or not isinstance(value.get('dataset_source_ref'), str) \
            or not ID_RE.fullmatch(value['dataset_source_ref']):
        raise BenchmarkError('development_split_required')
    if not isinstance(value.get('dataset_path'), str) or not value['dataset_path']:
        raise BenchmarkError('invalid_config')
    if not isinstance(value.get('screening_policy_path'), str) or not value['screening_policy_path']:
        raise BenchmarkError('invalid_config')
    if (not isinstance(value.get('declared_subscriptions'), list)
            or any(not isinstance(item, str) or not item for item in value['declared_subscriptions'])
            or len(set(value['declared_subscriptions'])) != len(value['declared_subscriptions'])):
        raise BenchmarkError('invalid_config')
    if value.get('first_order') not in ('deterministic_first', 'owned_first'):
        raise BenchmarkError('invalid_config')
    env_name, token_file = value.get('owned_token_env'), value.get('owned_token_file')
    if (env_name is None) == (token_file is None):
        raise BenchmarkError('invalid_config')
    if env_name is not None and (not isinstance(env_name, str) or not ENV_RE.fullmatch(env_name)):
        raise BenchmarkError('invalid_config')
    if token_file is not None and (not isinstance(token_file, str) or not token_file):
        raise BenchmarkError('invalid_config')
    try:
        validate_loopback_url(value['target_base_url'])
    except KeyError:
        raise BenchmarkError('invalid_config') from None
    # Unknown extension keys remain inert: only the documented contract fields
    # are passed to benchmark behavior, while the complete config is hashed.
    return {key: value[key] for key in CONFIG_FIELDS}, raw, digest(raw)


def load_token(config: dict) -> str:
    if config['owned_token_env'] is not None:
        token = os.environ.get(config['owned_token_env'], '')
    else:
        path = _resolve_outside_repo(config['owned_token_file'], 'owned_token_unavailable')
        try:
            info = path.stat()
            if path.is_symlink() or info.st_size > 4096:
                raise BenchmarkError('owned_token_unavailable')
            if os.name == 'posix' and (info.st_uid != os.geteuid() or info.st_mode & 0o077):
                raise BenchmarkError('owned_token_permissions_unverified')
            if os.name == 'nt':
                verify_private_windows_acl(path)
            token = path.read_text(encoding='ascii').strip()
        except BenchmarkError:
            raise
        except (OSError, UnicodeError):
            raise BenchmarkError('owned_token_unavailable') from None
    if (len(token) < 32 or len(token) > 4096 or any(c.isspace() or ord(c) < 32 for c in token)
            or len(set(token)) < 12):
        raise BenchmarkError('owned_token_unavailable')
    return token


def verify_private_windows_acl(path: Path) -> None:
    """Fail closed unless a PowerShell ACL check confirms only private principals."""
    script = r'''
$ErrorActionPreference = 'Stop'
$p = $env:LFG_PRIVATE_CHECK_PATH
$acl = Get-Acl -LiteralPath $p
$userSid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$ownerSid = $acl.GetOwner([Security.Principal.SecurityIdentifier]).Value
if ($ownerSid -ne $userSid) { exit 31 }
$allowed = @($userSid, 'S-1-5-18', 'S-1-5-32-544')
foreach ($ace in $acl.Access) {
  if ($ace.AccessControlType -ne [Security.AccessControl.AccessControlType]::Allow) { continue }
  try { $sid = $ace.IdentityReference.Translate([Security.Principal.SecurityIdentifier]).Value }
  catch { exit 32 }
  if ($sid -notin $allowed) { exit 33 }
}
exit 0
'''
    try:
        result = subprocess.run(
            ['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', script],
            env={**os.environ, 'LFG_PRIVATE_CHECK_PATH': str(path)},
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise BenchmarkError('private_output_permissions_unverified') from None
    if result.returncode != 0:
        raise BenchmarkError('private_output_permissions_unverified')


def load_rows(config: dict, policy: dict, rules: list) -> tuple[list[dict[str, str]], str, str, str, dict, int]:
    path = _resolve_outside_repo(config['dataset_path'], 'dataset_unavailable')
    try:
        info = path.stat()
        if info.st_size > config['max_dataset_bytes']:
            raise BenchmarkError('dataset_size_limit')
        with path.open('rb') as source:
            raw = source.read(config['max_dataset_bytes'] + 1)
        if len(raw) > config['max_dataset_bytes']:
            raise BenchmarkError('dataset_size_limit')
        decoded = raw.decode('utf-8')
        parsed = [json.loads(line) for line in decoded.splitlines() if line.strip()]
    except BenchmarkError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise BenchmarkError('dataset_invalid') from None
    if not parsed or len(parsed) > config['max_source_records']:
        raise BenchmarkError('dataset_record_limit')
    rows: list[dict[str, str]] = []
    identifiers: set[str] = set()
    for record in parsed:
        if (not isinstance(record, dict) or not isinstance(record.get('id'), str)
                or not record['id'].strip() or not isinstance(record.get('text'), str)
                or not record['text'].strip() or record['id'] in identifiers):
            raise BenchmarkError('dataset_invalid')
        identifiers.add(record['id'])
        rows.append({'id': record['id'], 'text': record['text']})
    if len(rows) > config['max_records']:
        rows = random.Random(config['selection_seed']).sample(rows, config['max_records'])
    source_selected_hash = digest(json.dumps(
        [{'id_sha256': digest(row['id'].encode()), 'text_sha256': digest(row['text'].encode('utf-8'))}
         for row in rows], sort_keys=True, separators=(',', ':')).encode())
    redaction_counts: Counter = Counter()
    changed_records = 0
    for row in rows:
        raw_text = row['text'].encode('utf-8')
        if len(raw_text) > config['max_input_bytes']:
            raise BenchmarkError('input_size_limit')
        try:
            screened, found = screening._screen(row['text'], rules, policy['limits'])
        except screening.ScreenError as error:
            raise BenchmarkError(f'screening_{error}') from None
        if screened != row['text']:
            changed_records += 1
        row['text'] = screened
        redaction_counts.update(found)
        if len(screened.encode('utf-8')) > config['max_input_bytes']:
            raise BenchmarkError('screened_input_size_limit')
    selected_digest = digest(json.dumps(
        [{'id_sha256': digest(row['id'].encode()), 'text_sha256': digest(row['text'].encode('utf-8'))}
         for row in rows], sort_keys=True, separators=(',', ':')).encode())
    return rows, digest(raw), source_selected_hash, selected_digest, {
        'changed_records': changed_records, 'redaction_counts': dict(sorted(redaction_counts.items()))}, len(parsed)


def load_screening_policy(config: dict) -> tuple[dict, list, str]:
    path = _resolve_outside_repo(config['screening_policy_path'], 'screening_policy_unavailable')
    try:
        return screening.load_policy(path, MAX_SCREEN_POLICY_BYTES)
    except screening.ScreenError as error:
        raise BenchmarkError(f'screening_policy_{error}') from None


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def make_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())


def http_json(opener: urllib.request.OpenerDirector, url: str, *, method: str,
              payload: dict | None, token: str | None, timeout: float,
              max_request_bytes: int, max_response_bytes: int) -> HttpResult:
    body = None if payload is None else json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    if body is not None and len(body) > max_request_bytes:
        raise BenchmarkError('request_size_limit')
    headers = {'Accept': 'application/json', 'Cache-Control': 'no-store'}
    if body is not None:
        headers['Content-Type'] = 'application/json'
    if token is not None:
        headers['X-LFG-Session-Token'] = token
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    started = time.perf_counter_ns()
    try:
        with opener.open(request, timeout=timeout) as response:
            status = response.status
            raw = response.read(max_response_bytes + 1)
    except urllib.error.HTTPError as error:
        try:
            error.read(max_response_bytes + 1)
        except (socket.timeout, TimeoutError):
            raise ClientTimeout('client_timeout_contamination_unknown', time.perf_counter_ns() - started) from None
        except (OSError, http.client.HTTPException):
            raise BenchmarkError('http_transport_error') from None
        finally:
            error.close()
        elapsed = time.perf_counter_ns() - started
        return HttpResult(error.code, None, elapsed, f'http_{error.code}')
    except (socket.timeout, TimeoutError):
        raise ClientTimeout('client_timeout_contamination_unknown', time.perf_counter_ns() - started) from None
    except urllib.error.URLError as error:
        if isinstance(error.reason, (socket.timeout, TimeoutError)):
            raise ClientTimeout('client_timeout_contamination_unknown', time.perf_counter_ns() - started) from None
        raise BenchmarkError('http_transport_error') from None
    except OSError:
        raise BenchmarkError('http_transport_error') from None
    except http.client.HTTPException:
        raise BenchmarkError('http_transport_error') from None
    elapsed = time.perf_counter_ns() - started
    if len(raw) > max_response_bytes:
        return HttpResult(status, None, elapsed, 'response_too_large')
    try:
        payload_value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError):
        return HttpResult(status, None, elapsed, 'malformed_response')
    if not isinstance(payload_value, dict):
        return HttpResult(status, None, elapsed, 'malformed_response')
    return HttpResult(status, payload_value, elapsed)


def percentile(samples: list[int], percent: int) -> int | None:
    if not samples:
        return None
    ordered = sorted(samples)
    return ordered[math.ceil(percent * len(ordered) / 100) - 1]


def summarize(samples: list[int], denominator: int) -> dict:
    return {
        'count': len(samples), 'denominator': denominator,
        'minimum_ns': min(samples) if samples else None,
        'median_ns': statistics.median(samples) if samples else None,
        'p95_ns': percentile(samples, 95),
        'maximum_ns': max(samples) if samples else None,
    }


def model_identity(value: object) -> dict | None:
    if not isinstance(value, dict) or not isinstance(value.get('id'), str):
        return None
    return {key: value[key] for key in ENGINE_ID_FIELDS
            if isinstance(value.get(key), str) or type(value.get(key)) in (int, bool)}


def validated_status_identity(opener, base_url, token, cfg) -> tuple[dict, dict]:
    status_result = http_json(
        opener, base_url + '/status', method='GET', payload=None, token=None,
        timeout=cfg['client_timeout_seconds'], max_request_bytes=cfg['max_request_bytes'],
        max_response_bytes=cfg['max_response_bytes'])
    owned_result = http_json(
        opener, base_url + '/api/lfg/assess/owned/status', method='GET', payload=None, token=token,
        timeout=cfg['client_timeout_seconds'], max_request_bytes=cfg['max_request_bytes'],
        max_response_bytes=cfg['max_response_bytes'])
    if status_result.status != 200 or status_result.payload is None:
        raise BenchmarkError('deterministic_status_unavailable')
    if owned_result.status != 200 or owned_result.payload is None:
        raise BenchmarkError('owned_status_unavailable')
    server = status_result.payload
    owned = owned_result.payload
    engine = model_identity(owned.get('engine'))
    limits = owned.get('limits')
    if (owned.get('enabled') is not True or owned.get('status') != 'ready' or engine is None
            or not isinstance(limits, dict) or not is_int(limits.get('timeoutMs'), 1, 600000)):
        raise BenchmarkError('owned_service_not_ready')
    required_timeout = limits['timeoutMs'] / 1000 + cfg['server_timeout_overhead_ms'] / 1000
    if cfg['client_timeout_seconds'] <= required_timeout:
        raise BenchmarkError('client_timeout_not_longer_than_server_bound')
    deterministic_identity = {
        'phase': server.get('phase') if isinstance(server.get('phase'), str) else None,
        'operationsDigest': server.get('operationsDigest') if isinstance(server.get('operationsDigest'), str) else None,
        'encoder': model_identity(server.get('encoder')),
    }
    if deterministic_identity['encoder'] is None:
        raise BenchmarkError('deterministic_identity_unavailable')
    return deterministic_identity, {'engine': engine, 'limits': {key: limits.get(key) for key in (
        'backend', 'timeoutMs', 'maxRequestBytes', 'maxContextBytes', 'maxGeneratedBytes',
        'maxConcurrent', 'workerThreads', 'maxCacheStateBytesPerRequest', 'maxCachedGraphBytes')
        if isinstance(limits.get(key), (str, int, float, bool))}}


def validate_assessment(payload: dict, session_id: str, revision: int, identity: dict | None) -> tuple[str, dict | None, int | None]:
    if (payload.get('schemaVersion') != 'lfg.assessment.v1' or payload.get('sessionId') != session_id
            or not is_int(payload.get('revision'), 0, sys.maxsize) or payload.get('revision') != revision
            or not isinstance(payload.get('configDigest'), str)
            or not SHA_RE.fullmatch(payload['configDigest']) or not isinstance(payload.get('configRevision'), str)
            or not is_int(payload.get('durationMicros'), 0, sys.maxsize)):
        return 'malformed_response', identity, None
    next_identity = {'configRevision': payload['configRevision'], 'configDigest': payload['configDigest']}
    if identity is not None and identity != next_identity:
        raise BenchmarkError('deterministic_identity_changed')
    return 'ok', next_identity, payload['durationMicros']


def validate_owned(payload: dict, expected_engine: dict) -> tuple[str, int | None]:
    status = payload.get('status')
    if (payload.get('schemaVersion') != 'lfg.owned-assessment.v1'
            or not isinstance(status, str) or status not in KNOWN_OWNED_STATUSES
            or 'projectContextSha256' in payload):
        return 'malformed_response', None
    if model_identity(payload.get('engine')) != expected_engine:
        raise BenchmarkError('owned_identity_changed')
    measurement = payload.get('measurement')
    handler_ms = None
    if isinstance(measurement, dict) and measurement.get('schemaVersion') == 'lfg.owned-run-measurement.v1':
        if (measurement.get('status') != status or measurement.get('engineId') != expected_engine['id']
                or not is_int(measurement.get('handlerElapsedMs'))):
            return 'malformed_response', None
        handler_ms = measurement['handlerElapsedMs']
    return status, handler_ms


def safe_status(result: HttpResult) -> str:
    if result.status is None:
        return result.response_code or 'transport_error'
    if result.status != 200:
        return f'http_{result.status}'
    return result.response_code or 'malformed_response'


def empty_path_stats() -> dict:
    return {'status_counts': Counter(), 'warmup_status_counts': Counter(),
            'latency_ns': [], 'handler_time_us': [], 'handler_missing': 0,
            'attempts': 0, 'completed_responses': 0}


def record_response(stats: dict, phase: str, status: str, result: HttpResult,
                    handler_value: int | None, handler_scale: int) -> None:
    stats['attempts'] += 1
    (stats['warmup_status_counts'] if phase == 'warmup' else stats['status_counts'])[status] += 1
    if result.status is not None:
        stats['completed_responses'] += 1
        stats['latency_ns'].append(result.elapsed_ns)
    if phase == 'measured':
        if handler_value is None:
            stats['handler_missing'] += 1
        else:
            stats['handler_time_us'].append(handler_value * handler_scale)


def public_stats(stats: dict) -> dict:
    return {
        'attempts': stats['attempts'],
        'completed_responses': stats['completed_responses'],
        'status_counts': dict(sorted(stats['status_counts'].items())),
        'warmup_status_counts': dict(sorted(stats['warmup_status_counts'].items())),
        'end_to_end_latency': summarize(stats['latency_ns'], stats['completed_responses']),
        'server_handler_time_us': summarize(stats['handler_time_us'], stats['attempts']),
        'server_handler_time_missing': stats['handler_missing'],
    }


def build_report_base(cfg: dict, config_hash: str, dataset_hash: str,
                      source_selected_hash: str, selected_hash: str, screening_changes: dict,
                      policy_hash: str, source_record_count: int, rows: list[dict],
                      deterministic_id: dict, owned_id: dict) -> dict:
    return {
        'schema_version': REPORT_SCHEMA,
        'complete': False,
        'abort_code': None,
        'contamination_state': 'clean',
        'promotion_eligible': False,
        'quality_evaluated': False,
        'source': {
            'source_ref': cfg['dataset_source_ref'],
            'split': cfg['dataset_split'],
            'split_basis': 'operator_configured_development_claim; not independently verified; no holdout determination',
            'screening_policy_sha256': policy_hash,
            'screening': 'operator-supplied policy applied locally; does not verify prior screening or labels',
            'screening_changes': screening_changes,
            'dataset_sha256': dataset_hash,
            'source_selected_id_text_digest': source_selected_hash,
            'screened_selected_id_text_digest': selected_hash,
            'source_records': source_record_count,
            'selected_records': len(rows),
            'selection': 'seeded sample without replacement' if len(rows) < source_record_count else 'all bounded source records',
            'selection_seed': cfg['selection_seed'],
        },
        'service': {
            'target': cfg['target_base_url'],
            'deterministic': deterministic_id,
            'owned': owned_id,
            'server_binary_sha256': None,
            'server_binary_identity_basis': 'not attested by the HTTP contracts',
        },
        'settings': {
            'benchmark_config_sha256': config_hash,
            'warmup_rounds_per_record': cfg['warmup_rounds'],
            'measured_rounds_per_record': cfg['measured_rounds'],
            'declared_subscriptions_sent_to_deterministic': cfg['declared_subscriptions'],
            'owned_context': 'none',
            'owned_request_text_matches_deterministic_text': True,
            'order': 'single-flight alternating pairs; configured first path per pair cycle',
            'client_timeout_seconds': cfg['client_timeout_seconds'],
            'owned_server_timeout_ms': owned_id['limits']['timeoutMs'],
            'client_timeout_overhead_ms': cfg['server_timeout_overhead_ms'],
            'max_input_bytes': cfg['max_input_bytes'],
            'max_request_bytes': cfg['max_request_bytes'],
            'max_response_bytes': cfg['max_response_bytes'],
        },
        'measurement_scope': {
            'end_to_end': 'client timer starts immediately before HTTP open after JSON/header preparation; stops after bounded response body read, including bounded HTTP error bodies',
            'deterministic_handler': 'endpoint-reported durationMicros; separate from client transport time',
            'owned_handler': 'endpoint-reported handlerElapsedMs; starts after Axum body extraction and ends before response serialization/context-digest attachment; may include child startup, not model-only time',
            'cpu': None,
            'cpu_unavailable_reason': 'No separately configured and identity-verified isolated cgroup interval was measured.',
            'cpu_cycles': None,
            'cpu_cycles_unavailable_reason': 'No separately configured and identity-verified hardware-counter interval was measured.',
            'memory': None,
            'memory_unavailable_reason': 'No per-process or isolated interval memory measurement was established.',
            'hardware': {
                'platform': platform.platform(), 'machine': platform.machine(),
                'processor': platform.processor() or None, 'logical_cpus': os.cpu_count(),
                'python_version': platform.python_version(),
            },
        },
        'paths': {'deterministic': None, 'owned': None},
    }


def verify_run_inputs(dataset_path: Path, dataset_hash: str, config_path: Path,
                      config_raw: bytes, config_hash: str, policy_path: Path,
                      policy_hash: str, cfg: dict) -> None:
    dataset_raw = read_bounded(dataset_path, cfg['max_dataset_bytes'], 'benchmark_input_changed')
    final_config_raw = read_bounded(config_path, MAX_CONFIG_BYTES, 'benchmark_input_changed')
    policy_raw = read_bounded(policy_path, MAX_SCREEN_POLICY_BYTES, 'benchmark_input_changed')
    if (digest(dataset_raw) != dataset_hash or digest(final_config_raw) != config_hash
            or final_config_raw != config_raw or digest(policy_raw) != policy_hash):
        raise BenchmarkError('benchmark_input_changed')


def run_benchmark(config_path: Path) -> dict:
    cfg, config_raw, config_hash = load_config(config_path)
    base_url = validate_loopback_url(cfg['target_base_url'])
    dataset_path = _resolve_outside_repo(cfg['dataset_path'], 'dataset_unavailable')
    policy_path = _resolve_outside_repo(cfg['screening_policy_path'], 'screening_policy_unavailable')
    token = load_token(cfg)
    policy, rules, policy_hash = load_screening_policy(cfg)
    rows, dataset_hash, source_selected_hash, selected_hash, screening_changes, source_count = load_rows(cfg, policy, rules)
    opener = make_opener()
    deterministic_id, owned_id = validated_status_identity(opener, base_url, token, cfg)
    report = build_report_base(cfg, config_hash, dataset_hash, source_selected_hash, selected_hash,
                               screening_changes, policy_hash, source_count, rows, deterministic_id, owned_id)
    stats = {'deterministic': empty_path_stats(), 'owned': empty_path_stats()}
    report['paths'] = stats
    session_id = 'paired-' + uuid.uuid4().hex
    assessment_identity = None
    start_first = 'deterministic' if cfg['first_order'] == 'deterministic_first' else 'owned'
    total_rounds = cfg['warmup_rounds'] + cfg['measured_rounds']
    if len(rows) * total_rounds > sys.maxsize:
        raise BenchmarkError('request_count_arithmetic_limit')
    interrupted = False
    for round_index in range(total_rounds):
        phase = 'warmup' if round_index < cfg['warmup_rounds'] else 'measured'
        for row_index, row in enumerate(rows):
            pair_index = round_index * len(rows) + row_index
            first = start_first if pair_index % 2 == 0 else ('owned' if start_first == 'deterministic' else 'deterministic')
            order = (first, 'owned' if first == 'deterministic' else 'deterministic')
            revision = pair_index + 1
            pair_payloads = {
                'deterministic': {'sessionId': session_id, 'revision': revision, 'text': row['text'],
                                  'declaredSubscriptions': cfg['declared_subscriptions']},
                'owned': {'text': row['text']},
            }
            if any(len(json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))
                   > cfg['max_request_bytes'] for payload in pair_payloads.values()):
                raise BenchmarkError('request_size_limit')
            for path_name in order:
                if path_name == 'deterministic':
                    payload = pair_payloads['deterministic']
                    endpoint = base_url + '/api/lfg/assess'
                    request_token = None
                else:
                    payload = pair_payloads['owned']
                    endpoint = base_url + '/api/lfg/assess/owned'
                    request_token = token
                try:
                    result = http_json(
                        opener, endpoint, method='POST', payload=payload, token=request_token,
                        timeout=cfg['client_timeout_seconds'], max_request_bytes=cfg['max_request_bytes'],
                        max_response_bytes=cfg['max_response_bytes'])
                except ClientTimeout as error:
                    report['abort_code'] = 'client_timeout_contamination_unknown'
                    report['contamination_state'] = 'unknown; no further requests issued'
                    result = HttpResult(None, None, error.elapsed_ns, 'client_timeout')
                    record_response(stats[path_name], phase, 'client_timeout', result, None, 1)
                    report['paths'] = {name: public_stats(value) for name, value in stats.items()}
                    interrupted = True
                    break
                except BenchmarkError as error:
                    if error.code == 'http_transport_error':
                        report['abort_code'] = 'http_transport_contamination_unknown'
                        report['contamination_state'] = 'unknown; no further requests issued'
                        result = HttpResult(None, None, 0, error.code)
                        record_response(stats[path_name], phase, error.code, result, None, 1)
                        interrupted = True
                        break
                    result = HttpResult(None, None, 0, error.code)
                handler_value = None
                if result.status != 200 or result.payload is None:
                    outcome = safe_status(result)
                elif path_name == 'deterministic':
                    outcome, next_identity, handler_value = validate_assessment(
                        result.payload, session_id, revision, assessment_identity)
                    if next_identity is not None:
                        assessment_identity = next_identity
                        report['service']['deterministic']['assessmentConfig'] = next_identity
                    if outcome == 'malformed_response':
                        handler_value = None
                else:
                    outcome, handler_value = validate_owned(result.payload, owned_id['engine'])
                if phase == 'measured':
                    record_response(stats[path_name], phase, outcome, result, handler_value,
                                    1 if path_name == 'deterministic' else 1000)
                else:
                    record_response(stats[path_name], phase, outcome, result, None, 1)
            if interrupted:
                break
        if interrupted:
            break
    if interrupted:
        verify_run_inputs(dataset_path, dataset_hash, config_path, config_raw, config_hash,
                          policy_path, policy_hash, cfg)
        report['paths'] = {name: public_stats(value) for name, value in stats.items()}
        return report
    verify_run_inputs(dataset_path, dataset_hash, config_path, config_raw, config_hash,
                      policy_path, policy_hash, cfg)
    report['paths'] = {name: public_stats(value) for name, value in stats.items()}
    report['complete'] = True
    return report


def verify_private_path(path: Path) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
        info = resolved.stat()
    except OSError:
        raise BenchmarkError('private_output_directory_unavailable') from None
    if not resolved.is_dir() or inside(resolved, REPO_ROOT):
        raise BenchmarkError('private_output_directory_invalid')
    if os.name == 'posix':
        if info.st_uid != os.geteuid() or info.st_mode & 0o077:
            raise BenchmarkError('private_output_permissions_unverified')
    elif os.name == 'nt':
        verify_private_windows_acl(resolved)
    else:
        raise BenchmarkError('private_output_permissions_unverified')
    return resolved


def write_report(path: Path, report: dict, max_bytes: int) -> None:
    parent = verify_private_path(path.parent)
    target = parent / path.name
    if target.resolve(strict=False) != target or target.name in ('', '.', '..'):
        raise BenchmarkError('private_output_path_invalid')
    raw = (json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
    if len(raw) > max_bytes:
        raise BenchmarkError('report_size_limit')
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, 'O_BINARY'):
        flags |= os.O_BINARY
    try:
        fd = os.open(target, flags, 0o600)
    except FileExistsError:
        raise BenchmarkError('report_already_exists') from None
    except OSError:
        raise BenchmarkError('report_create_failed') from None
    try:
        if os.name == 'posix':
            os.fchmod(fd, 0o600)
            info = os.fstat(fd)
            if info.st_uid != os.geteuid() or info.st_mode & 0o077:
                raise BenchmarkError('private_output_permissions_unverified')
        elif os.name == 'nt':
            verify_private_windows_acl(target)
        with os.fdopen(fd, 'wb', closefd=False) as output:
            output.write(raw)
            output.flush()
            os.fsync(fd)
    except BenchmarkError:
        os.close(fd)
        try:
            target.unlink()
        except OSError:
            pass
        raise
    except OSError:
        os.close(fd)
        try:
            target.unlink()
        except OSError:
            pass
        raise BenchmarkError('report_write_failed') from None
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        config, _, _ = load_config(args.config)
        report = run_benchmark(args.config)
        write_report(args.output, report, config['max_report_bytes'])
        return 2 if not report['complete'] else 0
    except BenchmarkError as error:
        sys.stderr.write(f'paired benchmark failed: {error.code}\n')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
