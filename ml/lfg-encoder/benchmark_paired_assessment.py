#!/usr/bin/env python3
"""Compare deterministic and experimental owned LFG HTTP assessment paths."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import copy
import hashlib
import http.client
import io
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
import stat
import subprocess
import sys
import threading
import time
import urllib.parse
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
import screen_private_corpus as screening


SCHEMA = 'lfg.paired-http-benchmark-config.v1'
REPORT_SCHEMA = 'lfg.paired-http-assessment-benchmark.v1'
CONFIG_FIELDS = {
    'schema_version', 'target_base_url', 'owned_token_env', 'owned_token_file',
    'dataset_path', 'dataset_split', 'dataset_source_ref', 'selection_seed', 'screening_policy_path',
    'source_manifest_path',
    'max_records', 'max_source_records', 'max_dataset_bytes', 'max_input_bytes',
    'max_request_bytes', 'max_response_bytes', 'max_report_bytes',
    'client_timeout_seconds', 'server_timeout_overhead_ms', 'warmup_rounds',
    'measured_rounds', 'declared_subscriptions', 'first_order',
}
MAX_CONFIG_BYTES = 64 * 1024
MAX_SCREEN_POLICY_BYTES = 1024 * 1024
MAX_SCREEN_MANIFEST_BYTES = 1024 * 1024
SCREEN_MANIFEST_SCHEMA = 'lfg.private-corpus-screen-manifest.v1'
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
    'serviceConfigSha256', 'cachedManifestSha256', 'inferenceBackend',
    'modelIdentityKind', 'prefillOnnxSha256', 'decodeOnnxSha256',
    'sourceWeightsSha256', 'contextBytes',
)
# JSON extraction/size checks, peer/Host/Origin/token authorization, and route
# matching reject before owned worker dispatch for these statuses.
SAFE_REJECTION_HTTP_STATUSES = {400, 401, 403, 404, 413}
OWNED_WORKER_RESULT_STATUSES = {
    'ok', 'truncated', 'timeout', 'incomplete_output', 'malformed_output', 'inference_error',
}


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


class DeadlineSocket:
    """Socket facade that reapplies a total request deadline on every I/O."""

    def __init__(self, sock: socket.socket, deadline: float):
        self._sock = sock
        self._deadline = deadline

    def _remaining(self) -> float:
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise socket.timeout('deadline expired')
        return remaining

    def recv(self, size: int, flags: int = 0) -> bytes:
        self._sock.settimeout(self._remaining())
        return self._sock.recv(size, flags)

    def sendall(self, data: bytes, flags: int = 0) -> None:
        view = memoryview(data)
        while view:
            self._sock.settimeout(self._remaining())
            sent = self._sock.send(view, flags)
            if sent <= 0:
                raise OSError('socket closed')
            view = view[sent:]

    def makefile(self, mode: str = 'r', buffering: int | None = None, *, encoding=None,
                 errors=None, newline=None):
        if 'r' not in mode or 'b' not in mode:
            return self._sock.makefile(mode, buffering, encoding=encoding, errors=errors, newline=newline)
        reader = DeadlineRawReader(DeadlineSocket(self._sock.dup(), self._deadline))
        return io.BufferedReader(reader, buffer_size=io.DEFAULT_BUFFER_SIZE if buffering in (None, -1) else buffering)

    def __getattr__(self, name):
        return getattr(self._sock, name)


class DeadlineRawReader(io.RawIOBase):
    def __init__(self, sock: DeadlineSocket):
        super().__init__()
        self._deadline_sock = sock

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        data = self._deadline_sock.recv(len(buffer))
        buffer[:len(data)] = data
        return len(data)

    def close(self) -> None:
        self._deadline_sock.close()
        super().close()


class DeadlineHTTPConnection(http.client.HTTPConnection):
    def __init__(self, host: str, deadline: float, timeout=None):
        self._deadline = deadline
        super().__init__(host, timeout=timeout)

    def connect(self):
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise socket.timeout('deadline expired')
        raw = socket.create_connection((self.host, self.port), timeout=remaining,
                                       source_address=self.source_address)
        self.sock = DeadlineSocket(raw, self._deadline)


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


def _resolve_outside_repo(value: str, code: str) -> Path:
    path = Path(value).expanduser()
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        raise BenchmarkError(code) from None
    if in_git_worktree(resolved):
        raise BenchmarkError('private_input_inside_repository')
    return resolved


def in_git_worktree(path: Path) -> bool:
    probe = path if path.is_dir() else path.parent
    if any(ancestor.name == '.git' or (ancestor / '.git').exists()
           for ancestor in (probe, *probe.parents)):
        return True
    try:
        result = subprocess.run(['git', '-C', str(probe), 'rev-parse', '--git-dir'],
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        raise BenchmarkError('git_privacy_unverified') from None
    return result.returncode == 0


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
    if not isinstance(value.get('source_manifest_path'), str) or not value['source_manifest_path']:
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


def load_source_manifest(config: dict) -> tuple[Path, bytes, str, str, int]:
    path = _resolve_outside_repo(config['source_manifest_path'], 'source_manifest_unavailable')
    try:
        raw = read_bounded(path, MAX_SCREEN_MANIFEST_BYTES, 'source_manifest_size_limit')
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError):
        raise BenchmarkError('source_manifest_invalid') from None
    if not isinstance(value, dict) or value.get('schema_version') != SCREEN_MANIFEST_SCHEMA:
        raise BenchmarkError('source_manifest_invalid')
    outputs = value.get('outputs')
    counts = value.get('counts')
    if (not isinstance(outputs, dict) or not isinstance(counts, dict)
            or not isinstance(outputs.get('train'), str) or not SHA_RE.fullmatch(outputs['train'])
            or not is_int(counts.get('train'), 1, sys.maxsize)):
        raise BenchmarkError('source_manifest_train_missing')
    _resolve_outside_repo(config['dataset_path'], 'dataset_unavailable')
    if counts['train'] > config['max_source_records']:
        raise BenchmarkError('dataset_record_limit')
    return path, raw, digest(raw), outputs['train'], counts['train']


def load_rows(config: dict, policy: dict, rules: list, manifest_train_sha: str,
              manifest_train_count: int) -> tuple[list[dict[str, str]], str, str, str, dict, int]:
    path = _resolve_outside_repo(config['dataset_path'], 'dataset_unavailable')
    try:
        info = path.stat()
        if info.st_size > config['max_dataset_bytes']:
            raise BenchmarkError('dataset_size_limit')
        with path.open('rb') as source:
            raw = source.read(config['max_dataset_bytes'] + 1)
        if len(raw) > config['max_dataset_bytes']:
            raise BenchmarkError('dataset_size_limit')
        if digest(raw) != manifest_train_sha:
            raise BenchmarkError('manifest_train_artifact_mismatch')
        decoded = raw.decode('utf-8')
        parsed = []
        for line_index, line in enumerate(decoded.splitlines()):
            if line.strip():
                parsed.append((json.loads(line), line.encode('utf-8'), line_index + 1))
    except BenchmarkError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise BenchmarkError('dataset_invalid') from None
    if not parsed or len(parsed) > config['max_source_records']:
        raise BenchmarkError('dataset_record_limit')
    if len(parsed) != manifest_train_count:
        raise BenchmarkError('manifest_train_count_mismatch')
    rows: list[dict[str, str]] = []
    identifiers: set[str] = set()
    for record, raw_line, line_number in parsed:
        if (not isinstance(record, dict) or not isinstance(record.get('id'), str)
                or not record['id'].strip() or not isinstance(record.get('text'), str)
                or not record['text'].strip() or record['id'] in identifiers):
            raise BenchmarkError('dataset_invalid')
        row_split = record.get('split')
        if not isinstance(row_split, str) or row_split != 'train':
            raise BenchmarkError('dataset_split_mismatch')
        upstream_hashes = {}
        for field in ('candidate_file_sha256', 'source_record_sha256', 'source_text_sha256',
                      'screen_policy_sha256', 'screened_text_sha256'):
            field_value = record.get(field)
            if field_value is not None:
                if not isinstance(field_value, str) or not SHA_RE.fullmatch(field_value):
                    raise BenchmarkError('dataset_invalid')
                upstream_hashes[field] = field_value
        identifiers.add(record['id'])
        provenance = {
            'source_line_number': line_number,
            'dataset_record_sha256': digest(raw_line),
            'record_id_sha256': digest(record['id'].encode('utf-8')),
            'row_split_claim': row_split,
        }
        for field in ('family', 'project', 'project_group'):
            value = record.get(field)
            if value is not None:
                if not isinstance(value, str):
                    raise BenchmarkError('dataset_invalid')
                provenance[f'{field}_sha256'] = digest(value.encode('utf-8'))
        provenance.update({f'upstream_{key}': value for key, value in upstream_hashes.items()})
        rows.append({'id': record['id'], 'text': record['text'], 'provenance': provenance})
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
        row['provenance']['screened_text_sha256'] = digest(screened.encode('utf-8'))
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


def http_json(url: str, *, method: str,
              payload: dict | None, token: str | None, timeout: float,
              max_request_bytes: int, max_response_bytes: int) -> HttpResult:
    body = None if payload is None else json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    if body is not None and len(body) > max_request_bytes:
        raise BenchmarkError('request_size_limit')
    try:
        parsed = urllib.parse.urlsplit(url)
        validate_loopback_url(f'{parsed.scheme}://{parsed.netloc}')
    except (ValueError, BenchmarkError):
        raise BenchmarkError('invalid_loopback_target') from None
    if not parsed.path.startswith('/') or parsed.query or parsed.fragment:
        raise BenchmarkError('invalid_loopback_target')
    headers = {'Accept': 'application/json', 'Cache-Control': 'no-store'}
    if body is not None:
        headers['Content-Type'] = 'application/json'
    if token is not None:
        headers['X-LFG-Session-Token'] = token
    headers['Connection'] = 'close'
    started = time.perf_counter_ns()
    deadline = time.monotonic() + timeout
    connection = DeadlineHTTPConnection(parsed.netloc, deadline, timeout=timeout)
    try:
        connection.request(method, parsed.path or '/', body=body, headers=headers)
        response = connection.getresponse()
        status = response.status
        raw = response.read(max_response_bytes + 1)
    except (socket.timeout, TimeoutError):
        raise ClientTimeout('client_timeout_contamination_unknown', time.perf_counter_ns() - started) from None
    except (OSError, http.client.HTTPException):
        raise BenchmarkError('http_transport_error') from None
    finally:
        connection.close()
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


def model_identity(value: object, *, owned: bool = False) -> dict | None:
    if not isinstance(value, dict) or not isinstance(value.get('id'), str) or not value['id']:
        return None
    result = {key: value[key] for key in ENGINE_ID_FIELDS
              if isinstance(value.get(key), str) or type(value.get(key)) in (int, bool)}
    if not owned:
        return result
    backend = value.get('inferenceBackend')
    identity_kind = value.get('modelIdentityKind')
    if backend == 'cached_onnx' and identity_kind == 'source_weights_sha256':
        required_sha = (
            'modelSha256', 'modelManifestSha256', 'cachedManifestSha256',
            'prefillOnnxSha256', 'decodeOnnxSha256', 'sourceWeightsSha256',
            'trainingManifestSha256', 'framingSha256', 'runtimeConfigSha256',
            'serviceConfigSha256',
        )
    elif backend == 'legacy_full_prefix' and identity_kind == 'onnx_graph_sha256':
        required_sha = ('modelSha256', 'modelManifestSha256', 'trainingManifestSha256',
                        'framingSha256', 'runtimeConfigSha256', 'serviceConfigSha256')
    else:
        return None
    if (any(not isinstance(value.get(key), str) or not SHA_RE.fullmatch(value[key])
            for key in required_sha)
            or not is_int(value.get('contextBytes'), 1, sys.maxsize)):
        return None
    return result


def validated_status_identity(base_url, token, cfg) -> tuple[dict, dict]:
    status_result = http_json(
        base_url + '/api/shippin/intelligence/lfg/status', method='GET', payload=None, token=None,
        timeout=cfg['client_timeout_seconds'], max_request_bytes=cfg['max_request_bytes'],
        max_response_bytes=cfg['max_response_bytes'])
    owned_result = http_json(
        base_url + '/api/lfg/assess/owned/status', method='GET', payload=None, token=token,
        timeout=cfg['client_timeout_seconds'], max_request_bytes=cfg['max_request_bytes'],
        max_response_bytes=cfg['max_response_bytes'])
    if status_result.status != 200 or status_result.payload is None:
        raise BenchmarkError('deterministic_status_unavailable')
    if owned_result.status != 200 or owned_result.payload is None:
        raise BenchmarkError('owned_status_unavailable')
    server = status_result.payload
    owned = owned_result.payload
    engine = model_identity(owned.get('engine'), owned=True)
    limits = owned.get('limits')
    if (owned.get('enabled') is not True or owned.get('status') != 'ready' or engine is None
            or not isinstance(limits, dict) or not is_int(limits.get('timeoutMs'), 1, sys.maxsize)):
        if engine is None:
            raise BenchmarkError('owned_identity_unavailable')
        raise BenchmarkError('owned_service_not_ready')
    required_timeout_ms = limits['timeoutMs'] + cfg['server_timeout_overhead_ms']
    if (required_timeout_ms > threading.TIMEOUT_MAX * 1000
            or cfg['client_timeout_seconds'] * 1000 <= required_timeout_ms):
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


def validate_owned(payload: dict, expected_engine: dict) -> tuple[str, int | None, str]:
    status = payload.get('status')
    if (payload.get('schemaVersion') != 'lfg.owned-assessment.v1'
            or not isinstance(status, str) or status not in KNOWN_OWNED_STATUSES
            or 'projectContextSha256' in payload):
        return 'malformed_response', None, 'unknown'
    if model_identity(payload.get('engine'), owned=True) != expected_engine:
        raise BenchmarkError('owned_identity_changed')
    measurement = payload.get('measurement')
    handler_ms = None
    lifecycle = 'unknown'
    if isinstance(measurement, dict) and measurement.get('schemaVersion') == 'lfg.owned-run-measurement.v1':
        if (measurement.get('status') != status or measurement.get('engineId') != expected_engine['id']
                or not is_int(measurement.get('handlerElapsedMs'))):
            return 'malformed_response', None, 'unknown'
        started = measurement.get('workerStarted')
        completed = measurement.get('workerCompletedSuccessfully')
        if ((started is not None and type(started) is not bool)
                or (completed is not None and type(completed) is not bool)
                or (started is False and completed is True)):
            return 'malformed_response', None, 'unknown'
        if started is False:
            if status in ('ok', 'truncated', 'timeout', 'incomplete_output', 'malformed_output'):
                return 'malformed_response', None, 'unknown'
            lifecycle = 'worker_not_started'
        elif started is True and completed is not None:
            if status in ('ok', 'truncated', 'incomplete_output', 'malformed_output') and completed is not True:
                return 'malformed_response', None, 'unknown'
            lifecycle = ('worker_exit_observed_success' if completed
                         else 'worker_exit_observed_failure')
        handler_ms = measurement['handlerElapsedMs']
    return status, handler_ms, lifecycle


def safe_status(result: HttpResult) -> str:
    if result.status is None:
        return result.response_code or 'transport_error'
    if result.status != 200:
        return f'http_{result.status}'
    return result.response_code or 'malformed_response'


def empty_path_stats() -> dict:
    def empty_phase() -> dict:
        return {'status_counts': Counter(), 'latency_ns': [],
                'attempts': 0, 'completed_responses': 0,
                'worker_lifecycle_counts': Counter()}

    return {**empty_phase(), 'handler_time_us': [], 'handler_missing': 0,
            'warmup': empty_phase()}


def record_response(stats: dict, phase: str, status: str, result: HttpResult,
                    handler_value: int | None, handler_scale: int,
                    worker_lifecycle: str | None = None) -> None:
    phase_stats = stats['warmup'] if phase == 'warmup' else stats
    phase_stats['attempts'] += 1
    phase_stats['status_counts'][status] += 1
    if worker_lifecycle is not None:
        phase_stats['worker_lifecycle_counts'][worker_lifecycle] += 1
    if result.status is not None:
        phase_stats['completed_responses'] += 1
        phase_stats['latency_ns'].append(result.elapsed_ns)
    if phase == 'measured':
        if handler_value is None:
            stats['handler_missing'] += 1
        else:
            stats['handler_time_us'].append(handler_value * handler_scale)


def public_stats(stats: dict) -> dict:
    warmup = stats['warmup']
    return {
        'attempts': stats['attempts'],
        'completed_responses': stats['completed_responses'],
        'status_counts': dict(sorted(stats['status_counts'].items())),
        'worker_lifecycle_counts': dict(sorted(stats['worker_lifecycle_counts'].items())),
        'end_to_end_latency': summarize(stats['latency_ns'], stats['completed_responses']),
        'server_handler_time_us': summarize(stats['handler_time_us'], stats['attempts']),
        'server_handler_time_missing': stats['handler_missing'],
        'warmup': {
            'attempts': warmup['attempts'],
            'completed_responses': warmup['completed_responses'],
            'status_counts': dict(sorted(warmup['status_counts'].items())),
            'worker_lifecycle_counts': dict(sorted(warmup['worker_lifecycle_counts'].items())),
            'end_to_end_latency': summarize(warmup['latency_ns'], warmup['completed_responses']),
        },
    }


def build_report_base(cfg: dict, config_hash: str, dataset_hash: str,
                      source_selected_hash: str, selected_hash: str, screening_changes: dict,
                      policy_hash: str, manifest_hash: str, manifest_train_sha: str,
                      manifest_train_count: int, source_record_count: int, rows: list[dict],
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
            'split': 'train',
            'operator_split_claim': cfg['dataset_split'],
            'split_basis': 'selected artifact digest/count are bound to the manifest train output; underlying source split and independence remain unverified',
            'manifest_schema_version': SCREEN_MANIFEST_SCHEMA,
            'screen_manifest_sha256': manifest_hash,
            'manifest_train_artifact_sha256': manifest_train_sha,
            'manifest_train_record_count': manifest_train_count,
            'screening_policy_sha256': policy_hash,
            'screening': 'operator-supplied policy applied locally; does not verify prior screening or labels',
            'screening_changes': screening_changes,
            'dataset_sha256': dataset_hash,
            'source_selected_id_text_digest': source_selected_hash,
            'screened_selected_id_text_digest': selected_hash,
            'source_records': source_record_count,
            'selected_records': len(rows),
            'selected_record_provenance': [row['provenance'] for row in rows],
            'selection': 'seeded sample without replacement' if len(rows) < source_record_count else 'all bounded source records',
            'selection_seed': cfg['selection_seed'],
        },
        'service': {
            'target': cfg['target_base_url'],
            'deterministic': deterministic_id,
            'owned': owned_id,
            'identity_after': None,
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
                      policy_hash: str, manifest_path: Path, manifest_hash: str,
                      cfg: dict) -> None:
    dataset_raw = read_bounded(dataset_path, cfg['max_dataset_bytes'], 'benchmark_input_changed')
    final_config_raw = read_bounded(config_path, MAX_CONFIG_BYTES, 'benchmark_input_changed')
    policy_raw = read_bounded(policy_path, MAX_SCREEN_POLICY_BYTES, 'benchmark_input_changed')
    manifest_raw = read_bounded(manifest_path, MAX_SCREEN_MANIFEST_BYTES, 'benchmark_input_changed')
    if (digest(dataset_raw) != dataset_hash or digest(final_config_raw) != config_hash
            or final_config_raw != config_raw or digest(policy_raw) != policy_hash
            or digest(manifest_raw) != manifest_hash):
        raise BenchmarkError('benchmark_input_changed')


def run_benchmark(config_path: Path) -> dict:
    cfg, config_raw, config_hash = load_config(config_path)
    base_url = validate_loopback_url(cfg['target_base_url'])
    manifest_path, manifest_raw, manifest_hash, manifest_train_sha, manifest_train_count = load_source_manifest(cfg)
    dataset_path = _resolve_outside_repo(cfg['dataset_path'], 'dataset_unavailable')
    policy_path = _resolve_outside_repo(cfg['screening_policy_path'], 'screening_policy_unavailable')
    token = load_token(cfg)
    policy, rules, policy_hash = load_screening_policy(cfg)
    rows, dataset_hash, source_selected_hash, selected_hash, screening_changes, source_count = load_rows(
        cfg, policy, rules, manifest_train_sha, manifest_train_count)
    deterministic_id, owned_id = validated_status_identity(base_url, token, cfg)
    deterministic_id_before = copy.deepcopy(deterministic_id)
    owned_id_before = copy.deepcopy(owned_id)
    report = build_report_base(cfg, config_hash, dataset_hash, source_selected_hash, selected_hash,
                               screening_changes, policy_hash, manifest_hash, manifest_train_sha,
                               manifest_train_count, source_count, rows, deterministic_id, owned_id)
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
                        endpoint, method='POST', payload=payload, token=request_token,
                        timeout=cfg['client_timeout_seconds'], max_request_bytes=cfg['max_request_bytes'],
                        max_response_bytes=cfg['max_response_bytes'])
                except ClientTimeout as error:
                    report['abort_code'] = 'client_timeout_contamination_unknown'
                    report['contamination_state'] = 'unknown; no further requests issued'
                    result = HttpResult(None, None, error.elapsed_ns, 'client_timeout')
                    record_response(stats[path_name], phase, 'client_timeout', result, None, 1,
                                    'unknown' if path_name == 'owned' else None)
                    report['paths'] = {name: public_stats(value) for name, value in stats.items()}
                    interrupted = True
                    break
                except BenchmarkError as error:
                    if error.code == 'http_transport_error':
                        report['abort_code'] = 'http_transport_contamination_unknown'
                        report['contamination_state'] = 'unknown; no further requests issued'
                        result = HttpResult(None, None, 0, error.code)
                        record_response(stats[path_name], phase, error.code, result, None, 1,
                                        'unknown' if path_name == 'owned' else None)
                        interrupted = True
                        break
                    result = HttpResult(None, None, 0, error.code)
                handler_value = None
                worker_lifecycle = None
                if result.status != 200 or result.payload is None:
                    outcome = safe_status(result)
                    if path_name == 'owned' and result.status == 200:
                        worker_lifecycle = 'unknown'
                elif path_name == 'deterministic':
                    outcome, next_identity, handler_value = validate_assessment(
                        result.payload, session_id, revision, assessment_identity)
                    if next_identity is not None:
                        assessment_identity = next_identity
                        report['service']['deterministic']['assessmentConfig'] = next_identity
                    if outcome == 'malformed_response':
                        handler_value = None
                else:
                    outcome, handler_value, worker_lifecycle = validate_owned(result.payload, owned_id['engine'])
                if phase == 'measured':
                    record_response(stats[path_name], phase, outcome, result, handler_value,
                                    1 if path_name == 'deterministic' else 1000, worker_lifecycle)
                else:
                    record_response(stats[path_name], phase, outcome, result, None, 1, worker_lifecycle)
                ambiguous_http = (result.status is not None and result.status != 200
                                  and result.status not in SAFE_REJECTION_HTTP_STATUSES)
                owned_busy = path_name == 'owned' and result.status == 200 and outcome == 'busy'
                lifecycle_unknown = (path_name == 'owned' and result.status == 200
                                     and worker_lifecycle == 'unknown'
                                     and (outcome in OWNED_WORKER_RESULT_STATUSES
                                          or outcome in ('malformed_response', 'response_too_large')))
                if ambiguous_http or owned_busy or lifecycle_unknown:
                    report['abort_code'] = ('owned_busy_contamination_unknown' if owned_busy else
                                            'owned_worker_lifecycle_unknown' if lifecycle_unknown else
                                            'http_response_contamination_unknown')
                    report['contamination_state'] = 'unknown; no further requests issued'
                    interrupted = True
                    break
            if interrupted:
                break
        if interrupted:
            break
    if interrupted:
        verify_run_inputs(dataset_path, dataset_hash, config_path, config_raw, config_hash,
                          policy_path, policy_hash, manifest_path, manifest_hash, cfg)
        report['paths'] = {name: public_stats(value) for name, value in stats.items()}
        return report
    deterministic_after, owned_after = validated_status_identity(base_url, token, cfg)
    if deterministic_after != deterministic_id_before or owned_after != owned_id_before:
        raise BenchmarkError('service_identity_changed')
    report['service']['identity_after'] = {
        'deterministic': deterministic_after,
        'owned': owned_after,
    }
    verify_run_inputs(dataset_path, dataset_hash, config_path, config_raw, config_hash,
                      policy_path, policy_hash, manifest_path, manifest_hash, cfg)
    report['paths'] = {name: public_stats(value) for name, value in stats.items()}
    report['complete'] = True
    return report


def verify_private_path(path: Path) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
        info = resolved.stat()
    except OSError:
        raise BenchmarkError('private_output_directory_unavailable') from None
    if not resolved.is_dir() or in_git_worktree(resolved):
        raise BenchmarkError('private_output_directory_invalid')
    if os.name == 'posix':
        if info.st_uid != os.geteuid() or info.st_mode & 0o077:
            raise BenchmarkError('private_output_permissions_unverified')
    elif os.name == 'nt':
        verify_private_windows_acl(resolved)
    else:
        raise BenchmarkError('private_output_permissions_unverified')
    return resolved


def open_private_output_directory(path: Path) -> tuple[Path, int | None]:
    resolved = verify_private_path(path)
    if os.name != 'posix':
        return resolved, None
    if (not hasattr(os, 'O_DIRECTORY') or not hasattr(os, 'O_NOFOLLOW')
            or os.open not in os.supports_dir_fd):
        raise BenchmarkError('private_output_permissions_unverified')
    descriptor = None
    try:
        expected = resolved.stat()
        parts = resolved.parts
        descriptor = os.open(parts[0], os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        for index, component in enumerate(parts[1:], start=1):
            next_descriptor = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                       dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
            info = os.fstat(descriptor)
            leaf = index == len(parts) - 1
            if not stat.S_ISDIR(info.st_mode):
                raise BenchmarkError('private_output_directory_invalid')
            if leaf:
                if info.st_uid != os.geteuid() or info.st_mode & 0o077:
                    raise BenchmarkError('private_output_permissions_unverified')
                if (info.st_dev, info.st_ino) != (expected.st_dev, expected.st_ino):
                    raise BenchmarkError('private_output_directory_changed')
            elif (info.st_uid not in (0, os.geteuid())
                  or (info.st_mode & 0o022 and not (info.st_uid == 0 and info.st_mode & stat.S_ISVTX))):
                raise BenchmarkError('private_output_ancestor_unsafe')
        return resolved, descriptor
    except BenchmarkError:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except OSError:
        if descriptor is not None:
            os.close(descriptor)
        raise BenchmarkError('private_output_permissions_unverified') from None


def write_report(path: Path, report: dict, max_bytes: int) -> None:
    target_name = path.name
    if target_name in ('', '.', '..') or Path(target_name).name != target_name:
        raise BenchmarkError('private_output_path_invalid')
    raw = (json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
    if len(raw) > max_bytes:
        raise BenchmarkError('report_size_limit')
    parent, directory_fd = open_private_output_directory(path.parent)
    target = parent / target_name
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, 'O_BINARY'):
        flags |= os.O_BINARY
    if os.name == 'posix':
        flags |= os.O_NOFOLLOW
    fd = None
    try:
        try:
            fd = (os.open(target_name, flags, 0o600, dir_fd=directory_fd)
                  if directory_fd is not None else os.open(target, flags, 0o600))
        except FileExistsError:
            raise BenchmarkError('report_already_exists') from None
        except OSError:
            raise BenchmarkError('report_create_failed') from None
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
        if fd is not None:
            os.close(fd)
            try:
                if directory_fd is not None:
                    os.unlink(target_name, dir_fd=directory_fd)
                else:
                    target.unlink()
            except OSError:
                pass
        raise
    except OSError:
        if fd is not None:
            os.close(fd)
            try:
                if directory_fd is not None:
                    os.unlink(target_name, dir_fd=directory_fd)
                else:
                    target.unlink()
            except OSError:
                pass
        raise BenchmarkError('report_write_failed') from None
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if directory_fd is not None:
            os.close(directory_fd)


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
