#!/usr/bin/env python3
"""Serve a private corpus-review workspace on an authenticated loopback address."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import secrets
import socket
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import corpus_audit
import evaluate
import inspect_codex_inventory
import inventory_checkpoint_prompts
import review_corpus
import train

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'review-web'
DEFAULT_CONFIG = ROOT / 'review-web-config.v1.json'
DEFAULT_GUIDE = ROOT / 'label-guide.v1.json'


def load_guide(path: Path, taxonomy: dict[str, set[str]]) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if (not isinstance(value, dict) or set(value) != {'schema_version', 'revision', 'notice', 'labels'}
            or value['schema_version'] != 'lfg.review-label-guide.v1'
            or not isinstance(value['revision'], str) or not value['revision'].strip()
            or not isinstance(value['notice'], str) or not value['notice'].strip()
            or not isinstance(value['labels'], dict) or set(value['labels']) != set(evaluate.FIELDS)):
        raise ValueError('invalid label guide')
    for field in evaluate.FIELDS:
        labels = value['labels'][field]
        if not isinstance(labels, dict) or set(labels) != taxonomy[field]:
            raise ValueError('label guide does not match workspace taxonomy')
        for description in labels.values():
            if (not isinstance(description, dict) or set(description) != {'apply', 'boundary'}
                    or any(not isinstance(text, str) or not text.strip() for text in description.values())):
                raise ValueError('each label requires application and boundary guidance')
    if path.read_bytes() != raw:
        raise ValueError('label guide changed while loading')
    return value, hashlib.sha256(raw).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    fields = {'max_request_bytes', 'max_prompt_bytes', 'request_timeout_seconds'}
    if not isinstance(value, dict) or not fields | {'schema_version'} <= set(value) or \
            set(value) - (fields | {'schema_version', 'source_verification', 'source_context', 'checkpoint_verification', 'redaction_suggestions'}) or \
            value['schema_version'] != 'lfg.review-web-config.v1' or \
            any(type(value[name]) is not int or value[name] < 1 for name in fields):
        raise ValueError('invalid local review web configuration')
    if 'source_verification' in value:
        limits = value['source_verification']
        if (not isinstance(limits, dict) or set(limits) != {'max_session_bytes', 'max_line_bytes'}
                or any(type(limit) is not int or limit < 1 for limit in limits.values())):
            raise ValueError('invalid source verification limits')
    if 'source_context' in value:
        limits = value['source_context']
        if ('source_verification' not in value or not isinstance(limits, dict)
                or set(limits) != {'max_messages', 'max_context_bytes'}
                or any(type(limit) is not int or limit < 1 for limit in limits.values())):
            raise ValueError('invalid source context limits')
    if 'checkpoint_verification' in value:
        limits = value['checkpoint_verification']
        if (not isinstance(limits, dict) or set(limits) != {'repo', 'max_history_bytes'}
                or not isinstance(limits['repo'], str) or not Path(limits['repo']).is_absolute()
                or type(limits['max_history_bytes']) is not int or limits['max_history_bytes'] < 1):
            raise ValueError('invalid checkpoint verification configuration')
    if 'redaction_suggestions' in value:
        rules = value['redaction_suggestions']
        if not isinstance(rules, list):
            raise ValueError('invalid redaction suggestions')
        ids = set()
        for rule in rules:
            if (not isinstance(rule, dict)
                    or set(rule) != {'id', 'label', 'start_marker', 'end_marker', 'trailing_text', 'replacement'}
                    or any(not isinstance(rule[key], str) for key in rule)
                    or any(not rule[key].strip() for key in ('id', 'label', 'start_marker', 'end_marker', 'trailing_text'))
                    or rule['id'] in ids):
                raise ValueError('invalid redaction suggestion rule')
            ids.add(rule['id'])
    return value


def redaction_suggestions(text: str, rules: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Return optional prefix-removal previews in Unicode code-point offsets."""
    result = []
    for rule in rules:
        start_marker, end_marker, trailing = rule['start_marker'], rule['end_marker'], rule['trailing_text']
        if text.count(start_marker) != 1 or text.count(end_marker) != 1:
            continue
        start, close = text.find(start_marker), text.find(end_marker)
        after = close + len(end_marker)
        if start < 0 or close <= start or text[:start].strip() or not text.startswith(trailing, after):
            continue
        end = after + len(trailing)
        candidate = rule['replacement'] + text[end:]
        if not candidate.strip() or candidate == text:
            continue
        result.append({'id': rule['id'], 'label': rule['label'], 'start': 0,
                       'end': end, 'replacement': rule['replacement']})
    return result


def verify_source(workspace: Path, identifier: str, expected_digest: str, limits: dict[str, int] | None,
                  context_limits: dict[str, int] | None = None,
                  checkpoint_limits: dict[str, Any] | None = None) -> str | dict[str, Any]:
    """Check the source attached to a pending record, without changing the review."""
    plan, _, ledger = review_corpus.load_workspace(workspace)
    item = ledger.get(identifier)
    if item is None or item['decision'] != 'pending':
        raise ValueError('only pending records can be checked')
    _, rows, index, source = review_corpus.find_row(workspace, plan, identifier)
    row = rows[index]
    _, taxonomy_hash = evaluate.load_taxonomy(review_corpus.private_path(workspace, plan['taxonomy']))
    current = corpus_audit.annotation_digest(row, source['project'], source['split'], taxonomy_hash)
    if current != expected_digest or current != item['annotation_sha256'] or row['verified']:
        raise ValueError('record changed or an earlier write was interrupted; review it again')
    original = row.get('source')
    if not isinstance(original, dict):
        raise ValueError('automatic source verification is unavailable for this source type')
    if original.get('kind') == 'codex-session-user-text':
        if limits is None:
            raise ValueError('source verification is not configured')
        entry = {**original, 'schema_version': 'lfg.unreviewed-codex-message.v2'}
        if context_limits is None:
            text = inspect_codex_inventory.verify(entry, limits['max_session_bytes'], limits['max_line_bytes'])
        else:
            text, messages, truncated = inspect_codex_inventory.prior_user_context(
                entry, limits['max_session_bytes'], limits['max_line_bytes'],
                context_limits['max_messages'], context_limits['max_context_bytes'])
        status = 'source_prefix_and_original_text_match'
    elif original.get('kind') == 'git-checkpoint-user-prompt' and context_limits is None:
        if checkpoint_limits is None:
            raise ValueError('checkpoint source verification is not configured')
        revision = original.get('history_revision')
        raw = inventory_checkpoint_prompts.read_history(Path(checkpoint_limits['repo']), revision,
                                                        checkpoint_limits['max_history_bytes'])
        if hashlib.sha256(raw).hexdigest() != original.get('history_stream_sha256'):
            raise ValueError('Git history differs from the recorded source')
        _, records = inventory_checkpoint_prompts.parse_history(raw, source['project'], revision)
        candidates = [candidate for candidate in records if candidate['id'] == original.get('commit')]
        if len(candidates) != 1:
            raise ValueError('checkpoint commit is absent from the recorded history')
        candidate = candidates[0]
        if (any(original.get(key) != value for key, value in candidate['source'].items())
                or original.get('text_sha256') != candidate['text_sha256']
                or row['id'] != 'checkpoint-' + candidate['id']
                or row['family'] != 'checkpoint-session-' + (candidate['session_sha256'] or 'unknown-session')):
            raise ValueError('checkpoint source metadata differs from the recorded history')
        text = candidate['text']
        status = 'checkpoint_history_and_original_text_match'
    else:
        raise ValueError('automatic source verification is unavailable for this source type')
    if text != row['text'] or row.get('review_transforms'):
        raise ValueError('review copy differs from the original source')
    # Recheck the review record after the potentially lengthy file read.
    plan_after, _, ledger_after = review_corpus.load_workspace(workspace)
    _, rows_after, index_after, source_after = review_corpus.find_row(workspace, plan_after, identifier)
    _, taxonomy_after = evaluate.load_taxonomy(review_corpus.private_path(workspace, plan_after['taxonomy']))
    if (ledger_after.get(identifier) != item or taxonomy_after != taxonomy_hash
            or corpus_audit.annotation_digest(rows_after[index_after], source_after['project'],
                                               source_after['split'], taxonomy_after) != current):
        raise ValueError('review record changed during source verification')
    if context_limits is not None:
        return {'status': 'source_prefix_and_original_text_match', 'messages': messages, 'truncated': truncated}
    return status


class ReviewServer(HTTPServer):
    def __init__(self, workspace: Path, reviewer: str, port: int, config: dict[str, Any], token: str,
                 guide_path: Path | None = None):
        if not reviewer.strip() or '\n' in reviewer or '\r' in reviewer:
            raise ValueError('reviewer must be a nonempty single-line identifier')
        self.workspace = workspace.resolve(strict=True)
        self.reviewer = reviewer.strip()
        self.config = config
        self.token = token
        plan, _, _ = review_corpus.load_workspace(self.workspace)
        taxonomy, self.taxonomy_hash = evaluate.load_taxonomy(review_corpus.private_path(self.workspace, plan['taxonomy']))
        self.guide_path = guide_path.resolve(strict=True) if guide_path is not None else None
        self.guide, self.guide_sha256 = load_guide(self.guide_path, taxonomy) if self.guide_path else (None, None)
        super().__init__(('127.0.0.1', port), ReviewHandler)

    @property
    def origin(self) -> str:
        return f'http://127.0.0.1:{self.server_port}'


class ReviewHandler(BaseHTTPRequestHandler):
    server: ReviewServer

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(self.server.config['request_timeout_seconds'])

    def log_message(self, format: str, *args: Any) -> None:
        # Requests and source text should not be copied to access logs.
        return

    def reply(self, code: int, content: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(content)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy', "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(content)

    def json_reply(self, code: int, value: dict[str, Any]) -> None:
        self.reply(code, json.dumps(value, ensure_ascii=True, allow_nan=False).encode('utf-8'), 'application/json; charset=utf-8')

    def reject_unauthorized_post(self, max_request_bytes: int) -> None:
        # Reply before draining. On Windows, closing with an unread small body
        # can reset the connection before the client receives the rejection.
        self.close_connection = True
        self.json_reply(403, {'error': 'not authorized'})
        lengths = self.headers.get_all('Content-Length', [])
        if self.headers.get_all('Transfer-Encoding', []) or len(lengths) != 1 or not lengths[0].isdigit():
            return
        try:
            length = int(lengths[0])
            if length <= max_request_bytes:
                self.rfile.read(length)
        except (OSError, ValueError):
            # The existing socket timeout bounds an incomplete rejected body.
            return

    def allowed_host(self) -> bool:
        return self.headers.get_all('Host', []) == [f'127.0.0.1:{self.server.server_port}']

    def authorized(self, post: bool = False) -> bool:
        origins = self.headers.get_all('Origin', [])
        auth = self.headers.get_all('Authorization', [])
        expected = ('Bearer ' + self.server.token).encode('utf-8')
        return self.allowed_host() and (origins == [self.server.origin] if post else origins in ([], [self.server.origin])) and \
            len(auth) == 1 and hmac.compare_digest(auth[0].encode('utf-8'), expected)

    def do_GET(self) -> None:
        if not self.allowed_host():
            self.json_reply(403, {'error': 'invalid host'})
            return
        parsed = urlsplit(self.path)
        assets = {'/': ('index.html', 'text/html; charset=utf-8'),
                  '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                  '/app.css': ('app.css', 'text/css; charset=utf-8')}
        if parsed.path in assets and not parsed.query:
            filename, mime = assets[parsed.path]
            self.reply(200, (ASSETS / filename).read_bytes(), mime)
            return
        if not self.authorized():
            self.json_reply(403, {'error': 'not authorized'})
            return
        try:
            if parsed.path == '/api/status' and not parsed.query:
                self.json_reply(200, {'reviewer': self.server.reviewer, **review_corpus.status(self.server.workspace)})
                return
            if parsed.path == '/api/item':
                query = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=True)
                if (set(query) not in ({'offset'}, {'id'}) or len(next(iter(query.values()))) != 1
                        or ('offset' in query and not query['offset'][0].isdigit())
                        or ('id' in query and not query['id'][0].strip())):
                    raise ValueError('invalid pending-record offset')
                plan, _, ledger = review_corpus.load_workspace(self.server.workspace)
                pending = [identifier for identifier, item in ledger.items() if item['decision'] == 'pending']
                if 'id' in query:
                    if query['id'][0] not in pending:
                        self.json_reply(200, {'item': None, 'pending': len(pending), 'offset': 0, 'missing_id': True})
                        return
                    offset = pending.index(query['id'][0])
                else:
                    offset = int(query['offset'][0])
                if offset >= len(pending):
                    self.json_reply(200, {'item': None, 'pending': len(pending), 'offset': offset})
                    return
                identifier = pending[offset]
                _, rows, index, source = review_corpus.find_row(self.server.workspace, plan, identifier)
                row = rows[index]
                taxonomy, taxonomy_hash = evaluate.load_taxonomy(
                    review_corpus.private_path(self.server.workspace, plan['taxonomy']))
                if taxonomy_hash != self.server.taxonomy_hash:
                    raise ValueError('workspace taxonomy changed; restart review')
                if len(row['text'].encode('utf-8')) > self.server.config['max_prompt_bytes']:
                    self.json_reply(200, {'item': None, 'blocked': 'Request exceeds configured display limit.',
                                          'pending': len(pending), 'offset': offset})
                    return
                annotation = corpus_audit.annotation_digest(row, source['project'], source['split'], taxonomy_hash)
                stale = (not row['verified'] and ledger[identifier]['annotation_sha256'] != annotation) or row['verified']
                suggestions = [] if stale else redaction_suggestions(row['text'], self.server.config.get('redaction_suggestions', []))
                self.json_reply(200, {'item': {'id': identifier, 'text': row['text'], 'source': row['source'],
                                               'review_note': row.get('review_note'), 'labels': {k: row[k] for k in evaluate.FIELDS},
                                               'labels_pending': row.get('labels_pending', False),
                                               'project': source['project'], 'split': source['split'],
                                               'annotation_sha256': annotation,
                                               'redaction_suggestions': suggestions,
                                               'pending_changed': not row['verified'] and ledger[identifier]['annotation_sha256'] != annotation,
                                               'interrupted_write': bool(row['verified'])},
                                      'taxonomy': {k: sorted(v) for k, v in taxonomy.items()},
                                      'guide': self.server.guide, 'guide_sha256': self.server.guide_sha256,
                                      'pending': len(pending), 'offset': offset})
                return
            self.json_reply(404, {'error': 'not found'})
        except (OSError, ValueError, KeyError, TypeError, UnicodeError) as error:
            self.json_reply(400, {'error': str(error)})

    def do_POST(self) -> None:
        if not self.authorized(post=True):
            self.reject_unauthorized_post(self.server.config['max_request_bytes'])
            return
        if self.path not in ('/api/decision', '/api/verify-source', '/api/source-context') or self.headers.get('Transfer-Encoding') or \
                self.headers.get_content_type() != 'application/json':
            self.json_reply(400, {'error': 'invalid request'})
            return
        try:
            lengths = self.headers.get_all('Content-Length', [])
            if len(lengths) != 1 or not lengths[0].isdigit():
                raise ValueError('invalid content length')
            length = int(lengths[0])
            if length < 1 or length > self.server.config['max_request_bytes']:
                raise ValueError('request exceeds configured limit')
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError('incomplete request')
            value = json.loads(raw)
            if self.path in ('/api/verify-source', '/api/source-context'):
                if (not isinstance(value, dict) or set(value) != {'id', 'annotation_sha256'}
                        or not isinstance(value['id'], str) or not isinstance(value['annotation_sha256'], str)):
                    raise ValueError('invalid source verification request')
                if self.path == '/api/source-context':
                    limits = self.server.config.get('source_context')
                    if limits is None:
                        raise ValueError('source context is not configured')
                    result = verify_source(self.server.workspace, value['id'], value['annotation_sha256'],
                                           self.server.config.get('source_verification'), limits)
                    self.json_reply(200, result)
                else:
                    status = verify_source(self.server.workspace, value['id'], value['annotation_sha256'],
                                           self.server.config.get('source_verification'),
                                           checkpoint_limits=self.server.config.get('checkpoint_verification'))
                    self.json_reply(200, {'status': status})
                return
            required = {'id', 'annotation_sha256', 'decision', 'confirmation', 'labels'}
            if not isinstance(value, dict) or not required <= set(value) or set(value) - (required | {'redactions', 'guide_sha256'}):
                raise ValueError('invalid decision')
            decision = value['decision']
            if decision not in ('approved', 'rejected') or value['confirmation'] != ('APPROVE' if decision == 'approved' else 'REJECT'):
                raise ValueError('explicit decision confirmation required')
            if not isinstance(value['id'], str) or not isinstance(value['annotation_sha256'], str):
                raise ValueError('invalid record identity')
            labels = value['labels'] if decision == 'approved' else None
            if decision == 'approved' and not isinstance(labels, dict):
                raise ValueError('label groups are required')
            guide_sha256 = None
            if decision == 'approved' and self.server.guide_path is not None:
                plan, _, _ = review_corpus.load_workspace(self.server.workspace)
                taxonomy, taxonomy_hash = evaluate.load_taxonomy(review_corpus.private_path(self.server.workspace, plan['taxonomy']))
                _, current_guide = load_guide(self.server.guide_path, taxonomy)
                if (taxonomy_hash != self.server.taxonomy_hash or current_guide != self.server.guide_sha256
                        or value.get('guide_sha256') != current_guide):
                    raise ValueError('label guide or taxonomy changed; restart review')
                guide_sha256 = current_guide
            review_corpus.record_decision(self.server.workspace, value['id'], self.server.reviewer,
                                          decision, value['annotation_sha256'], labels,
                                          value.get('redactions'), self.server.config['max_prompt_bytes'], guide_sha256)
            self.json_reply(200, {'saved': True, 'decision': decision})
        except (OSError, ValueError, KeyError, TypeError, UnicodeError, socket.timeout) as error:
            self.json_reply(400, {'error': str(error)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--reviewer', required=True)
    parser.add_argument('--port', type=int, default=0, help='loopback port; 0 selects an available port')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--label-guide', type=Path, default=DEFAULT_GUIDE,
                        help='label definitions matching the workspace taxonomy')
    args = parser.parse_args()
    try:
        if not 0 <= args.port <= 65535:
            raise ValueError('invalid port')
        config = load_config(args.config)
        token = secrets.token_urlsafe(32)
        with train.run_lock(args.workspace / 'review'):
            with ReviewServer(args.workspace, args.reviewer, args.port, config, token, args.label_guide) as server:
                print(f'{server.origin}/#{token}', flush=True)
                print('Local review is running. Open the URL in your browser; press Ctrl+C to stop.', flush=True)
                try:
                    server.serve_forever()
                except KeyboardInterrupt:
                    pass
    except (OSError, ValueError, KeyError, TypeError, UnicodeError) as error:
        raise SystemExit(f'local review server rejected: {error}') from error


if __name__ == '__main__':
    main()
