#!/usr/bin/env python3
"""Serve a private corpus-review workspace on an authenticated loopback address."""

from __future__ import annotations

import argparse
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
import review_corpus
import train

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'review-web'
DEFAULT_CONFIG = ROOT / 'review-web-config.v1.json'


def load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    fields = {'max_request_bytes', 'max_prompt_bytes', 'request_timeout_seconds'}
    if not isinstance(value, dict) or set(value) != fields | {'schema_version'} or \
            value['schema_version'] != 'lfg.review-web-config.v1' or \
            any(type(value[name]) is not int or value[name] < 1 for name in fields):
        raise ValueError('invalid local review web configuration')
    return value


class ReviewServer(HTTPServer):
    def __init__(self, workspace: Path, reviewer: str, port: int, config: dict[str, Any], token: str):
        if not reviewer.strip() or '\n' in reviewer or '\r' in reviewer:
            raise ValueError('reviewer must be a nonempty single-line identifier')
        self.workspace = workspace.resolve(strict=True)
        self.reviewer = reviewer.strip()
        self.config = config
        self.token = token
        review_corpus.load_workspace(self.workspace)
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
                if set(query) != {'offset'} or len(query['offset']) != 1 or not query['offset'][0].isdigit():
                    raise ValueError('invalid pending-record offset')
                offset = int(query['offset'][0])
                plan, _, ledger = review_corpus.load_workspace(self.server.workspace)
                pending = [identifier for identifier, item in ledger.items() if item['decision'] == 'pending']
                if offset >= len(pending):
                    self.json_reply(200, {'item': None, 'pending': len(pending), 'offset': offset})
                    return
                identifier = pending[offset]
                _, rows, index, source = review_corpus.find_row(self.server.workspace, plan, identifier)
                row = rows[index]
                taxonomy, taxonomy_hash = evaluate.load_taxonomy(
                    review_corpus.private_path(self.server.workspace, plan['taxonomy']))
                if len(row['text'].encode('utf-8')) > self.server.config['max_prompt_bytes']:
                    self.json_reply(200, {'item': None, 'blocked': 'Request exceeds configured display limit.',
                                          'pending': len(pending), 'offset': offset})
                    return
                annotation = corpus_audit.annotation_digest(row, source['project'], source['split'], taxonomy_hash)
                self.json_reply(200, {'item': {'id': identifier, 'text': row['text'], 'source': row['source'],
                                               'review_note': row.get('review_note'), 'labels': {k: row[k] for k in evaluate.FIELDS},
                                               'labels_pending': row.get('labels_pending', False),
                                               'project': source['project'], 'split': source['split'],
                                               'annotation_sha256': annotation,
                                               'pending_changed': not row['verified'] and ledger[identifier]['annotation_sha256'] != annotation,
                                               'interrupted_write': bool(row['verified'])},
                                      'taxonomy': {k: sorted(v) for k, v in taxonomy.items()},
                                      'pending': len(pending), 'offset': offset})
                return
            self.json_reply(404, {'error': 'not found'})
        except (OSError, ValueError, KeyError, TypeError, UnicodeError) as error:
            self.json_reply(400, {'error': str(error)})

    def do_POST(self) -> None:
        if not self.authorized(post=True):
            self.json_reply(403, {'error': 'not authorized'})
            return
        if self.path != '/api/decision' or self.headers.get('Transfer-Encoding') or \
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
            if not isinstance(value, dict) or set(value) != {'id', 'annotation_sha256', 'decision', 'confirmation', 'labels'}:
                raise ValueError('invalid decision')
            decision = value['decision']
            if decision not in ('approved', 'rejected') or value['confirmation'] != ('APPROVE' if decision == 'approved' else 'REJECT'):
                raise ValueError('explicit decision confirmation required')
            if not isinstance(value['id'], str) or not isinstance(value['annotation_sha256'], str):
                raise ValueError('invalid record identity')
            labels = value['labels'] if decision == 'approved' else None
            if decision == 'approved' and not isinstance(labels, dict):
                raise ValueError('label groups are required')
            review_corpus.record_decision(self.server.workspace, value['id'], self.server.reviewer,
                                          decision, value['annotation_sha256'], labels)
            self.json_reply(200, {'saved': True, 'decision': decision})
        except (OSError, ValueError, KeyError, TypeError, UnicodeError, socket.timeout) as error:
            self.json_reply(400, {'error': str(error)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--reviewer', required=True)
    parser.add_argument('--port', type=int, default=0, help='loopback port; 0 selects an available port')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    try:
        if not 0 <= args.port <= 65535:
            raise ValueError('invalid port')
        config = load_config(args.config)
        token = secrets.token_urlsafe(32)
        with train.run_lock(args.workspace / 'review'):
            with ReviewServer(args.workspace, args.reviewer, args.port, config, token) as server:
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
