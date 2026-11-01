#!/usr/bin/env python3
"""Review a private per-node ledger on an authenticated loopback server."""
from __future__ import annotations

import argparse
import json
import secrets
import socket
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import node_review
import review_web
import inspect_codex_inventory
import inspect_claude_inventory

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'node-review-web'
DEFAULT_CONFIG = ROOT / 'node-review-web-config.example.json'


def read_limited(path: Path, limit: int) -> bytes:
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError('file exceeds configured limit')
    return raw


def load_config(path: Path) -> dict:
    value = json.loads(path.read_bytes())
    fields = {'maxRequestBytes', 'maxPromptBytes', 'maxQueueBytes', 'maxLedgerBytes',
              'maxGuideBytes', 'maxSourceBytes', 'requestTimeoutSeconds'}
    if (not isinstance(value, dict) or set(value) != fields | {'schemaVersion'}
            or value['schemaVersion'] != 'lfg.node-review-web-config.v1'
            or any(type(value[field]) is not int or value[field] < 1 for field in fields)):
        raise ValueError('invalid node review web configuration')
    return value


def snapshot_choices(snapshot: Path) -> tuple[dict[str, list[str]], str]:
    raw = snapshot.read_bytes()
    spec = json.loads(raw)
    if not isinstance(spec, dict) or spec.get('schemaVersion') != 'lfg.kernel-snapshot.v1' or not isinstance(spec.get('nodes'), dict):
        raise ValueError('invalid snapshot')
    nodes = {}
    for name, value in spec['nodes'].items():
        if isinstance(value, dict) and value.get('kind') == 'choice':
            options = value.get('options')
            if not node_review.nonempty(name) or not isinstance(options, dict) or not options or any(not node_review.nonempty(key) for key in options):
                raise ValueError('invalid snapshot choice')
            nodes[name] = sorted(options)
    if not nodes or snapshot.read_bytes() != raw:
        raise ValueError('snapshot has no choices or changed while reading')
    return nodes, node_review.digest(raw)


def template(snapshot: Path, output: Path, selected: list[str] | None = None) -> dict:
    nodes, snapshot_sha = snapshot_choices(snapshot)
    if selected:
        if len(set(selected)) != len(selected) or set(selected) - set(nodes):
            raise ValueError('template selection contains duplicate or unknown nodes')
        nodes = {name: nodes[name] for name in selected}
    value = {'schemaVersion': 'lfg.node-review-guide.v1', 'revision': 'draft', 'status': 'draft',
             'notice': 'Draft template; define and review every choice before enabling approvals.',
             'snapshotSha256': snapshot_sha, 'nodes': {
                 name: {'question': '', 'options': {option: {'meaning': '', 'boundary': ''} for option in choices}}
                 for name, choices in nodes.items()}}
    raw = node_review.encode(value) + b'\n'
    node_review.create_new(output, raw)
    return {'guideSha256': node_review.digest(raw), 'status': 'draft', 'nodes': len(nodes)}


def load_guide(path: Path, snapshot: Path, limit: int) -> tuple[dict, str]:
    raw = read_limited(path, limit)
    value = json.loads(raw)
    nodes, snapshot_sha = snapshot_choices(snapshot)
    if (not isinstance(value, dict) or set(value) != {'schemaVersion', 'revision', 'status', 'notice', 'snapshotSha256', 'nodes'}
            or value['schemaVersion'] != 'lfg.node-review-guide.v1' or not node_review.nonempty(value['revision'])
            or value['status'] not in ('draft', 'ready') or not isinstance(value['notice'], str)
            or not value['notice'].strip() or value['snapshotSha256'] != snapshot_sha
            or not isinstance(value['nodes'], dict) or not value['nodes'] or set(value['nodes']) - set(nodes)):
        raise ValueError('invalid or snapshot-incompatible node review guide')
    for name, entry in value['nodes'].items():
        if (not isinstance(entry, dict) or set(entry) != {'question', 'options'}
                or not isinstance(entry['question'], str) or not isinstance(entry['options'], dict)
                or set(entry['options']) != set(nodes[name])):
            raise ValueError('guide choices differ from snapshot')
        texts = [entry['question']]
        for option in entry['options'].values():
            if (not isinstance(option, dict) or set(option) != {'meaning', 'boundary'}
                    or any(not isinstance(option[key], str) for key in ('meaning', 'boundary'))):
                raise ValueError('invalid choice guidance')
            texts.extend((option['meaning'], option['boundary']))
        if value['status'] == 'ready' and any(not text.strip() for text in texts):
            raise ValueError('ready guide requires a question and guidance for every option')
    if read_limited(path, limit) != raw:
        raise ValueError('guide changed while reading')
    return value, node_review.digest(raw)


def source_metadata(path: Path | None, item: dict, limit: int) -> dict:
    if path is None:
        return {'status': 'not_configured'}
    raw = read_limited(path, limit)
    if node_review.digest(raw) != item['sourceFileSha256']:
        return {'status': 'source_digest_mismatch'}
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    matches = [row for row in rows if isinstance(row, dict) and row.get('id') == item['requestId']]
    if len(matches) != 1 or matches[0].get('text') != item['text']:
        return {'status': 'source_record_mismatch'}
    if read_limited(path, limit) != raw:
        raise ValueError('source changed while reading')
    row = matches[0]
    return {'status': 'source_file_and_text_match', 'source': row.get('source'),
            'family': row.get('family'), 'reviewNote': row.get('review_note')}


class NodeServer(HTTPServer):
    def __init__(self, queue: Path, ledger: Path, snapshot: Path, guide: Path, source: Path | None,
                 reviewer: str, port: int, config: dict, token: str, context_config: dict | None = None):
        if not node_review.nonempty(reviewer):
            raise ValueError('reviewer must be a nonempty single-line identifier')
        self.queue, self.ledger, self.snapshot, self.guide = [path.resolve(strict=True) for path in (queue, ledger, snapshot, guide)]
        self.source = source.resolve(strict=True) if source is not None else None
        self.reviewer, self.config, self.token = reviewer, config, token
        self.context_config = context_config
        if context_config is not None and (self.source is None or 'source_verification' not in context_config
                                           or 'source_context' not in context_config):
            raise ValueError('source context requires a source file and configured limits')
        self.load()
        super().__init__(('127.0.0.1', port), NodeHandler)

    @property
    def origin(self) -> str:
        return f'http://127.0.0.1:{self.server_port}'

    def load(self) -> tuple[list[dict], dict, dict, str]:
        read_limited(self.queue, self.config['maxQueueBytes'])
        read_limited(self.ledger, self.config['maxLedgerBytes'])
        items, ledger, _ = node_review.load_ledger(self.queue, self.snapshot, self.ledger)
        guide, guide_sha = load_guide(self.guide, self.snapshot, self.config['maxGuideBytes'])
        if any(item['node'] not in guide['nodes'] for item in items):
            raise ValueError('guide does not cover every queue node')
        return items, ledger, guide, guide_sha


class NodeHandler(review_web.ReviewHandler):
    server: NodeServer

    def setup(self) -> None:
        BaseHTTPRequestHandler.setup(self)
        self.connection.settimeout(self.server.config['requestTimeoutSeconds'])

    def do_GET(self) -> None:
        if not self.allowed_host():
            self.json_reply(403, {'error': 'invalid host'})
            return
        parsed = urlsplit(self.path)
        assets = {'/': (ASSETS / 'index.html', 'text/html; charset=utf-8'),
                  '/app.js': (ASSETS / 'app.js', 'text/javascript; charset=utf-8'),
                  '/app.css': (ROOT / 'review-web/app.css', 'text/css; charset=utf-8'),
                  '/node.css': (ASSETS / 'node.css', 'text/css; charset=utf-8')}
        if parsed.path in assets and not parsed.query:
            path, mime = assets[parsed.path]
            self.reply(200, path.read_bytes(), mime)
            return
        if not self.authorized():
            self.json_reply(403, {'error': 'not authorized'})
            return
        try:
            items, ledger, guide, guide_sha = self.server.load()
            decisions = {row['itemSha256']: row['decision'] for row in ledger['records']}
            pending = [item for item in items if decisions[node_review.identity(item)] == 'pending']
            if parsed.path == '/api/status' and not parsed.query:
                counts = {state: sum(value == state for value in decisions.values()) for state in ('pending', 'approved', 'rejected')}
                self.json_reply(200, {'reviewer': self.server.reviewer, 'decisions': counts,
                                      'guideStatus': guide['status'], 'guideSha256': guide_sha,
                                      'sourceContextEnabled': self.server.context_config is not None})
                return
            if parsed.path == '/api/item':
                query = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=True)
                if set(query) != {'offset'} or len(query['offset']) != 1 or not query['offset'][0].isdigit():
                    raise ValueError('invalid pending-record offset')
                offset = int(query['offset'][0])
                if offset >= len(pending):
                    self.json_reply(200, {'item': None, 'pending': len(pending), 'offset': offset})
                    return
                item = pending[offset]
                if len(item['text'].encode('utf-8')) > self.server.config['maxPromptBytes']:
                    self.json_reply(200, {'item': None, 'blocked': 'Request exceeds configured display limit.',
                                          'pending': len(pending), 'offset': offset})
                    return
                source = source_metadata(self.server.source, item, self.server.config['maxSourceBytes'])
                self.json_reply(200, {'item': {'itemSha256': node_review.identity(item), 'text': item['text'],
                    'node': item['node'], 'inputSha256': item['inputSha256'], 'sourceFileSha256': item['sourceFileSha256'],
                    'observationSequence': item['observationSequence'], 'journalTailSha256': item['journalTailSha256'],
                    'source': source}, 'guide': guide['nodes'][item['node']], 'guideStatus': guide['status'],
                    'guideNotice': guide['notice'], 'guideSha256': guide_sha, 'pending': len(pending), 'offset': offset})
                return
            self.json_reply(404, {'error': 'not found'})
        except (OSError, ValueError, KeyError, TypeError, UnicodeError, socket.timeout) as error:
            self.json_reply(400, {'error': str(error)})

    def do_POST(self) -> None:
        if not self.authorized(post=True):
            self.reject_unauthorized_post(self.server.config['maxRequestBytes'])
            return
        if self.path not in ('/api/decision', '/api/source-context') or self.headers.get('Transfer-Encoding') or self.headers.get_content_type() != 'application/json':
            self.json_reply(400, {'error': 'invalid request'})
            return
        try:
            lengths = self.headers.get_all('Content-Length', [])
            if len(lengths) != 1 or not lengths[0].isdigit():
                raise ValueError('invalid content length')
            length = int(lengths[0])
            if length < 1 or length > self.server.config['maxRequestBytes']:
                raise ValueError('request exceeds configured limit')
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError('incomplete request')
            value = json.loads(raw)
            if self.path == '/api/source-context':
                if not isinstance(value, dict) or set(value) != {'itemSha256'} or not isinstance(value['itemSha256'], str):
                    raise ValueError('invalid source context request')
                self.json_reply(200, self.source_context(value['itemSha256']))
                return
            common = {'itemSha256', 'guideSha256', 'decision', 'confirmation', 'attest'}
            approved = {'choice', 'project', 'projectGroup', 'family', 'split'}
            if (not isinstance(value, dict) or value.get('decision') not in ('approved', 'rejected')
                    or set(value) != (common | approved if value['decision'] == 'approved' else common)
                    or value.get('attest') is not True
                    or value.get('confirmation') != ('APPROVE' if value['decision'] == 'approved' else 'REJECT')
                    or not isinstance(value.get('itemSha256'), str)):
                raise ValueError('explicit decision confirmation is required')
            _, _, guide, guide_sha = self.server.load()
            if value['guideSha256'] != guide_sha:
                raise ValueError('guide changed; reload before deciding')
            if value['decision'] == 'approved' and guide['status'] != 'ready':
                raise ValueError('draft guide cannot be used for approval')
            args = argparse.Namespace(queue=self.server.queue, ledger=self.server.ledger, snapshot=self.server.snapshot,
                item_sha256=value['itemSha256'], reviewer=self.server.reviewer, decision=value['decision'], attest=True,
                choice=value.get('choice'), project=value.get('project'), project_group=value.get('projectGroup'),
                family=value.get('family'), split=value.get('split'),
                guide_sha256=guide_sha if value['decision'] == 'approved' else None)
            node_review.decide(args)
            self.json_reply(200, {'saved': True, 'decision': value['decision']})
        except (OSError, ValueError, KeyError, TypeError, UnicodeError, socket.timeout) as error:
            self.json_reply(400, {'error': str(error)})

    def source_context(self, identifier: str) -> dict:
        config = self.server.context_config
        if config is None or self.server.source is None:
            raise ValueError('source context is not configured')
        items, ledger, _, _ = self.server.load()
        matches = [item for item in items if node_review.identity(item) == identifier]
        decisions = [row for row in ledger['records'] if row['itemSha256'] == identifier]
        if len(matches) != 1 or len(decisions) != 1 or decisions[0]['decision'] != 'pending':
            raise ValueError('only a pending item can be inspected')
        item = matches[0]
        metadata = source_metadata(self.server.source, item, self.server.config['maxSourceBytes'])
        if metadata['status'] != 'source_file_and_text_match' or not isinstance(metadata.get('source'), dict):
            raise ValueError('source file and request text must match before inspecting context')
        source = metadata['source']
        limits, context = config['source_verification'], config['source_context']
        if source.get('kind') == 'codex-session-user-text':
            entry = {**source, 'schema_version': 'lfg.unreviewed-codex-message.v2'}
            text, messages, truncated = inspect_codex_inventory.prior_user_context(
                entry, limits['max_session_bytes'], limits['max_line_bytes'],
                context['max_messages'], context['max_context_bytes'])
            scope = 'prior_user_messages_in_session'
        elif source.get('kind') == 'claude-session-human-text':
            if 'max_ancestry_records' not in context:
                raise ValueError('Claude context requires a configured ancestry limit')
            entry = {**source, 'schema_version': 'lfg.unreviewed-claude-message.v1'}
            text, messages, truncated = inspect_claude_inventory.prior_user_context(
                entry, limits['max_session_bytes'], limits['max_line_bytes'],
                context['max_messages'], context['max_context_bytes'], context['max_ancestry_records'])
            scope = 'same_session_source_parent_chain'
        else:
            raise ValueError('context is unavailable for this source type')
        if text != item['text']:
            raise ValueError('review text differs from the verified original')
        items_after, ledger_after, _, _ = self.server.load()
        if (items_after != items or ledger_after != ledger
                or source_metadata(self.server.source, item, self.server.config['maxSourceBytes']) != metadata):
            raise ValueError('review or source changed during context inspection')
        return {'status': 'source_prefix_and_original_text_match', 'contextScope': scope,
                'messages': messages, 'truncated': truncated}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    draft = sub.add_parser('template')
    draft.add_argument('--snapshot', type=Path, required=True)
    draft.add_argument('--output', type=Path, required=True)
    draft.add_argument('--node', action='append', help='include a choice node; omit to include all')
    serve = sub.add_parser('serve')
    for field in ('queue', 'ledger', 'snapshot', 'guide'):
        serve.add_argument('--' + field, type=Path, required=True)
    serve.add_argument('--source', type=Path)
    serve.add_argument('--context-config', type=Path, help='opt in to bounded source context using review-web configuration')
    serve.add_argument('--reviewer', required=True)
    serve.add_argument('--port', type=int, default=0)
    serve.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    if args.command == 'template':
        print(json.dumps(template(args.snapshot, args.output, args.node)))
    else:
        if not 0 <= args.port <= 65535:
            raise ValueError('invalid port')
        config = load_config(args.config)
        context_config = review_web.load_config(args.context_config) if args.context_config else None
        token = secrets.token_urlsafe(32)
        with NodeServer(args.queue, args.ledger, args.snapshot, args.guide, args.source,
                        args.reviewer, args.port, config, token, context_config) as server:
            print(f'{server.origin}/#{token}', flush=True)
            print('Local node review is running. Open the URL; press Ctrl+C to stop.', flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, UnicodeError) as error:
        raise SystemExit(f'node review web: {error}') from error
