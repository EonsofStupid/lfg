#!/usr/bin/env python3
"""Serve aggregate local-teacher progress on loopback without exposing records."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
from pathlib import Path
import re
import subprocess


def status(selection: Path, output: Path, unit: str) -> dict:
    try:
        with (selection / 'manifest.json').open('rb') as stream:
            raw = stream.read(1048577)
        if len(raw) > 1048576:
            raise ValueError('selection manifest too large')
        manifest = json.loads(raw)
        total = manifest['requests']
        if manifest.get('schema_version') != 'lfg.teacher-request-selection-result.v1' or type(total) is not int or total < 0:
            raise ValueError('invalid selection manifest')
        def count(folder):
            if not folder.exists():
                return 0
            if folder.is_symlink() or not folder.is_dir():
                raise ValueError('invalid collection directory')
            return sum(1 for path in folder.iterdir() if re.fullmatch(r'[0-9]{6}\.json', path.name)
                       and path.is_file() and not path.is_symlink())
        completed = count(output / 'records')
        failed = count(output / 'failures')
        if completed + failed > total:
            raise ValueError('collection count exceeds selection')
        p = subprocess.run(['systemctl', '--user', 'is-active', unit], capture_output=True,
                           text=True, timeout=2, check=False)
        state = p.stdout.strip()
        if state not in ('active', 'inactive', 'activating', 'deactivating', 'failed'):
            state = 'unavailable'
        return {'state': state, 'completed': completed, 'failed': failed, 'requested': total}
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        return {'state': 'unavailable', 'completed': 0, 'failed': 0, 'requested': 0}


def handler(selection, output, unit):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != '/api/status':
                self.send_error(404)
                return
            data = json.dumps(status(selection, output, unit), allow_nan=False).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format, *args):
            pass
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--unit', required=True)
    parser.add_argument('--listen', default='127.0.0.1')
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()
    if not ipaddress.ip_address(args.listen).is_loopback:
        parser.error('status service must bind to loopback')
    ThreadingHTTPServer((args.listen, args.port), handler(args.selection, args.output, args.unit)).serve_forever()


if __name__ == '__main__':
    main()
