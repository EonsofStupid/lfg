#!/usr/bin/env python3
"""Accept authenticated dataset submissions on a loopback-only HTTP endpoint."""
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import hmac
import ipaddress
import json
from pathlib import Path
import socket

import receive_teacher_dataset as receiver


def handler(inbox: Path, token_file: Path, max_payload: int, max_dataset: int, timeout: int):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(timeout)

        def reply(self, status: int, body: dict):
            data = json.dumps(body, sort_keys=True).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            if self.path != '/api/datasets':
                self.reply(404, {'error': 'not found'})
                return
            try:
                token = token_file.read_bytes()
                supplied = self.headers.get('Authorization', '')
                expected = 'Bearer ' + token.decode('ascii')
                if (not 32 <= len(token) <= 512 or not hmac.compare_digest(supplied, expected)):
                    self.reply(401, {'error': 'unauthorized'})
                    return
                length_header = self.headers.get('Content-Length', '')
                if self.headers.get('Transfer-Encoding') or not length_header.isdecimal():
                    self.reply(400, {'error': 'invalid content length'})
                    return
                length = int(length_header)
                if length < 1 or length > max_payload:
                    self.reply(413, {'error': 'invalid payload size'})
                    return
                raw = self.rfile.read(length)
                if len(raw) != length:
                    self.reply(400, {'error': 'incomplete payload'})
                    return
                result = receiver.receive(raw, inbox, max_payload, max_dataset)
                self.reply(200, result)
            except (OSError, ValueError, TypeError, UnicodeError, json.JSONDecodeError, socket.timeout):
                self.reply(400, {'error': 'dataset rejected'})

        def log_message(self, format, *args):
            pass
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inbox', type=Path, required=True)
    parser.add_argument('--token-file', type=Path, required=True)
    parser.add_argument('--max-payload-bytes', type=int, required=True)
    parser.add_argument('--max-dataset-bytes', type=int, required=True)
    parser.add_argument('--timeout-seconds', type=int, required=True)
    parser.add_argument('--listen', default='127.0.0.1')
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()
    if (not ipaddress.ip_address(args.listen).is_loopback or not args.inbox.is_absolute()
            or not args.token_file.is_absolute() or min(args.max_payload_bytes, args.max_dataset_bytes,
                                                        args.timeout_seconds) < 1):
        parser.error('loopback, absolute paths and positive limits are required')
    HTTPServer((args.listen, args.port), handler(args.inbox, args.token_file, args.max_payload_bytes,
                                                args.max_dataset_bytes, args.timeout_seconds)).serve_forever()


if __name__ == '__main__':
    main()
