#!/usr/bin/env python3
"""Read-only loopback dashboard for one owned-model training service."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
from pathlib import Path
import time


PAGE = b'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LFG training</title><style>
body{background:#111827;color:#f9fafb;font:16px system-ui;margin:0;padding:2rem}
main{max-width:720px;margin:auto}section{background:#1f2937;border-radius:12px;padding:1.5rem;margin:1rem 0}
h1{margin-bottom:.3rem}p{line-height:1.55;color:#d1d5db}strong{color:#fff}
progress{width:100%;height:1.4rem;accent-color:#38bdf8}small{color:#d1d5db}
</style><main><h1>LFG training</h1><p>Private developer view</p><section>
<p>Status: <strong id="state">Connecting</strong></p><progress id="progress" max="1" value="0"></progress>
<p id="steps">Waiting for the trainer</p><small id="updated"></small></section>
<section><p>This is an experimental owned-model run. A completed training run does not establish assessment quality or release readiness.</p></section>
</main><script>
async function refresh(){try{const r=await fetch('api/status',{cache:'no-store'});if(!r.ok)throw Error();
const d=await r.json();document.getElementById('state').textContent=d.state;
const p=document.getElementById('progress');p.max=d.total_steps||1;p.value=d.step||0;
document.getElementById('steps').textContent=(d.step||0).toLocaleString()+' / '+(d.total_steps||0).toLocaleString()+' steps';
document.getElementById('updated').textContent=d.updated_unix?'Last update: '+new Date(d.updated_unix*1000).toLocaleString():'No status yet';
}catch(e){document.getElementById('state').textContent='Status unavailable'}}refresh();setInterval(refresh,5000);
</script></html>'''


def status(path: Path) -> dict:
    try:
        with path.open('rb') as stream:
            raw = stream.read(65537)
        if len(raw) > 65536:
            raise ValueError('oversized status')
        value = json.loads(raw)
        if value.get('schema_version') != 'lfg.owned-training-status.v1':
            raise ValueError('invalid status')
        state = value.get('state')
        step = value.get('step', 0)
        total = value.get('total_steps', step)
        updated = value.get('updated_unix')
        if (state not in ('training', 'complete', 'failed', 'needs_attention')
                or type(step) is not int or step < 0 or type(total) is not int or total < step
                or type(updated) not in (int, float) or not 0 < updated <= time.time() + 300):
            raise ValueError('invalid status fields')
        return {'state': state, 'step': step, 'total_steps': total, 'updated_unix': updated}
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {'state': 'unavailable', 'step': 0, 'total_steps': 0, 'updated_unix': None}


def handler(path: Path):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/':
                content, content_type = PAGE, 'text/html; charset=utf-8'
            elif self.path == '/api/status':
                content = json.dumps(status(path), allow_nan=False).encode('utf-8')
                content_type = 'application/json; charset=utf-8'
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, format, *args):
            pass
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--status', type=Path, required=True)
    parser.add_argument('--listen', default='127.0.0.1')
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()
    if not ipaddress.ip_address(args.listen).is_loopback:
        parser.error('dashboard must bind to a loopback address')
    ThreadingHTTPServer((args.listen, args.port), handler(args.status)).serve_forever()


if __name__ == '__main__':
    main()
