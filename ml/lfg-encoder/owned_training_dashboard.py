#!/usr/bin/env python3
"""Read-only loopback dashboard for one owned-model training service."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit

import owned_training_metrics

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
<section id="metrics" hidden><h2>Initial training metrics</h2><p id="metrics-result">Waiting</p>
<small>Training loss is not assessment accuracy.</small></section>
<section id="teacher" hidden><h2>Local teacher collection</h2><p id="teacher-state">Connecting</p>
<progress id="teacher-progress" max="1" value="0"></progress><p id="teacher-count"></p></section>
<section id="evaluation" hidden><h2>Development format check</h2><p id="evaluation-result">Waiting</p>
<small>This checks JSON structure, not assessment accuracy.</small></section>
<section id="queue" hidden><h2>Continual training queue</h2><p id="queue-result">Waiting</p></section>
<section id="additional" hidden><h2>Additional training run</h2><p id="additional-result">Waiting</p></section>
<section><p>This is an experimental owned-model run. A completed training run does not establish assessment quality or release readiness.</p></section>
</main><script>
const teacherUrl=__TEACHER_URL__;
async function refresh(){try{const r=await fetch('api/status',{cache:'no-store'});if(!r.ok)throw Error();
const d=await r.json();document.getElementById('state').textContent=d.state;
const p=document.getElementById('progress');p.max=d.total_steps||1;p.value=d.step||0;
document.getElementById('steps').textContent=(d.step||0).toLocaleString()+' / '+(d.total_steps||0).toLocaleString()+' steps';
document.getElementById('updated').textContent=d.updated_unix?'Last update: '+new Date(d.updated_unix*1000).toLocaleString():'No status yet';
}catch(e){document.getElementById('state').textContent='Status unavailable'}
if(teacherUrl){document.getElementById('teacher').hidden=false;try{
const r=await fetch(teacherUrl,{cache:'no-store'});if(!r.ok)throw Error();const d=await r.json();
document.getElementById('teacher-state').textContent='Status: '+d.state;
const p=document.getElementById('teacher-progress');p.max=d.requested||1;p.value=(d.completed||0)+(d.failed||0);
document.getElementById('teacher-count').textContent=(d.completed||0)+' completed, '+(d.failed||0)+' incomplete / '+(d.requested||0)+' requests';
}catch(e){document.getElementById('teacher-state').textContent='Status unavailable'}}
try{const r=await fetch('api/evaluation',{cache:'no-store'});if(r.ok){const d=await r.json();
document.getElementById('evaluation').hidden=false;document.getElementById('evaluation-result').textContent=
d.state==='complete'?(d.required_fields_present+' / '+d.total+' outputs contain the required fields'):
(d.state==='pending'?'Waiting for the follow-on evaluation':
(d.state==='evaluating'?'Evaluating the latest update':'Evaluation status unavailable'));}}
catch(e){}
try{const r=await fetch('api/metrics',{cache:'no-store'});if(r.ok){const d=await r.json();
document.getElementById('metrics').hidden=false;document.getElementById('metrics-result').textContent=
d.state==='available'?'Mean loss over '+d.loss_steps+' steps: '+d.mean_training_loss.toFixed(4)+'; '+
d.byte_positions.toLocaleString()+' byte positions':'Waiting for a completed checkpoint';}}
catch(e){}
try{const r=await fetch('api/queue',{cache:'no-store'});if(r.ok){const d=await r.json();
document.getElementById('queue').hidden=false;document.getElementById('queue-result').textContent=
'Status: '+d.state+'; '+d.records+' records; '+d.completed_updates+' completed updates'+
(d.total_steps?'; '+d.step+' / '+d.total_steps+' steps':'');}}
catch(e){}
try{const r=await fetch('api/additional-status',{cache:'no-store'});if(r.ok){const d=await r.json();
document.getElementById('additional').hidden=false;document.getElementById('additional-result').textContent=
'Status: '+d.state+'; '+d.step+' / '+d.total_steps+' steps';}}
catch(e){} }refresh();setInterval(refresh,5000);
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


def render_page(teacher_status_url: str | None) -> bytes:
    url = json.dumps(teacher_status_url).replace('<', '\\u003c').encode('utf-8')
    return PAGE.replace(b'__TEACHER_URL__', url)


def current_status(path: Path, next_status: Path | None) -> dict:
    return status(next_status if next_status is not None and next_status.exists() else path)


def evaluation_status(path: Path) -> dict:
    try:
        with path.open('rb') as stream:
            raw = stream.read(1048577)
        if len(raw) > 1048576:
            raise ValueError('evaluation report too large')
        value = json.loads(raw)
        counts = value.get('format_counts')
        if (value.get('schema_version') != 'lfg.owned-torch-format-evaluation.v1' or not isinstance(counts, dict)
                or any(key not in ('invalid_json', 'missing_required_fields', 'required_fields_present')
                       or type(count) is not int or count < 0 for key, count in counts.items())):
            raise ValueError('invalid evaluation report')
        return {'state': 'complete', 'required_fields_present': counts.get('required_fields_present', 0),
                'total': sum(counts.values())}
    except FileNotFoundError:
        return {'state': 'pending', 'required_fields_present': 0, 'total': 0}
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {'state': 'unavailable', 'required_fields_present': 0, 'total': 0}


def queue_status(path: Path) -> dict:
    try:
        with path.open('rb') as stream:
            raw = stream.read(65537)
        if len(raw) > 65536:
            raise ValueError('queue status too large')
        value = json.loads(raw)
        state = value.get('state')
        records, updates = value.get('records', 0), value.get('completed_updates', 0)
        step, total = value.get('step', 0), value.get('total_steps', 0)
        if (value.get('schema_version') != 'lfg.owned-dataset-queue-status.v1'
                or state not in ('waiting_for_initial_model', 'idle', 'training', 'failed')
                or type(records) is not int or records < 0 or type(updates) is not int or updates < 0
                or type(step) is not int or type(total) is not int or not 0 <= step <= total):
            raise ValueError('invalid queue status')
        return {'state': state, 'records': records, 'completed_updates': updates, 'step': step, 'total_steps': total}
    except FileNotFoundError:
        return {'state': 'pending', 'records': 0, 'completed_updates': 0, 'step': 0, 'total_steps': 0}
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {'state': 'unavailable', 'records': 0, 'completed_updates': 0, 'step': 0, 'total_steps': 0}


def current_evaluation(path: Path, update_status: Path | None) -> dict:
    if update_status is not None and update_status.exists():
        try:
            with update_status.open('rb') as stream:
                raw = stream.read(65537)
            if len(raw) > 65536:
                raise ValueError('update evaluation status too large')
            value = json.loads(raw)
            state = value.get('state')
            if value.get('schema_version') != 'lfg.owned-update-evaluator-status.v1':
                raise ValueError('invalid update evaluation status')
            if state in ('evaluating', 'failed'):
                return {'state': state, 'required_fields_present': 0, 'total': 0}
            if state == 'complete':
                valid, total = value.get('required_fields_present'), value.get('total')
                if type(valid) is not int or type(total) is not int or not 0 <= valid <= total:
                    raise ValueError('invalid update evaluation counts')
                return {'state': 'complete', 'required_fields_present': valid, 'total': total}
            if state != 'waiting':
                raise ValueError('invalid update evaluation state')
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return {'state': 'unavailable', 'required_fields_present': 0, 'total': 0}
    return evaluation_status(path)


def handler(path: Path, teacher_status_url: str | None = None, next_status: Path | None = None,
            evaluation_report: Path | None = None, queue_status_path: Path | None = None,
            update_evaluation_status: Path | None = None, metrics_root: Path | None = None,
            next_metrics_root: Path | None = None, metrics_max_bytes: int = 33554432,
            metrics_window: int = 100, metrics_refresh_seconds: int = 30,
            additional_status: Path | None = None):
    page = render_page(teacher_status_url)
    cache, cache_lock = {'time': 0.0, 'value': {'state': 'pending'}}, threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/':
                content, content_type = page, 'text/html; charset=utf-8'
            elif self.path == '/api/status':
                content = json.dumps(current_status(path, next_status), allow_nan=False).encode('utf-8')
                content_type = 'application/json; charset=utf-8'
            elif self.path == '/api/evaluation' and evaluation_report is not None:
                content = json.dumps(current_evaluation(evaluation_report, update_evaluation_status), allow_nan=False).encode('utf-8')
                content_type = 'application/json; charset=utf-8'
            elif self.path == '/api/queue' and queue_status_path is not None:
                content = json.dumps(queue_status(queue_status_path), allow_nan=False).encode('utf-8')
                content_type = 'application/json; charset=utf-8'
            elif self.path == '/api/additional-status' and additional_status is not None:
                content = json.dumps(status(additional_status), allow_nan=False).encode('utf-8')
                content_type = 'application/json; charset=utf-8'
            elif self.path == '/api/metrics' and metrics_root is not None:
                with cache_lock:
                    if time.monotonic() - cache['time'] >= metrics_refresh_seconds:
                        selected = next_metrics_root if next_metrics_root is not None and next_metrics_root.exists() else metrics_root
                        cache['value'] = owned_training_metrics.metrics(selected, metrics_max_bytes, metrics_window)
                        cache['time'] = time.monotonic()
                    content = json.dumps(cache['value'], allow_nan=False).encode('utf-8')
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
    parser.add_argument('--next-status', type=Path, help='optional follow-on job status, used once present')
    parser.add_argument('--evaluation-report', type=Path, help='optional follow-on development format report')
    parser.add_argument('--queue-status', type=Path, help='optional continual training queue status')
    parser.add_argument('--additional-status', type=Path, help='optional additional training run status')
    parser.add_argument('--update-evaluation-status', type=Path, help='optional evaluation status for continual updates')
    parser.add_argument('--metrics-root', type=Path, help='optional current job checkpoint directory')
    parser.add_argument('--next-metrics-root', type=Path, help='optional follow-on job checkpoint directory')
    parser.add_argument('--metrics-max-bytes', type=int, default=33554432)
    parser.add_argument('--metrics-window', type=int, default=100)
    parser.add_argument('--metrics-refresh-seconds', type=int, default=30)
    parser.add_argument('--teacher-status-url', help='optional relative URL for an aggregate teacher status endpoint')
    args = parser.parse_args()
    if not ipaddress.ip_address(args.listen).is_loopback:
        parser.error('dashboard must bind to a loopback address')
    if min(args.metrics_max_bytes, args.metrics_window, args.metrics_refresh_seconds) < 1:
        parser.error('metrics limits must be positive')
    if args.teacher_status_url and (urlsplit(args.teacher_status_url).scheme or urlsplit(args.teacher_status_url).netloc
                                    or '\\' in args.teacher_status_url):
        parser.error('teacher status URL must be relative')
    ThreadingHTTPServer((args.listen, args.port), handler(args.status, args.teacher_status_url, args.next_status,
                                                        args.evaluation_report, args.queue_status,
                                                        args.update_evaluation_status, args.metrics_root,
                                                        args.next_metrics_root, args.metrics_max_bytes, args.metrics_window,
                                                        args.metrics_refresh_seconds, args.additional_status)).serve_forever()


if __name__ == '__main__':
    main()
