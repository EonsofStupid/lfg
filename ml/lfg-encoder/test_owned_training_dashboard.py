import json
from http.server import ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import time
import unittest
import urllib.request

import owned_training_dashboard as dashboard


class DashboardStatusTests(unittest.TestCase):
    def test_additional_run_endpoint_exposes_only_sanitized_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'status.json'
            path.write_text(json.dumps({'schema_version': 'lfg.owned-training-status.v1', 'state': 'training',
                                        'step': 12, 'total_steps': 100, 'updated_unix': time.time(),
                                        'private_text': 'must not leave the host'}))
            server = ThreadingHTTPServer(('127.0.0.1', 0), dashboard.handler(path, additional_status=path))
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{server.server_port}/api/additional-status') as response:
                    result = json.load(response)
                self.assertEqual(result['step'], 12)
                self.assertNotIn('private_text', result)
            finally:
                server.shutdown()
                server.server_close()
                worker.join()

    def test_only_public_progress_fields_are_exposed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'status.json'
            path.write_text(json.dumps({'schema_version': 'lfg.owned-training-status.v1',
                'state': 'training', 'step': 10, 'total_steps': 100, 'updated_unix': time.time(),
                'private_path': '/secret/corpus', 'private_text': 'not public'}))
            result = dashboard.status(path)
            self.assertEqual(set(result), {'state', 'step', 'total_steps', 'updated_unix'})
            self.assertEqual(result['step'], 10)
            path.write_text('{')
            self.assertEqual(dashboard.status(path)['state'], 'unavailable')

    def test_teacher_url_is_escaped_when_embedded_in_page(self):
        page = dashboard.render_page('</script><script>bad</script>')
        self.assertNotIn(b'</script><script>bad</script>', page)
        self.assertIn(b'\\u003c/script>', page)

    def test_switches_to_follow_on_status_when_it_appears(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / 'first', Path(directory) / 'second'
            data = {'schema_version': 'lfg.owned-training-status.v1', 'state': 'training',
                    'step': 4, 'total_steps': 10, 'updated_unix': time.time()}
            first.write_text(json.dumps(data))
            self.assertEqual(dashboard.current_status(first, second)['step'], 4)
            data['step'] = 1
            second.write_text(json.dumps(data))
            self.assertEqual(dashboard.current_status(first, second)['step'], 1)

    def test_evaluation_summary_omits_private_report_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            self.assertEqual(dashboard.evaluation_status(path)['state'], 'pending')
            path.write_text(json.dumps({'schema_version': 'lfg.owned-torch-format-evaluation.v1',
                'format_counts': {'invalid_json': 9, 'required_fields_present': 1},
                'private_path': '/not/exposed'}))
            result = dashboard.evaluation_status(path)
            self.assertEqual(result, {'state': 'complete', 'required_fields_present': 1, 'total': 10})
            self.assertNotIn('private', json.dumps(result))

    def test_queue_summary_exposes_only_aggregate_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'queue.json'
            self.assertEqual(dashboard.queue_status(path)['state'], 'pending')
            path.write_text(json.dumps({'schema_version': 'lfg.owned-dataset-queue-status.v1',
                'state': 'training', 'records': 42, 'completed_updates': 2, 'private_path': '/not/exposed'}))
            self.assertEqual(dashboard.queue_status(path), {'state': 'training', 'records': 42, 'completed_updates': 2,
                                                            'step': 0, 'total_steps': 0})

    def test_latest_update_evaluation_replaces_prior_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            first, latest = Path(directory) / 'first.json', Path(directory) / 'latest.json'
            first.write_text(json.dumps({'schema_version': 'lfg.owned-torch-format-evaluation.v1',
                                         'format_counts': {'invalid_json': 9, 'required_fields_present': 1}}))
            self.assertEqual(dashboard.current_evaluation(first, latest)['required_fields_present'], 1)
            latest.write_text(json.dumps({'schema_version': 'lfg.owned-update-evaluator-status.v1',
                                          'state': 'complete', 'required_fields_present': 3, 'total': 10,
                                          'private_path': '/not/exposed'}))
            result = dashboard.current_evaluation(first, latest)
            self.assertEqual(result, {'state': 'complete', 'required_fields_present': 3, 'total': 10})


if __name__ == '__main__':
    unittest.main()
