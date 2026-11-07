"""Protect the merge decision against unrelated or incomplete CI results."""
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch


path = Path(__file__).resolve().parents[2] / 'ci' / 'check_ci_run.py'
spec = importlib.util.spec_from_file_location('check_ci_run', path)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class CheckedMergeCITests(unittest.TestCase):
    def open_pr(self, head, base):
        return {'state': 'open', 'draft': False, 'base': {'ref': 'main', 'sha': base},
                'head': {'sha': head, 'repo': {'full_name': 'owner/repo'}},
                'labels': [{'name': 'ready-to-auto-merge'}]}

    def test_stale_or_ineligible_pull_requests_stop_waiting(self):
        head, base = 'a' * 40, 'b' * 40
        original = self.open_pr(head, base)
        self.assertTrue(gate.still_eligible(original, 'owner/repo', head, base))
        for change in ({'state': 'closed'}, {'draft': True}, {'labels': []},
                       {'base': {'ref': 'main', 'sha': 'c' * 40}},
                       {'head': {'sha': 'd' * 40, 'repo': {'full_name': 'owner/repo'}}},
                       {'head': {'sha': head, 'repo': None}}):
            self.assertFalse(gate.still_eligible({**original, **change}, 'owner/repo', head, base))

    def run_record(self, number, head, base, identifier):
        return {'id': identifier, 'event': 'pull_request', 'head_sha': head,
                'pull_requests': [{'number': number, 'head': {'sha': head}, 'base': {'sha': base}}]}

    def test_only_exact_pr_head_and_base_count_and_newest_is_used(self):
        runs = [self.run_record(7, 'h', 'old', 8), self.run_record(8, 'h', 'b', 9),
                self.run_record(7, 'other', 'b', 10), self.run_record(7, 'h', 'b', 11),
                self.run_record(7, 'h', 'b', 12)]
        self.assertEqual(gate.matching_run(runs, 7, 'h', 'b')['id'], 12)
        self.assertIsNone(gate.matching_run(runs, 7, 'h', 'different'))

    def test_every_required_job_must_complete_successfully_once(self):
        required = ['rust', 'python-ml', 'ui', 'container']
        good = [{'name': name, 'status': 'completed', 'conclusion': 'success'} for name in required]
        self.assertTrue(gate.jobs_passed(good, required))
        self.assertFalse(gate.jobs_passed(good[:-1], required))
        self.assertFalse(gate.jobs_passed(good + [good[0]], required))
        for value in ('failure', 'skipped', 'cancelled', None):
            changed = [dict(item) for item in good]
            changed[-1]['conclusion'] = value
            self.assertFalse(gate.jobs_passed(changed, required))
        running = [dict(item) for item in good]
        running[-1]['status'] = 'in_progress'
        self.assertFalse(gate.jobs_passed(running, required))

    def test_cli_checks_the_successful_run_attempt_and_rejects_failure(self):
        head, base = 'a' * 40, 'b' * 40
        run = self.run_record(7, head, base, 12)
        run.update(status='completed', conclusion='success', run_attempt=2)
        required = ['rust', 'python-ml', 'ui', 'container']
        jobs = [{'name': name, 'status': 'completed', 'conclusion': 'success'} for name in required]
        args = ['check_ci_run.py', '--repository', 'owner/repo', '--pull-request', '7', '--head', head,
                '--base', base, '--timeout-seconds', '5', '--poll-seconds', '1']
        for name in required:
            args.extend(['--required-job', name])
        pr = self.open_pr(head, base)
        branch = {'commit': {'sha': base}}
        with patch.object(sys, 'argv', args), patch.object(gate, 'api', side_effect=[pr, branch, {'workflow_runs': [run]}, {'jobs': jobs}]) as api, redirect_stdout(io.StringIO()):
            gate.main()
        self.assertIn('/runs/12/attempts/2/jobs?', api.call_args_list[3].args[0])
        run['conclusion'] = 'failure'
        with patch.object(sys, 'argv', args), patch.object(gate, 'api', side_effect=[pr, branch, {'workflow_runs': [run]}]):
            with self.assertRaisesRegex(SystemExit, 'concluded failure'):
                gate.main()
        with patch.object(sys, 'argv', args), patch.object(gate, 'api', return_value={**pr, 'state': 'closed'}):
            with self.assertRaisesRegex(SystemExit, 'stop waiting'):
                gate.main()
        with patch.object(sys, 'argv', args), patch.object(gate, 'api', side_effect=[pr, {'commit': {'sha': 'c' * 40}}]):
            with self.assertRaisesRegex(SystemExit, 'main changed'):
                gate.main()

    def test_api_failure_reports_http_status_without_provider_diagnostic(self):
        diagnostic = 'BlobNotFound PRIVATE_TOKEN_DO_NOT_LOG (HTTP 404)'
        result = gate.subprocess.CompletedProcess(['gh', 'api'], 1, '', diagnostic)
        with patch.object(gate.subprocess, 'run', return_value=result):
            with self.assertRaises(gate.ApiFailure) as caught:
                gate.api('repos/owner/repo/pulls/7')
        self.assertEqual(caught.exception.status, 404)
        self.assertNotIn('PRIVATE_TOKEN_DO_NOT_LOG', str(caught.exception))
        self.assertNotIn('BlobNotFound', str(caught.exception))
        malformed = gate.subprocess.CompletedProcess(['gh', 'api'], 0, 'PRIVATE_BODY_DO_NOT_LOG', '')
        with patch.object(gate.subprocess, 'run', return_value=malformed):
            with self.assertRaises(gate.ApiFailure) as invalid:
                gate.api('repos/owner/repo/pulls/7')
        self.assertTrue(invalid.exception.__suppress_context__)
        self.assertNotIn('PRIVATE_BODY_DO_NOT_LOG', str(invalid.exception))

    def test_repeated_403_or_404_fails_actionably_after_retries(self):
        head, base = 'a' * 40, 'b' * 40
        args = ['check_ci_run.py', '--repository', 'owner/repo', '--pull-request', '7', '--head', head,
                '--base', base, '--timeout-seconds', '10', '--poll-seconds', '1',
                '--required-job', 'rust']
        for status in (401, 403, 404):
            with self.subTest(status=status):
                output = io.StringIO()
                with (patch.object(sys, 'argv', args), patch.object(gate, 'api',
                        side_effect=gate.ApiFailure(status=status, kind='http')) as api,
                     patch.object(gate.time, 'monotonic', return_value=0), patch.object(gate.time, 'sleep'),
                     redirect_stdout(output)):
                    guidance = ('verify GitHub authentication' if status == 401 else 'verify .*repository access')
                    with self.assertRaisesRegex(SystemExit, f'HTTP {status}.*{guidance}'):
                        gate.main()
                self.assertEqual(api.call_count, 3)
                self.assertIn(f'HTTP {status}; retry 1/3', output.getvalue())
                self.assertNotIn('BlobNotFound', output.getvalue())

    def test_retryable_http_and_rate_limit_statuses_retry_and_recover(self):
        head, base = 'a' * 40, 'b' * 40
        run = self.run_record(7, head, base, 12)
        run.update(status='completed', conclusion='success', run_attempt=1)
        pr, branch = self.open_pr(head, base), {'commit': {'sha': base}}
        jobs = [{'name': 'rust', 'status': 'completed', 'conclusion': 'success'}]
        args = ['check_ci_run.py', '--repository', 'owner/repo', '--pull-request', '7', '--head', head,
                '--base', base, '--timeout-seconds', '10', '--poll-seconds', '1', '--required-job', 'rust']
        for failure in (gate.ApiFailure(status=408, kind='http'),
                        gate.ApiFailure(status=425, kind='http'),
                        gate.ApiFailure(status=429, kind='http'),
                        gate.ApiFailure(status=403, kind='rate-limit')):
            with self.subTest(status=failure.label):
                output = io.StringIO()
                with (patch.object(sys, 'argv', args), patch.object(gate, 'api', side_effect=[
                        failure, pr, branch, {'workflow_runs': [run]}, {'jobs': jobs}]) as api,
                     patch.object(gate.time, 'monotonic', return_value=0), patch.object(gate.time, 'sleep'),
                     redirect_stdout(output)):
                    gate.main()
                self.assertEqual(api.call_count, 5)
                self.assertIn(f'CI status unavailable ({failure.label}); retrying', output.getvalue())
                self.assertIn('matching CI run 12 attempt 1 passed', output.getvalue())

    def test_transient_timeout_is_reported_safely_then_exact_check_retries(self):
        head, base = 'a' * 40, 'b' * 40
        run = self.run_record(7, head, base, 12)
        run.update(status='completed', conclusion='success', run_attempt=1)
        pr, branch = self.open_pr(head, base), {'commit': {'sha': base}}
        jobs = [{'name': 'rust', 'status': 'completed', 'conclusion': 'success'}]
        args = ['check_ci_run.py', '--repository', 'owner/repo', '--pull-request', '7', '--head', head,
                '--base', base, '--timeout-seconds', '10', '--poll-seconds', '1', '--required-job', 'rust']
        output = io.StringIO()
        with (patch.object(sys, 'argv', args), patch.object(gate, 'api', side_effect=[
                subprocess.TimeoutExpired('gh api', 30), pr, branch,
                {'workflow_runs': [run]}, {'jobs': jobs}]) as api,
             patch.object(gate.time, 'monotonic', return_value=0), patch.object(gate.time, 'sleep'),
             redirect_stdout(output)):
            gate.main()
        self.assertEqual(api.call_count, 5)
        self.assertIn('CI status unavailable (timeout); retrying', output.getvalue())
        self.assertIn('matching CI run 12 attempt 1 passed', output.getvalue())


if __name__ == '__main__':
    unittest.main()
