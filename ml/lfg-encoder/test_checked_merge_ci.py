"""Protect the merge decision against unrelated or incomplete CI results."""
import importlib.util
import io
from pathlib import Path
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch


path = Path(__file__).resolve().parents[2] / 'ci' / 'check_ci_run.py'
spec = importlib.util.spec_from_file_location('check_ci_run', path)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class CheckedMergeCITests(unittest.TestCase):
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
        with patch.object(sys, 'argv', args), patch.object(gate, 'api', side_effect=[{'workflow_runs': [run]}, {'jobs': jobs}]) as api, redirect_stdout(io.StringIO()):
            gate.main()
        self.assertIn('/runs/12/attempts/2/jobs?', api.call_args_list[1].args[0])
        run['conclusion'] = 'failure'
        with patch.object(sys, 'argv', args), patch.object(gate, 'api', return_value={'workflow_runs': [run]}):
            with self.assertRaisesRegex(SystemExit, 'concluded failure'):
                gate.main()


if __name__ == '__main__':
    unittest.main()
