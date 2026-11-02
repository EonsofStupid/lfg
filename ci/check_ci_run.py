#!/usr/bin/env python3
"""Wait for the full CI run for one exact pull-request head and base."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time


def matching_run(runs: list[dict], pr: int, head: str, base: str) -> dict | None:
    candidates = []
    for run in runs:
        if run.get('event') != 'pull_request' or run.get('head_sha') != head:
            continue
        if any(item.get('number') == pr and item.get('head', {}).get('sha') == head
               and item.get('base', {}).get('sha') == base for item in run.get('pull_requests', [])):
            candidates.append(run)
    return max(candidates, key=lambda item: item['id']) if candidates else None


def jobs_passed(jobs: list[dict], required: list[str]) -> bool:
    for name in required:
        matches = [job for job in jobs if job.get('name') == name]
        if len(matches) != 1 or matches[0].get('status') != 'completed' or matches[0].get('conclusion') != 'success':
            return False
    return True


def still_eligible(pr: dict, repository: str, head: str, base: str) -> bool:
    """Stop waiting once this exact checked merge can no longer be performed."""
    return (pr.get('state') == 'open' and pr.get('draft') is False
            and pr.get('base', {}).get('ref') == 'main'
            and pr.get('base', {}).get('sha') == base
            and pr.get('head', {}).get('sha') == head
            and (pr.get('head', {}).get('repo') or {}).get('full_name') == repository
            and any(label.get('name') == 'ready-to-auto-merge' for label in pr.get('labels', [])))


def api(endpoint: str) -> dict:
    result = subprocess.run(['gh', 'api', endpoint], capture_output=True, text=True, check=True, timeout=30)
    return json.loads(result.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', required=True)
    parser.add_argument('--pull-request', type=int, required=True)
    parser.add_argument('--head', required=True)
    parser.add_argument('--base', required=True)
    parser.add_argument('--timeout-seconds', type=int, required=True)
    parser.add_argument('--poll-seconds', type=int, required=True)
    parser.add_argument('--required-job', action='append', required=True)
    args = parser.parse_args()
    if (not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', args.repository)
            or args.pull_request < 1 or any(not re.fullmatch(r'[0-9a-f]{40}', value) for value in (args.head, args.base))
            or args.timeout_seconds < 1 or args.poll_seconds < 1
            or any(not name.strip() for name in args.required_job)
            or len(set(args.required_job)) != len(args.required_job)):
        raise SystemExit('invalid CI gate arguments')
    deadline = time.monotonic() + args.timeout_seconds
    endpoint = (f'repos/{args.repository}/actions/workflows/ci.yml/runs'
                f'?event=pull_request&head_sha={args.head}&per_page=100')
    while time.monotonic() < deadline:
        try:
            pr = api(f'repos/{args.repository}/pulls/{args.pull_request}')
            if not still_eligible(pr, args.repository, args.head, args.base):
                raise SystemExit('pull request is closed, changed, or no longer eligible; stop waiting')
            live_base = api(f'repos/{args.repository}/branches/main')['commit']['sha']
            if live_base != args.base:
                raise SystemExit('main changed since verification started; stop waiting')
            run = matching_run(api(endpoint)['workflow_runs'], args.pull_request, args.head, args.base)
            if run and run.get('status') == 'completed':
                if run.get('conclusion') != 'success':
                    raise SystemExit(f"matching CI run {run['id']} concluded {run.get('conclusion')}")
                attempt = run.get('run_attempt')
                if type(attempt) is not int or attempt < 1:
                    raise SystemExit('matching CI run has no valid attempt')
                jobs = api(f"repos/{args.repository}/actions/runs/{run['id']}/attempts/{attempt}/jobs?per_page=100")['jobs']
                if not jobs_passed(jobs, args.required_job):
                    raise SystemExit('matching CI run is missing a successful required job')
                print(f"matching CI run {run['id']} attempt {attempt} passed all required jobs")
                return
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError, json.JSONDecodeError, KeyError):
            print('CI status unavailable; retrying', flush=True)
        time.sleep(min(args.poll_seconds, max(0, deadline - time.monotonic())))
    raise SystemExit('timed out waiting for a matching successful CI run')


if __name__ == '__main__':
    main()
