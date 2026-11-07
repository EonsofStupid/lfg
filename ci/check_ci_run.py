#!/usr/bin/env python3
"""Wait for the full CI run for one exact pull-request head and base."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time


class ApiFailure(Exception):
    """Sanitized API failure metadata; never retains provider error text."""

    def __init__(self, *, status: int | None = None, kind: str = 'gh-cli') -> None:
        self.status = status
        self.kind = kind
        super().__init__(self.label)

    @property
    def label(self) -> str:
        if self.status is not None:
            return f'HTTP {self.status}' + (' rate-limit' if self.kind == 'rate-limit' else '')
        return self.kind


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
    result = subprocess.run(['gh', 'api', endpoint], capture_output=True, text=True, check=False, timeout=30)
    if result.returncode != 0:
        diagnostic = (result.stderr or '') + '\n' + (result.stdout or '')
        match = re.search(r'\bHTTP\s+([1-5][0-9]{2})\b', diagnostic, re.IGNORECASE)
        status = int(match.group(1)) if match else None
        kind = ('rate-limit' if status == 403 and 'rate limit' in diagnostic.lower()
                else 'http' if status is not None else 'gh-cli')
        raise ApiFailure(status=status, kind=kind)
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        raise ApiFailure(kind='invalid-json') from None


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
    last_permanent_failure = None
    repeated_permanent_failures = 0
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
        except ApiFailure as error:
            permanent_client_error = (error.status is not None and 400 <= error.status < 500
                                      and error.status not in (408, 425, 429)
                                      and error.kind != 'rate-limit')
            if permanent_client_error:
                failure = (error.status, error.kind)
                if failure == last_permanent_failure:
                    repeated_permanent_failures += 1
                else:
                    last_permanent_failure = failure
                    repeated_permanent_failures = 1
                if repeated_permanent_failures >= 3:
                    if error.status == 401:
                        guidance = 'verify GitHub authentication and repository access'
                    elif error.status == 403:
                        guidance = 'verify GitHub permissions and repository access'
                    else:
                        guidance = 'verify the API request, repository access, workflow visibility, and PR'
                    raise SystemExit(
                        f'GitHub API repeatedly returned HTTP {error.status}; {guidance} before retrying '
                        'the exact check')
                print(f'CI status unavailable ({error.label}; retry {repeated_permanent_failures}/3)', flush=True)
            else:
                last_permanent_failure = None
                repeated_permanent_failures = 0
                print(f'CI status unavailable ({error.label}); retrying', flush=True)
        except (subprocess.TimeoutExpired, OSError, json.JSONDecodeError, KeyError) as error:
            last_permanent_failure = None
            repeated_permanent_failures = 0
            label = 'timeout' if isinstance(error, subprocess.TimeoutExpired) else type(error).__name__
            print(f'CI status unavailable ({label}); retrying', flush=True)
        time.sleep(min(args.poll_seconds, max(0, deadline - time.monotonic())))
    raise SystemExit('timed out waiting for a matching successful CI run')


if __name__ == '__main__':
    main()
