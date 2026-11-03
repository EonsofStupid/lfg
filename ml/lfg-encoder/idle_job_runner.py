#!/usr/bin/env python3
"""Run one configured user service only while this machine's GPU is otherwise idle.

Whole-GPU utilization decides when to start. Once the job runs, that figure
includes the job itself, and per-process GPU figures are often missing on a
GB10, so the signal to stop is CPU use by the other processes holding the GPU.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from owned_training_service import atomic_json

RUNNING = ('active', 'activating', 'deactivating')
NEEDS_ATTENTION = 3


def load_config(path: Path) -> dict:
    value = json.loads(path.read_text(encoding='utf-8'))
    counts = ('poll_seconds', 'idle_samples_to_start', 'busy_samples_to_stop')
    memory = ('min_available_memory_bytes', 'stop_below_available_memory_bytes')
    if (not isinstance(value, dict) or value.get('schema_version') != 'lfg.idle-job.v1'
            or not isinstance(value.get('job_unit'), str)
            or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9@_.:-]*\.service', value['job_unit'])
            or any(not isinstance(value.get(k), str) or not Path(value[k]).is_absolute()
                   for k in ('complete_marker', 'status_file'))
            or any(type(value.get(k)) is not int or value[k] < 1 for k in (*counts, *memory))
            or value['stop_below_available_memory_bytes'] > value['min_available_memory_bytes']
            or type(value.get('max_idle_gpu_percent')) is not int or not 0 <= value['max_idle_gpu_percent'] <= 100
            or type(value.get('max_foreign_cpu_percent')) not in (int, float) or value['max_foreign_cpu_percent'] < 0):
        raise ValueError('invalid idle job configuration')
    return value


def nvidia_query(query: str) -> list[str]:
    result = subprocess.run(['nvidia-smi', query, '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, check=True, timeout=30)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def read_gpu_percent() -> int:
    return max(int(value) for value in nvidia_query('--query-gpu=utilization.gpu'))


def read_compute_pids() -> list[int]:
    return [int(value) for value in nvidia_query('--query-compute-apps=pid')]


def owned_by_job(pid: int, unit: str) -> bool:
    try:
        return unit in Path(f'/proc/{pid}/cgroup').read_text(encoding='utf-8')
    except OSError:
        return False


def read_cpu_ticks(pid: int) -> int | None:
    try:
        fields = Path(f'/proc/{pid}/stat').read_text(encoding='utf-8').rsplit(')', 1)[1].split()
    except OSError:
        return None
    return int(fields[11]) + int(fields[12])


def read_available_memory() -> int:
    for line in Path('/proc/meminfo').read_text(encoding='utf-8').splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise ValueError('MemAvailable is missing')


class Sampler:
    def __init__(self, unit: str, *, gpu=read_gpu_percent, pids=read_compute_pids, owned=owned_by_job,
                 ticks=read_cpu_ticks, memory=read_available_memory, clock=time.monotonic,
                 ticks_per_second: int | None = None):
        self.unit, self.gpu, self.pids, self.owned = unit, gpu, pids, owned
        self.ticks, self.memory, self.clock = ticks, memory, clock
        self.ticks_per_second = ticks_per_second or os.sysconf('SC_CLK_TCK')
        self.previous, self.previous_time = {}, None

    def sample(self) -> dict | None:
        """Return the current readings, or None if any of them is unavailable."""
        try:
            gpu, pids, memory = self.gpu(), self.pids(), self.memory()
        except (OSError, ValueError, subprocess.SubprocessError):
            return None
        now = self.clock()
        current = {}
        for pid in pids:
            if not self.owned(pid, self.unit):
                ticks = self.ticks(pid)
                if ticks is not None:
                    current[pid] = ticks
        # The first reading has nothing to compare with, and a process seen for
        # the first time only counts from the next reading.
        foreign = None
        if self.previous_time is not None and now > self.previous_time:
            used = sum(max(0, ticks - self.previous.get(pid, ticks)) for pid, ticks in current.items())
            foreign = used / self.ticks_per_second / (now - self.previous_time) * 100
        self.previous, self.previous_time = current, now
        return {'gpu_percent': gpu, 'foreign_cpu_percent': foreign, 'available_memory_bytes': memory}


def decide(running: bool, sample: dict | None, counts: dict, config: dict) -> str | None:
    """Update the idle and busy counts in place and return 'start', 'stop' or None."""
    if running:
        counts['idle'] = 0
        busy = (sample is None or sample['available_memory_bytes'] < config['stop_below_available_memory_bytes']
                or (sample['foreign_cpu_percent'] or 0) > config['max_foreign_cpu_percent'])
        counts['busy'] = counts['busy'] + 1 if busy else 0
        if counts['busy'] >= config['busy_samples_to_stop']:
            counts['busy'] = 0
            return 'stop'
        return None
    counts['busy'] = 0
    idle = (sample is not None and sample['foreign_cpu_percent'] is not None
            and sample['gpu_percent'] <= config['max_idle_gpu_percent']
            and sample['foreign_cpu_percent'] <= config['max_foreign_cpu_percent']
            and sample['available_memory_bytes'] >= config['min_available_memory_bytes'])
    counts['idle'] = counts['idle'] + 1 if idle else 0
    if counts['idle'] >= config['idle_samples_to_start']:
        counts['idle'] = 0
        return 'start'
    return None


def job_state(unit: str) -> str:
    return subprocess.run(['systemctl', '--user', 'is-active', unit], capture_output=True, text=True,
                          check=False, timeout=60).stdout.strip()


def start_job(unit: str) -> None:
    subprocess.run(['systemctl', '--user', 'start', '--no-block', unit], check=True, timeout=60)


def stop_job(unit: str) -> None:
    subprocess.run(['systemctl', '--user', 'stop', unit], check=True, timeout=300)
    # A job stopped by signal can report failure; this stop was deliberate.
    subprocess.run(['systemctl', '--user', 'reset-failed', unit], check=False, timeout=60)


def run(config: dict, sampler: Sampler, *, state=job_state, start=start_job, stop=stop_job, sleep=time.sleep) -> int:
    unit, marker, status = config['job_unit'], Path(config['complete_marker']), Path(config['status_file'])
    status.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    counts, starts, stops = {'idle': 0, 'busy': 0}, 0, 0

    def report(name: str, sample: dict | None = None) -> None:
        atomic_json(status, {'schema_version': 'lfg.idle-job-status.v1', 'state': name, 'job_unit': unit,
                             'starts': starts, 'stops': stops, 'sample': sample, 'updated_at': time.time()})

    try:
        while True:
            current = state(unit)
            if marker.exists():
                report('complete')
                return 0
            if current == 'failed':
                report('needs_attention')
                return NEEDS_ATTENTION
            sample = sampler.sample()
            running = current in RUNNING
            action = decide(running, sample, counts, config)
            if action == 'start':
                start(unit)
                starts, running = starts + 1, True
            elif action == 'stop':
                stop(unit)
                stops, running = stops + 1, False
            report('running' if running else 'waiting', sample)
            sleep(config['poll_seconds'])
    finally:
        # Never leave the job running without a watcher.
        if state(unit) in RUNNING:
            stop(unit)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)

    def terminate(_signum, _frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, terminate)
    return run(config, Sampler(config['job_unit']))


if __name__ == '__main__':
    raise SystemExit(main())
