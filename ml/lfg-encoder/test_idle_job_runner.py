"""Idle job decisions with fixed readings; no GPU or systemd is used."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import idle_job_runner as runner

GIB = 1024 ** 3


def sample(gpu=0, foreign=5.0, memory=64 * GIB):
    return {'gpu_percent': gpu, 'foreign_cpu_percent': foreign, 'available_memory_bytes': memory}


class IdleJobTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = {'schema_version': 'lfg.idle-job.v1', 'job_unit': 'lfg-example-job.service',
                       'complete_marker': str(self.root / 'complete.json'), 'status_file': str(self.root / 'status.json'),
                       'poll_seconds': 30, 'idle_samples_to_start': 2, 'busy_samples_to_stop': 2,
                       'max_idle_gpu_percent': 5, 'max_foreign_cpu_percent': 40,
                       'min_available_memory_bytes': 32 * GIB, 'stop_below_available_memory_bytes': 4 * GIB}

    def load(self, **changes):
        path = self.root / 'config.json'
        path.write_text(json.dumps({**self.config, **changes}), encoding='utf-8')
        return runner.load_config(path)

    def test_configuration_rejects_unsafe_values_and_accepts_extra_fields(self):
        self.assertEqual(self.load(note='ignored')['note'], 'ignored')
        for changes in ({'job_unit': '--all.service'}, {'job_unit': 'job'}, {'complete_marker': 'relative'},
                        {'poll_seconds': 0}, {'max_idle_gpu_percent': 101},
                        {'stop_below_available_memory_bytes': 64 * GIB}):
            with self.assertRaises(ValueError, msg=changes):
                self.load(**changes)

    def test_start_needs_consecutive_idle_samples(self):
        counts = {'idle': 0, 'busy': 0}
        self.assertIsNone(runner.decide(False, sample(), counts, self.config))
        self.assertIsNone(runner.decide(False, sample(gpu=96), counts, self.config))
        self.assertEqual(counts['idle'], 0)
        self.assertIsNone(runner.decide(False, sample(), counts, self.config))
        self.assertEqual(runner.decide(False, sample(), counts, self.config), 'start')

    def test_does_not_start_on_any_unknown_or_busy_reading(self):
        for reading in (None, sample(foreign=None), sample(gpu=6), sample(foreign=41), sample(memory=31 * GIB)):
            counts = {'idle': 0, 'busy': 0}
            for _ in range(5):
                self.assertIsNone(runner.decide(False, reading, counts, self.config), reading)

    def test_own_gpu_use_does_not_stop_the_job(self):
        counts = {'idle': 0, 'busy': 0}
        for _ in range(5):
            self.assertIsNone(runner.decide(True, sample(gpu=100), counts, self.config))

    def test_stops_for_other_work_low_memory_or_lost_readings(self):
        for reading in (sample(foreign=100), sample(memory=3 * GIB), None):
            counts = {'idle': 0, 'busy': 0}
            self.assertIsNone(runner.decide(True, reading, counts, self.config), reading)
            self.assertEqual(runner.decide(True, reading, counts, self.config), 'stop', reading)

    def test_brief_other_work_does_not_stop_the_job(self):
        counts = {'idle': 0, 'busy': 0}
        self.assertIsNone(runner.decide(True, sample(foreign=100), counts, self.config))
        self.assertIsNone(runner.decide(True, sample(), counts, self.config))
        self.assertIsNone(runner.decide(True, sample(foreign=100), counts, self.config))

    def sampler(self, ticks, *, pids=(10,), clock=None, owned=lambda pid, unit: False):
        times = iter(clock or range(0, 1000, 10))
        return runner.Sampler('job.service', gpu=lambda: 0, pids=lambda: list(pids), owned=owned,
                              ticks=lambda pid: ticks[pid], memory=lambda: 64 * GIB,
                              clock=lambda: next(times), ticks_per_second=100)

    def test_sampler_measures_other_processes_only(self):
        ticks = {10: 1000, 20: 5000}
        sampler = self.sampler(ticks, pids=(10, 20), owned=lambda pid, unit: pid == 20)
        self.assertIsNone(sampler.sample()['foreign_cpu_percent'])
        ticks.update({10: 2000, 20: 9000})
        self.assertEqual(sampler.sample()['foreign_cpu_percent'], 100.0)

    def test_sampler_counts_a_new_process_from_its_next_reading(self):
        ticks = {10: 0, 30: 500}
        sampler = self.sampler(ticks)
        sampler.sample()
        sampler.pids = lambda: [10, 30]
        self.assertEqual(sampler.sample()['foreign_cpu_percent'], 0.0)
        ticks[30] = 1500
        self.assertEqual(sampler.sample()['foreign_cpu_percent'], 100.0)

    def test_sampler_returns_nothing_when_a_reading_fails(self):
        sampler = self.sampler({10: 0})
        def fail():
            raise subprocess.CalledProcessError(1, 'nvidia-smi')
        sampler.gpu = fail
        self.assertIsNone(sampler.sample())
        sampler.gpu = lambda: int('[N/A]')
        self.assertIsNone(sampler.sample())

    class Machine:
        """A fake systemd unit that a test advances through fixed samples."""
        def __init__(self, marker, samples, complete_after=None):
            self.marker, self.samples, self.complete_after = marker, iter(samples), complete_after
            self.active, self.calls = False, []

        def state(self, _unit):
            return 'active' if self.active else 'inactive'

        def start(self, _unit):
            self.active = True
            self.calls.append('start')

        def stop(self, _unit):
            self.active = False
            self.calls.append('stop')

        def sample(self):
            try:
                return next(self.samples)
            except StopIteration:
                self.marker.write_text('{}', encoding='utf-8')
                return sample()

    def run_machine(self, machine):
        fake = type('FakeSampler', (), {'sample': staticmethod(machine.sample)})
        return runner.run(self.config, fake, state=machine.state, start=machine.start, stop=machine.stop,
                          sleep=lambda _seconds: None)

    def test_pauses_for_other_work_then_resumes_until_complete(self):
        machine = self.Machine(self.root / 'complete.json',
                               [sample(), sample(), sample(gpu=100), sample(gpu=100, foreign=100),
                                sample(gpu=100, foreign=100), sample(), sample(), sample(gpu=100)])
        self.assertEqual(self.run_machine(machine), 0)
        self.assertEqual(machine.calls[:3], ['start', 'stop', 'start'])
        self.assertFalse(machine.active, 'the job must not outlive its watcher')
        self.assertEqual(json.loads((self.root / 'status.json').read_text(encoding='utf-8'))['state'], 'complete')

    def test_failed_job_needs_attention_instead_of_restarting(self):
        machine = self.Machine(self.root / 'complete.json', [sample()])
        machine.state = lambda _unit: 'failed'
        self.assertEqual(self.run_machine(machine), runner.NEEDS_ATTENTION)
        self.assertEqual(machine.calls, [])

    def test_watcher_shutdown_stops_a_running_job(self):
        machine = self.Machine(self.root / 'complete.json', [sample(), sample(), sample()])
        fake = type('FakeSampler', (), {'sample': staticmethod(machine.sample)})
        polls = []
        def sleep(_seconds):
            polls.append(1)
            if len(polls) == 3:
                raise SystemExit(0)
        with self.assertRaises(SystemExit):
            runner.run(self.config, fake, state=machine.state, start=machine.start, stop=machine.stop, sleep=sleep)
        self.assertEqual(machine.calls, ['start', 'stop'])


if __name__ == '__main__':
    unittest.main()
