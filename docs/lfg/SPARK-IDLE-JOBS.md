# Running jobs on idle Sparks

`idle_job_runner.py` runs one configured systemd user service only while the
machine's GPU is otherwise idle, so training can use spare Spark time without
competing with the services already there. The job itself must be able to stop
at any time and continue later.

## How it decides

The runner starts the job after a configured number of consecutive samples in
which whole-GPU use and CPU use by other GPU processes are both low and enough
memory is available. It stops the job when other GPU processes stay busy, when
available memory falls below a floor, or when it can no longer read the machine's
state. It waits for idle again and restarts the job, and it stops the job if the
runner itself shuts down. When the completion marker appears it exits. If the job
fails on its own, the runner reports `needs_attention` and does not restart it.

Whole-GPU use cannot separate other work from the job's own once the job runs.
Per-process GPU figures were also unreliable: on a GB10 at 96% GPU use,
`nvidia-smi pmon` reported no per-process figures. CPU use by other GPU processes
is the stop signal instead. On the GB10s sampled on 2026-11-03, idle vLLM processes
used about 3% of a core each and a process actively using the GPU used about 100%.
Work that uses the GPU without keeping a CPU core busy would not be noticed, and
a job that stops may lose the work since its last checkpoint.

The Spark's memory is shared between CPU and GPU. Set the job's own memory limit
and `min_available_memory_bytes` for each host, because services such as vLLM can
hold most of it. The [runner configuration](../../ml/lfg-encoder/idle-job.example.json)
and [systemd unit](../../ml/lfg-encoder/systemd/idle-job-runner.service.example)
are examples to configure for the host, not recommended fixed limits.

```bash
python ml/lfg-encoder/idle_job_runner.py --config PRIVATE_IDLE_JOB_CONFIG.json
```
