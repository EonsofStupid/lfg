//! Bounded, process-local development traces. Prompt content is never retained.

use std::{collections::VecDeque, sync::Mutex, time::Instant};

use lfg_core::now_rfc3339;
use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Operation {
    Preflight,
    Classification,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Trace {
    sequence: u64,
    recorded_at: String,
    operation: Operation,
    outcome: &'static str,
    total_micros: u64,
    queue_micros: Option<u64>,
    work_micros: Option<u64>,
}

#[derive(Default)]
struct Inner {
    total: u64,
    traces: VecDeque<Trace>,
}

pub struct DevStats {
    enabled: bool,
    capacity: usize,
    inner: Mutex<Inner>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Summary {
    operation: Operation,
    samples: usize,
    errors: usize,
    p50_total_micros: Option<u64>,
    p95_total_micros: Option<u64>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Snapshot {
    schema_version: &'static str,
    enabled: bool,
    capacity: usize,
    total_recorded: u64,
    evicted: u64,
    cpu_time_micros: Option<u64>,
    operations_digest: String,
    encoder_digest: String,
    classifier_digest: String,
    summaries: Vec<Summary>,
    traces: Vec<Trace>,
}

impl Snapshot {
    pub fn with_context(
        mut self,
        operations_digest: String,
        encoder_digest: String,
        classifier_digest: String,
    ) -> Self {
        self.operations_digest = operations_digest;
        self.encoder_digest = encoder_digest;
        self.classifier_digest = classifier_digest;
        self
    }
}

fn micros(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_micros()).unwrap_or(u64::MAX)
}

impl DevStats {
    pub fn new(enabled: bool, capacity: usize) -> Self {
        Self {
            enabled,
            capacity,
            inner: Mutex::new(Inner::default()),
        }
    }

    pub fn start(&self) -> Option<Instant> {
        self.enabled.then(Instant::now)
    }

    pub fn elapsed(start: Option<Instant>) -> Option<u64> {
        start.map(micros)
    }

    pub fn record(
        &self,
        operation: Operation,
        started: Option<Instant>,
        queue_micros: Option<u64>,
        work_micros: Option<u64>,
        succeeded: bool,
    ) {
        let Some(started) = started else { return };
        let total_micros = micros(started);
        let mut inner = self.inner.lock().unwrap_or_else(|error| error.into_inner());
        inner.total = inner.total.saturating_add(1);
        let sequence = inner.total;
        if inner.traces.len() == self.capacity {
            inner.traces.pop_front();
        }
        inner.traces.push_back(Trace {
            sequence,
            recorded_at: now_rfc3339(),
            operation,
            outcome: if succeeded { "ok" } else { "error" },
            total_micros,
            queue_micros,
            work_micros,
        });
    }

    pub fn snapshot(&self) -> Snapshot {
        let inner = self.inner.lock().unwrap_or_else(|error| error.into_inner());
        let traces: Vec<_> = inner.traces.iter().cloned().collect();
        let total_recorded = inner.total;
        drop(inner);
        let summaries = [Operation::Preflight, Operation::Classification]
            .into_iter()
            .map(|operation| {
                let relevant: Vec<_> = traces
                    .iter()
                    .filter(|trace| trace.operation == operation)
                    .collect();
                let mut times: Vec<_> = relevant.iter().map(|trace| trace.total_micros).collect();
                times.sort_unstable();
                let percentile = |numerator: usize| -> Option<u64> {
                    if times.is_empty() {
                        None
                    } else {
                        let rank = (times.len() / 100) * numerator
                            + ((times.len() % 100) * numerator).div_ceil(100);
                        Some(times[rank.saturating_sub(1)])
                    }
                };
                Summary {
                    operation,
                    samples: relevant.len(),
                    errors: relevant
                        .iter()
                        .filter(|trace| trace.outcome == "error")
                        .count(),
                    p50_total_micros: percentile(50),
                    p95_total_micros: percentile(95),
                }
            })
            .collect();
        Snapshot {
            schema_version: "lfg.dev-stats.v1",
            enabled: self.enabled,
            capacity: self.capacity,
            total_recorded,
            evicted: total_recorded.saturating_sub(traces.len() as u64),
            cpu_time_micros: None,
            operations_digest: String::new(),
            encoder_digest: String::new(),
            classifier_digest: String::new(),
            summaries,
            traces,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn traces_are_bounded_and_disabled_collection_is_empty() {
        let stats = DevStats::new(true, 2);
        for success in [true, false, true] {
            stats.record(
                Operation::Preflight,
                stats.start(),
                Some(0),
                Some(1),
                success,
            );
        }
        let snapshot = stats.snapshot();
        assert_eq!(snapshot.total_recorded, 3);
        assert_eq!(snapshot.evicted, 1);
        assert_eq!(snapshot.traces.len(), 2);
        assert_eq!(snapshot.summaries[0].errors, 1);
        assert!(snapshot.cpu_time_micros.is_none());
        let disabled = DevStats::new(false, 0);
        disabled.record(
            Operation::Classification,
            disabled.start(),
            None,
            None,
            false,
        );
        assert!(disabled.snapshot().traces.is_empty());
    }
}
