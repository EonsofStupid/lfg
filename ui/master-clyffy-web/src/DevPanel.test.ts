import { expect, test } from 'vitest'
import { diagnosticExport } from './DevPanel'
import type { DevStats } from './types'

test('diagnostic export excludes unexpected prompt fields at every level', () => {
  const stats = { schemaVersion: 'lfg.dev-stats.v1', enabled: true, capacity: 1, totalRecorded: 1, evicted: 0,
    cpuTimeMicros: null, operationsDigest: 'a', encoderDigest: 'b', classifierDigest: 'c', prompt: 'private',
    summaries: [{ operation: 'preflight', samples: 1, errors: 0, p50TotalMicros: 1, p95TotalMicros: 1, prompt: 'private' }],
    traces: [{ sequence: 1, recordedAt: '2026-11-01T00:00:00Z', operation: 'preflight', outcome: 'ok',
      totalMicros: 1, queueMicros: null, workMicros: 1, prompt: 'private' }],
  } as unknown as DevStats
  const result = diagnosticExport(stats)
  expect(result.snapshot.traces).toHaveLength(1)
  expect(JSON.stringify(result)).not.toContain('private')
})
