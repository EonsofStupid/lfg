import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import type { OwnedStatus } from './ownedAssessmentApi'
import { ScorecardPanel } from './ScorecardPanel'
import { compareReportedIdentity, parsePairedReport, scanFeedbackJournal } from './scorecard'

const engine = {
  id: 'cached-local-engine', inferenceBackend: 'cached_onnx', modelIdentityKind: 'source_weights_sha256', contextBytes: 2048,
  modelSha256: '1'.repeat(64), modelManifestSha256: '2'.repeat(64), cachedManifestSha256: '3'.repeat(64),
  prefillOnnxSha256: '4'.repeat(64), decodeOnnxSha256: '5'.repeat(64), sourceWeightsSha256: '6'.repeat(64),
  trainingManifestSha256: '7'.repeat(64), framingSha256: '8'.repeat(64), runtimeConfigSha256: '9'.repeat(64),
  serviceConfigSha256: 'a'.repeat(64),
}
const deterministic = { phase: 'ready', operationsDigest: 'c'.repeat(64),
  encoder: { id: 'rules', sha256: 'b'.repeat(64) } }
function latency(count: number, denominator = count) {
  return count ? { count, denominator, minimum_ns: 1_000_000, median_ns: 2_000_000,
    p95_ns: 3_000_000, maximum_ns: 3_000_000 } :
    { count, denominator, minimum_ns: null, median_ns: null, p95_ns: null, maximum_ns: null }
}
function pathSummary(worker = true) {
  return {
    attempts: 2, completed_responses: 2,
    status_counts: { ok: 2 }, worker_lifecycle_counts: worker ? { worker_exit_observed_success: 2 } : {},
    end_to_end_latency: latency(2), server_handler_time_us: latency(2), server_handler_time_missing: 0,
    warmup: { attempts: 1, completed_responses: 1, status_counts: { ok: 1 },
      worker_lifecycle_counts: worker ? { worker_exit_observed_success: 1 } : {}, end_to_end_latency: latency(1) },
  }
}
function makeReport(): Record<string, any> {
  const report: Record<string, any> = {
    schema_version: 'lfg.paired-http-assessment-benchmark.v1', complete: true, abort_code: null,
    contamination_state: 'clean', promotion_eligible: false, quality_evaluated: false,
    source: {
      source_ref: 'private-source-location', split: 'train', operator_split_claim: 'development',
      split_basis: 'bound train digest; source independence unknown', manifest_schema_version: 'lfg.private-corpus-screen-manifest.v1',
      screen_manifest_sha256: 'd'.repeat(64), manifest_train_artifact_sha256: '0'.repeat(64), manifest_train_record_count: 5,
      screening_policy_sha256: 'f'.repeat(64), dataset_sha256: '0'.repeat(64),
      source_selected_id_text_digest: '1'.repeat(64), screened_selected_id_text_digest: '2'.repeat(64),
      source_records: 5, selected_records: 1, selected_record_provenance: [{ private_record_text: 'must stay hidden' }],
      selection: 'seeded sample without replacement', selection_seed: 42,
    },
    service: { target: 'http://127.0.0.1:43127', deterministic, owned: { engine, limits: { timeoutMs: 3000 } },
      identity_after: { deterministic, owned: { engine } }, server_binary_sha256: null,
      server_binary_identity_basis: 'not attested by the HTTP contracts' },
    settings: { benchmark_config_sha256: '3'.repeat(64), warmup_rounds_per_record: 1,
      measured_rounds_per_record: 2, declared_subscriptions_sent_to_deterministic: ['declared'],
      owned_context: 'none', owned_request_text_matches_deterministic_text: true,
      order: 'single-flight alternating pairs; configured first path per pair cycle', client_timeout_seconds: 5, owned_server_timeout_ms: 3000,
      client_timeout_overhead_ms: 500, max_input_bytes: 1024, max_request_bytes: 2048, max_response_bytes: 4096,
      extension_secret: 'private extension must not display' },
    measurement_scope: { end_to_end: 'bounded response wall time', deterministic_handler: 'server duration',
      owned_handler: 'may include child startup', cpu: null, cpu_cycles: null, memory: null,
      hardware: { platform: 'test', machine: 'fixture', processor: null, logical_cpus: 4 } },
    paths: { deterministic: pathSummary(false), owned: pathSummary() },
    created_at: 'not a supported field; must not display',
  }
  return report
}

const currentStatus: OwnedStatus = { enabled: true, status: 'ready', engine }

function responseJson(value: unknown, status = 200): Response {
  const body = JSON.stringify(value)
  return new Response(body, { status, headers: { 'Content-Type': 'application/json', 'Content-Length': String(new TextEncoder().encode(body).length) } })
}

beforeEach(() => { vi.stubGlobal('fetch', vi.fn()) })
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals() })

test('validates and preserves report source while displaying bounded measured evidence only', async () => {
  const report = makeReport()
  expect(parsePairedReport(report)).toBe(report)
  expect(compareReportedIdentity(parsePairedReport(report), currentStatus)).toBe('matching_reported_identity')
  render(<ScorecardPanel maxBytes={100_000} maxFeedbackPages={2} maxFeedbackPageSize={10} currentOwnedStatus={currentStatus} />)
  const file = new File([JSON.stringify(report)], 'private-report.json', { type: 'application/json' })
  fireEvent.change(screen.getByLabelText(/Paired report JSON file/), { target: { files: [file] } })
  await screen.findByText(/Operator-supplied paired benchmark report/)
  expect(screen.getAllByText(/2 measured samples \/ 2 attempts/)).toHaveLength(2)
  expect(screen.getAllByText(/1 attempts, 1 completed responses/)).toHaveLength(2)
  expect(screen.getByText(/reported fields match/)).toBeInTheDocument()
  expect(screen.getByText(/Server executable identity/).nextElementSibling).toHaveTextContent('Unknown')
  expect(document.body.textContent).not.toContain('must stay hidden')
  expect(document.body.textContent).not.toContain('private extension must not display')
  expect(document.body.textContent).not.toContain('private-source-location')
  expect(document.body.textContent).not.toContain('not a supported field')
})

test('rejects unsupported or internally inconsistent report shapes', async () => {
  const invalidSchema = makeReport(); invalidSchema.schema_version = 'other'
  expect(() => parsePairedReport(invalidSchema)).toThrow(/unsupported or inconsistent/)
  const invalidCounts = makeReport(); invalidCounts.paths.owned.status_counts.ok = 1
  expect(() => parsePairedReport(invalidCounts)).toThrow(/unsupported or inconsistent/)
  const invalidWarmup = makeReport(); invalidWarmup.paths.owned.warmup.end_to_end_latency.denominator = 0
  expect(() => parsePairedReport(invalidWarmup)).toThrow(/unsupported or inconsistent/)
})

test('accepts an aborted partial report with unknown ending identity and real partial denominators', () => {
  const partial = makeReport()
  partial.complete = false
  partial.abort_code = 'owned_worker_lifecycle_unknown'
  partial.contamination_state = 'unknown; no further requests issued'
  partial.service.identity_after = null
  partial.paths.owned.attempts = 1
  partial.paths.owned.completed_responses = 1
  partial.paths.owned.status_counts = { malformed_response: 1 }
  partial.paths.owned.worker_lifecycle_counts = { unknown: 1 }
  partial.paths.owned.end_to_end_latency = latency(1)
  partial.paths.owned.server_handler_time_us = latency(1)
  partial.paths.owned.warmup.attempts = 0
  partial.paths.owned.warmup.completed_responses = 0
  partial.paths.owned.warmup.status_counts = {}
  partial.paths.owned.warmup.worker_lifecycle_counts = {}
  partial.paths.owned.warmup.end_to_end_latency = latency(0)
  expect(parsePairedReport(partial)).toBe(partial)
})

test('report size limit is applied before file parsing and new selection clears old evidence', async () => {
  render(<ScorecardPanel maxBytes={2} maxFeedbackPages={2} maxFeedbackPageSize={10} currentOwnedStatus={null} />)
  const input = screen.getByLabelText(/Paired report JSON file/)
  fireEvent.change(input, { target: { files: [new File([JSON.stringify(makeReport())], 'too-large.json')] } })
  expect(await screen.findByRole('alert')).toHaveTextContent('configured 2 UTF-8 byte limit')
  expect(screen.queryByText(/Operator-supplied paired benchmark report/)).not.toBeInTheDocument()
})

test('a late file read cannot replace the newest selected report', async () => {
  const firstReport = makeReport()
  const lateDigest = 'abcdef'.repeat(10) + 'abcd'
  firstReport.source.dataset_sha256 = lateDigest
  firstReport.source.manifest_train_artifact_sha256 = lateDigest
  const realRead = FileReader.prototype.readAsArrayBuffer
  let firstReader: FileReader | null = null
  let firstBlob: Blob | null = null
  vi.spyOn(FileReader.prototype, 'readAsArrayBuffer').mockImplementation(function (this: FileReader, blob: Blob) {
    if (!firstReader) { firstReader = this; firstBlob = blob; return }
    return realRead.call(this, blob)
  })
  render(<ScorecardPanel maxBytes={100_000} maxFeedbackPages={2} maxFeedbackPageSize={10} currentOwnedStatus={null} />)
  const input = screen.getByLabelText(/Paired report JSON file/)
  fireEvent.change(input, { target: { files: [new File([JSON.stringify(firstReport)], 'older.json')] } })
  fireEvent.change(input, { target: { files: [new File([JSON.stringify(makeReport())], 'newer.json')] } })
  await waitFor(() => expect(screen.getAllByText('0'.repeat(64))).toHaveLength(2))
  expect(firstReader).not.toBeNull()
  realRead.call(firstReader!, firstBlob!)
  await waitFor(() => expect(screen.getAllByText('0'.repeat(64))).toHaveLength(2))
  expect(screen.queryByText(lateDigest)).not.toBeInTheDocument()
})

test('missing and changed artifact identity never creates a match', () => {
  const report = parsePairedReport(makeReport())
  expect(compareReportedIdentity(report, null)).toBe('unavailable')
  const changed: OwnedStatus = { enabled: true, status: 'ready', engine: { ...engine, decodeOnnxSha256: '0'.repeat(64) } }
  expect(compareReportedIdentity(report, changed)).toBe('different_reported_identity')
  expect(compareReportedIdentity(report, { ...currentStatus, engine: { id: engine.id } })).toBe('unavailable')
})

test('bounded feedback traversal counts claims and corrections but retains no record text', async () => {
  const privateText = 'private prompt and generated output'
  const fetcher = vi.fn()
    .mockResolvedValueOnce(responseJson({ schemaVersion: 'lfg.feedback-list.v1', hasMore: true,
      nextCursor: 'cursor-two', records: [
        { feedbackId: 'one', outcome: { status: 'confirmed', observation: privateText }, ownerCorrection: { note: privateText } },
        { feedbackId: 'two', outcome: { status: 'unknown' }, ownerCorrection: null },
        { feedbackId: 'three', outcome: { status: 'success' }, ownerCorrection: {} },
      ] }))
    .mockResolvedValueOnce(responseJson({ schemaVersion: 'lfg.feedback-list.v1', hasMore: false,
      nextCursor: null, records: [] }))
  const result = await scanFeedbackJournal('secret-token', {
    maxPages: 2, maxResponseBytes: 20_000, maxPageSize: 10, signal: new AbortController().signal, fetcher,
  })
  expect(result).toMatchObject({ loadedRecords: 3, operatorReportedObservations: 1, unknownOutcomes: 2,
    recordsWithCorrections: 1, pagesLoaded: 2, scanCapReached: false, state: 'complete' })
  expect(JSON.stringify(result)).not.toContain(privateText)
  expect(String(fetcher.mock.calls[0][0])).not.toContain('secret-token')
  expect(fetcher.mock.calls[0][1]?.headers).toMatchObject({ 'X-LFG-Session-Token': 'secret-token' })
})

test('scan cap and later page failure stay partial with no false completion', async () => {
  const first = { schemaVersion: 'lfg.feedback-list.v1', hasMore: true, nextCursor: 'next', records: [
    { feedbackId: 'one', outcome: { status: 'confirmed' }, ownerCorrection: null },
  ] }
  const capped = await scanFeedbackJournal('token', { maxPages: 1, maxResponseBytes: 2000, maxPageSize: 5,
    signal: new AbortController().signal, fetcher: vi.fn(async () => responseJson(first)) })
  expect(capped).toMatchObject({ loadedRecords: 1, pagesLoaded: 1, scanCapReached: true, state: 'partial' })
  const onProgress = vi.fn()
  const laterFailure = scanFeedbackJournal('token', { maxPages: 2, maxResponseBytes: 2000, maxPageSize: 5,
    signal: new AbortController().signal,
    fetcher: vi.fn().mockResolvedValueOnce(responseJson(first)).mockResolvedValueOnce(responseJson({}, 403)),
    onProgress,
  })
  await expect(laterFailure).rejects.toThrow(/HTTP 403/)
  expect(onProgress).toHaveBeenLastCalledWith(expect.objectContaining({ loadedRecords: 1, state: 'partial' }))
})

test('rejects a feedback response over the configured body bound', async () => {
  const oversized = responseJson({ schemaVersion: 'lfg.feedback-list.v1', hasMore: false, nextCursor: null, records: [] })
  const scan = scanFeedbackJournal('token', { maxPages: 1, maxResponseBytes: 1, maxPageSize: 2,
    signal: new AbortController().signal, fetcher: vi.fn(async () => oversized) })
  await expect(scan).rejects.toThrow(/configured response-byte limit/)
})

test('disconnect cancels an in-flight journal page traversal', async () => {
  let resolvePage!: (response: Response) => void
  vi.mocked(fetch).mockImplementation(async (url) => {
    if (String(url).endsWith('/api/lfg/feedback/status')) return responseJson({ schemaVersion: 'lfg.feedback-status.v1',
      enabled: true, available: true, defaultPageSize: 20, maxPageSize: 20 })
    return await new Promise<Response>((resolve) => { resolvePage = resolve })
  })
  render(<ScorecardPanel maxBytes={100_000} maxFeedbackPages={2} maxFeedbackPageSize={20} currentOwnedStatus={null} />)
  fireEvent.change(screen.getByLabelText('Local feedback token'), { target: { value: 'private-token' } })
  fireEvent.click(screen.getByRole('button', { name: 'Connect feedback' }))
  await screen.findByRole('button', { name: 'Scan current journal' })
  fireEvent.click(screen.getByRole('button', { name: 'Scan current journal' }))
  await waitFor(() => expect(resolvePage).toBeDefined())
  fireEvent.click(screen.getByRole('button', { name: 'Disconnect feedback' }))
  resolvePage(responseJson({ schemaVersion: 'lfg.feedback-list.v1', hasMore: false, nextCursor: null,
    records: [{ feedbackId: 'late', outcome: { status: 'confirmed' }, ownerCorrection: null }] }))
  await waitFor(() => expect(screen.queryByLabelText('Feedback aggregate counts')).not.toBeInTheDocument())
  expect(screen.getByRole('button', { name: 'Connect feedback' })).toBeInTheDocument()
})

test('denied feedback access stays visible as denied with no record details', async () => {
  vi.mocked(fetch).mockResolvedValue(responseJson({ private: 'do not display' }, 403))
  render(<ScorecardPanel maxBytes={10_000} maxFeedbackPages={2} maxFeedbackPageSize={10} currentOwnedStatus={null} />)
  fireEvent.change(screen.getByLabelText('Local feedback token'), { target: { value: 'denied-token' } })
  fireEvent.click(screen.getByRole('button', { name: 'Connect feedback' }))
  expect(await screen.findByText('Feedback connection state: denied.')).toBeInTheDocument()
  expect(document.body.textContent).not.toContain('do not display')
  expect(String(vi.mocked(fetch).mock.calls[0][0])).not.toContain('denied-token')
})
