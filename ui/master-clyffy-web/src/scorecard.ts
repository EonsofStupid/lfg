import type { OwnedEngine, OwnedStatus } from './ownedAssessmentApi'

export type JsonObject = Record<string, unknown>
export type LatencySummary = {
  count: number
  denominator: number
  minimum_ns: number | null
  median_ns: number | null
  p95_ns: number | null
  maximum_ns: number | null
}
export type PathSummary = {
  attempts: number
  completed_responses: number
  status_counts: Record<string, number>
  worker_lifecycle_counts: Record<string, number>
  end_to_end_latency: LatencySummary
  server_handler_time_us: LatencySummary
  server_handler_time_missing: number
  warmup: {
    attempts: number
    completed_responses: number
    status_counts: Record<string, number>
    worker_lifecycle_counts: Record<string, number>
    end_to_end_latency: LatencySummary
  }
}
export type PairedReport = JsonObject & {
  schema_version: 'lfg.paired-http-assessment-benchmark.v1'
  complete: boolean
  abort_code: string | null
  contamination_state: string
  source: JsonObject
  service: JsonObject
  settings: JsonObject
  measurement_scope: JsonObject
  paths: { deterministic: PathSummary | null; owned: PathSummary | null }
}

const SHA256 = /^[0-9a-f]{64}$/
const SAFE_LABEL = /^[A-Za-z0-9._ -]{1,128}$/
const MAX_SAFE_INTEGER = Number.MAX_SAFE_INTEGER
const MAX_STATUS_NAME = 80

function object(value: unknown): value is JsonObject {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}
function string(value: unknown, max = 4096): value is string {
  return typeof value === 'string' && value.length <= max
}
function integer(value: unknown): value is number {
  return Number.isSafeInteger(value) && Number(value) >= 0
}
function finite(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= MAX_SAFE_INTEGER
}
function digestOrNull(value: unknown): boolean {
  return value === null || (typeof value === 'string' && SHA256.test(value))
}
function assert(condition: unknown, message = 'The paired report has an unsupported or inconsistent shape.'): asserts condition {
  if (!condition) throw new Error(message)
}

function validateCounts(value: unknown, attempts: number): value is Record<string, number> {
  if (!object(value)) return false
  const keys = Object.keys(value)
  if (keys.length > 64 || !keys.every((key) => key.length <= MAX_STATUS_NAME && integer(value[key]))) return false
  const total = keys.reduce((sum, key) => sum + Number(value[key]), 0)
  return Number.isSafeInteger(total) && total === attempts
}

function lifecycleCountsMatch(value: JsonObject, attempts: number, workerExpected: boolean): boolean {
  const counts = Object.values(value)
  if (counts.length > 8 || !counts.every(integer)) return false
  const total = counts.reduce((sum, count) => sum + Number(count), 0)
  return Number.isSafeInteger(total) && (workerExpected ? total <= attempts : total === 0)
}

function validateLatency(value: unknown, expectedDenominator: number): value is LatencySummary {
  if (!object(value) || !integer(value.count) || !integer(value.denominator)
      || value.denominator !== expectedDenominator || value.count > value.denominator) return false
  const measures = [value.minimum_ns, value.median_ns, value.p95_ns, value.maximum_ns]
  if (value.count === 0) return measures.every((item) => item === null)
  if (!measures.every(finite)) return false
  const nums = measures as number[]
  return nums[0] <= nums[1] && nums[1] <= nums[2] && nums[2] <= nums[3]
}

function validatePhase(value: unknown, workerExpected: boolean): value is PathSummary['warmup'] {
  if (!object(value) || !integer(value.attempts) || !integer(value.completed_responses)
      || value.completed_responses > value.attempts || !validateCounts(value.status_counts, value.attempts)
      || !object(value.worker_lifecycle_counts)
      || !lifecycleCountsMatch(value.worker_lifecycle_counts, value.attempts, workerExpected)
      || !validateLatency(value.end_to_end_latency, value.completed_responses)) return false
  return value.end_to_end_latency.count === value.completed_responses
}

function validatePath(value: unknown, workerExpected: boolean): value is PathSummary {
  if (!object(value) || !integer(value.attempts) || !integer(value.completed_responses)
      || value.completed_responses > value.attempts || !validateCounts(value.status_counts, value.attempts)
      || !object(value.worker_lifecycle_counts)
      || !lifecycleCountsMatch(value.worker_lifecycle_counts, value.attempts, workerExpected)
      || !validateLatency(value.end_to_end_latency, value.completed_responses)
      || !integer(value.server_handler_time_missing)
      || !validateLatency(value.server_handler_time_us, value.attempts)
      || value.server_handler_time_us.count + value.server_handler_time_missing !== value.attempts
      || !validatePhase(value.warmup, workerExpected)) return false
  return value.end_to_end_latency.count === value.completed_responses
}

function validateIdentity(value: unknown): value is JsonObject {
  if (!object(value) || !string(value.id, 128) || !/^[A-Za-z0-9._:-]{1,128}$/.test(value.id)) return false
  for (const key of ['sha256', 'modelSha256', 'configSha256', 'modelManifestSha256', 'cachedManifestSha256', 'prefillOnnxSha256',
    'decodeOnnxSha256', 'sourceWeightsSha256', 'trainingManifestSha256', 'framingSha256',
    'runtimeConfigSha256', 'serviceConfigSha256']) {
    if (value[key] !== undefined && value[key] !== null && !SHA256.test(String(value[key]))) return false
  }
  return true
}

function validateDeterministicIdentity(value: unknown): value is JsonObject {
  return object(value) && (value.phase === null || string(value.phase, 128))
    && digestOrNull(value.operationsDigest) && validateIdentity(value.encoder)
}

export function parsePairedReport(value: unknown): PairedReport {
  assert(object(value) && value.schema_version === 'lfg.paired-http-assessment-benchmark.v1')
  assert(typeof value.complete === 'boolean' && (value.abort_code === null || [
    'client_timeout_contamination_unknown', 'http_transport_contamination_unknown',
    'owned_busy_contamination_unknown', 'owned_worker_lifecycle_unknown',
    'http_response_contamination_unknown',
  ].includes(String(value.abort_code))))
  if (value.complete) assert(value.abort_code === null)
  assert(value.contamination_state === 'clean' || value.contamination_state === 'unknown; no further requests issued')
  if (value.complete) assert(value.contamination_state === 'clean')
  else assert(value.abort_code !== null && value.contamination_state === 'unknown; no further requests issued')
  assert(value.promotion_eligible === false && value.quality_evaluated === false)
  assert(object(value.source) && object(value.service) && object(value.settings) && object(value.measurement_scope)
    && object(value.paths))
  const source = value.source
  assert(string(source.source_ref, 512) && source.split === 'train'
    && ['train', 'training', 'dev', 'development'].includes(String(source.operator_split_claim))
    && string(source.split_basis, 512))
  assert(source.manifest_schema_version === 'lfg.private-corpus-screen-manifest.v1')
  for (const key of ['screen_manifest_sha256', 'manifest_train_artifact_sha256',
    'screening_policy_sha256', 'dataset_sha256', 'source_selected_id_text_digest',
    'screened_selected_id_text_digest']) assert(typeof source[key] === 'string' && SHA256.test(source[key] as string))
  assert(integer(source.manifest_train_record_count) && integer(source.source_records)
    && integer(source.selected_records) && source.selected_records > 0
    && source.source_records >= source.selected_records && source.manifest_train_record_count === source.source_records)
  assert(source.dataset_sha256 === source.manifest_train_artifact_sha256)
  assert(Array.isArray(source.selected_record_provenance)
    && source.selected_record_provenance.length === source.selected_records)
  assert(['seeded sample without replacement', 'all bounded source records'].includes(String(source.selection))
    && integer(source.selection_seed))

  const service = value.service
  assert(string(service.target, 512))
  assert(validateDeterministicIdentity(service.deterministic)
    && object(service.owned) && validateIdentity(service.owned.engine))
  assert(service.identity_after === null || object(service.identity_after))
  assert(digestOrNull(service.server_binary_sha256))
  assert(string(service.server_binary_identity_basis, 256))
  assert(service.server_binary_sha256 === null || typeof service.server_binary_sha256 === 'string')
  const settings = value.settings
  assert(typeof settings.benchmark_config_sha256 === 'string' && SHA256.test(settings.benchmark_config_sha256))
  assert(integer(settings.warmup_rounds_per_record) && integer(settings.measured_rounds_per_record)
    && settings.measured_rounds_per_record > 0)
  assert(Array.isArray(settings.declared_subscriptions_sent_to_deterministic)
    && settings.declared_subscriptions_sent_to_deterministic.length <= 256
    && settings.declared_subscriptions_sent_to_deterministic.every((entry) => string(entry, 256)))
  assert(settings.owned_context === 'none' && settings.owned_request_text_matches_deterministic_text === true)
  assert(settings.order === 'single-flight alternating pairs; configured first path per pair cycle')
  assert(finite(settings.client_timeout_seconds) && settings.client_timeout_seconds > 0)
  for (const key of ['owned_server_timeout_ms', 'client_timeout_overhead_ms',
    'max_input_bytes', 'max_request_bytes', 'max_response_bytes']) assert(integer(settings[key]))

  const scope = value.measurement_scope
  assert(string(scope.end_to_end, 512) && string(scope.deterministic_handler, 512)
    && string(scope.owned_handler, 512))
  assert(object(scope.hardware) && string(scope.hardware.platform, 256)
    && string(scope.hardware.machine, 256)
    && (scope.hardware.processor === null || string(scope.hardware.processor, 256))
    && (scope.hardware.logical_cpus === null || integer(scope.hardware.logical_cpus)))
  assert(SAFE_LABEL.test(scope.hardware.platform as string) && SAFE_LABEL.test(scope.hardware.machine as string))
  for (const metric of ['cpu', 'cpu_cycles', 'memory']) assert(scope[metric] === null || finite(scope[metric]))
  for (const name of ['deterministic', 'owned']) {
    const path = value.paths[name]
    assert(path === null || validatePath(path, name === 'owned'))
    if (value.complete) {
      assert(path !== null)
      const expectedAttempts = Number(source.selected_records) * Number(settings.measured_rounds_per_record)
      const expectedWarmups = Number(source.selected_records) * Number(settings.warmup_rounds_per_record)
      assert(Number.isSafeInteger(expectedAttempts) && Number.isSafeInteger(expectedWarmups)
        && path.attempts === expectedAttempts && path.warmup.attempts === expectedWarmups)
    }
  }
  if (value.complete) {
    const identityAfter = value.service.identity_after
    assert(object(identityAfter) && validateDeterministicIdentity(identityAfter.deterministic)
      && object(identityAfter.owned) && validateIdentity(identityAfter.owned.engine))
  }
  return value as PairedReport
}

const IDENTITY_FIELDS = [
  'id', 'inferenceBackend', 'modelIdentityKind', 'modelSha256', 'modelManifestSha256',
  'cachedManifestSha256', 'prefillOnnxSha256', 'decodeOnnxSha256', 'sourceWeightsSha256',
  'trainingManifestSha256', 'framingSha256', 'runtimeConfigSha256', 'serviceConfigSha256',
] as const

function cachedIdentity(value: unknown): string[] | null {
  if (!object(value) || value.inferenceBackend !== 'cached_onnx'
      || value.modelIdentityKind !== 'source_weights_sha256') return null
  const parts: string[] = []
  for (const key of IDENTITY_FIELDS) {
    const item = value[key]
    if (typeof item !== 'string' || !item || (key !== 'inferenceBackend' && key !== 'modelIdentityKind'
      && key !== 'id' && !SHA256.test(item))) return null
    parts.push(item)
  }
  return parts
}

export type IdentityMatch = 'matching_reported_identity' | 'different_reported_identity' | 'unavailable'
export function compareReportedIdentity(report: PairedReport, current: OwnedStatus | null): IdentityMatch {
  const currentIdentity = cachedIdentity(current?.status === 'ready' ? current.engine : null)
  const reportIdentity = cachedIdentity((report.service.owned as JsonObject).engine)
  const after = report.service.identity_after
  const afterOwned = object(after) && object(after.owned) ? after.owned : null
  const afterIdentity = cachedIdentity(afterOwned && object(afterOwned.engine) ? afterOwned.engine : null)
  if (!currentIdentity || !reportIdentity || !afterIdentity) return 'unavailable'
  if (reportIdentity.join('|') !== afterIdentity.join('|')) return 'different_reported_identity'
  return currentIdentity.join('|') === reportIdentity.join('|')
    ? 'matching_reported_identity' : 'different_reported_identity'
}

export type FeedbackAggregate = {
  loadedRecords: number
  operatorReportedObservations: number
  unknownOutcomes: number
  recordsWithCorrections: number
  pagesLoaded: number
  scanCapReached: boolean
  state: 'complete' | 'partial'
}

async function boundedJson(response: Response, maxBytes: number): Promise<unknown> {
  const lengthHeader = response.headers.get('content-length')
  if (lengthHeader && integer(Number(lengthHeader)) && Number(lengthHeader) > maxBytes)
    throw new Error('A feedback response exceeded the configured response-byte limit.')
  if (!response.body) throw new Error('The feedback response body was unavailable.')
  const reader = response.body.getReader()
  const chunks: Uint8Array[] = []
  let byteCount = 0
  try {
    while (true) {
      const { done, value: chunk } = await reader.read()
      if (done) break
      byteCount += chunk.byteLength
      if (byteCount > maxBytes) {
        await reader.cancel()
        throw new Error('A feedback response exceeded the configured response-byte limit.')
      }
      chunks.push(chunk)
    }
  } finally { reader.releaseLock() }
  const bytes = new Uint8Array(byteCount)
  let offset = 0
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength }
  try { return JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes)) }
  catch { throw new Error('The feedback response could not be validated.') }
}

export async function feedbackStatusBounded(token: string, maxResponseBytes: number,
  signal: AbortSignal, fetcher: typeof fetch = fetch): Promise<{ available: boolean; maxPageSize: number }> {
  const response = await fetcher('/api/lfg/feedback/status', {
    method: 'GET', cache: 'no-store', credentials: 'same-origin', signal,
    headers: { Accept: 'application/json', 'X-LFG-Session-Token': token },
  })
  if (!response.ok) throw new Error(`Feedback access failed (HTTP ${response.status}).`)
  const value = await boundedJson(response, maxResponseBytes)
  if (!object(value) || value.schemaVersion !== 'lfg.feedback-status.v1'
      || typeof value.enabled !== 'boolean' || typeof value.available !== 'boolean'
      || !integer(value.defaultPageSize) || !integer(value.maxPageSize) || value.maxPageSize < 1)
    throw new Error('The feedback status response could not be validated.')
  return { available: value.enabled && value.available, maxPageSize: value.maxPageSize }
}

export async function scanFeedbackJournal(token: string, options: {
  maxPages: number
  maxResponseBytes: number
  maxPageSize: number
  signal: AbortSignal
  fetcher?: typeof fetch
  onProgress?: (aggregate: FeedbackAggregate) => void
}): Promise<FeedbackAggregate> {
  if (!string(token, 4096) || token.length === 0 || !integer(options.maxPages) || options.maxPages < 1
      || !integer(options.maxResponseBytes) || options.maxResponseBytes < 1
      || !integer(options.maxPageSize) || options.maxPageSize < 1
      || !Number.isSafeInteger(options.maxPages * options.maxPageSize))
    throw new Error('Feedback scan limits are unavailable or invalid.')
  let cursor: string | null = null
  let loadedRecords = 0
  let operatorReportedObservations = 0
  let unknownOutcomes = 0
  let recordsWithCorrections = 0
  let pagesLoaded = 0
  const fetcher = options.fetcher ?? fetch
  while (pagesLoaded < options.maxPages) {
    const query = new URLSearchParams({ limit: String(options.maxPageSize) })
    if (cursor) query.set('cursor', cursor)
    const response = await fetcher(`/api/lfg/feedback?${query.toString()}`, {
      method: 'GET', cache: 'no-store', credentials: 'same-origin', signal: options.signal,
      headers: { Accept: 'application/json', 'X-LFG-Session-Token': token },
    })
    if (!response.ok) throw new Error(`Feedback access failed (HTTP ${response.status}).`)
    const page = await boundedJson(response, options.maxResponseBytes)
    if (!object(page) || page.schemaVersion !== 'lfg.feedback-list.v1' || !Array.isArray(page.records)
        || page.records.length > options.maxPageSize || typeof page.hasMore !== 'boolean'
        || !(page.nextCursor === null || (string(page.nextCursor, 256) && page.nextCursor.length > 0))
        || page.hasMore !== (page.nextCursor !== null)) throw new Error('The feedback page response could not be validated.')
    for (const record of page.records) {
      if (!object(record) || !string(record.feedbackId, 128)
          || !(record.outcome === null || record.outcome === undefined || object(record.outcome))
          || !(record.ownerCorrection === null || record.ownerCorrection === undefined || object(record.ownerCorrection)))
        throw new Error('A feedback record could not be summarized safely.')
      const outcomeStatus = object(record.outcome) ? record.outcome.status : undefined
      if (outcomeStatus === 'confirmed') operatorReportedObservations += 1
      else unknownOutcomes += 1
      if (object(record.ownerCorrection) && hasOwnerCorrection(record.ownerCorrection)) recordsWithCorrections += 1
      loadedRecords += 1
    }
    pagesLoaded += 1
    cursor = page.nextCursor as string | null
    const progress: FeedbackAggregate = {
      loadedRecords, operatorReportedObservations, unknownOutcomes, recordsWithCorrections,
      pagesLoaded, scanCapReached: pagesLoaded === options.maxPages && page.hasMore,
      state: page.hasMore ? 'partial' : 'complete',
    }
    options.onProgress?.(progress)
    if (!page.hasMore) return progress
  }
  return { loadedRecords, operatorReportedObservations, unknownOutcomes, recordsWithCorrections,
    pagesLoaded, scanCapReached: true, state: 'partial' }
}

function hasOwnerCorrection(value: JsonObject): boolean {
  for (const field of ['note', 'intent', 'complexity', 'model']) {
    if (typeof value[field] === 'string' && value[field].trim().length > 0) return true
  }
  return typeof value.agentCount === 'number' && Number.isFinite(value.agentCount)
}
