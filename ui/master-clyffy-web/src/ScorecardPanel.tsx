import { useEffect, useRef, useState } from 'react'
import type { OwnedStatus } from './ownedAssessmentApi'
import { compareReportedIdentity, feedbackStatusBounded, parsePairedReport, scanFeedbackJournal,
  type FeedbackAggregate, type PairedReport, type PathSummary } from './scorecard'
import { compareArtifacts, digestSortedIdSet, parseClassifierReport,
  type ClassifierArtifact, type ComparisonLink } from './classifierComparison'

const DISPLAY_STATUSES = [
  'ok', 'busy', 'timeout', 'client_timeout', 'truncated', 'incomplete_output', 'malformed_output',
  'inference_error', 'invalid_input', 'context_limit', 'unavailable', 'malformed_response', 'response_too_large',
  'http_transport_error', 'transport_error',
  'response_too_large', 'http_400', 'http_401', 'http_403', 'http_404', 'http_413',
  'http_500', 'http_502', 'http_503', 'http_504', 'http_transport_contamination_unknown',
]

function countUnknownStatuses(counts: Record<string, number>): number {
  return Object.entries(counts).reduce((total, [name, count]) =>
    total + (isDisplayStatus(name) ? 0 : count), 0)
}

function isDisplayStatus(name: string): boolean {
  return DISPLAY_STATUSES.includes(name) || /^http_[1-5]\d\d$/.test(name)
}

function ms(value: number | null): string {
  return value === null ? 'Unavailable' : `${(value / 1_000_000).toFixed(3)} ms`
}

function safeFeedbackError(error: unknown): string {
  const message = error instanceof Error ? error.message : ''
  return /^(Feedback access failed \(HTTP \d{3}\)\.|A feedback (?:response|page) exceeded the configured response-byte limit\.|The feedback (?:status |page )?response (?:body was unavailable|could not be validated)\.|A feedback record could not be summarized safely\.|Feedback scan limits are unavailable or invalid\.)$/.test(message)
    ? message : 'Feedback scan stopped; response details were discarded.'
}

function feedbackFailureState(error: unknown): 'denied' | 'unavailable' {
  const message = error instanceof Error ? error.message : ''
  return /^Feedback access failed \(HTTP (?:401|403)\)\.$/.test(message) ? 'denied' : 'unavailable'
}

function Latency({ summary, label }: { summary: PathSummary['end_to_end_latency']; label: string }) {
  return <div>
    <strong>{label}</strong>
    <p>{summary.count} measured samples / {summary.denominator} attempts</p>
    <p>Median {ms(summary.median_ns)} · p95 {ms(summary.p95_ns)} · range {ms(summary.minimum_ns)}–{ms(summary.maximum_ns)}</p>
  </div>
}

function HandlerTime({ summary }: { summary: PathSummary['server_handler_time_us'] }) {
  const format = (value: number | null) => value === null ? 'Unavailable' : `${(value / 1000).toFixed(3)} ms`
  return <div><strong>Endpoint-reported handler wall time</strong>
    <p>{summary.count} samples / {summary.denominator} attempts · median {format(summary.median_ns)} · p95 {format(summary.p95_ns)}</p>
  </div>
}

function StatusCounts({ counts }: { counts: Record<string, number> }) {
  return <ul>{Object.keys(counts).filter(isDisplayStatus).sort().map((status) =>
    <li key={status}>{status}: {counts[status]}</li>)}
  {countUnknownStatuses(counts) > 0 && <li>Other status categories: {countUnknownStatuses(counts)}</li>}</ul>
}

function Identity({ label, value }: { label: string; value: unknown }) {
  if (typeof value !== 'string' || !/^[0-9a-f]{64}$/.test(value)) return <li>{label}: unavailable</li>
  return <li>{label}: <code>{value}</code></li>
}

function ReportView({ report, currentOwnedStatus }: { report: PairedReport; currentOwnedStatus: OwnedStatus | null }) {
  const service = report.service
  const deterministicInfo = service.deterministic as Record<string, unknown>
  const deterministic = deterministicInfo.encoder as Record<string, unknown>
  const owned = service.owned as Record<string, unknown>
  const engine = owned.engine as Record<string, unknown>
  const splitSource = report.source
  const workload = report.settings
  const scope = report.measurement_scope
  const identityMatch = compareReportedIdentity(report, currentOwnedStatus)
  return <div className="scorecard-report" aria-label="Imported paired report">
    <p role="status">Operator-supplied paired benchmark report · {report.complete ? 'run completed' : 'partial run'}</p>
    <p>Report schema: {report.schema_version}.</p>
    <p>Paired paths perform different work: deterministic assessment includes configured declarations; owned generation has no context. Timing does not establish model-only time, quality, source independence, or promotion readiness. No automatic promotion occurs.</p>
    <h3>Run outcome and workload</h3>
    <dl>
      <dt>Abort</dt><dd>{report.abort_code ?? 'None reported'}</dd>
      <dt>Contamination</dt><dd>{report.contamination_state}</dd>
      <dt>Source split claim</dt><dd>{String(splitSource.split)} · operator configured: {String(splitSource.operator_split_claim)}</dd>
      <dt>Source independence</dt><dd>Unverified</dd>
      <dt>Warmup rounds per record</dt><dd>{String(workload.warmup_rounds_per_record)}</dd>
      <dt>Measured rounds per record</dt><dd>{String(workload.measured_rounds_per_record)}</dd>
      <dt>Selected source records</dt><dd>{String(splitSource.selected_records)} of {String(splitSource.source_records)}</dd>
      <dt>Selection method</dt><dd>{String(splitSource.selection)}</dd>
      <dt>Owned context</dt><dd>{String(workload.owned_context)}</dd>
      <dt>Deterministic declaration count</dt><dd>{(workload.declared_subscriptions_sent_to_deterministic as unknown[]).length}</dd>
      <dt>Pair ordering</dt><dd>Configured alternating single-flight pairs</dd>
      <dt>Client timeout</dt><dd>{String(workload.client_timeout_seconds)} seconds</dd>
      <dt>Request / response byte limits</dt><dd>{String(workload.max_request_bytes)} / {String(workload.max_response_bytes)}</dd>
      <dt>Server executable identity</dt><dd>{service.server_binary_sha256 === null ? 'Unknown; endpoint does not attest the binary.' : 'Reported digest is available.'}</dd>
    </dl>
    <h3>Reported artifact and provenance digests</h3>
    <ul>
      <li>Deterministic engine: {String(deterministic.id)}</li>
      <Identity label="Deterministic model" value={deterministic.modelSha256 ?? deterministic.sha256} />
      <Identity label="Deterministic operations config" value={deterministicInfo.operationsDigest} />
      <li>Owned engine: {String(engine.id)}</li>
      <Identity label="Model" value={engine.modelSha256} />
      <Identity label="Model manifest" value={engine.modelManifestSha256} />
      <Identity label="Cached manifest" value={engine.cachedManifestSha256} />
      <Identity label="Prefill graph" value={engine.prefillOnnxSha256} />
      <Identity label="Decode graph" value={engine.decodeOnnxSha256} />
      <Identity label="Source weights" value={engine.sourceWeightsSha256} />
      <Identity label="Training manifest" value={engine.trainingManifestSha256} />
      <Identity label="Framing" value={engine.framingSha256} />
      <Identity label="Runtime config" value={engine.runtimeConfigSha256} />
      <Identity label="Service config" value={engine.serviceConfigSha256} />
      <Identity label="Screen manifest" value={splitSource.screen_manifest_sha256} />
      <Identity label="Train artifact" value={splitSource.manifest_train_artifact_sha256} />
      <Identity label="Dataset" value={splitSource.dataset_sha256} />
      <Identity label="Screening policy" value={splitSource.screening_policy_sha256} />
      <Identity label="Selected source ID/text set" value={splitSource.source_selected_id_text_digest} />
      <Identity label="Selected screened ID/text set" value={splitSource.screened_selected_id_text_digest} />
      <Identity label="Benchmark config" value={workload.benchmark_config_sha256} />
      {typeof service.server_binary_sha256 === 'string' && <Identity label="Server executable" value={service.server_binary_sha256} />}
    </ul>
    <p>These are report-author-supplied identities. Current owned engine comparison: {
      identityMatch === 'matching_reported_identity' ? 'reported fields match.'
        : identityMatch === 'different_reported_identity' ? 'reported fields differ.'
          : 'unavailable because a complete supported identity is missing.'
    }</p>
    {report.paths.deterministic && <PathView name="Deterministic assessment" summary={report.paths.deterministic} />}
    {report.paths.owned && <PathView name="Owned assessment" summary={report.paths.owned} workerPath />}
    <h3>Measurement limits</h3>
    <dl>
      <dt>CPU</dt><dd>{scope.cpu === null ? 'Unknown' : `${String(scope.cpu)}; report supplied, scope and units unverified`}</dd>
      <dt>CPU cycles</dt><dd>{scope.cpu_cycles === null ? 'Unknown' : `${String(scope.cpu_cycles)}; report supplied, scope and units unverified`}</dd>
      <dt>Memory</dt><dd>{scope.memory === null ? 'Unknown' : `${String(scope.memory)}; report supplied, scope and units unverified`}</dd>
      <dt>Reported platform</dt><dd>{String((scope.hardware as Record<string, unknown>).platform)}</dd>
      <dt>Reported machine</dt><dd>{String((scope.hardware as Record<string, unknown>).machine)}</dd>
      <dt>Logical CPUs</dt><dd>{(scope.hardware as Record<string, unknown>).logical_cpus === null ? 'Unknown' : String((scope.hardware as Record<string, unknown>).logical_cpus)}</dd>
      <dt>Timing boundaries</dt><dd>Client wall time covers the request through bounded response read. Owned handler time starts after body extraction, ends before response serialization, and may include child startup.</dd>
    </dl>
    <p>Per-request history and a report creation timestamp were not supplied; none is inferred. Selected-record metadata and text are hidden.</p>
  </div>
}

function PathView({ name, summary, workerPath = false }: { name: string; summary: PathSummary; workerPath?: boolean }) {
  const observedLifecycleCount = (counts: Record<string, number>) =>
    ['worker_not_started', 'worker_exit_observed_success', 'worker_exit_observed_failure']
      .reduce((sum, category) => sum + (counts[category] ?? 0), 0)
  const lifecycleAttempts = observedLifecycleCount(summary.worker_lifecycle_counts)
  const warmupLifecycleAttempts = observedLifecycleCount(summary.warmup.worker_lifecycle_counts)
  return <section aria-label={`${name} measured summary`}>
    <h3>{name}</h3>
    <p>Measured: {summary.attempts} attempts, {summary.completed_responses} completed responses.</p>
    <Latency label="End-to-end response latency" summary={summary.end_to_end_latency} />
    <HandlerTime summary={summary.server_handler_time_us} />
    <p>Endpoint handler time missing: {summary.server_handler_time_missing} attempts.</p>
    <h4>Measured statuses</h4><StatusCounts counts={summary.status_counts} />
    {workerPath && <p>Worker lifecycle observations: {lifecycleAttempts} of {summary.attempts}; {summary.attempts - lifecycleAttempts} unknown or unavailable.</p>}
    <h4>Warmup, shown separately</h4>
    <p>{summary.warmup.attempts} attempts, {summary.warmup.completed_responses} completed responses.</p>
    <Latency label="Warmup end-to-end response latency" summary={summary.warmup.end_to_end_latency} />
    <StatusCounts counts={summary.warmup.status_counts} />
    {workerPath && <p>Warmup worker lifecycle observations: {warmupLifecycleAttempts} of {summary.warmup.attempts}; {summary.warmup.attempts - warmupLifecycleAttempts} unknown or unavailable.</p>}
  </section>
}

export function ScorecardPanel({ maxBytes, maxFeedbackPages, maxFeedbackPageSize, currentOwnedStatus }: {
  maxBytes: number
  maxFeedbackPages: number
  maxFeedbackPageSize: number
  currentOwnedStatus: OwnedStatus | null
}) {
  const [report, setReport] = useState<PairedReport | null>(null)
  const [comparisonReports, setComparisonReports] = useState<{ classical: ClassifierArtifact | null; local: ClassifierArtifact | null; owned: ClassifierArtifact | null }>({ classical: null, local: null, owned: null })
  const [comparisonError, setComparisonError] = useState('')
  const [comparisonLink, setComparisonLink] = useState<ComparisonLink>({ state: 'unavailable', reasons: ['Select the three reports to inspect reported links.'] })
  const [fileError, setFileError] = useState('')
  const [feedbackTokenInput, setFeedbackTokenInput] = useState('')
  const [feedbackToken, setFeedbackToken] = useState<string | null>(null)
  const [feedbackState, setFeedbackState] = useState<'disconnected' | 'connecting' | 'ready' | 'disabled' | 'denied' | 'unavailable' | 'scanning' | 'partial' | 'complete'>('disconnected')
  const [feedbackError, setFeedbackError] = useState('')
  const [feedbackAggregate, setFeedbackAggregate] = useState<FeedbackAggregate | null>(null)
  const importGeneration = useRef(0)
  const comparisonGenerations = useRef({ classical: 0, local: 0, owned: 0 })
  const comparisonReaders = useRef<{ classical: FileReader | null; local: FileReader | null; owned: FileReader | null }>({
    classical: null, local: null, owned: null,
  })
  const feedbackGeneration = useRef(0)
  const feedbackAbort = useRef<AbortController | null>(null)
  const fileReader = useRef<FileReader | null>(null)
  const feedbackAggregateRef = useRef<FeedbackAggregate | null>(null)
  const identityRef = useRef<string | null>(null)

  function clearFeedbackScan() {
    feedbackGeneration.current += 1
    feedbackAbort.current?.abort()
    feedbackAbort.current = null
    feedbackAggregateRef.current = null
    setFeedbackAggregate(null)
    setFeedbackError('')
    setFeedbackState(feedbackToken ? 'ready' : 'disconnected')
  }

  useEffect(() => () => {
    importGeneration.current += 1
    comparisonGenerations.current.classical += 1
    comparisonGenerations.current.local += 1
    comparisonGenerations.current.owned += 1
    comparisonReaders.current.classical?.abort()
    comparisonReaders.current.local?.abort()
    comparisonReaders.current.owned?.abort()
    feedbackGeneration.current += 1
    feedbackAbort.current?.abort()
    fileReader.current?.abort()
  }, [])

  useEffect(() => {
    let active = true
    const classical = comparisonReports.classical
    const digestPromise = classical?.validationIdSetSha256
      ? Promise.resolve(classical.validationIdSetSha256)
      : classical?.validationIds ? digestSortedIdSet(classical.validationIds) : Promise.resolve(null)
    void digestPromise.then((idDigest) => {
      if (active) setComparisonLink(compareArtifacts(classical, comparisonReports.local, comparisonReports.owned, idDigest))
    }).catch(() => {
      if (active) setComparisonLink({ state: 'unavailable', reasons: ['Browser-local report linkage could not be computed.'] })
    })
    return () => { active = false }
  }, [comparisonReports])

  const identityKey = currentOwnedStatus?.status === 'ready' && currentOwnedStatus.engine
    ? JSON.stringify([currentOwnedStatus.engine.inferenceBackend, currentOwnedStatus.engine.modelIdentityKind,
      currentOwnedStatus.engine.modelSha256, currentOwnedStatus.engine.modelManifestSha256,
      currentOwnedStatus.engine.cachedManifestSha256, currentOwnedStatus.engine.prefillOnnxSha256,
      currentOwnedStatus.engine.decodeOnnxSha256, currentOwnedStatus.engine.framingSha256,
      currentOwnedStatus.engine.runtimeConfigSha256, currentOwnedStatus.engine.serviceConfigSha256]) : null
  useEffect(() => {
    if (identityRef.current !== null && identityRef.current !== identityKey) clearFeedbackScan()
    identityRef.current = identityKey
  }, [identityKey])

  async function importFile(file?: File) {
    const current = ++importGeneration.current
    fileReader.current?.abort()
    fileReader.current = null
    clearFeedbackScan()
    setReport(null)
    setFileError('')
    if (!file) return
    if (!Number.isSafeInteger(maxBytes) || maxBytes < 1 || file.size > maxBytes) {
      setFileError(`Choose a report no larger than the configured ${maxBytes.toLocaleString()} UTF-8 byte limit.`)
      return
    }
    try {
      const buffer = await new Promise<ArrayBuffer>((resolve, reject) => {
        const reader = new FileReader()
        fileReader.current = reader
        reader.onload = () => resolve(reader.result as ArrayBuffer)
        reader.onerror = () => reject(new Error('The selected report could not be read.'))
        reader.onabort = () => reject(new Error('Report import was canceled.'))
        reader.readAsArrayBuffer(file)
      })
      if (importGeneration.current !== current) return
      fileReader.current = null
      let value: unknown
      try { value = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(buffer)) as unknown }
      catch { throw new Error('The selected file is not valid UTF-8 JSON.') }
      const parsed = parsePairedReport(value)
      if (importGeneration.current === current) setReport(parsed)
    } catch {
      if (importGeneration.current === current) setFileError('The selected report could not be read or validated.')
    }
  }

  async function importComparisonFile(slot: 'classical' | 'local' | 'owned', file?: File) {
    const current = ++comparisonGenerations.current[slot]
    comparisonReaders.current[slot]?.abort()
    comparisonReaders.current[slot] = null
    setComparisonError('')
    setComparisonLink({ state: 'unavailable', reasons: ['Rechecking selected report links.'] })
    setComparisonReports((previous) => ({ ...previous, [slot]: null }))
    if (!file) return
    const maxCombined = Number.isSafeInteger(maxBytes * 3) ? maxBytes * 3 : Number.MAX_SAFE_INTEGER
    const retainedSize = (['classical', 'local', 'owned'] as const)
      .filter((key) => key !== slot).reduce((sum, key) => sum + (comparisonReports[key]?.rawSize ?? 0), 0)
    if (!Number.isSafeInteger(maxBytes) || maxBytes < 1 || file.size > maxBytes
        || !Number.isSafeInteger(retainedSize + file.size) || retainedSize + file.size > maxCombined) {
      setComparisonError('A selected comparison report exceeds the configured per-file or combined byte limit.')
      return
    }
    let buffer: ArrayBuffer | null = null
    try {
      buffer = await new Promise<ArrayBuffer>((resolve, reject) => {
        const reader = new FileReader()
        comparisonReaders.current[slot] = reader
        reader.onload = () => resolve(reader.result as ArrayBuffer)
        reader.onerror = () => reject(new Error('unreadable file'))
        reader.onabort = () => reject(new Error('cancelled file'))
        reader.readAsArrayBuffer(file)
      })
      if (comparisonGenerations.current[slot] !== current) return
      comparisonReaders.current[slot] = null
      if (buffer.byteLength > maxBytes || buffer.byteLength > maxCombined - retainedSize) throw new Error('oversized file')
      let rawSha256: string | null = null
      if (globalThis.crypto?.subtle) {
        try {
          const hash = await globalThis.crypto.subtle.digest('SHA-256', buffer)
          if (comparisonGenerations.current[slot] !== current) return
          rawSha256 = [...new Uint8Array(hash)].map((byte) => byte.toString(16).padStart(2, '0')).join('')
        } catch { rawSha256 = null }
      }
      let parsed: unknown
      try { parsed = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(buffer)) as unknown }
      catch { throw new Error('invalid report') }
      let artifact = parseClassifierReport(parsed, rawSha256)
      let validationIdSetSha256 = artifact.validationIdSetSha256
      if (artifact.validationIds && globalThis.crypto?.subtle) {
        try { validationIdSetSha256 = await digestSortedIdSet(artifact.validationIds) }
        catch { validationIdSetSha256 = null }
      } else if (artifact.validationIds) validationIdSetSha256 = null
      if (comparisonGenerations.current[slot] !== current) return
      artifact = { ...artifact, validationIds: null, validationIdSetSha256 }
      const rawSize = buffer.byteLength
      setComparisonReports((previous) => ({ ...previous, [slot]: { ...artifact, rawSize } }))
      setComparisonError('')
    } catch {
      if (comparisonGenerations.current[slot] === current) setComparisonError('The selected comparison report could not be read or validated.')
    } finally {
      if (buffer) new Uint8Array(buffer).fill(0)
    }
  }

  async function connectFeedback() {
    const token = feedbackTokenInput
    if (!token) return
    const current = ++feedbackGeneration.current
    setFeedbackState('connecting')
    setFeedbackError('')
    setFeedbackAggregate(null)
    feedbackAggregateRef.current = null
    const controller = new AbortController()
    feedbackAbort.current?.abort()
    feedbackAbort.current = controller
    try {
      const result = await feedbackStatusBounded(token, maxBytes, controller.signal)
      if (feedbackGeneration.current !== current || controller.signal.aborted) return
      if (!result.available) {
        setFeedbackToken(null)
        setFeedbackState('disabled')
        setFeedbackError('Local feedback storage is disabled or unavailable.')
        return
      }
      setFeedbackToken(token)
      setFeedbackTokenInput('')
      setFeedbackState('ready')
    } catch (error) {
      if (feedbackGeneration.current === current && !controller.signal.aborted) {
        setFeedbackState(feedbackFailureState(error))
        setFeedbackError(safeFeedbackError(error))
      }
    }
  }

  async function scanFeedback() {
    if (!feedbackToken) return
    const current = ++feedbackGeneration.current
    const controller = new AbortController()
    feedbackAbort.current?.abort()
    feedbackAbort.current = controller
    setFeedbackState('scanning')
    setFeedbackAggregate(null)
    feedbackAggregateRef.current = null
    setFeedbackError('')
    try {
      const status = await feedbackStatusBounded(feedbackToken, maxBytes, controller.signal)
      if (feedbackGeneration.current !== current || controller.signal.aborted) return
      if (!status.available) {
        setFeedbackState('disabled')
        setFeedbackError('Local feedback storage is disabled or unavailable.')
        return
      }
      const result = await scanFeedbackJournal(feedbackToken, {
        maxPages: maxFeedbackPages,
        maxResponseBytes: maxBytes,
        maxPageSize: Math.min(status.maxPageSize, maxFeedbackPageSize),
        signal: controller.signal,
        onProgress: (next) => {
          if (feedbackGeneration.current === current && !controller.signal.aborted) {
            setFeedbackAggregate(next)
            feedbackAggregateRef.current = next
            setFeedbackState(next.state === 'complete' ? 'complete' : 'partial')
          }
        },
      })
      if (feedbackGeneration.current === current && !controller.signal.aborted) {
        setFeedbackAggregate(result)
        feedbackAggregateRef.current = result
        setFeedbackState(result.state)
      }
    } catch (error) {
      if (feedbackGeneration.current === current && !controller.signal.aborted) {
        setFeedbackState(feedbackAggregateRef.current ? 'partial' : feedbackFailureState(error))
        setFeedbackError(safeFeedbackError(error))
      }
    }
  }

  function disconnectFeedback() {
    feedbackGeneration.current += 1
    feedbackAbort.current?.abort()
    feedbackAbort.current = null
    setFeedbackToken(null)
    setFeedbackTokenInput('')
    setFeedbackAggregate(null)
    feedbackAggregateRef.current = null
    setFeedbackError('')
    setFeedbackState('disconnected')
  }

  return <section className="dev-panel scorecard-panel" aria-label="Alpha evidence scorecard">
    <div className="dev-heading"><div><p className="eyebrow">ALPHA EVIDENCE</p><h2>Paired run scorecard</h2></div></div>
    <p>Import a local paired report to inspect measured evidence. The file remains in browser memory only; it is not uploaded, persisted, or written to logs.</p>
    <label>Paired report JSON file (up to {maxBytes.toLocaleString()} bytes)
      <input type="file" accept="application/json,.json" onChange={(event) => {
        const file = event.currentTarget.files?.[0]
        event.currentTarget.value = ''
        void importFile(file)
      }} />
    </label>
    {fileError && <p role="alert">{fileError}</p>}
    {report && <ReportView report={report} currentOwnedStatus={currentOwnedStatus} />}
    <section className="classifier-comparison" aria-label="Imported classifier comparison">
      <h3>Exploratory classifier comparison</h3>
      <p>Select the three aggregate reports explicitly. Each file is read locally in bounded memory. Report identities, screening and labels are author claims; this panel does not verify source content.</p>
      <div className="classifier-file-inputs">
        <label>Classical report JSON
          <input type="file" accept="application/json,.json" onChange={(event) => {
            const file = event.currentTarget.files?.[0]; event.currentTarget.value = ''; void importComparisonFile('classical', file)
          }} />
        </label>
        <label>Local model report JSON
          <input type="file" accept="application/json,.json" onChange={(event) => {
            const file = event.currentTarget.files?.[0]; event.currentTarget.value = ''; void importComparisonFile('local', file)
          }} />
        </label>
        <label>Owned LFG report JSON
          <input type="file" accept="application/json,.json" onChange={(event) => {
            const file = event.currentTarget.files?.[0]; event.currentTarget.value = ''; void importComparisonFile('owned', file)
          }} />
        </label>
      </div>
      <p>Each file is limited to {maxBytes.toLocaleString()} bytes; combined imported bytes are limited to {Number.isSafeInteger(maxBytes * 3) ? (maxBytes * 3).toLocaleString() : 'the safe configured maximum'}.</p>
      {comparisonError && <p role="alert">{comparisonError}</p>}
      <p role="status">Reported linkage: {comparisonLink.state === 'matching_reported_fields'
        ? 'selected reports agree on the checked report-supplied fields.'
        : comparisonLink.state === 'mismatch' ? 'one or more checked report-supplied fields differ.'
          : 'unavailable until all reports and browser-local digests are available.'}</p>
      {comparisonLink.reasons.length > 0 && <ul aria-label="Linkage details">{comparisonLink.reasons.map((reason) => <li key={reason}>{reason}</li>)}</ul>}
      <ClassifierComparisonView reports={comparisonReports} link={comparisonLink} />
    </section>
    <section aria-label="Feedback journal summary">
      <h3>Local feedback journal summary</h3>
      <p>This is a bounded count of current records traversed. The journal can change during traversal; this is not a snapshot. Confirmed outcomes are operator-reported observations, not verified success. Records can refer to the same assessment, so counts are not independent tasks or accuracy.</p>
      {!feedbackToken ? <>
        {(feedbackState === 'disabled' || feedbackState === 'denied' || feedbackState === 'unavailable')
          && <p role="status">Feedback connection state: {feedbackState}.</p>}
        <label>Local feedback token <input type="password" autoComplete="off" value={feedbackTokenInput}
          onChange={(event) => {
            feedbackGeneration.current += 1
            feedbackAbort.current?.abort()
            setFeedbackState('disconnected')
            setFeedbackTokenInput(event.target.value)
          }} /></label>
        <button type="button" disabled={feedbackState === 'connecting' || !feedbackTokenInput}
          onClick={() => void connectFeedback()}>{feedbackState === 'connecting' ? 'Connecting…' : 'Connect feedback'}</button>
      </> : <>
        <p role="status">Feedback connection: {feedbackState}</p>
        <button type="button" disabled={feedbackState === 'scanning'} onClick={() => void scanFeedback()}>Scan current journal</button>
        <button type="button" onClick={disconnectFeedback}>Disconnect feedback</button>
      </>}
      {feedbackError && <p role="alert">{feedbackError}</p>}
      {feedbackAggregate && <dl aria-label="Feedback aggregate counts">
        <dt>Current records loaded</dt><dd>{feedbackAggregate.loadedRecords}</dd>
        <dt>Operator-reported observations</dt><dd>{feedbackAggregate.operatorReportedObservations}</dd>
        <dt>Unknown outcomes</dt><dd>{feedbackAggregate.unknownOutcomes}</dd>
        <dt>Records with owner corrections</dt><dd>{feedbackAggregate.recordsWithCorrections}</dd>
        <dt>Pages loaded</dt><dd>{feedbackAggregate.pagesLoaded}</dd>
        <dt>Traversal state</dt><dd>{feedbackAggregate.state}{feedbackAggregate.scanCapReached ? ' · configured page cap reached' : ''}</dd>
      </dl>}
      {feedbackToken && <p>Feedback token is held in this browser component's memory only. Feedback counts are journal-wide and are not linked to the imported report.</p>}
    </section>
  </section>
}

function formatComparisonMetric(key: string, value: number | null | undefined): string {
  if (value === null || value === undefined) return 'Unknown'
  if (['accuracy', 'f1', 'precision', 'recall', 'coverage', 'macro_f1', 'macro_precision', 'macro_recall',
    'micro_f1_on_covered', 'micro_precision_on_covered', 'micro_recall_on_covered', 'overall_coverage',
    'prediction_coverage_on_known_targets', 'target_coverage', 'subset_accuracy_on_fully_scored'].includes(key))
    return `${(value * 100).toFixed(2)}%`
  return Number.isInteger(value) ? String(value) : value.toPrecision(4)
}

function TargetSupport({ support }: { support: ClassifierArtifact['targetSupport'] }) {
  if (!support) return <p>Target support counts: Unknown.</p>
  const rows = Object.entries(support as Record<string, Record<string, number>>)
  if (!rows.length) return <p>Target support counts: Unknown.</p>
  return <div><strong>Reported target observation support</strong><ul>
    {rows.map(([label, counts]) => <li key={label}>{label}: {Object.entries(counts).map(([kind, count]) => `${kind.replaceAll('_', ' ')} ${count}`).join(' · ')}</li>)}
  </ul></div>
}

function ClassifierComparisonView({ reports, link }: {
  reports: { classical: ClassifierArtifact | null; local: ClassifierArtifact | null; owned: ClassifierArtifact | null }
  link: ComparisonLink
}) {
  const classical = reports.classical
  const local = reports.local
  const owned = reports.owned
  if (!classical && !local && !owned) return <p>No comparison reports selected.</p>
  const metricList = (metrics: ClassifierArtifact['metrics'][string] | undefined) => {
    if (!metrics) return <p>No report selected.</p>
    const entries = Object.entries(metrics.values).filter(([key]) => key !== 'labels')
    return entries.length === 0 ? <p>Metrics unavailable.</p> : <dl>{entries.map(([key, value]) =>
      <div key={key}><dt>{key.replaceAll('_', ' ')}</dt><dd>{formatComparisonMetric(key, value)}</dd></div>)}</dl>
  }
  const algorithmSections = classical ? Object.entries(classical.algorithms).map(([name, algorithm]) =>
    <section key={name} aria-label={`Classical algorithm ${name}`}><h5>{name}</h5>
      <p>Reported status: {algorithm.status}. Vectorize, fit and bulk-predict elapsed: {formatComparisonMetric('elapsed_ms', algorithm.elapsedMs)} ms.</p>
      {metricList(algorithm.metrics)}
    </section>) : null
  const ownCoverage = owned?.metrics.owned?.values.coverage ?? owned?.metrics.owned?.values.overall_coverage ?? null
  const directAssessment = link.state === 'matching_reported_fields' && ownCoverage !== null && ownCoverage > 0
  const imported = [classical, local, owned].filter((value): value is ClassifierArtifact => value !== null)
  const unreviewedClaim = imported.some((value) => value.labelsUnverifiedClaim === true)
  const reviewedClaim = imported.length > 0 && imported.every((value) => value.labelsUnverifiedClaim === false)
  const priorUseClaim = imported.some((value) => value.validationPreviouslyUsedClaim === true)
  const independenceUnverified = imported.some((value) => value.validationIndependenceVerifiedClaim === false)
  const independenceClaim = imported.some((value) => value.validationIndependenceVerifiedClaim === true)
  return <div className="classifier-comparison-results">
    <p><strong>Task:</strong> {classical?.taskField ?? local?.taskField ?? owned?.taskField ?? 'Unknown'}
      {' · '}<strong>Validation rows:</strong> {classical?.validationRows ?? local?.validationRows ?? owned?.validationRows ?? 'Unknown'}
      {' · '}<strong>Labels:</strong> {(classical?.labels ?? local?.labels ?? owned?.labels ?? []).join(', ') || 'Unknown'}</p>
    <p>Report label-review claim: {unreviewedClaim ? 'metadata says labels are unreviewed.'
      : reviewedClaim ? 'metadata says labels were reviewed; this is not independently verified.' : 'not established.'}
      {' '}Validation history claim: {priorUseClaim ? 'a report states the development validation set had prior use.'
        : independenceUnverified ? 'independence is not verified.'
          : independenceClaim ? 'a report claims independent validation; this is not independently verified.' : 'not established.'}
      {' '}Reports do not carry per-row target values or the per-row prediction intersection, so paired significance and winner claims are unavailable.</p>
    <p>{directAssessment ? 'Reported fields link and the owned report records some coverage; this permits only an aggregate inspection.'
      : 'Direct LFG competition is not assessable from these reports: reported coverage is absent/zero or report linkage is incomplete.'}</p>
    <div className="classifier-comparators">
      <section aria-label="Classical comparison results"><h4>Classical baselines</h4>
        <p>{classical ? `${Object.keys(classical.algorithms).length} dynamically reported algorithms · ${classical.validationRows} validation rows` : 'Not selected.'}</p>
        {classical && <TargetSupport support={classical.targetSupport} />}
        {algorithmSections}
      </section>
      <section aria-label="Local model comparison results"><h4>Local model</h4>
        <p>{local ? `${local.identity ?? 'Model identity unknown'} · ${local.validationRows} validation rows` : 'Not selected.'}</p>
        {local && <p>Model identity is report-supplied and does not establish binary identity.</p>}
        {metricList(local?.metrics.local_12b)}
        {local && <TargetSupport support={local.targetSupport} />}
        {local && <p>Reported request latency: median {formatComparisonMetric('median_ms', local.timing.median_ms)} ms; p95 {formatComparisonMetric('p95_ms', local.timing.p95_ms)} ms. Reported scope only.</p>}
        {local && <p>Observed answer outcomes: {Object.entries(local.statuses).map(([k, v]) => `${k} ${v}`).join(' · ') || 'Unknown'}.</p>}
      </section>
      <section aria-label="Owned LFG comparison results"><h4>Owned LFG</h4>
        <p>{owned ? `${owned.identity ?? 'Model identity unknown'} · ${owned.validationRows} validation rows` : 'Not selected.'}</p>
        {owned && <p>Model identity is report-supplied and does not establish binary identity. Reported end-to-end timing: median {formatComparisonMetric('median_ms', owned.timing.median_ms)} ms; p95 {formatComparisonMetric('p95_ms', owned.timing.p95_ms)} ms.</p>}
        {metricList(owned?.metrics.owned)}
        {owned && <TargetSupport support={owned.targetSupport} />}
        {owned && <p>Observed response statuses: {Object.entries(owned.statuses).map(([k, v]) => `${k} ${v}`).join(' · ') || 'Unknown'}.</p>}
      </section>
    </div>
    <p>Differences in covered subsets remain. No metric is inferred when absent or null. Elapsed values are report-supplied scopes, not CPU time, model-only time, or cost. This exploratory comparison does not qualify a model or promote it.</p>
  </div>
}

export default ScorecardPanel
