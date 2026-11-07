export type SafeObject = Record<string, unknown>

export type MetricSet = {
  values: Record<string, number | null>
  counts: Record<string, number>
}

export type ClassifierArtifact = {
  schema: 'lfg.classifier-comparison-report.v1' | 'lfg.local-12b-comparison-report.v1' | 'lfg.owned-matched-validation-report.v1'
  rawSha256: string | null
  rawSize?: number
  taskField: string
  labels: string[]
  validationRows: number
  validationIds: string[] | null
  validationIdSetSha256: string | null
  targetMaskSha256: string | null
  targetSupport: SafeObject | null
  provenance: Record<string, string | boolean | null>
  metrics: Record<string, MetricSet>
  algorithms: Record<string, { status: string; metrics: MetricSet; elapsedMs: number | null }>
  identity: string | null
  statuses: Record<string, number>
  timing: Record<string, number | null>
  labelsUnverifiedClaim: boolean | null
  validationIndependenceVerifiedClaim: boolean | null
  validationPreviouslyUsedClaim: boolean | null
}

const DIGEST = /^[0-9a-f]{64}$/
const FIELD = /^[A-Za-z0-9_.-]{1,128}$/
const MAX_LABELS = 512
const MAX_ALGORITHMS = 64
const METRIC_FIELDS = new Set(['examples', 'macro_f1', 'macro_f1_supported_labels', 'macro_metric_label_coverage',
  'macro_precision', 'macro_precision_supported_labels', 'macro_recall', 'macro_recall_supported_labels',
  'micro_f1_on_covered', 'micro_precision_on_covered', 'micro_recall_on_covered', 'overall_coverage',
  'prediction_coverage_on_known_targets', 'scored_label_predictions', 'subset_accuracy_on_fully_scored',
  'target_coverage', 'total_label_predictions', 'unknown_or_abstained', 'unknown_positive_targets',
  'unknown_target_cells', 'coverage', 'covered_known_target_cells', 'known_target_cells', 'tp', 'tn', 'fp', 'fn',
  'precision', 'recall', 'f1'])
const RATE_FIELDS = new Set(['macro_f1', 'macro_precision', 'macro_recall', 'micro_f1_on_covered',
  'micro_precision_on_covered', 'micro_recall_on_covered', 'overall_coverage',
  'prediction_coverage_on_known_targets', 'subset_accuracy_on_fully_scored', 'target_coverage', 'coverage',
  'macro_f1_supported_labels', 'macro_precision_supported_labels', 'macro_recall_supported_labels',
  'precision', 'recall', 'f1'])
const CLASSIFIER_STATUSES = new Set(['completed', 'constant_target_fallback_only', 'no_supported_targets', 'failed'])
const REPORT_STATUSES = new Set(['answered', 'abstain', 'ok', 'busy', 'timeout', 'client_timeout', 'truncated',
  'incomplete_output', 'malformed_output', 'inference_error', 'invalid_input', 'context_limit', 'unavailable',
  'malformed_response', 'response_too_large', 'http_transport_error', 'transport_error', 'http_400', 'http_401',
  'http_403', 'http_404', 'http_413', 'http_500', 'http_502', 'http_503', 'http_504', 'unknown_unmapped_taxonomy',
  'no_terminal_response', 'unknown_no_terminal_response'])

function record(value: unknown): value is SafeObject {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}
function fail(): never { throw new Error('The selected comparison report could not be validated.') }
function need<T>(value: unknown, predicate: (item: unknown) => item is T): T
function need<T>(value: unknown, predicate: (item: unknown) => boolean): T
function need<T>(value: unknown, predicate: (item: unknown) => boolean): T {
  if (value === null || value === undefined || !predicate(value)) fail()
  return value as T
}
function digest(value: unknown): string {
  return need<string>(value, (item) => typeof item === 'string' && DIGEST.test(item))
}
function safeName(value: unknown): string {
  return need<string>(value, (item) => typeof item === 'string' && FIELD.test(item))
}
function finiteOrNull(value: unknown): number | null {
  if (value === null) return null
  return need<number>(value, (item) => typeof item === 'number' && Number.isFinite(item) && item >= 0)
}
function safeCount(value: unknown): number {
  return need<number>(value, (item) => typeof item === 'number' && Number.isSafeInteger(item) && item >= 0)
}
function safeStringOrNull(value: unknown): string | null {
  if (value === null || typeof value !== 'string' || value.length < 1 || value.length > 128
      || !/^[A-Za-z0-9][A-Za-z0-9._:/@+-]*$/.test(value)
      || /^[A-Za-z]:[\\/]/.test(value) || value.startsWith('/')
      || /(^|[\\/])\.\.([\\/]|$)/.test(value)) return null
  return value
}
function parseLabels(value: unknown): string[] {
  if (!Array.isArray(value) || value.length < 1 || value.length > MAX_LABELS) fail()
  const labels = value.map(safeName)
  if (new Set(labels).size !== labels.length) fail()
  return labels
}
function parseMetricObject(value: unknown): MetricSet {
  if (!record(value)) fail()
  const values: Record<string, number | null> = {}
  const counts: Record<string, number> = {}
  for (const [key, item] of Object.entries(value)) {
    if (!METRIC_FIELDS.has(key)) continue
    if (item === null || typeof item === 'number') {
      const numeric = finiteOrNull(item)
      if (numeric !== null && RATE_FIELDS.has(key) && numeric > 1) fail()
      if (numeric !== null && !RATE_FIELDS.has(key) && !Number.isSafeInteger(numeric)) fail()
      values[key] = numeric
    }
    else if (record(item)) counts[key] = Object.keys(item).length
    else fail()
  }
  const validateRatio = (ratio: string, numerator: string, denominator: string) => {
    const n = values[numerator], d = values[denominator], r = values[ratio]
    if (typeof n === 'number' && typeof d === 'number') {
      if (n > d) fail()
      if (d === 0) {
        if (r !== undefined && r !== null) fail()
      } else if (typeof r === 'number' && Math.abs(r - n / d) > 0.00001) fail()
    }
  }
  validateRatio('coverage', 'covered_known_target_cells', 'known_target_cells')
  validateRatio('overall_coverage', 'scored_label_predictions', 'total_label_predictions')
  const scored = values.scored_label_predictions
  const total = values.total_label_predictions
  const unknown = values.unknown_target_cells
  const known = values.known_target_cells
  const predictedCoverage = values.prediction_coverage_on_known_targets
  if (typeof scored === 'number' && typeof total === 'number') {
    if (scored > total) fail()
    if (typeof unknown === 'number') {
      if (unknown > total) fail()
      const knownFromUnknown = total - unknown
      if (typeof known === 'number' && known !== knownFromUnknown) fail()
      if (typeof values.target_coverage === 'number' && total > 0
          && Math.abs(values.target_coverage - knownFromUnknown / total) > 0.00001) fail()
    }
    const knownDenominator = typeof known === 'number' ? known
      : typeof unknown === 'number' ? total - unknown : null
    if (knownDenominator === 0) {
      if (scored > 0 || (predictedCoverage !== undefined && predictedCoverage !== null)) fail()
    } else if (typeof predictedCoverage === 'number' && knownDenominator !== null
        && Math.abs(predictedCoverage - scored / knownDenominator) > 0.00001) fail()
  }
  if (typeof known === 'number' && typeof total === 'number' && total > 0) {
    if (known > total) fail()
    if (typeof values.target_coverage === 'number' && Math.abs(values.target_coverage - known / total) > 0.00001) fail()
  } else if (typeof known === 'number' && typeof total === 'number' && total === 0) {
    if (known > 0 || (values.target_coverage !== undefined && values.target_coverage !== null)) fail()
  }
  return { values, counts }
}
function parseOptionalMetricObject(value: unknown): MetricSet {
  return value === null || value === undefined ? { values: {}, counts: {} } : parseMetricObject(value)
}
function parseAlgorithms(value: unknown): Record<string, ClassifierArtifact['algorithms'][string]> {
  if (!record(value) || Object.keys(value).length > MAX_ALGORITHMS) fail()
  const output: Record<string, ClassifierArtifact['algorithms'][string]> = {}
  for (const [key, raw] of Object.entries(value)) {
    if (!FIELD.test(key) || !record(raw)) fail()
    const metrics = parseOptionalMetricObject(raw.metrics)
    const status = typeof raw.status === 'string' && CLASSIFIER_STATUSES.has(raw.status) ? raw.status : 'unknown'
    output[key] = {
      status, metrics,
      elapsedMs: finiteOrNull(raw.fit_vectorize_predict_elapsed_ms),
    }
  }
  return output
}
function parseSupport(value: unknown): SafeObject | null {
  if (value === null || value === undefined) return null
  if (!record(value)) fail()
  const result: SafeObject = {}
  for (const [label, counts] of Object.entries(value)) {
    if (!FIELD.test(label) || !record(counts)) fail()
    const clean: SafeObject = {}
    for (const [key, count] of Object.entries(counts)) {
      if (!FIELD.test(key)) fail()
      clean[key] = safeCount(count)
    }
    result[label] = clean
  }
  return result
}
function parseStatuses(value: unknown): Record<string, number> {
  if (value === null || value === undefined) return {}
  if (!record(value)) fail()
  const result: Record<string, number> = {}
  let other = 0
  for (const [key, count] of Object.entries(value)) {
    if (!FIELD.test(key)) fail()
    const safeValue = safeCount(count)
    if (REPORT_STATUSES.has(key)) result[key] = safeValue
    else other += safeValue
  }
  if (other > 0) result.other_status_categories = safeCount(other)
  return result
}
function provenanceKeys(data: SafeObject, source: SafeObject, baseline: SafeObject = {}): Record<string, string | boolean | null> {
  const result: Record<string, string | boolean | null> = {}
  const mappings: Record<string, [string, string, string]> = {
    manifest_sha256: ['manifest_sha256', 'manifest_sha256', 'screened_manifest_sha256'],
    plan_sha256: ['plan_sha256', 'plan_sha256', 'plan_sha256'],
    policy_sha256: ['policy_sha256', 'policy_sha256', 'screening_policy_sha256'],
    review_ledger_sha256: ['review_ledger_sha256', 'review_ledger_sha256', 'review_ledger_sha256'],
    taxonomy_sha256: ['taxonomy_sha256', 'taxonomy_sha256', 'taxonomy_sha256'],
    train_sha256: ['train_sha256', 'train_sha256', 'train_sha256'],
    validation_sha256: ['validation_sha256', 'validation_sha256', 'validation_sha256'],
    screen_code_sha256: ['screen_code_sha256', 'screen_code_sha256', 'screening_code_sha256'],
    baseline_report_sha256: ['baseline_report_sha256', 'baseline_report_sha256', 'classical_report_sha256'],
    comparison_12b_report_sha256: ['comparison_12b_report_sha256', 'comparison_12b_report_sha256', 'comparison_12b_report_sha256'],
  }
  for (const [key, fields] of Object.entries(mappings)) {
    const sourceAlias = key === 'manifest_sha256' ? source.screened_manifest_sha256
      : key === 'policy_sha256' ? source.screening_policy_sha256
        : key === 'screen_code_sha256' ? source.screening_code_sha256 : undefined
    const value = data[fields[0]] ?? source[fields[1]] ?? sourceAlias ?? baseline[fields[2]]
    const baselineAlias = key === 'baseline_report_sha256'
      ? baseline.report_sha256 ?? baseline.classical_report_sha256
      : key === 'comparison_12b_report_sha256' ? baseline.comparison_12b_report_sha256 : undefined
    const resolved = value ?? baselineAlias
    if (resolved === undefined || resolved === null) result[key] = null
    else result[key] = digest(resolved)
  }
  return result
}
function safeTiming(value: unknown): Record<string, number | null> {
  if (!record(value)) return {}
  const result: Record<string, number | null> = {}
  for (const key of ['count', 'min_ms', 'median_ms', 'p95_ms', 'max_ms']) {
    if (key in value) result[key] = key === 'count' && value[key] !== null ? safeCount(value[key]) : finiteOrNull(value[key])
  }
  return result
}
function boolOrNull(value: unknown): boolean | null {
  return typeof value === 'boolean' ? value : null
}
function previouslyUsedClaim(value: unknown): boolean | null {
  if (typeof value !== 'string') return null
  const normalized = value.trim().toLowerCase().replace(/[-_]+/g, ' ').replace(/\s+/g, ' ')
  return ['previously used', 'used previously', 'prior use', 'previous use'].includes(normalized) ? true : null
}

export function parseClassifierReport(value: unknown, rawSha256: string | null): ClassifierArtifact {
  if (rawSha256 !== null) digest(rawSha256)
  if (!record(value)) fail()
  const schema = value.schema_version
  const task = need(value.task, record)
  const field = safeName(task.field)
  const labels = parseLabels(task.labels)
  if (schema === 'lfg.classifier-comparison-report.v1') {
    const data = need(value.data, record)
    const counts = need(data.counts, record)
    const rows = safeCount(counts.validation)
    const ids = need(data.validation_record_id_sha256, (x): x is string[] => Array.isArray(x)
      && x.length === rows && x.length <= 100_000 && x.every((id) => typeof id === 'string' && DIGEST.test(id)))
    const idDigest = null // calculated by the component using the producer's canonical JSON representation
    const manifestClaims = record(data.manifest_claims) ? data.manifest_claims : {}
    const algorithms = parseAlgorithms(value.algorithms)
    const algorithmMetrics: Record<string, MetricSet> = {}
    for (const [name, algo] of Object.entries(algorithms)) algorithmMetrics[name] = algo.metrics
    return {
      schema, rawSha256, taskField: field, labels, validationRows: rows, validationIds: ids,
      validationIdSetSha256: idDigest, targetMaskSha256: null,
      targetSupport: parseSupport(task.validation_support), provenance: provenanceKeys(data, {}),
      metrics: algorithmMetrics, algorithms, identity: null, statuses: {}, timing: {},
      labelsUnverifiedClaim: typeof manifestClaims.human_reviewed === 'boolean' ? !manifestClaims.human_reviewed : null,
      validationIndependenceVerifiedClaim: boolOrNull(manifestClaims.independent_split_verified),
      validationPreviouslyUsedClaim: previouslyUsedClaim(data.validation_group_use_claim),
    }
  }
  if (schema === 'lfg.local-12b-comparison-report.v1') {
    const baseline = need(value.baseline, record)
    const provenance = need(value.provenance, record)
    const baselineReportHash = digest(baseline.report_sha256)
    const provenanceBaselineHash = digest(provenance.baseline_report_sha256)
    if (baselineReportHash !== provenanceBaselineHash) fail()
    const rows = safeCount(task.validation_rows)
    const ids = need(task.validation_record_id_sha256, (x): x is string[] => Array.isArray(x)
      && x.length === rows && x.length <= 100_000 && x.every((id) => typeof id === 'string' && DIGEST.test(id)))
    const metrics = parseOptionalMetricObject(value.metrics)
    const inference = need(value.inference, record)
    return {
      schema, rawSha256, taskField: field, labels, validationRows: rows, validationIds: ids,
      validationIdSetSha256: null, targetMaskSha256: digest(need(task.target_mask_sha256, record).validation),
      targetSupport: parseSupport(task.target_observation_support), provenance: provenanceKeys({}, provenance, baseline),
      metrics: { local_12b: metrics }, algorithms: {}, identity: safeStringOrNull(need(value.model, record).identity),
      statuses: parseStatuses(inference.outcome_counts), timing: safeTiming(inference.timing_ms),
      labelsUnverifiedClaim: boolOrNull(provenance.labels_unverified),
      validationIndependenceVerifiedClaim: boolOrNull(provenance.validation_independence_verified),
      validationPreviouslyUsedClaim: previouslyUsedClaim(provenance.validation_history_claim),
    }
  }
  if (schema === 'lfg.owned-matched-validation-report.v1') {
    const baseline = need(value.baseline, record)
    const provenance = need(value.provenance, record)
    const rows = safeCount(task.validation_rows)
    const owned = need(value.owned, record)
    const deterministic = need(value.deterministic, record)
    const ownMetric = parseOptionalMetricObject(owned.metrics)
    const deterministicMetric = parseMetricObject(deterministic)
    const classical = need(baseline.classical_aggregate_metrics, record)
    const algorithms: Record<string, ClassifierArtifact['algorithms'][string]> = {}
    const algorithmMetrics: Record<string, MetricSet> = {}
    if (!Object.keys(classical).length || Object.keys(classical).length > MAX_ALGORITHMS) fail()
    for (const [name, item] of Object.entries(classical)) {
      if (!FIELD.test(name) || !record(item)) fail()
      const metrics = parseMetricObject(item)
      algorithmMetrics[name] = metrics
      algorithms[name] = { status: 'reported aggregate', metrics, elapsedMs: null }
    }
    const model = owned.engine_identity === null ? null : need(owned.engine_identity, record)
    const historyClaims = record(provenance.baseline_review_and_history_claims)
      ? provenance.baseline_review_and_history_claims : {}
    return {
      schema, rawSha256, taskField: field, labels, validationRows: rows, validationIds: null,
      validationIdSetSha256: digest(task.validation_record_id_set_sha256),
      targetMaskSha256: digest(task.target_mask_sha256), targetSupport: parseSupport(task.target_observation_support),
      provenance: provenanceKeys(task, provenance, baseline),
      metrics: { owned: ownMetric, deterministic: deterministicMetric, ...algorithmMetrics },
      algorithms, identity: safeStringOrNull(model?.id), statuses: parseStatuses(owned.status_counts),
      timing: safeTiming(record(owned.latency_ms) ? owned.latency_ms.end_to_end : null),
      labelsUnverifiedClaim: boolOrNull(historyClaims.legacy_labels_unverified),
      validationIndependenceVerifiedClaim: boolOrNull(historyClaims.validation_independence_verified),
      validationPreviouslyUsedClaim: previouslyUsedClaim(historyClaims.validation_history_claim),
    }
  }
  fail()
}

export async function digestSortedIdSet(ids: string[]): Promise<string> {
  if (ids.length > 100_000 || !ids.every((id) => DIGEST.test(id))) fail()
  const sorted = [...ids].sort()
  if (new Set(sorted).size !== sorted.length) fail()
  const bytes = new TextEncoder().encode(JSON.stringify(sorted))
  const hash = await crypto.subtle.digest('SHA-256', bytes)
  return [...new Uint8Array(hash)].map((byte) => byte.toString(16).padStart(2, '0')).join('')
}

export type ComparisonLink = { state: 'matching_reported_fields' | 'mismatch' | 'unavailable'; reasons: string[] }

export function compareArtifacts(classical: ClassifierArtifact | null, local: ClassifierArtifact | null,
  owned: ClassifierArtifact | null, canonicalIdDigest: string | null): ComparisonLink {
  if (!classical || !local || !owned || !canonicalIdDigest || !classical.rawSha256 || !local.rawSha256 || !owned.rawSha256)
    return { state: 'unavailable', reasons: ['One or more reports or browser digests are unavailable.'] }
  const reasons: string[] = []
  if (local.provenance.baseline_report_sha256 !== classical.rawSha256
      || owned.provenance.baseline_report_sha256 !== classical.rawSha256) reasons.push('Reported baseline report hashes differ.')
  if (owned.provenance.comparison_12b_report_sha256 !== local.rawSha256) reasons.push('Reported local-model report hash differs.')
  if (local.taskField !== classical.taskField || owned.taskField !== classical.taskField) reasons.push('Task fields differ.')
  if (local.labels.join('\0') !== classical.labels.join('\0') || owned.labels.join('\0') !== classical.labels.join('\0')) reasons.push('Selected labels differ.')
  if (classical.validationRows !== local.validationRows || local.validationRows !== owned.validationRows) reasons.push('Validation row counts differ.')
  if (canonicalIdDigest !== classical.validationIdSetSha256 || canonicalIdDigest !== local.validationIdSetSha256
      || canonicalIdDigest !== owned.validationIdSetSha256) reasons.push('Reported validation ID set differs.')
  if (!local.targetMaskSha256 || local.targetMaskSha256 !== owned.targetMaskSha256) reasons.push('Reported local and owned target masks differ or are unavailable.')
  const digestKeys = ['manifest_sha256', 'plan_sha256', 'policy_sha256', 'review_ledger_sha256', 'taxonomy_sha256', 'train_sha256', 'validation_sha256', 'screen_code_sha256']
  for (const key of digestKeys) {
    const a = classical.provenance[key], b = local.provenance[key], c = owned.provenance[key]
    if (!a || !b || !c) reasons.push(`Reported ${key.replaceAll('_', ' ')} is unavailable.`)
    else if (a !== b || b !== c) reasons.push(`Reported ${key.replaceAll('_', ' ')} differs.`)
  }
  const canonical = (value: unknown): string => {
    if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`
    if (record(value)) return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${canonical(value[key])}`).join(',')}}`
    return JSON.stringify(value)
  }
  if (!classical.targetSupport || !local.targetSupport || !owned.targetSupport
      || canonical(classical.targetSupport) !== canonical(local.targetSupport)
      || canonical(local.targetSupport) !== canonical(owned.targetSupport)) reasons.push('Reported target support differs or is unavailable.')
  return { state: reasons.length ? 'mismatch' : 'matching_reported_fields', reasons }
}
