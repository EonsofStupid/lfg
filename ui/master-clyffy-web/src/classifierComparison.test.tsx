import { afterEach, expect, test, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { compareArtifacts, digestSortedIdSet, parseClassifierReport } from './classifierComparison'
import { ScorecardPanel } from './ScorecardPanel'

afterEach(() => { cleanup(); vi.restoreAllMocks() })

const d = (n: string) => n.repeat(64)
const support = { 'build.software': { positive: 4, negative: 7, unknown_rows: 0 } }
const common = { field: 'intents', labels: ['build.software'], taxonomy_sha256: d('a') }
const classifier = (raw = d('1')) => parseClassifierReport({
  schema_version: 'lfg.classifier-comparison-report.v1', task: { ...common, validation_support: support },
  data: { counts: { validation: 2 }, validation_record_id_sha256: [d('b'), d('c')],
    manifest_sha256: d('d'), plan_sha256: d('e'), policy_sha256: d('f'), review_ledger_sha256: d('1'),
    taxonomy_sha256: d('a'), train_sha256: d('2'), validation_sha256: d('3'), screen_code_sha256: d('9') },
  algorithms: { 'logistic_regression': { status: 'completed', fit_vectorize_predict_elapsed_ms: 2.5,
    metrics: { examples: 2, macro_f1: null, coverage: 0.5 } } },
}, raw)
const local = (baseHash: string, raw = d('4')) => parseClassifierReport({
  schema_version: 'lfg.local-12b-comparison-report.v1', task: { ...common, validation_rows: 2,
    validation_record_id_sha256: [d('b'), d('c')], target_mask_sha256: { train: d('5'), validation: d('6') },
    target_observation_support: support },
  baseline: { report_sha256: baseHash },
  provenance: { manifest_sha256: d('d'), plan_sha256: d('e'), policy_sha256: d('f'), review_ledger_sha256: d('1'),
    taxonomy_sha256: d('a'), train_sha256: d('2'), validation_sha256: d('3'), screen_code_sha256: d('9'), baseline_report_sha256: baseHash },
  metrics: { macro_f1: null, coverage: 0.5, unknown_target_cells: 1 },
  inference: { outcome_counts: { answered: 1, abstain: 1 }, timing_ms: { count: 2, median_ms: null } },
  model: { identity: 'local-model-report-identity' },
}, raw)
const owned = (classicalHash: string, localHash: string, idDigest = d('8'), raw = d('7')) => parseClassifierReport({
  schema_version: 'lfg.owned-matched-validation-report.v1', task: { ...common, validation_rows: 2,
    validation_record_id_set_sha256: idDigest, target_mask_sha256: d('6'), target_observation_support: support },
  baseline: { classical_report_sha256: classicalHash, comparison_12b_report_sha256: localHash,
    classical_aggregate_metrics: { logistic_regression: { macro_f1: null, coverage: 0.5 } } },
  provenance: { screened_manifest_sha256: d('d'), plan_sha256: d('e'), screening_policy_sha256: d('f'),
    review_ledger_sha256: d('1'), taxonomy_sha256: d('a'), train_sha256: d('2'), validation_sha256: d('3'),
    screening_code_sha256: d('9') },
  owned: { engine_identity: { id: 'lfg-report-identity' }, metrics: { coverage: 0, f1: null, covered_known_target_cells: 0 },
    status_counts: { malformed_output: 1, truncated: 1 } },
  deterministic: { coverage: 0, f1: null, known_target_cells: 11, covered_known_target_cells: 0 },
}, raw)

test('links three versioned reports using actual raw hashes, sorted IDs, shared target mask and support', async () => {
  const c = classifier()
  const l = local(c.rawSha256!)
  const idDigest = await digestSortedIdSet([d('b'), d('c')])
  c.validationIdSetSha256 = idDigest
  l.validationIdSetSha256 = idDigest
  const o = owned(c.rawSha256!, l.rawSha256!, idDigest)
  expect(c.algorithms.logistic_regression.status).toBe('completed')
  expect(l.metrics.local_12b.values.macro_f1).toBeNull()
  expect(o.metrics.owned.values.coverage).toBe(0)
  expect(compareArtifacts(c, l, o, idDigest)).toEqual({ state: 'matching_reported_fields', reasons: [] })
})

test('mismatched baseline, task, rows, labels, IDs, mask, support or provenance never links', async () => {
  const c = classifier()
  const l = local(c.rawSha256!)
  const idDigest = await digestSortedIdSet([d('b'), d('c')])
  c.validationIdSetSha256 = idDigest
  l.validationIdSetSha256 = idDigest
  const o = owned(c.rawSha256!, l.rawSha256!, idDigest)
  expect(compareArtifacts(c, local(d('9')), o, idDigest).state).toBe('mismatch')
  expect(compareArtifacts(c, l, owned(c.rawSha256!, d('9')), idDigest).state).toBe('mismatch')
  const changed = local(c.rawSha256!)
  changed.taskField = 'complexity'
  expect(compareArtifacts(c, changed, o, idDigest).state).toBe('mismatch')
  expect(compareArtifacts(c, l, o, d('8')).state).toBe('mismatch')
  const screenDrift = local(c.rawSha256!)
  screenDrift.provenance.screen_code_sha256 = d('0')
  expect(compareArtifacts(c, screenDrift, o, idDigest).state).toBe('mismatch')
  const taxonomyDrift = owned(c.rawSha256!, l.rawSha256!, idDigest)
  taxonomyDrift.provenance.taxonomy_sha256 = d('0')
  expect(compareArtifacts(c, l, taxonomyDrift, idDigest).state).toBe('mismatch')
  expect(compareArtifacts(null, l, o, idDigest).state).toBe('unavailable')
})

test('validates safe allowlisted primitives and refuses malformed schema or digests', () => {
  const value = { schema_version: 'lfg.classifier-comparison-report.v1', task: { ...common,
    validation_support: support }, data: { counts: { validation: 2 }, validation_record_id_sha256: [d('b'), d('c')],
    manifest_sha256: d('d'), plan_sha256: d('e'), policy_sha256: d('f'), review_ledger_sha256: d('1'),
    taxonomy_sha256: d('a'), train_sha256: d('2'), validation_sha256: d('3') },
    algorithms: { 'unknown_algorithm': { status: 'private prompt phrase', fit_vectorize_predict_elapsed_ms: null,
      metrics: { coverage: null, privateText: 'do not render' } } } }
  const parsedUnknown = parseClassifierReport(value, null)
  expect(parsedUnknown.algorithms.unknown_algorithm.metrics.values.coverage).toBeNull()
  expect(parsedUnknown.algorithms.unknown_algorithm.status).toBe('unknown')
  expect(JSON.stringify(parsedUnknown)).not.toContain('private prompt phrase')
  expect(() => parseClassifierReport({ ...value, schema_version: 'wrong' }, null)).toThrow(/could not be validated/)
  expect(() => parseClassifierReport(value, 'not-a-digest')).toThrow(/could not be validated/)
  const malformed = structuredClone(value) as Record<string, any>
  malformed.algorithms.unknown_algorithm.fit_vectorize_predict_elapsed_ms = Infinity
  expect(() => parseClassifierReport(malformed, null)).toThrow(/could not be validated/)
  const outOfRange = structuredClone(value) as Record<string, any>
  outOfRange.algorithms.unknown_algorithm.metrics.coverage = 1.5
  expect(() => parseClassifierReport(outOfRange, null)).toThrow(/could not be validated/)
  const fractionalCount = structuredClone(value) as Record<string, any>
  fractionalCount.algorithms.unknown_algorithm.metrics.examples = 0.5
  expect(() => parseClassifierReport(fractionalCount, null)).toThrow(/could not be validated/)
  const mixedKnown = structuredClone(value) as Record<string, any>
  mixedKnown.algorithms.unknown_algorithm.metrics = { prediction_coverage_on_known_targets: 0.5,
    overall_coverage: 0.25, target_coverage: 0.5, scored_label_predictions: 1,
    total_label_predictions: 4, unknown_target_cells: 2 }
  expect(parseClassifierReport(mixedKnown, null).algorithms.unknown_algorithm.metrics.values.prediction_coverage_on_known_targets).toBe(0.5)
  mixedKnown.algorithms.unknown_algorithm.metrics.prediction_coverage_on_known_targets = 0.25
  expect(() => parseClassifierReport(mixedKnown, null)).toThrow(/could not be validated/)
  const zeroDenominator = structuredClone(value) as Record<string, any>
  zeroDenominator.algorithms.unknown_algorithm.metrics = { coverage: 1, covered_known_target_cells: 1, known_target_cells: 0 }
  expect(() => parseClassifierReport(zeroDenominator, null)).toThrow(/could not be validated/)
  zeroDenominator.algorithms.unknown_algorithm.metrics = { coverage: null, covered_known_target_cells: 0, known_target_cells: 0 }
  expect(parseClassifierReport(zeroDenominator, null).algorithms.unknown_algorithm.metrics.values.coverage).toBeNull()
  zeroDenominator.algorithms.unknown_algorithm.metrics = { prediction_coverage_on_known_targets: 0,
    scored_label_predictions: 0, total_label_predictions: 2, unknown_target_cells: 2 }
  expect(() => parseClassifierReport(zeroDenominator, null)).toThrow(/could not be validated/)
  const claim = structuredClone(value) as Record<string, any>
  claim.data.validation_group_use_claim = 'previously used'
  expect(parseClassifierReport(claim, null).validationPreviouslyUsedClaim).toBe(true)
  claim.data.validation_group_use_claim = 'not previously used'
  expect(parseClassifierReport(claim, null).validationPreviouslyUsedClaim).toBeNull()
  claim.data.validation_group_use_claim = 'historically reviewed and probably independent'
  expect(parseClassifierReport(claim, null).validationPreviouslyUsedClaim).toBeNull()
  const privateIdentity = { schema_version: 'lfg.local-12b-comparison-report.v1', task: { ...common, validation_rows: 0,
    validation_record_id_sha256: [], target_mask_sha256: { validation: d('6') }, target_observation_support: {} },
    baseline: { report_sha256: d('1') }, provenance: { manifest_sha256: d('d'), plan_sha256: d('e'), policy_sha256: d('f'),
      review_ledger_sha256: d('1'), taxonomy_sha256: d('a'), train_sha256: d('2'), validation_sha256: d('3'),
      screen_code_sha256: d('9'), baseline_report_sha256: d('1') }, metrics: {},
    inference: { outcome_counts: {}, timing_ms: {} }, model: { identity: 'private prompt phrase' } }
  expect(parseClassifierReport(privateIdentity, null).identity).toBeNull()
  privateIdentity.model.identity = 'library/mistral-nemo:12b-instruct-q4_0'
  expect(parseClassifierReport(privateIdentity, null).identity).toBe('library/mistral-nemo:12b-instruct-q4_0')
  privateIdentity.model.identity = 'C:/private/path'
  expect(parseClassifierReport(privateIdentity, null).identity).toBeNull()
})

test('retains unknown owned worker identity and metrics as unavailable on failure reports', () => {
  const value = { schema_version: 'lfg.owned-matched-validation-report.v1',
    task: { ...common, validation_rows: 2, validation_record_id_set_sha256: d('8'), target_mask_sha256: d('6'), target_observation_support: support },
    baseline: { classical_report_sha256: d('1'), comparison_12b_report_sha256: d('4'),
      classical_aggregate_metrics: { logistic_regression: { macro_f1: null, coverage: null } } },
    provenance: { screened_manifest_sha256: d('d'), plan_sha256: d('e'), screening_policy_sha256: d('f'),
      review_ledger_sha256: d('1'), taxonomy_sha256: d('a'), train_sha256: d('2'),
      validation_sha256: d('3'), screening_code_sha256: d('9') },
    owned: { engine_identity: null, metrics: null, status_counts: {}, latency_ms: null }, deterministic: {} }
  const parsed = parseClassifierReport(value, d('7'))
  expect(parsed.identity).toBeNull()
  expect(parsed.metrics.owned.values).toEqual({})
  expect(parsed.timing).toEqual({})
})

test('parses unknown reference guide separately from explicit-null model guide and requires provisional policy', () => {
  const value = { schema_version: 'lfg.owned-matched-validation-report.v1',
    task: { ...common, validation_rows: 1, validation_record_id_set_sha256: d('8'), target_mask_sha256: d('6'), target_observation_support: support },
    baseline: { classical_report_sha256: d('1'), comparison_12b_report_sha256: d('4'),
      classical_aggregate_metrics: { logistic_regression: { macro_f1: null, coverage: null } } },
    provenance: { screened_manifest_sha256: d('d'), plan_sha256: d('e'), screening_policy_sha256: d('f'),
      review_ledger_sha256: d('1'), taxonomy_sha256: d('a'), train_sha256: d('2'), validation_sha256: d('3'),
      development_reference_guide: { status: 'not_declared', sha256: null },
      model_response_guide: { status: 'explicitly_none', sha256: null },
      reference_guide_policy: { schema_version: 'lfg.owned-unknown-reference-guide-policy.v1', allow_provisional_agreement: true },
      reference_guide_semantic_comparability: 'unknown' },
    owned: { engine_identity: null, metrics: { coverage: 0, f1: null, covered_known_target_cells: 0 }, status_counts: {},
      reference_guide_assessment: { reference_guide: { status: 'not_declared', sha256: null },
        model_response_guide: { status: 'explicitly_none', sha256: null },
        policy: { schema_version: 'lfg.owned-unknown-reference-guide-policy.v1', allow_provisional_agreement: true },
        semantic_comparability: 'unknown', provisional_comparison_enabled: true,
        claim_interpretation: 'report-supplied target claims; review and history remain report claims; not adjudicated truth or qualification' } },
    deterministic: { coverage: null, f1: null, known_target_cells: 0, covered_known_target_cells: 0 } }
  const withExtensions = { ...value, future_report_extension: 'private extension sentinel',
    owned: { ...value.owned, reference_guide_assessment: { ...value.owned.reference_guide_assessment,
      future_policy_extension: { diagnostic: 'private policy sentinel' } } } }
  const parsed = parseClassifierReport(withExtensions, d('7'))
  expect(parsed.referenceGuideAssessment).toEqual({ referenceStatus: 'not_declared', referenceSha256: null,
    modelStatus: 'explicitly_none', modelSha256: null, semanticComparability: 'unknown', provisionalComparisonEnabled: true,
    policy: { schemaVersion: 'lfg.owned-unknown-reference-guide-policy.v1', allowProvisionalAgreement: true } })
  expect(JSON.stringify(parsed)).not.toContain('private')
  const verifiedInactive = structuredClone(value) as Record<string, any>
  verifiedInactive.owned.reference_guide_assessment.reference_guide = { status: 'verified_sha256', sha256: d('9') }
  verifiedInactive.owned.reference_guide_assessment.model_response_guide = { status: 'verified_sha256', sha256: d('9') }
  verifiedInactive.owned.reference_guide_assessment.semantic_comparability = 'guide_identity_bound'
  verifiedInactive.owned.reference_guide_assessment.provisional_comparison_enabled = false
  verifiedInactive.provenance.development_reference_guide = { status: 'verified_sha256', sha256: d('9') }
  verifiedInactive.provenance.model_response_guide = { status: 'verified_sha256', sha256: d('9') }
  verifiedInactive.provenance.reference_guide_semantic_comparability = 'guide_identity_bound'
  const verifiedParsed = parseClassifierReport(verifiedInactive, d('7'))
  expect(verifiedParsed.referenceGuideAssessment?.policy?.allowProvisionalAgreement).toBe(true)
  expect(verifiedParsed.referenceGuideAssessment?.provisionalComparisonEnabled).toBe(false)
  const conflictingGuides = structuredClone(verifiedInactive) as Record<string, any>
  conflictingGuides.owned.reference_guide_assessment.model_response_guide.sha256 = d('8')
  expect(() => parseClassifierReport(conflictingGuides, d('7'))).toThrow(/could not be validated/)
  const conflictingProvenance = structuredClone(verifiedInactive) as Record<string, any>
  conflictingProvenance.provenance.development_reference_guide.sha256 = d('8')
  expect(() => parseClassifierReport(conflictingProvenance, d('7'))).toThrow(/could not be validated/)
  const explicitNoneInactive = structuredClone(value) as Record<string, any>
  explicitNoneInactive.owned.reference_guide_assessment.reference_guide = { status: 'explicitly_none', sha256: null }
  explicitNoneInactive.owned.reference_guide_assessment.semantic_comparability = 'guide_identity_bound'
  explicitNoneInactive.owned.reference_guide_assessment.provisional_comparison_enabled = false
  explicitNoneInactive.provenance.development_reference_guide = { status: 'explicitly_none', sha256: null }
  explicitNoneInactive.provenance.reference_guide_semantic_comparability = 'guide_identity_bound'
  const explicitNoneParsed = parseClassifierReport(explicitNoneInactive, d('7'))
  expect(explicitNoneParsed.referenceGuideAssessment?.policy?.allowProvisionalAgreement).toBe(true)
  expect(explicitNoneParsed.referenceGuideAssessment?.provisionalComparisonEnabled).toBe(false)
  const missingOptIn = structuredClone(value) as Record<string, any>
  missingOptIn.owned.reference_guide_assessment.policy = null
  expect(() => parseClassifierReport(missingOptIn, d('7'))).toThrow(/could not be validated/)
  const missingModelIdentity = structuredClone(value) as Record<string, any>
  missingModelIdentity.owned.reference_guide_assessment.model_response_guide = null
  expect(() => parseClassifierReport(missingModelIdentity, d('7'))).toThrow(/could not be validated/)
  const malformedVerified = structuredClone(value) as Record<string, any>
  malformedVerified.owned.reference_guide_assessment.reference_guide = { status: 'verified_sha256', sha256: null }
  expect(() => parseClassifierReport(malformedVerified, d('7'))).toThrow(/could not be validated/)
})

test('computes Python-compatible sorted opaque ID set digest and detects duplicates', async () => {
  const first = await digestSortedIdSet([d('c'), d('b')])
  expect(first).toBe(await digestSortedIdSet([d('b'), d('c')]))
  await expect(digestSortedIdSet([d('b'), d('b')])).rejects.toThrow(/could not be validated/)
})

test('imports all three aggregate reports locally, displays safe dynamic summaries and never uploads or persists them', async () => {
  const ids = [d('b'), d('c')]
  const idDigest = await digestSortedIdSet(ids)
  const shared = { manifest_sha256: d('d'), plan_sha256: d('e'), policy_sha256: d('f'), review_ledger_sha256: d('1'),
    taxonomy_sha256: d('a'), train_sha256: d('2'), validation_sha256: d('3'), screen_code_sha256: d('9') }
  const classicalText = JSON.stringify({ schema_version: 'lfg.classifier-comparison-report.v1',
    task: { ...common, validation_support: support },
    data: { counts: { validation: 2 }, validation_record_id_sha256: ids, ...shared,
      manifest_claims: { human_reviewed: false, independent_split_verified: false },
      validation_group_use_claim: 'previously used' },
    algorithms: { 'logistic_regression': { status: 'completed', fit_vectorize_predict_elapsed_ms: 12,
      metrics: { examples: 2, macro_f1: null, overall_coverage: 0.5, privateNumericExtension: 999,
        privateText: 'private prompt and output must not display' } } } })
  const sha = async (text: string) => [...new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text)))].map((v) => v.toString(16).padStart(2, '0')).join('')
  const classicalHash = await sha(classicalText)
  const localText = JSON.stringify({ schema_version: 'lfg.local-12b-comparison-report.v1',
    task: { ...common, validation_rows: 2, validation_record_id_sha256: ids,
      target_mask_sha256: { train: d('5'), validation: d('6') }, target_observation_support: support },
    baseline: { report_sha256: classicalHash }, provenance: { ...shared, baseline_report_sha256: classicalHash,
      labels_unverified: true, validation_independence_verified: false },
    metrics: { examples: 2, macro_f1: null, coverage: 0.5 },
    inference: { outcome_counts: { answered: 1, abstain: 1 }, timing_ms: { count: 2, median_ms: null } },
    model: { identity: 'local-model-identity' } })
  const localHash = await sha(localText)
  const ownedText = JSON.stringify({ schema_version: 'lfg.owned-matched-validation-report.v1',
    task: { ...common, validation_rows: 2, validation_record_id_set_sha256: idDigest,
      target_mask_sha256: d('6'), target_observation_support: support },
    baseline: { classical_report_sha256: classicalHash, comparison_12b_report_sha256: localHash,
      classical_aggregate_metrics: { 'logistic_regression': { macro_f1: null, coverage: 0.5 } } },
    provenance: { screened_manifest_sha256: d('d'), plan_sha256: d('e'), screening_policy_sha256: d('f'), screening_code_sha256: d('9'),
      review_ledger_sha256: d('1'), taxonomy_sha256: d('a'), train_sha256: d('2'), validation_sha256: d('3'),
      development_reference_guide: { status: 'not_declared', sha256: null },
      model_response_guide: { status: 'explicitly_none', sha256: null },
      reference_guide_policy: { schema_version: 'lfg.owned-unknown-reference-guide-policy.v1', allow_provisional_agreement: true },
      reference_guide_semantic_comparability: 'unknown' },
    owned: { engine_identity: { id: 'lfg-owned-model-identity' }, metrics: { coverage: 0, f1: null, covered_known_target_cells: 0 },
      status_counts: { malformed_output: 1, truncated: 1 }, latency_ms: { end_to_end: { count: 2, median_ms: null } },
      reference_guide_assessment: { reference_guide: { status: 'not_declared', sha256: null },
        model_response_guide: { status: 'explicitly_none', sha256: null },
        policy: { schema_version: 'lfg.owned-unknown-reference-guide-policy.v1', allow_provisional_agreement: true },
        semantic_comparability: 'unknown', provisional_comparison_enabled: true,
        claim_interpretation: 'report-supplied target claims; review and history remain report claims; not adjudicated truth or qualification' } },
    deterministic: { coverage: 0, f1: null, known_target_cells: 11, covered_known_target_cells: 0 } })
  const classicalParsed = parseClassifierReport(JSON.parse(classicalText), classicalHash)
  const localParsed = parseClassifierReport(JSON.parse(localText), localHash)
  expect(parseClassifierReport(JSON.parse(ownedText), await sha(ownedText)).schema).toContain('owned-matched')
  expect(classicalParsed.algorithms.logistic_regression).toBeDefined()
  expect(localParsed.metrics.local_12b).toBeDefined()
  const fetchSpy = vi.spyOn(globalThis, 'fetch')
  const storageSpy = vi.spyOn(Storage.prototype, 'setItem')
  const logSpy = vi.spyOn(console, 'log')
  render(<ScorecardPanel maxBytes={100_000} maxFeedbackPages={2} maxFeedbackPageSize={10} currentOwnedStatus={null} />)
  const select = async (label: string, content: string) => {
    fireEvent.change(screen.getByLabelText(label), { target: { files: [new File([content], 'private-source-file.json')] } })
  }
  await select('Owned LFG report JSON', ownedText)
  await select('Classical report JSON', classicalText)
  await select('Local model report JSON', localText)
  expect(await screen.findByText(/selected reports agree on the checked report-supplied fields/)).toBeInTheDocument()
  expect(screen.getByText(/metadata says labels are unreviewed/)).toBeInTheDocument()
  expect(screen.getByText(/development validation set had prior use/)).toBeInTheDocument()
  expect(screen.getByRole('note')).toHaveTextContent(/Reference labeling-guide provenance is unknown/)
  expect(screen.getByRole('note')).toHaveTextContent(/provisional comparisons against report-supplied target claims/)
  expect(screen.getByText(/review and history remain report claims/)).toBeInTheDocument()
  expect(screen.getByRole('heading', { name: 'logistic_regression' })).toBeInTheDocument()
  expect(screen.getByText(/Direct LFG competition is not assessable/)).toBeInTheDocument()
  expect(screen.getAllByText(/2 validation rows/)).toHaveLength(3)
  expect(document.body.textContent).not.toContain('private prompt and output')
  expect(document.body.textContent).not.toContain('privateNumericExtension')
  expect(document.body.textContent).not.toContain('private-source-file')
  expect(fetchSpy).not.toHaveBeenCalled()
  expect(storageSpy).not.toHaveBeenCalled()
  expect(logSpy).not.toHaveBeenCalled()
})

test('imports configured but inactive policy on known reference guides without provisional warning', async () => {
  const makeKnown = (referenceStatus: 'verified_sha256' | 'explicitly_none') => ({ schema_version: 'lfg.owned-matched-validation-report.v1',
    task: { ...common, validation_rows: 1, validation_record_id_set_sha256: d('8'), target_mask_sha256: d('6'), target_observation_support: support },
    baseline: { classical_report_sha256: d('1'), comparison_12b_report_sha256: d('2'),
      classical_aggregate_metrics: { logistic_regression: { macro_f1: null, coverage: 0 } } },
    provenance: { screened_manifest_sha256: d('d'), plan_sha256: d('e'), screening_policy_sha256: d('f'),
      review_ledger_sha256: d('1'), taxonomy_sha256: d('a'), train_sha256: d('2'), validation_sha256: d('3'),
      baseline_review_and_history_claims: { legacy_labels_unverified: null,
        validation_independence_verified: null, validation_history_claim: 'not established' },
      development_reference_guide: { status: referenceStatus, sha256: referenceStatus === 'verified_sha256' ? d('9') : null },
      model_response_guide: { status: referenceStatus, sha256: referenceStatus === 'verified_sha256' ? d('9') : null },
      reference_guide_policy: { schema_version: 'lfg.owned-unknown-reference-guide-policy.v1', allow_provisional_agreement: true },
      reference_guide_semantic_comparability: 'guide_identity_bound' },
    owned: { engine_identity: null, metrics: { coverage: 0, f1: null, covered_known_target_cells: 0 }, status_counts: {},
      reference_guide_assessment: { reference_guide: { status: referenceStatus, sha256: referenceStatus === 'verified_sha256' ? d('9') : null },
        model_response_guide: { status: referenceStatus, sha256: referenceStatus === 'verified_sha256' ? d('9') : null },
        policy: { schema_version: 'lfg.owned-unknown-reference-guide-policy.v1', allow_provisional_agreement: true },
        semantic_comparability: 'guide_identity_bound', provisional_comparison_enabled: false,
        claim_interpretation: 'report-supplied target claims; review and history remain report claims; not adjudicated truth or qualification' } },
    deterministic: { coverage: null, f1: null, known_target_cells: 0, covered_known_target_cells: 0 } })
  const text = JSON.stringify(makeKnown('verified_sha256'))
  render(<ScorecardPanel maxBytes={100_000} maxFeedbackPages={2} maxFeedbackPageSize={10} currentOwnedStatus={null} />)
  fireEvent.change(screen.getByLabelText('Owned LFG report JSON'), {
    target: { files: [new File([text], 'known-report.json')] },
  })
  expect(await screen.findByText(/Guide provenance: development reference verified_sha256/)).toBeInTheDocument()
  expect(screen.getByText(/configured but inactive/)).toBeInTheDocument()
  expect(screen.queryByRole('note')).not.toBeInTheDocument()
  fireEvent.change(screen.getByLabelText('Owned LFG report JSON'), {
    target: { files: [new File([JSON.stringify(makeKnown('explicitly_none'))], 'explicit-none-report.json')] },
  })
  expect(await screen.findByText(/Guide provenance: development reference explicitly_none/)).toBeInTheDocument()
  expect(screen.getByText(/configured but inactive/)).toBeInTheDocument()
  expect(screen.queryByRole('note')).not.toBeInTheDocument()
})

test('a stale digest cannot replace a newer selection and malformed or oversized files stay private', async () => {
  const classicalFile = (name: string) => JSON.stringify({ schema_version: 'lfg.classifier-comparison-report.v1',
    task: { ...common, validation_support: support }, data: { counts: { validation: 2 },
      validation_record_id_sha256: [d('b'), d('c')], manifest_sha256: d('d'), plan_sha256: d('e'), policy_sha256: d('f'),
      review_ledger_sha256: d('1'), taxonomy_sha256: d('a'), train_sha256: d('2'), validation_sha256: d('3') },
    algorithms: { [name]: { status: 'completed', fit_vectorize_predict_elapsed_ms: 2, metrics: { coverage: 0.25 } } } })
  const first = classicalFile('logistic_regression')
  const second = classicalFile('current_dynamic_model')
  const originalDigest = crypto.subtle.digest.bind(crypto.subtle)
  let finishOld: ((value: ArrayBuffer) => void) | null = null
  let digestCalls = 0
  vi.spyOn(crypto.subtle, 'digest').mockImplementation((algorithm, data) => {
    digestCalls += 1
    if (digestCalls === 1) return new Promise((resolve) => { finishOld = resolve })
    return originalDigest(algorithm, data)
  })
  const view = render(<ScorecardPanel maxBytes={20_000} maxFeedbackPages={2} maxFeedbackPageSize={10} currentOwnedStatus={null} />)
  const input = screen.getByLabelText('Classical report JSON')
  fireEvent.change(input, { target: { files: [new File([first], 'first.json')] } })
  await vi.waitFor(() => expect(finishOld).not.toBeNull())
  fireEvent.change(input, { target: { files: [new File([second], 'second.json')] } })
  expect(await screen.findByRole('heading', { name: 'current_dynamic_model' })).toBeInTheDocument()
  finishOld!(new Uint8Array(32).buffer)
  await new Promise((resolve) => setTimeout(resolve, 0))
  expect(screen.getByRole('heading', { name: 'current_dynamic_model' })).toBeInTheDocument()
  expect(screen.queryByRole('heading', { name: 'logistic_regression' })).not.toBeInTheDocument()
  view.unmount()

  render(<ScorecardPanel maxBytes={4} maxFeedbackPages={2} maxFeedbackPageSize={10} currentOwnedStatus={null} />)
  fireEvent.change(screen.getByLabelText('Classical report JSON'), {
    target: { files: [new File([new Uint8Array([0xff, 0xfe])], 'bad-utf8.json')] },
  })
  expect(await screen.findByRole('alert')).toHaveTextContent('could not be read or validated')
  fireEvent.change(screen.getByLabelText('Local model report JSON'), {
    target: { files: [new File(['larger than configured'], 'oversize.json')] },
  })
  expect(await screen.findByRole('alert')).toHaveTextContent('per-file or combined byte limit')
})
