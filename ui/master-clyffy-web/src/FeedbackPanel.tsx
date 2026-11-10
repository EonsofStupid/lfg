import { useEffect, useState } from 'react'
import type { Assessment } from './types'
import { feedbackClient, type FeedbackClient, type FeedbackRecord, type FeedbackState } from './feedbackApi'
import type { OwnedResult } from './ownedAssessmentApi'

// Assessment.prompt and matched terms may contain user text. Never save the full
// assessment implicitly: capture only the output fields needed for supervision.
async function feedbackBase(reference: string, prompt: string): Promise<FeedbackState> {
  if (!globalThis.crypto?.subtle) throw new Error('A secure local browser context is required to identify the original request.')
  const digest = await globalThis.crypto.subtle.digest('SHA-256', new TextEncoder().encode(prompt))
  const sourceRef = 'sha256:' + Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('')
  return {
    assessmentRef: reference, provenance: { engineId: 'unknown', sourceRef }, originalRecommendation: {},
    ownerCorrection: null, outcome: { status: 'unknown' },
  }
}

export async function feedbackFromAssessment(assessment: Assessment, prompt: string): Promise<FeedbackState> {
  const base = await feedbackBase(`${assessment.sessionId}:${assessment.revision}`, prompt)
  return {
    ...base,
    provenance: { ...base.provenance, engineId: 'deterministic-assessment', configDigest: assessment.configDigest },
    originalRecommendation: {
      intent: assessment.intent.primary,
      complexity: assessment.complexity.bracket,
      complexityScore: assessment.complexity.score,
      agentCount: assessment.agents.total,
      model: assessment.models.suggested?.model ?? null,
      subscription: assessment.models.suggested?.subscriptionId ?? null,
      role: assessment.models.role,
      effort: assessment.models.effort,
      reasonCodes: assessment.rules,
      scoringTruncated: !!assessment.prompt.scoringTruncated,
    },
  }
}

function asText(value: unknown): string { return typeof value === 'string' ? value : '' }
function displayValue(value: unknown): string {
  if (value === null || value === undefined) return 'unknown'
  return typeof value === 'object' ? JSON.stringify(value) : String(value)
}

export function feedbackFromOwned(state: FeedbackState, result: OwnedResult): FeedbackState {
  const output = result.output ?? {}
  const { configDigest, ...baseProvenance } = state.provenance
  return {
    ...state,
    provenance: {
      ...baseProvenance, engineId: result.engine?.id ?? 'unknown',
      ...(configDigest ? { comparisonRulesConfigDigest: configDigest } : {}),
      ...(result.engine?.modelSha256 ? { modelDigest: result.engine.modelSha256 } : {}),
      ...(result.engine?.modelManifestSha256 ? { modelManifestSha256: result.engine.modelManifestSha256 } : {}),
      ...(result.engine?.trainingManifestSha256 ? { trainingManifestSha256: result.engine.trainingManifestSha256 } : {}),
      ...(result.engine?.framingSha256 ? { framingSha256: result.engine.framingSha256 } : {}),
      ...(result.engine?.runtimeConfigSha256 ? { runtimeConfigSha256: result.engine.runtimeConfigSha256 } : {}),
      ...(result.engine?.serviceConfigSha256 ? { serviceConfigSha256: result.engine.serviceConfigSha256 } : {}),
      ...(typeof result.modelListSha256 === 'string' ? { modelListSha256: result.modelListSha256 } : {}),
      ...(typeof result.projectContextSha256 === 'string' && /^[0-9a-f]{64}$/.test(result.projectContextSha256)
        ? { projectContextSha256: result.projectContextSha256 } : {}),
    },
    originalRecommendation: {
      intent: output.intent ?? null, complexity: output.complexity ?? null,
      model: output.suggested_models ?? null, agentCount: output.agent_count ?? null,
      reason: output.reason ?? null, modelOutput: output,
      generationStatus: result.status, stopReason: result.stopReason ?? null,
      ...(result.outputText ? { rawOutput: result.outputText } : {}),
    },
  }
}

export function FeedbackPanel({ assessment, prompt, client = feedbackClient, ownedResult = null, assessmentRef }: {
  assessment: Assessment | null
  prompt: string
  client?: FeedbackClient
  ownedResult?: OwnedResult | null
  assessmentRef?: string
}) {
  const [token, setToken] = useState<string | null>(null)
  const [tokenInput, setTokenInput] = useState('')
  const [available, setAvailable] = useState(false)
  const [unavailableReason, setUnavailableReason] = useState('')
  const [entries, setEntries] = useState<FeedbackRecord[]>([])
  const [nextCursor, setNextCursor] = useState<string | null>(null)
  const [record, setRecord] = useState<FeedbackRecord | null>(null)
  const [history, setHistory] = useState<FeedbackRecord[]>([])
  const [historyNextBefore, setHistoryNextBefore] = useState<number | null>(null)
  const [selectedRevision, setSelectedRevision] = useState('')
  const [correction, setCorrection] = useState('')
  const [intent, setIntent] = useState('')
  const [complexity, setComplexity] = useState('')
  const [model, setModel] = useState('')
  const [agentCount, setAgentCount] = useState('')
  const [observed, setObserved] = useState('')
  const [confirmer, setConfirmer] = useState('')
  const [confirmed, setConfirmed] = useState(false)
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [source, setSource] = useState<'rules' | 'owned'>('rules')
  const canSaveOwned = !!ownedResult && (!!ownedResult.output || !!ownedResult.outputText)
  useEffect(() => { if (!canSaveOwned) setSource('rules') }, [canSaveOwned])

  function show(next: FeedbackRecord, revisions: FeedbackRecord[], before: number | null) {
    setRecord(next); setHistory(revisions); setSelectedRevision(String(next.revision))
    setHistoryNextBefore(before)
    setCorrection(asText(next.ownerCorrection?.note))
    setIntent(asText(next.ownerCorrection?.intent))
    setComplexity(asText(next.ownerCorrection?.complexity))
    setModel(asText(next.ownerCorrection?.model))
    setAgentCount(typeof next.ownerCorrection?.agentCount === 'number' ? String(next.ownerCorrection.agentCount) : '')
    setConfirmed(next.outcome?.status === 'confirmed')
    setObserved(asText(next.outcome?.observation))
    setConfirmer(asText(next.outcome?.confirmer))
  }

  async function run(action: () => Promise<void>) {
    setBusy(true); setError(''); setMessage('')
    try { await action() } catch (value) { setError(value instanceof Error ? value.message : 'Feedback could not be saved.') }
    finally { setBusy(false) }
  }

  async function refresh(selected?: FeedbackRecord) {
    if (!token) return
    const page = await client.list(token)
    setEntries(selected && !page.records.some((entry) => entry.feedbackId === selected.feedbackId) ? [selected, ...page.records] : page.records)
    setNextCursor(page.nextCursor)
  }

  async function loadMore() {
    if (!token || !nextCursor) return
    const page = await client.list(token, nextCursor)
    setEntries((current) => [...current, ...page.records.filter((entry) => !current.some((old) => old.feedbackId === entry.feedbackId))])
    setNextCursor(page.nextCursor)
  }

  async function open(id: string) {
    if (!token) throw new Error('Connect local feedback first.')
    const result = await client.read(id, token)
    show(result.record, result.history, result.historyNextBefore)
  }

  async function loadOlderHistory() {
    if (!token || !record || historyNextBefore === null) return
    const result = await client.read(record.feedbackId, token, historyNextBefore)
    setHistory((current) => [...current, ...result.history.filter((entry) => !current.some((old) => old.revision === entry.revision))])
    setHistoryNextBefore(result.historyNextBefore)
  }

  function revisedState(current: FeedbackRecord): FeedbackState {
    const state: FeedbackState = { ...current }
    for (const key of ['feedbackId', 'revision', 'createdAt', 'updatedAt', 'parentRevision', 'action', 'restoredRevision']) delete state[key]
    const nextOutcome = { ...current.outcome }
    const nextCorrection = { ...current.ownerCorrection }
    for (const [key, value] of Object.entries({ note: correction, intent, complexity, model })) {
      if (value.trim()) nextCorrection[key] = value.trim()
      else delete nextCorrection[key]
    }
    if (agentCount.trim()) nextCorrection.agentCount = Number(agentCount)
    else delete nextCorrection.agentCount
    if (!confirmed) {
      delete nextOutcome.observation; delete nextOutcome.confirmer; delete nextOutcome.confirmedAt
    }
    return {
      ...state,
      ownerCorrection: Object.keys(nextCorrection).length ? nextCorrection : null,
      outcome: confirmed ? { ...nextOutcome, status: 'confirmed', observation: observed.trim(), confirmer: confirmer.trim() } : { ...nextOutcome, status: 'unknown' },
    }
  }

  const selectedHistory = history.find((entry) => String(entry.revision) === selectedRevision)

  return <section className="dev-panel feedback-panel" aria-label="Assessment feedback">
    <div className="dev-heading"><div><p className="eyebrow">LOCAL SUPERVISION</p><h2>Review an assessment</h2></div></div>
    <p>Save the recommendation, add your correction, and record an outcome only when you have observed it. A digest can help match the original prompt to a separately available source; it is not copied automatically. Anything you enter here is saved locally. Model output may repeat private prompt or supplied project context; a digest identifies bytes and does not screen saved output.</p>
    {!available || !token ? <>
      <p>Connect using the private feedback token configured for your local server. It stays in this browser session and is not put in the address bar.</p>
      <label>Local feedback token <input type="password" autoComplete="off" value={tokenInput} onChange={(event) => setTokenInput(event.target.value)} /></label>
      <button type="button" disabled={busy || !tokenInput} onClick={() => void run(async () => {
        const status = await client.status(tokenInput)
        if (!status.enabled || !status.available) { setUnavailableReason(status.reason ?? 'Check the private feedback configuration.'); return }
        setAvailable(true); setToken(tokenInput); setTokenInput(''); setUnavailableReason(''); setMessage('Local feedback connected.')
      })}>Connect feedback</button>
      {unavailableReason && <p role="status">Local feedback is unavailable. {unavailableReason}</p>}
    </> : <>
      <div className="feedback-actions">
        {canSaveOwned && <label>Assessment source <select value={source} onChange={(event) => setSource(event.target.value === 'owned' ? 'owned' : 'rules')}><option value="rules">Rules assessment</option><option value="owned">{typeof ownedResult?.engine?.label === 'string' ? ownedResult.engine.label : 'Experimental owned model'}</option></select></label>}
        <button type="button" disabled={busy || (source === 'rules' ? !assessment : !canSaveOwned || !(assessmentRef || assessment))} onClick={() => void run(async () => {
          if (source === 'rules' && !assessment) return
          let state = assessment ? await feedbackFromAssessment(assessment, prompt) : await feedbackBase(assessmentRef!, prompt)
          if (source === 'owned' && ownedResult) state = feedbackFromOwned(state, ownedResult)
          const created = await client.create(state, token)
          await open(created.feedbackId); await refresh(created); setMessage('Assessment saved for review.')
        })}>Save current assessment</button>
        <button type="button" disabled={busy} onClick={() => void run(() => refresh(record ?? undefined))}>Load saved assessments</button>
        <button type="button" disabled={busy} onClick={() => { setToken(null); setAvailable(false); setEntries([]); setNextCursor(null); setRecord(null); setHistory([]); setHistoryNextBefore(null); setMessage('Local feedback disconnected.') }}>Disconnect</button>
      </div>
      {source === 'owned' && canSaveOwned && <p>The model's actual output, including any partial text and extra fields, will be saved locally. It may repeat parts of the request; it remains unverified.</p>}
      {entries.length > 0 && <label>Saved assessment <select aria-label="Saved assessment" value={record?.feedbackId ?? ''} onChange={(event) => {
        if (event.target.value) void run(() => open(event.target.value))
      }}><option value="">Choose an assessment</option>{entries.map((entry) => <option key={entry.feedbackId} value={entry.feedbackId}>{entry.feedbackId} · revision {entry.revision}</option>)}</select></label>}
      {nextCursor && <button type="button" disabled={busy} onClick={() => void run(loadMore)}>Load more assessments</button>}
      {record && <div className="feedback-editor">
        <p>Engine: {record.provenance.engineId} · Assessment: {record.assessmentRef} · Revision {record.revision}</p>
        <details><summary>View provenance</summary><p>Model: {displayValue(record.provenance.modelDigest)} · Model manifest: {displayValue(record.provenance.modelManifestSha256)} · Training manifest: {displayValue(record.provenance.trainingManifestSha256)} · Framing: {displayValue(record.provenance.framingSha256)} · Runtime config: {displayValue(record.provenance.runtimeConfigSha256)} · Service config: {displayValue(record.provenance.serviceConfigSha256)} · Source reference: {displayValue(record.provenance.sourceRef)} · Project context digest: {displayValue(record.provenance.projectContextSha256)}</p><p>These identify recorded artifacts and submitted bytes; they do not verify project facts or the correctness of an assessment.</p></details>
        <p>Original suggestion: intent {displayValue(record.originalRecommendation.intent)}, complexity {displayValue(record.originalRecommendation.complexity)}, {displayValue(record.originalRecommendation.agentCount)} agents, model {displayValue(record.originalRecommendation.model)}.</p>
        <p>Fill in only the fields you want to correct; leave others blank.</p>
        <label>Corrected intent <input value={intent} onChange={(event) => setIntent(event.target.value)} /></label>
        <label>Corrected complexity <input value={complexity} onChange={(event) => setComplexity(event.target.value)} /></label>
        <label>Corrected model <input value={model} onChange={(event) => setModel(event.target.value)} /></label>
        <label>Corrected agent count <input type="number" min="1" step="1" value={agentCount} onChange={(event) => setAgentCount(event.target.value)} /></label>
        <label>Owner correction <textarea value={correction} onChange={(event) => setCorrection(event.target.value)} rows={3} /></label>
        <label><input type="checkbox" checked={confirmed} onChange={(event) => setConfirmed(event.target.checked)} /> I observed the outcome</label>
        {record.outcome?.kind !== undefined && record.outcome?.kind !== null && <p>Recorded outcome type: {displayValue(record.outcome.kind)}</p>}
        {confirmed ? <>
          <label>Observed outcome <textarea value={observed} onChange={(event) => setObserved(event.target.value)} rows={2} /></label>
          <label>Confirmed by <input value={confirmer} onChange={(event) => setConfirmer(event.target.value)} /></label>
        </> : <p>Outcome: unknown</p>}
        <div className="feedback-actions"><button type="button" disabled={busy || (confirmed && (!observed.trim() || !confirmer.trim())) || (!!agentCount && (!Number.isSafeInteger(Number(agentCount)) || Number(agentCount) < 1))} onClick={() => void run(async () => {
          const updated = await client.revise(record.feedbackId, record.revision, revisedState(record), token)
          await open(updated.feedbackId); await refresh(updated); setMessage('Review saved as a new revision.')
        })}>Save review</button>
          <label>Earlier revision <select value={selectedRevision} onChange={(event) => setSelectedRevision(event.target.value)}>{history.map((entry) => <option key={entry.revision} value={entry.revision}>Revision {entry.revision}</option>)}</select></label>
          <button type="button" disabled={busy || !selectedRevision || Number(selectedRevision) === record.revision} onClick={() => void run(async () => {
            const restored = await client.rollback(record.feedbackId, record.revision, Number(selectedRevision), token)
            await open(restored.feedbackId); await refresh(restored); setMessage('Earlier state restored as a new revision.')
          })}>Restore selected revision</button>
          {historyNextBefore !== null && <button type="button" disabled={busy} onClick={() => void run(loadOlderHistory)}>Load older revisions</button>}
        </div>
        {selectedHistory && <details><summary>View selected revision</summary>
          <p>Revision {selectedHistory.revision} · {selectedHistory.action} · {selectedHistory.createdAt}</p>
          <p>Correction: {asText(selectedHistory.ownerCorrection?.note) || 'No note'}.</p>
          <p>Intent: {asText(selectedHistory.ownerCorrection?.intent) || 'not corrected'} · Complexity: {asText(selectedHistory.ownerCorrection?.complexity) || 'not corrected'} · Model: {asText(selectedHistory.ownerCorrection?.model) || 'not corrected'} · Agents: {String(selectedHistory.ownerCorrection?.agentCount ?? 'not corrected')}</p>
          <p>Outcome: {selectedHistory.outcome?.status === 'confirmed' ? asText(selectedHistory.outcome.observation) || 'confirmed' : 'unknown'}</p>
        </details>}
      </div>}
    </>}
    {message && <p role="status">{message}</p>}
    {error && <p role="alert">{error}</p>}
  </section>
}
