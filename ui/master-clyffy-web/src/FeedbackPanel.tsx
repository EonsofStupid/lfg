import { useState } from 'react'
import type { Assessment } from './types'
import { feedbackClient, type FeedbackClient, type FeedbackRecord, type FeedbackState } from './feedbackApi'

// Assessment.prompt and matched terms may contain user text. Never save the full
// assessment implicitly: capture only the output fields needed for supervision.
export async function feedbackFromAssessment(assessment: Assessment, prompt: string): Promise<FeedbackState> {
  if (!globalThis.crypto?.subtle) throw new Error('A secure local browser context is required to identify the original request.')
  const digest = await globalThis.crypto.subtle.digest('SHA-256', new TextEncoder().encode(prompt))
  const sourceRef = 'sha256:' + Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('')
  return {
    assessmentRef: `${assessment.sessionId}:${assessment.revision}`,
    provenance: { engineId: 'deterministic-assessment', configDigest: assessment.configDigest, sourceRef },
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
    ownerCorrection: null,
    outcome: { status: 'unknown' },
  }
}

function asText(value: unknown): string { return typeof value === 'string' ? value : '' }

export function FeedbackPanel({ assessment, prompt, client = feedbackClient }: {
  assessment: Assessment | null
  prompt: string
  client?: FeedbackClient
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
    <p>Save the recommendation, add your correction, and record an outcome only when you have observed it. A digest can help match the original prompt to a separately available source; its text is not copied. Anything you enter here is saved locally.</p>
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
        <button type="button" disabled={busy || !assessment} onClick={() => void run(async () => {
          if (!assessment) return
          const created = await client.create(await feedbackFromAssessment(assessment, prompt), token)
          await open(created.feedbackId); await refresh(created); setMessage('Assessment saved for review.')
        })}>Save current assessment</button>
        <button type="button" disabled={busy} onClick={() => void run(() => refresh(record ?? undefined))}>Load saved assessments</button>
        <button type="button" disabled={busy} onClick={() => { setToken(null); setAvailable(false); setEntries([]); setNextCursor(null); setRecord(null); setHistory([]); setHistoryNextBefore(null); setMessage('Local feedback disconnected.') }}>Disconnect</button>
      </div>
      {entries.length > 0 && <label>Saved assessment <select aria-label="Saved assessment" value={record?.feedbackId ?? ''} onChange={(event) => {
        if (event.target.value) void run(() => open(event.target.value))
      }}><option value="">Choose an assessment</option>{entries.map((entry) => <option key={entry.feedbackId} value={entry.feedbackId}>{entry.feedbackId} · revision {entry.revision}</option>)}</select></label>}
      {nextCursor && <button type="button" disabled={busy} onClick={() => void run(loadMore)}>Load more assessments</button>}
      {record && <div className="feedback-editor">
        <p>Engine: {record.provenance.engineId} · Assessment: {record.assessmentRef} · Revision {record.revision}</p>
        <p>Original suggestion: intent {String(record.originalRecommendation.intent ?? 'unknown')}, complexity {String(record.originalRecommendation.complexity ?? 'unknown')}, {String(record.originalRecommendation.agentCount ?? 'unknown')} agents, model {String(record.originalRecommendation.model ?? 'unknown')}.</p>
        <p>Fill in only the fields you want to correct; leave others blank.</p>
        <label>Corrected intent <input value={intent} onChange={(event) => setIntent(event.target.value)} /></label>
        <label>Corrected complexity <input value={complexity} onChange={(event) => setComplexity(event.target.value)} /></label>
        <label>Corrected model <input value={model} onChange={(event) => setModel(event.target.value)} /></label>
        <label>Corrected agent count <input type="number" min="1" step="1" value={agentCount} onChange={(event) => setAgentCount(event.target.value)} /></label>
        <label>Owner correction <textarea value={correction} onChange={(event) => setCorrection(event.target.value)} rows={3} /></label>
        <label><input type="checkbox" checked={confirmed} onChange={(event) => setConfirmed(event.target.checked)} /> I observed the outcome</label>
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
