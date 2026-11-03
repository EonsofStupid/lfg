import { useEffect, useRef, useState } from 'react'
import type { Assessment } from './types'
import { ownedAssessmentClient, type OwnedAssessmentClient, type OwnedResult, type OwnedStatus } from './ownedAssessmentApi'

const STATUS_MESSAGE: Record<OwnedResult['status'], string> = {
  ok: 'Model output received. It has not been verified.',
  truncated: 'The model stopped at an output or context limit before completing its response.',
  timeout: 'The model exceeded the configured time limit.',
  busy: 'All configured model workers are busy. Try again after a worker is free.',
  invalid_input: 'The request is empty, invalid, or exceeds the configured input limit.',
  context_limit: 'The request and model framing exceed the configured model context.',
  incomplete_output: 'The model returned an assessment with missing or invalid required fields.',
  malformed_output: 'The model returned an incomplete or malformed assessment.',
  inference_error: 'The model could not complete inference.',
  unavailable: 'The experimental model is unavailable.',
}

const REASON_MESSAGE: Record<string, string> = {
  service_disabled: 'The owned model is not enabled in the local configuration.',
  session_token_unavailable: 'Configure a private token for the owned model.',
  loopback_listener_required: 'Open the developer server on this machine using a loopback address.',
  owned_runtime_not_built: 'This executable was built without the owned-model runtime.',
  asset_unavailable: 'A configured model or framing file could not be opened.',
  invalid_configuration: 'The owned-model configuration is invalid.',
  invalid_runtime_configuration: 'The native runtime configuration is invalid.',
  runtime_limits_mismatch: 'The configured limits do not match the model runtime.',
  artifact_identity_mismatch: 'The model does not match its configured identity.',
  framing_identity_mismatch: 'The framing file does not match its configured identity.',
  provenance_mismatch: 'The model and its training or framing identities do not match.',
}

function display(value: unknown): string {
  if (value === null || value === undefined || value === '') return 'Unknown'
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') return String(value)
  return JSON.stringify(value)
}

export function OwnedAssessmentPanel({ prompt, assessment, client = ownedAssessmentClient, onResult }: {
  prompt: string
  assessment: Assessment | null
  client?: OwnedAssessmentClient
  onResult?: (result: OwnedResult | null) => void
}) {
  const [tokenInput, setTokenInput] = useState('')
  const [token, setToken] = useState<string | null>(null)
  const [status, setStatus] = useState<OwnedStatus | null>(null)
  const [result, setResult] = useState<OwnedResult | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const requestGeneration = useRef(0)
  useEffect(() => { requestGeneration.current += 1; setResult(null) }, [prompt])

  async function run(action: () => Promise<void>) {
    setBusy(true); setError('')
    try { await action() } catch (value) { setError(value instanceof Error ? value.message : 'The owned model request failed.') }
    finally { setBusy(false) }
  }

  const bytes = new TextEncoder().encode(prompt).length
  const bodyBytes = new TextEncoder().encode(JSON.stringify({ text: prompt })).length
  const tooLarge = status?.limits !== undefined && (bytes > status.limits.maxRequestBytes || (status.limits.maxHttpBodyBytes !== undefined && bodyBytes > status.limits.maxHttpBodyBytes))
  const proposal = result && ['ok', 'incomplete_output', 'truncated'].includes(result.status) ? result.output : null

  return <section className="dev-panel owned-assessment" aria-label="Experimental owned model">
    <div className="dev-heading"><div><p className="eyebrow">EXPERIMENTAL OWNED MODEL</p><h2>Compare LFG's model</h2></div></div>
    <p>Run the current request through the experimental owned model. Its output is a proposal, not a verified assessment; the existing rules remain available above. This model path does not yet include the project's subscription profile.</p>
    {!token ? <>
      <label>Owned model token <input type="password" autoComplete="off" value={tokenInput} onChange={(event) => setTokenInput(event.target.value)} /></label>
      <button type="button" disabled={busy || !tokenInput} onClick={() => void run(async () => {
        const next = await client.status(tokenInput)
        setStatus(next); setToken(tokenInput); setTokenInput('')
      })}>Connect owned model</button>
      <p>The token is held only in this browser session. Configure it and the model on the local server first.</p>
    </> : <>
      <p role="status">Model status: {status?.status ?? 'unknown'}{status?.reason ? ` · ${REASON_MESSAGE[status.reason] ?? `Reason: ${status.reason}`}` : ''}</p>
      {status?.status === 'ready' && <p>Ready means the configuration and artifact identity were checked. Each run loads the model; its quality and speed are not yet qualified.</p>}
      {status?.engine && <p>Engine: {status.engine.id} · Model: {status.engine.modelSha256 ? status.engine.modelSha256.slice(0, 12) : 'identity unavailable'}</p>}
      {status?.limits && <p>Limit: {status.limits.maxRequestBytes.toLocaleString()} request bytes{status.limits.maxContextBytes ? ` · ${status.limits.maxContextBytes.toLocaleString()} context bytes including framing` : ''} · {status.limits.maxGeneratedBytes.toLocaleString()} output bytes · {status.limits.timeoutMs.toLocaleString()} ms · {status.limits.maxConcurrent} concurrent{status.limits.workerThreads ? ` · ${status.limits.workerThreads} threads requested` : ''}</p>}
      <div className="feedback-actions">
        <button type="button" disabled={busy || status?.status !== 'ready' || !prompt.trim() || tooLarge} onClick={() => void run(async () => {
          const current = ++requestGeneration.current
          setResult(null)
          const next = await client.assess(prompt, token)
          if (requestGeneration.current === current) { setResult(next); onResult?.(next) }
        })}>{busy ? 'Running model…' : 'Run owned model'}</button>
        <button type="button" disabled={busy} onClick={() => { requestGeneration.current += 1; setToken(null); setStatus(null); setResult(null); onResult?.(null) }}>Disconnect</button>
      </div>
      {tooLarge && <p role="alert">This request exceeds the model's configured input or HTTP body limit.</p>}
      {assessment && <p>Rules assessment: intent {assessment.intent.primary}, complexity {assessment.complexity.bracket}, {assessment.agents.total} agents.</p>}
      {result && <div className="owned-result">
        <p role="status">{STATUS_MESSAGE[result.status] ?? 'The model returned an unrecognized status.'} · {result.elapsedMs.toFixed(0)} ms{result.stopReason ? ` · ${result.stopReason}` : ''}</p>
        {proposal && <dl className="owned-fields">
          <div><dt>Intent</dt><dd>{display(proposal.intent)}</dd></div>
          <div><dt>Complexity</dt><dd>{display(proposal.complexity)}</dd></div>
          <div><dt>Suggested models</dt><dd>{Array.isArray(proposal.suggested_models) && proposal.suggested_models.length === 0 ? 'No models suggested' : display(proposal.suggested_models)}</dd></div>
          <div><dt>Agent count</dt><dd>{proposal.agent_count === 0 ? 'No agents suggested' : display(proposal.agent_count)}</dd></div>
          {proposal.reasons !== undefined && <div><dt>Reasons</dt><dd>{Array.isArray(proposal.reasons) ? <ul>{proposal.reasons.map((reason, index) => <li key={index}>{display(reason)}</li>)}</ul> : display(proposal.reasons)}</dd></div>}
          {proposal.reason !== undefined && <div><dt>Reason</dt><dd>{display(proposal.reason)}</dd></div>}
        </dl>}
        {result.outputText && <details><summary>Show raw model output</summary><pre>{result.outputText}</pre></details>}
      </div>}
    </>}
    {error && <p role="alert">{error}</p>}
  </section>
}
