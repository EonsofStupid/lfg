import { useEffect, useRef, useState } from 'react'
import type { Assessment } from './types'
import { ownedAssessmentBody, ownedAssessmentClient, sha256Utf8, type OwnedAssessmentClient, type OwnedResult, type OwnedStatus } from './ownedAssessmentApi'

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
  const [projectContext, setProjectContext] = useState('')
  const [contextBinding, setContextBinding] = useState<'verified' | 'missing' | 'mismatch' | null>(null)
  const [connecting, setConnecting] = useState(false)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState('')
  const requestGeneration = useRef(0)
  const busyGeneration = useRef(0)
  const connectionGeneration = useRef(0)
  const onResultRef = useRef(onResult)
  const promptRef = useRef(prompt)
  const projectContextRef = useRef(projectContext)
  const tokenInputRef = useRef(tokenInput)
  onResultRef.current = onResult
  promptRef.current = prompt
  projectContextRef.current = projectContext
  tokenInputRef.current = tokenInput

  function invalidate() {
    requestGeneration.current += 1
    busyGeneration.current += 1
    setRunning(false)
    setResult(null)
    setContextBinding(null)
    setError('')
    onResultRef.current?.(null)
  }

  useEffect(() => { invalidate() }, [prompt])

  useEffect(() => () => {
    requestGeneration.current += 1
    busyGeneration.current += 1
    connectionGeneration.current += 1
    onResultRef.current?.(null)
  }, [])

  async function run(action: () => Promise<void>, request?: number) {
    const current = ++busyGeneration.current
    setRunning(true)
    setError('')
    try { await action() } catch (value) {
      if (request === undefined || requestGeneration.current === request) {
        setError(value instanceof Error ? value.message : 'The owned model request failed.')
      }
    } finally {
      if (busyGeneration.current === current) setRunning(false)
    }
  }

  const promptBytes = new TextEncoder().encode(prompt).length
  const contextBytes = new TextEncoder().encode(projectContext).length
  const hasProjectContext = projectContext.length > 0
  // A lowercase SHA-256 digest always occupies 64 ASCII bytes. This gives an
  // exact size estimate before hashing; the actual serialized body is checked
  // again immediately before sending.
  const sizeBody = ownedAssessmentBody(prompt, hasProjectContext
    ? { context: projectContext, contextSha256: '0'.repeat(64) }
    : undefined)
  const bodyBytes = new TextEncoder().encode(JSON.stringify(sizeBody)).length
  const bodyLimit = status?.limits?.maxHttpBodyBytes
  const requestTooLarge = status?.limits !== undefined && promptBytes > status.limits.maxRequestBytes
  const contextTooLarge = hasProjectContext && status?.limits?.maxContextBytes !== undefined
    && contextBytes > status.limits.maxContextBytes
  const bodyTooLarge = bodyLimit !== undefined && bodyBytes > bodyLimit
  const tooLarge = requestTooLarge || contextTooLarge || bodyTooLarge
  const proposal = result && ['ok', 'incomplete_output', 'truncated'].includes(result.status) ? result.output : null

  function runOwnedAssessment() {
    const current = ++requestGeneration.current
    const submittedPrompt = prompt
    const submittedContext = projectContext
    setResult(null)
    setContextBinding(null)
    onResultRef.current?.(null)
    void run(async () => {
      let boundContext: { context: string; contextSha256: string } | undefined
      if (submittedContext.length > 0) {
        const contextSha256 = await sha256Utf8(submittedContext)
        if (requestGeneration.current !== current || promptRef.current !== submittedPrompt
            || projectContextRef.current !== submittedContext) return
        boundContext = { context: submittedContext, contextSha256 }
      }
      const body = ownedAssessmentBody(submittedPrompt, boundContext)
      const actualBodyBytes = new TextEncoder().encode(JSON.stringify(body)).length
      if (bodyLimit !== undefined && actualBodyBytes > bodyLimit) {
        throw new Error('The exact UTF-8 JSON request exceeds the configured HTTP body limit; nothing was sent.')
      }
      const next = boundContext
        ? await client.assess(submittedPrompt, token!, boundContext)
        : await client.assess(submittedPrompt, token!)
      if (requestGeneration.current !== current || promptRef.current !== submittedPrompt
          || projectContextRef.current !== submittedContext) return
      if (boundContext) {
        setResult(next)
        if (next.projectContextSha256 === boundContext.contextSha256) {
          setContextBinding('verified')
          onResultRef.current?.(next)
        } else {
          setContextBinding(next.projectContextSha256 ? 'mismatch' : 'missing')
          onResultRef.current?.(null)
        }
      } else {
        const unboundResult = { ...next }
        delete unboundResult.projectContextSha256
        setResult(unboundResult)
        onResultRef.current?.(unboundResult)
      }
    }, current)
  }

  async function connectOwnedModel() {
    const current = ++connectionGeneration.current
    const submittedToken = tokenInput
    setConnecting(true)
    setError('')
    try {
      const next = await client.status(submittedToken)
      if (connectionGeneration.current !== current || tokenInputRef.current !== submittedToken) return
      setStatus(next)
      setToken(submittedToken)
      setTokenInput('')
      tokenInputRef.current = ''
    } catch (value) {
      if (connectionGeneration.current === current) {
        setError(value instanceof Error ? value.message : 'The owned model request failed.')
      }
    } finally {
      if (connectionGeneration.current === current) setConnecting(false)
    }
  }

  return <section className="dev-panel owned-assessment" aria-label="Experimental owned model">
    <div className="dev-heading"><div><p className="eyebrow">EXPERIMENTAL OWNED MODEL</p><h2>Compare LFG's model</h2></div></div>
    <p>Run the current request through the experimental owned model. Its output is a proposal, not a verified assessment; the existing rules remain available above. Availability, project outcomes, and entitlement remain unknown unless separately verified. This run does not automatically include subscription settings or project profile answers.</p>
    <label>Optional project context (owner-supplied, unverified)
      <textarea rows={4} value={projectContext} onChange={(event) => {
        setProjectContext(event.target.value)
        invalidate()
      }} />
    </label>
    <p>Context is sent exactly as typed only when you run the model and stays in this component's memory. The digest identifies those bytes; it does not verify their truth or remove secrets. Model output may repeat the context. Saving feedback is a separate explicit action and can save model output, including text that repeats context.</p>
    {!token ? <>
      <label>Owned model token <input type="password" autoComplete="off" value={tokenInput} onChange={(event) => {
        const next = event.target.value
        tokenInputRef.current = next
        connectionGeneration.current += 1
        setConnecting(false)
        setTokenInput(next)
      }} /></label>
      <button type="button" disabled={connecting || !tokenInput} onClick={() => void connectOwnedModel()}>{connecting ? 'Connecting…' : 'Connect owned model'}</button>
      <p>The token is held only in this browser session. Configure it and the model on the local server first.</p>
    </> : <>
      <p role="status">Model status: {status?.status ?? 'unknown'}{status?.reason ? ` · ${REASON_MESSAGE[status.reason] ?? `Reason: ${status.reason}`}` : ''}</p>
      {status?.status === 'ready' && <p>Ready means the configuration and artifact identity were checked. Each run loads the model; its quality and speed are not yet qualified.</p>}
      {status?.engine && <p>Engine: {status.engine.id} · Model: {status.engine.modelSha256 ? status.engine.modelSha256.slice(0, 12) : 'identity unavailable'}</p>}
      {status?.limits && <p>Limit: {status.limits.maxRequestBytes.toLocaleString()} request bytes{status.limits.maxContextBytes ? ` · ${status.limits.maxContextBytes.toLocaleString()} context bytes including framing` : ''} · {status.limits.maxGeneratedBytes.toLocaleString()} output bytes · {status.limits.timeoutMs.toLocaleString()} ms · {status.limits.maxConcurrent} concurrent{status.limits.workerThreads ? ` · ${status.limits.workerThreads} threads requested` : ''}</p>}
      {bodyLimit !== undefined && <p>Serialized request body: {bodyBytes.toLocaleString()} / {bodyLimit.toLocaleString()} UTF-8 bytes.</p>}
      {bodyLimit === undefined && <p>Serialized request body: {bodyBytes.toLocaleString()} UTF-8 bytes. The owned route HTTP body limit is not advertised; the server may reject this request.</p>}
      <div className="feedback-actions">
        <button type="button" disabled={running || status?.status !== 'ready' || !prompt.trim() || tooLarge} onClick={runOwnedAssessment}>{running ? 'Running model…' : 'Run owned model'}</button>
        <button type="button" disabled={running} onClick={() => { invalidate(); connectionGeneration.current += 1; setToken(null); setStatus(null); setProjectContext('') }}>Disconnect</button>
      </div>
      {requestTooLarge && <p role="alert">The request exceeds the model's configured UTF-8 input limit.</p>}
      {contextTooLarge && <p role="alert">The supplied context exceeds the model's configured context byte limit.</p>}
      {bodyTooLarge && <p role="alert">The exact UTF-8 JSON request, including the context digest, exceeds the configured HTTP body limit.</p>}
      {hasProjectContext && <p>{contextBytes.toLocaleString()} UTF-8 context bytes. The complete system framing and context window are not available here; the server may return “context limit” and will not silently truncate this text.</p>}
      {assessment && <p>Rules assessment: intent {assessment.intent.primary}, complexity {assessment.complexity.bracket}, {assessment.agents.total} agents.</p>}
      {contextBinding === 'verified' && <p role="status">The server echoed the matching project context digest. It identifies the supplied bytes only; it does not verify project facts.</p>}
      {contextBinding === 'missing' && <p role="alert">The response did not include the expected context digest. The result is shown but cannot be saved as linked to this context.</p>}
      {contextBinding === 'mismatch' && <p role="alert">The response context digest did not match the submitted bytes. The result is shown but cannot be saved as linked to this context.</p>}
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
