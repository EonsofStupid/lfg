import { useEffect, useState } from 'react'
import { getAppContext, previewIntake } from './api'
import type { AppContext, ClassificationSnapshot, IntakePreview } from './types'

export function IntakePanel({ prompt, promptTooLarge, revision, classification, activeSubscriptions, debounceMs }: {
  prompt: string
  promptTooLarge: boolean
  revision: number
  classification: ClassificationSnapshot | null
  activeSubscriptions: string[]
  debounceMs: number
}) {
  const [preview, setPreview] = useState<IntakePreview | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [context, setContext] = useState<AppContext | null>(null)
  useEffect(() => {
    let stopped = false
    let timer: number | undefined
    let interval: number | null = null
    const poll = async () => {
      try {
        const result = await getAppContext()
        if (stopped) return
        if (result.schemaVersion !== 'lfg.app-context.v1' ||
          !['disabled', 'ready', 'unavailable'].includes(result.status) ||
          (result.status !== 'disabled' && (!Number.isSafeInteger(result.pollIntervalMs) || !result.pollIntervalMs || result.pollIntervalMs < 1 || result.pollIntervalMs > 2147483647)) ||
          (result.status === 'ready' && (!Array.isArray(result.providers) || result.providers.some(provider =>
            typeof provider.id !== 'string' || typeof provider.activated !== 'boolean' ||
            !Array.isArray(provider.capabilities) || provider.capabilities.some(capability => typeof capability !== 'string')))) ||
          (result.status !== 'ready' && result.providers !== null)) {
          throw new Error('Incompatible app context')
        }
        setContext(result)
        interval = result.pollIntervalMs
      } catch {
        if (stopped) return
        setContext({schemaVersion: 'lfg.app-context.v1', status: 'unavailable', revision: null, pollIntervalMs: interval, providers: null})
      }
      if (!stopped && interval !== null) timer = window.setTimeout(() => { void poll() }, interval)
    }
    void poll()
    return () => { stopped = true; if (timer !== undefined) window.clearTimeout(timer) }
  }, [])
  useEffect(() => {
    let stopped = false
    setPreview(null)
    setError(null)
    const current = classification?.revision === revision ? classification : null
    const facts: Record<string, unknown> = { request: promptTooLarge ? {} : { text: prompt } }
    if (current) {
      facts.project = { id: current.project }
      if (current.escalationCatalog.length > 0) {
        if (context?.status === 'disabled') facts.providers = current.escalationCatalog.map(({ target }) => ({
          provider: target.provider, subscriptionId: target.subscriptionId,
          userDeclaredAvailable: activeSubscriptions.includes(target.subscriptionId)
        }))
        facts.models = current.escalationCatalog.map(({ target }) => ({
          provider: target.provider, model: target.model, subscriptionId: target.subscriptionId,
          strengths: target.strengths, effortLevels: target.effortLevels, evidence: target.evidence,
          userDeclaredAvailable: activeSubscriptions.includes(target.subscriptionId)
        }))
      }
    }
    if (context?.status === 'ready' && context.providers) facts.providers = context.providers
    const timer = window.setTimeout(() => {
      void previewIntake(facts).then(value => {
        if (!stopped) { setPreview(value); setError(null) }
      }).catch((value: Error) => { if (!stopped) setError(value.message) })
    }, debounceMs)
    return () => { stopped = true; window.clearTimeout(timer) }
  }, [prompt, promptTooLarge, revision, classification, activeSubscriptions, debounceMs, context])

  return <section className="dev-panel" aria-label="Project intake">
    <div className="dev-heading"><div><p className="eyebrow">CONFIGURABLE QUESTIONS</p><h2>Project intake</h2></div></div>
    <p>{context?.status === 'ready'
      ? 'Providers come from the live DevPulse roster. Activation can mean history-only access; capabilities show what is qualified. Models are configured declarations.'
      : context?.status === 'unavailable'
        ? 'DevPulse context is unavailable; provider availability is unknown.'
        : 'Filled from this LFG session and its configured model catalog. Availability is a user declaration, not a verified subscription.'}</p>
    {error ? <p role="alert">{error}</p> : !preview ? <p>Preparing intake…</p> : <>
      <p>{preview.missingRequired} required questions need input · template {preview.templateRevision}</p>
      <dl className="intake-list">{preview.answers.map(answer => <div key={answer.id}>
        <dt>{answer.question}{answer.required ? ' · required' : ''}</dt>
        <dd>{answer.status === 'filled' ? <pre>{typeof answer.value === 'string' ? answer.value : JSON.stringify(answer.value, null, 2)}</pre> : 'Needs input'}</dd>
      </div>)}</dl>
    </>}
  </section>
}
