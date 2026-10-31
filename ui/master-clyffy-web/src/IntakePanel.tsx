import { useEffect, useState } from 'react'
import { previewIntake } from './api'
import type { ClassificationSnapshot, IntakePreview } from './types'

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
  useEffect(() => {
    let stopped = false
    setPreview(null)
    setError(null)
    const current = classification?.revision === revision ? classification : null
    const facts: Record<string, unknown> = { request: promptTooLarge ? {} : { text: prompt } }
    if (current) {
      facts.project = { id: current.project }
      if (current.escalationCatalog.length > 0) {
        facts.providers = current.escalationCatalog.map(({ target }) => ({
        provider: target.provider, subscriptionId: target.subscriptionId,
        userDeclaredAvailable: activeSubscriptions.includes(target.subscriptionId)
      }))
        facts.models = current.escalationCatalog.map(({ target }) => ({
        provider: target.provider, model: target.model, subscriptionId: target.subscriptionId,
        strengths: target.strengths, effortLevels: target.effortLevels, evidence: target.evidence
        }))
      }
    }
    const timer = window.setTimeout(() => {
      void previewIntake(facts).then(value => {
        if (!stopped) { setPreview(value); setError(null) }
      }).catch((value: Error) => { if (!stopped) setError(value.message) })
    }, debounceMs)
    return () => { stopped = true; window.clearTimeout(timer) }
  }, [prompt, promptTooLarge, revision, classification, activeSubscriptions, debounceMs])

  return <section className="dev-panel" aria-label="Project intake">
    <div className="dev-heading"><div><p className="eyebrow">CONFIGURABLE QUESTIONS</p><h2>Project intake</h2></div></div>
    <p>Filled from this LFG session and its configured model catalog. Availability is a user declaration, not a verified subscription.</p>
    {error ? <p role="alert">{error}</p> : !preview ? <p>Preparing intake…</p> : <>
      <p>{preview.missingRequired} required questions need input · template {preview.templateRevision}</p>
      <dl className="intake-list">{preview.answers.map(answer => <div key={answer.id}>
        <dt>{answer.question}{answer.required ? ' · required' : ''}</dt>
        <dd>{answer.status === 'filled' ? <pre>{typeof answer.value === 'string' ? answer.value : JSON.stringify(answer.value, null, 2)}</pre> : 'Needs input'}</dd>
      </div>)}</dl>
    </>}
  </section>
}
