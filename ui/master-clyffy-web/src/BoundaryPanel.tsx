import { useEffect, useRef, useState } from 'react'
import { getBoundaryCatalog } from './api'
import type { BoundaryCatalog } from './types'

type Evidence = { status: 'supplied' | 'missing' | 'not_required'; artifact: unknown; source: string | null; provenance: unknown }
type Imported = { sourceApp: string; revision: string; configSha256: string; entries: Map<string, Evidence> }

// The imported preview is operator-provided data, not an attestation of file contents.
export function readProjectPreview(raw: string, catalog: BoundaryCatalog): Imported {
  const value: unknown = JSON.parse(raw.replace(/^\uFEFF/, ''))
  if (!value || typeof value !== 'object') throw new Error('Invalid project preview')
  const top = value as Record<string, unknown>
  const preview = top.preview as Record<string, unknown> | null
  if (top.schemaVersion !== 'lfg.project-attunement-preview.v1' ||
      typeof top.configRevision !== 'string' || typeof top.configSha256 !== 'string' ||
      !preview || preview.schemaVersion !== 'lfg.boundary-attunement-preview.v1' ||
      preview.catalogDigest !== catalog.digest || typeof preview.sourceApp !== 'string' ||
      !Array.isArray(preview.questions) || !Array.isArray(top.sources)) throw new Error('Preview is invalid or uses a different question catalog')
  const questions = new Map(catalog.catalog.banks.flatMap(bank => bank.questions.map(question => [question.id, { bank: bank.id, question }] as const)))
  const entries = new Map<string, Evidence>()
  for (const item of preview.questions) {
    if (!item || typeof item !== 'object' || !item.question || typeof item.question.id !== 'string') throw new Error('Invalid question in project preview')
    const original = questions.get(item.question.id)
    if (!original || original.bank !== item.bankId || entries.has(item.question.id) ||
        !['supplied', 'missing', 'not_required'].includes(item.bindingStatus) ||
        (original.question.binding === null) !== (item.bindingStatus === 'not_required') ||
        (item.bindingStatus === 'supplied' && (item.artifact === null || item.artifact === undefined ||
          !['app_artifact', 'app_fact'].includes(item.artifactSource))) ||
        (item.bindingStatus !== 'supplied' && item.artifact != null)) throw new Error('Invalid question evidence in project preview')
    const provenance = top.sources.find((source: unknown) => source && typeof source === 'object' &&
      (source as Record<string, unknown>).binding === original.question.binding) ?? null
    entries.set(item.question.id, { status: item.bindingStatus, artifact: item.artifact, source: item.artifactSource ?? null, provenance })
  }
  return { sourceApp: preview.sourceApp, revision: top.configRevision, configSha256: top.configSha256, entries }
}

export function BoundaryPanel({ maxFileBytes }: { maxFileBytes: number }) {
  const [open, setOpen] = useState(false)
  const [catalog, setCatalog] = useState<BoundaryCatalog | null>(null)
  const [bankId, setBankId] = useState('')
  const [imported, setImported] = useState<Imported | null>(null)
  const [error, setError] = useState<string | null>(null)
  const fileRevision = useRef(0)
  useEffect(() => {
    if (!open) return
    let stopped = false
    void getBoundaryCatalog().then(value => {
      if (stopped) return
      if (value.catalog?.schemaVersion !== 'lfg.boundary-question-banks.v1' || !Array.isArray(value.catalog.banks) || !value.catalog.banks.length || typeof value.digest !== 'string') throw new Error('Incompatible question catalog')
      setCatalog(value); setBankId(value.catalog.banks[0].id); setImported(null); setError(null)
    }).catch((reason: Error) => { if (!stopped) setError(reason.message) })
    return () => { stopped = true; fileRevision.current++ }
  }, [open])
  const bank = catalog?.catalog.banks.find(item => item.id === bankId)
  const importFile = (file: File | undefined) => {
    const revision = ++fileRevision.current
    setImported(null); setError(null)
    if (!file || !catalog) return
    if (!Number.isSafeInteger(maxFileBytes) || maxFileBytes <= 0 || file.size > maxFileBytes) {
      setError('Preview exceeds the configured browser import limit or that limit is unavailable'); return
    }
    const reader = new FileReader()
    reader.onload = () => {
      if (revision !== fileRevision.current) return
      try { setImported(readProjectPreview(String(reader.result), catalog)) }
      catch (reason) { setError(reason instanceof Error ? reason.message : 'Could not read project preview') }
    }
    reader.onerror = () => { if (revision === fileRevision.current) setError('Could not read project preview') }
    reader.readAsText(file)
  }
  return <section className="dev-panel" aria-label="Authored boundary questions">
    <div className="dev-heading"><div><p className="eyebrow">OWNER-AUTHORED DRAFTS</p><h2>Boundary questions</h2></div>
      <button onClick={() => setOpen(!open)} aria-expanded={open}>{open ? 'Close questions' : 'Open questions'}</button></div>
    {open && <>
      <p>These are draft questions. Project artifacts provide context for a reviewer; they do not answer the questions.</p>
      {error && <p role="alert">{error}</p>}
      {!catalog ? <p>Loading question catalog…</p> : <>
        <p>{catalog.catalog.sourceNotice} Catalog {catalog.catalog.revision} · {catalog.digest}</p>
        <label>Question bank <select value={bankId} onChange={event => setBankId(event.target.value)}>
          {catalog.catalog.banks.map(item => <option key={item.id} value={item.id}>{item.title} ({item.questions.length})</option>)}
        </select></label>
        <p>Import a project preview generated by <code>boundary-project</code>. The file stays in this browser; imported evidence is not independently verified.</p>
        <label>Project preview JSON <input type="file" accept="application/json,.json" onChange={event => { const file = event.target.files?.[0]; event.target.value = ''; importFile(file) }} /></label>
        {imported && <p>Imported {imported.sourceApp} · configuration {imported.revision} · {imported.configSha256} <button onClick={() => { fileRevision.current++; setImported(null) }}>Clear evidence</button></p>}
        <dl className="intake-list">{bank?.questions.map(question => {
          const evidence = imported?.entries.get(question.id)
          return <div key={question.id}>
            <dt>{question.section}.{question.number} · {question.question}</dt>
            <dd><p>{question.sourceTerm}{question.binding ? ` · Binding: ${question.binding}` : ' · No artifact binding specified'}</p>
              {evidence?.status === 'supplied' ? <details><summary>Imported evidence ({evidence.source})</summary><pre>{typeof evidence.artifact === 'string' ? evidence.artifact : JSON.stringify(evidence.artifact, null, 2)}</pre>{evidence.provenance != null && <><p>Reported source</p><pre>{JSON.stringify(evidence.provenance, null, 2)}</pre></>}</details>
                : evidence?.status === 'missing' ? <p>Required artifact missing from preview</p>
                  : imported && !evidence ? <p>Not included in this project preview</p> : null}
            </dd>
          </div>
        })}</dl>
      </>}
    </>}
  </section>
}
