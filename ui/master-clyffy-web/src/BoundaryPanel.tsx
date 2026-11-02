import { useEffect, useRef, useState } from 'react'
import { getBoundaryCatalog } from './api'
import type { BoundaryCatalog } from './types'
import { getProfile, saveProfile } from './projectProfiles'
import type { DraftAnswer, ProfileRevision } from './projectProfiles'

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
  const [previewText, setPreviewText] = useState('')
  const [answers, setAnswers] = useState<Record<string, DraftAnswer>>({})
  const [projectId, setProjectId] = useState('')
  const [revisions, setRevisions] = useState<ProfileRevision[]>([])
  const [selectedRevision, setSelectedRevision] = useState(0)
  const [restoredFrom, setRestoredFrom] = useState<number | null>(null)
  const [notice, setNotice] = useState('')
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
    setImported(null); setPreviewText(''); setAnswers({}); setError(null)
    if (!file || !catalog) return
    if (!Number.isSafeInteger(maxFileBytes) || maxFileBytes <= 0 || file.size > maxFileBytes) {
      setError('Preview exceeds the configured browser import limit or that limit is unavailable'); return
    }
    const reader = new FileReader()
    reader.onload = () => {
      if (revision !== fileRevision.current) return
      try {
        const raw = String(reader.result)
        const next = readProjectPreview(raw, catalog)
        const proposed = Object.fromEntries([...next.entries].filter(([, item]) => item.status === 'supplied').map(([id, item]) =>
          [id, { value: typeof item.artifact === 'string' ? item.artifact : JSON.stringify(item.artifact, null, 2), status: 'proposed', origin: 'app_evidence' }] as const))
        setImported(next); setPreviewText(raw); setAnswers(proposed); setNotice('App evidence is a proposal until you review it.')
      }
      catch (reason) { setError(reason instanceof Error ? reason.message : 'Could not read project preview') }
    }
    reader.onerror = () => { if (revision === fileRevision.current) setError('Could not read project preview') }
    reader.readAsText(file)
  }
  const loadRevision = (version: ProfileRevision, head: ProfileRevision[]) => {
    if (!catalog || version.catalogDigest !== catalog.digest) throw new Error('Saved profile uses a different question catalog')
    const validIds = new Set(catalog.catalog.banks.flatMap(item => item.questions.map(question => question.id)))
    if (Object.keys(version.answers).some(id => !validIds.has(id))) throw new Error('Saved profile contains an unknown question')
    setImported(readProjectPreview(version.preview, catalog)); setPreviewText(version.preview); setAnswers(version.answers)
    setRevisions(head); setSelectedRevision(version.revision); setRestoredFrom(version.revision === head.length ? null : version.revision)
    setNotice(version.revision === head.length ? 'Loaded the latest saved revision.' : 'Loaded an earlier revision. Save it to create a new revision; history remains intact.')
  }
  const loadProfile = () => {
    try {
      const found = getProfile(window.localStorage, projectId, maxFileBytes)
      if (!found.length) throw new Error('No saved profile for that project ID')
      loadRevision(found[found.length - 1], found); setError(null)
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'Could not load profile') }
  }
  const saveCurrentProfile = () => {
    if (!catalog || !imported || !previewText) return
    try {
      const next = saveProfile(window.localStorage, projectId, catalog.digest, previewText, answers, revisions.length, restoredFrom, maxFileBytes)
      setRevisions(next); setSelectedRevision(next.length); setRestoredFrom(null); setError(null); setNotice(`Saved local revision ${next.length}.`)
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'Could not save profile') }
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
        {imported && <p>Imported {imported.sourceApp} · configuration {imported.revision} · {imported.configSha256} <button onClick={() => { fileRevision.current++; setImported(null); setPreviewText(''); setAnswers({}) }}>Clear evidence</button></p>}
        <div className="profile-controls">
          <label>Project ID <input value={projectId} onChange={event => { setProjectId(event.target.value); setRevisions([]); setSelectedRevision(0); setRestoredFrom(null); setNotice('') }} /></label>
          <button type="button" disabled={!projectId.trim()} onClick={loadProfile}>Load local profile</button>
          <button type="button" disabled={!projectId.trim() || !imported} onClick={saveCurrentProfile}>Save local profile</button>
          {revisions.length > 0 && <><label>Saved revision <select value={selectedRevision} onChange={event => setSelectedRevision(Number(event.target.value))}>
            {revisions.map(version => <option key={version.revision} value={version.revision}>{version.revision} · {version.savedAt}</option>)}
          </select></label><button type="button" onClick={() => { try { loadRevision(revisions[selectedRevision - 1], revisions); setError(null) } catch (reason) { setError(reason instanceof Error ? reason.message : 'Could not load revision') } }}>Load selected revision</button></>}
        </div>
        <p>Saving is explicit and stores the imported evidence and answers as plain text in this browser. Do not save secrets; anyone with access to this browser profile may be able to read them. Saving an older revision creates a new revision.</p>
        {notice && <p role="status">{notice}</p>}
        <dl className="intake-list">{bank?.questions.map(question => {
          const evidence = imported?.entries.get(question.id)
          const answer = Object.hasOwn(answers, question.id) ? answers[question.id] : undefined
          return <div key={question.id}>
            <dt>{question.section}.{question.number} · {question.question}</dt>
            <dd><p>{question.sourceTerm}{question.binding ? ` · Binding: ${question.binding}` : ' · No artifact binding specified'}</p>
              {evidence?.status === 'supplied' ? <details><summary>Imported evidence ({evidence.source})</summary><pre>{typeof evidence.artifact === 'string' ? evidence.artifact : JSON.stringify(evidence.artifact, null, 2)}</pre>{evidence.provenance != null && <><p>Reported source</p><pre>{JSON.stringify(evidence.provenance, null, 2)}</pre></>}</details>
                : evidence?.status === 'missing' ? <p>Required artifact missing from preview</p>
                  : imported && !evidence ? <p>Not included in this project preview</p> : null}
              {imported && <><label>Answer or proposed evidence <textarea aria-label={`Answer ${question.id}`} value={answer?.value ?? ''} onChange={event => setAnswers(current => ({ ...current, [question.id]: { value: event.target.value, status: 'proposed', origin: 'manual' } }))} /></label>
                <label><input type="checkbox" checked={answer?.status === 'reviewed'} disabled={!answer?.value.trim()} onChange={event => setAnswers(current => ({ ...current, [question.id]: { ...current[question.id], status: event.target.checked ? 'reviewed' : 'proposed' } }))} /> Reviewed by this user</label>
                {answer && <p>{answer.status === 'reviewed' ? 'Reviewed locally' : 'Proposed; review required'} · {answer.origin === 'app_evidence' ? 'from app evidence' : 'manual entry'}</p>}</>}
            </dd>
          </div>
        })}</dl>
      </>}
    </>}
  </section>
}
