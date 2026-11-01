import { useRef, useState } from 'react'

type ReviewStatus = { reviewer: string; decisions: { pending: number; approved: number; rejected: number } }
type Connection = { origin: string; token: string }

function parseReviewUrl(value: string): Connection {
  const url = new URL(value.trim())
  if (url.protocol !== 'http:' || url.hostname !== '127.0.0.1' || url.pathname !== '/' || url.search ||
      url.username || url.password || !/^#[A-Za-z0-9_-]+$/.test(url.hash)) {
    throw new Error('Use the complete local review URL printed by the review server.')
  }
  return { origin: url.origin, token: url.hash.slice(1) }
}

export function SupervisionPanel() {
  const [url, setUrl] = useState('')
  const [connection, setConnection] = useState<Connection | null>(null)
  const [status, setStatus] = useState<ReviewStatus | null>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const revision = useRef(0)
  const localUi = window.location.protocol === 'http:' && ['127.0.0.1', 'localhost'].includes(window.location.hostname)

  async function refresh(next: Connection) {
    const current = ++revision.current
    setBusy(true); setError(''); setStatus(null)
    try {
      const response = await fetch(`${next.origin}/api/status`, {
        headers: { Authorization: `Bearer ${next.token}` }, cache: 'no-store', credentials: 'omit',
      })
      if (!response.ok) throw new Error(`Review server returned ${response.status}.`)
      const data: ReviewStatus = await response.json()
      if (!data.decisions || !['pending', 'approved', 'rejected'].every((key) =>
        Number.isSafeInteger(data.decisions[key as keyof ReviewStatus['decisions']]) && data.decisions[key as keyof ReviewStatus['decisions']] >= 0)) {
        throw new Error('Review server returned an invalid status.')
      }
      if (current === revision.current) { setConnection(next); setStatus(data); setUrl('') }
    } catch (value) {
      if (current === revision.current) {
        setConnection(null)
        setError(value instanceof Error ? value.message : 'Could not reach the local review server.')
      }
    } finally { if (current === revision.current) setBusy(false) }
  }

  function connect() {
    try { void refresh(parseReviewUrl(url)) } catch (value) { setError(value instanceof Error ? value.message : 'Invalid review URL.') }
  }
  function disconnect() { revision.current++; setConnection(null); setStatus(null); setUrl(''); setError(''); setBusy(false) }

  return <section className="dev-panel supervision-panel" aria-label="Supervised learning review">
    <div className="dev-heading"><div><p className="eyebrow">HUMAN SUPERVISION</p><h2>Review training examples</h2></div><span>Local review workspace</span></div>
    <p>Review real requests, check their sources, remove private details, and supply the labels LFG can learn from. Approval records a review; it does not train or qualify a model.</p>
    {!localUi ? <p>Open Inspector on this machine using a loopback HTTP address to connect to local review.</p> : !connection ? <>
      <label className="supervision-connect">Local review URL <input type="password" autoComplete="off" value={url} onChange={(event) => setUrl(event.target.value)} placeholder="Paste the URL printed by the review server" /></label>
      <button type="button" disabled={busy || !url} onClick={connect}>{busy ? 'Connecting…' : 'Connect review workspace'}</button>
      <p>Start the review server with this UI origin explicitly allowed: <code>{window.location.origin}</code>. The URL is held only in this browser session.</p>
    </> : <>
      {status && <div className="supervision-counts" aria-label="Review decisions">
        <div><strong>{status.decisions.pending}</strong><span>Awaiting review</span></div>
        <div><strong>{status.decisions.approved}</strong><span>Approved by a reviewer</span></div>
        <div><strong>{status.decisions.rejected}</strong><span>Rejected</span></div>
      </div>}
      <p>Reviewer: {status?.reviewer || 'unknown'} · Open the review workspace to verify sources, edit labels and assessment targets, redact, and approve or reject each example.</p>
      <div className="supervision-actions"><a href={`${connection.origin}/#${connection.token}`} target="_blank" rel="noopener noreferrer">Open review workspace</a>
        <button type="button" disabled={busy} onClick={() => void refresh(connection)}>Refresh counts</button>
        <button type="button" onClick={disconnect}>Disconnect</button></div>
      <p>Reviewed counts are not model accuracy, evidence of training, or proof that an example has assessment targets.</p>
    </>}
    {error && <p role="alert">Could not connect. {error} Check the local server, its allowed UI origin, and the complete URL.</p>}
  </section>
}
