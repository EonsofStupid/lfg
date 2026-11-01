import { useEffect, useState } from 'react'
import { getDevStats, getKnowledge } from './api'
import type { DevStats, Knowledge, RuntimeStatus } from './types'

const formatMicros = (value: number | null) => value === null ? '—' : `${(value / 1000).toFixed(3)} ms`

export function DevPanel({ status, debounceMs, historyLimit, onDebounce, onHistory, onReset }: {
  status: RuntimeStatus
  debounceMs: number
  historyLimit: number
  onDebounce: (value: number) => void
  onHistory: (value: number) => void
  onReset: () => void
}) {
  const settings = status.operations.devTools
  const [stats, setStats] = useState<DevStats | null>(null)
  const [knowledge, setKnowledge] = useState<Knowledge | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!settings?.statsEnabled) { setStats(null); return }
    let stopped = false
    let timer: number | undefined
    const refresh = async () => {
      try {
        const [snapshot, learned] = await Promise.all([getDevStats(), getKnowledge()])
        if (!stopped) { setStats(snapshot); setKnowledge(learned); setError(null) }
      } catch (value) {
        if (!stopped) setError(value instanceof Error ? value.message : 'Could not load development statistics')
      } finally {
        if (!stopped) timer = window.setTimeout(() => void refresh(), settings.statsPollIntervalMs)
      }
    }
    void refresh()
    return () => { stopped = true; if (timer !== undefined) window.clearTimeout(timer) }
  }, [settings?.statsEnabled, settings?.statsPollIntervalMs])

  if (!settings) return <section className="dev-panel"><h2>Developer tools</h2><p>Developer tools are not configured for this server.</p></section>
  return <section className="dev-panel" aria-label="Developer statistics and tuning">
    <div className="dev-heading"><div><p className="eyebrow">PROCESS-LOCAL OBSERVABILITY</p><h2>Developer statistics</h2></div><span>Config {status.operations.revision} · {status.operationsDigest.slice(0, 12)}</span></div>
    <p>Elapsed handler time, not CPU time. Traces contain timings, outcomes and the IDs of the rules that fired, not prompts. They reset when the server restarts.</p>
    <div className="dev-controls">
      <label>Browser debounce <strong>{debounceMs} ms</strong><input aria-label="Browser debounce" type="range" min={settings.debounceMinMs} max={settings.debounceMaxMs} value={debounceMs} onChange={(event) => onDebounce(Number(event.target.value))} /></label>
      <label>Displayed history <strong>{historyLimit}</strong><input aria-label="Displayed history" type="range" min={settings.historyMinSamples} max={settings.historyMaxSamples} value={historyLimit} onChange={(event) => onHistory(Number(event.target.value))} /></label>
      <button type="button" onClick={onReset}>Use server defaults</button>
    </div>
    <p>These controls affect this browser session only; server configuration is unchanged.</p>
    {!settings.statsEnabled ? <p>Statistics collection is disabled in server configuration.</p> : error ? <p role="alert">{error}</p> : !stats ? <p>Loading statistics…</p> : <>
      <p>{stats.totalRecorded} traces recorded · {stats.traces.length} retained of {stats.capacity} · {stats.evicted} evicted · CPU time unavailable</p>
      <p>Encoder {stats.encoderDigest.slice(0, 12)} · classifier {stats.classifierDigest.slice(0, 12)} · operations {stats.operationsDigest.slice(0, 12)}{stats.assessmentDigest ? ` · assessment ${stats.assessmentDigest.slice(0, 12)}` : ''}</p>
      <div className="dev-table-wrap"><table><thead><tr><th>Operation</th><th>Samples</th><th>Errors</th><th>p50 elapsed</th><th>p95 elapsed</th></tr></thead><tbody>
        {stats.summaries.map((row) => <tr key={row.operation}><td>{row.operation}</td><td>{row.samples}</td><td>{row.errors}</td><td>{formatMicros(row.p50TotalMicros)}</td><td>{formatMicros(row.p95TotalMicros)}</td></tr>)}
      </tbody></table></div>
      <p>Percentiles use the retained samples shown above, including errors; small sample counts are not performance qualification.</p>
      <div className="dev-table-wrap"><table><thead><tr><th>Trace</th><th>Operation</th><th>Outcome</th><th>Elapsed</th><th>Queue</th><th>Work</th><th>Rules fired</th></tr></thead><tbody>
        {stats.traces.slice(-historyLimit).reverse().map((trace) => <tr key={trace.sequence}><td>{trace.sequence}</td><td>{trace.operation}</td><td>{trace.outcome}</td><td>{formatMicros(trace.totalMicros)}</td><td>{formatMicros(trace.queueMicros)}</td><td>{formatMicros(trace.workMicros)}</td><td className="dev-rules">{trace.rules?.join(' · ') ?? ''}</td></tr>)}
      </tbody></table></div>
      {knowledge && <>
        <h3>What LFG has learned</h3>
        <p>{knowledge.state === 'ready' ? `${knowledge.promptsSeen.toLocaleString()} requests · updated ${knowledge.refreshedAt ?? 'unknown'}` : `Learning is ${knowledge.state}.`} {knowledge.sources.map((source) => `${source.id}: ${source.files} files, ${source.prompts} requests`).join(' · ')}</p>
        <div className="dev-table-wrap"><table><thead><tr><th>Model</th><th>Subscription</th><th>Role</th><th>Source</th><th>Offered</th><th>Requests by bracket</th><th>Last used</th></tr></thead><tbody>
          {knowledge.models.map((model) => <tr key={`${model.subscriptionId}/${model.model}`}><td>{model.model}</td><td>{model.subscriptionId}</td><td>{model.role ?? 'unknown'}</td><td>{model.origin}</td><td>{model.available ? 'yes' : 'no'}</td><td>{model.usage.map((usage) => `${usage.bracket} ${usage.prompts}`).join(' · ') || 'none'}</td><td>{model.lastUsed ?? 'never'}</td></tr>)}
        </tbody></table></div>
      </>}
    </>}
  </section>
}
