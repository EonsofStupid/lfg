import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'

import { assess, classify, commitPlan, getGraph, getStatus } from './api'
import { layoutGraph } from './graph'
import { ParticleLayer } from './ParticleLayer'
import { AssessmentView } from './AssessmentView'
import { FeedbackPanel } from './FeedbackPanel'
import { OwnedAssessmentPanel } from './OwnedAssessmentPanel'
import type { OwnedResult } from './ownedAssessmentApi'
import { PromptBranchView } from './PromptBranchView'
import { DevPanel } from './DevPanel'
import { IntakePanel } from './IntakePanel'
import { BoundaryPanel } from './BoundaryPanel'
import { SupervisionPanel } from './SupervisionPanel'
import type { Assessment, ClassificationSnapshot, EscalationTarget, GraphNode, GraphSnapshot, PlanCommit, RoutingNeed, RuntimeStatus } from './types'

const SESSION_ID = `browser-${crypto.randomUUID()}`
const GraphCanvas = lazy(() => import('./GraphCanvas'))

type Mode = 'guided' | 'inspector'

export default function App() {
  const [status, setStatus] = useState<RuntimeStatus | null>(null)
  const [prompt, setPrompt] = useState('')
  const [revision, setRevision] = useState(1)
  const [assessment, setAssessment] = useState<Assessment | null>(null)
  const [assessmentSubscriptions, setAssessmentSubscriptions] = useState('')
  const [ownedSnapshot, setOwnedSnapshot] = useState<{ revision: number; result: OwnedResult } | null>(null)
  const [classification, setClassification] = useState<ClassificationSnapshot | null>(null)
  const [activeSubscriptions, setActiveSubscriptions] = useState<string[]>([])
  const [routingNeed, setRoutingNeed] = useState<RoutingNeed | null>(null)
  const [overrideTarget, setOverrideTarget] = useState<EscalationTarget | null>(null)
  const [commit, setCommit] = useState<PlanCommit | null>(null)
  const [graph, setGraph] = useState<GraphSnapshot | null>(null)
  const [flow, setFlow] = useState<Awaited<ReturnType<typeof layoutGraph>>>({ nodes: [], edges: [] })
  const [selected, setSelected] = useState<GraphNode | null>(null)
  const [mode, setMode] = useState<Mode>('guided')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [cursor, setCursor] = useState(0)
  const [debounceOverride, setDebounceOverride] = useState<number | null>(null)
  const [historyOverride, setHistoryOverride] = useState<number | null>(null)
  const requestRevision = useRef(0)
  const classifyRevision = useRef(0)
  const publicEmbed = new URLSearchParams(window.location.search).get('embed') === 'public'
  const promptBytes = useMemo(() => new TextEncoder().encode(prompt).length, [prompt])
  const promptTooLarge = status !== null && promptBytes > status.operations.maxPromptBytes
  const debounceMs = debounceOverride ?? status?.operations.uiDebounceMs
  const historyLimit = historyOverride ?? status?.operations.uiTimingHistorySamples

  useEffect(() => { void getStatus().then(setStatus).catch((value: Error) => setError(value.message)) }, [])

  useEffect(() => {
    const current = ++requestRevision.current
    const text = prompt
    if (!status || !debounceMs || !text.trim() || promptTooLarge) { setAssessment(null); return }
    const timer = window.setTimeout(() => {
      void assess(SESSION_ID, revision, text, activeSubscriptions).then((value) => {
        if (requestRevision.current === current) { setAssessment(value); setAssessmentSubscriptions(JSON.stringify(activeSubscriptions)) }
      }).catch((value: Error) => setError(value.message))
    }, debounceMs)
    return () => window.clearTimeout(timer)
  }, [prompt, revision, activeSubscriptions, status, promptTooLarge, debounceMs])

  useEffect(() => {
    const current = ++classifyRevision.current
    const text = prompt
    if (mode !== 'inspector' || !status || !debounceMs || !text.trim() || promptTooLarge) { setClassification(null); return }
    const timer = window.setTimeout(() => {
      void classify(SESSION_ID, revision, text, activeSubscriptions, routingNeed).then((value) => {
        if (classifyRevision.current === current) setClassification(value)
      }).catch((value: Error) => setError(value.message))
    }, debounceMs)
    return () => window.clearTimeout(timer)
  }, [mode, prompt, revision, activeSubscriptions, routingNeed, status, promptTooLarge, debounceMs])

  const toggleSubscription = useCallback((id: string) => {
    setActiveSubscriptions((current) => current.includes(id) ? current.filter((value) => value !== id) : [...current, id])
  }, [])

  useEffect(() => {
    if (!graph) { setFlow({ nodes: [], edges: [] }); return }
    void layoutGraph(graph).then(setFlow)
  }, [graph])

  useEffect(() => {
    if (!commit) return
    const source = new EventSource(`/api/shippin/intelligence/lfg/workflows/${encodeURIComponent(commit.workflowId)}/stream`)
    source.addEventListener('lfg.signal', () => { void getGraph(commit.workflowId).then((next) => { setGraph(next); setCursor(next.throughSequence) }) })
    return () => source.close()
  }, [commit])

  const handleCommit = useCallback(async () => {
    if (!status || !prompt.trim() || promptTooLarge || busy) return
    setBusy(true); setError(null); setSelected(null)
    try {
      const result = await commitPlan(SESSION_ID, revision, prompt)
      setCommit(result); setGraph(result.graph); setCursor(result.graph.throughSequence)
    } catch (value) {
      setError(value instanceof Error ? value.message : 'Plan compilation failed')
    } finally { setBusy(false) }
  }, [prompt, revision, busy, status, promptTooLarge])

  const handleReplay = useCallback(async (value: number) => {
    if (!commit) return
    setCursor(value)
    setGraph(await getGraph(commit.workflowId, value))
  }, [commit])

  const displayedTotals = graph?.totals ?? { predicted: assessment?.agents.total ?? 0, committed: 0, queued: 0, active: 0, blocked: 0, completed: 0, failed: 0 }
  const narrative = useMemo(() => guidedNarrative(assessment, graph), [assessment, graph])

  return (
    <div className={`lfg-shell ${publicEmbed ? 'is-embed' : ''}`}>
      <header className="masthead">
        <div className="brand-lockup"><span className="brand-mark">MC</span><div><strong>MasterClyffy</strong><span>Rust-authoritative LFG lab</span></div></div>
        {!publicEmbed && <div className="mode-switch" role="group" aria-label="View mode">
          <button className={mode === 'guided' ? 'is-active' : ''} onClick={() => setMode('guided')}>Guided</button>
          <button className={mode === 'inspector' ? 'is-active' : ''} onClick={() => setMode('inspector')}>Inspector</button>
        </div>}
        <span className="runtime-state"><i />{status?.phase ?? 'connecting'}</span>
      </header>

      <main>
        <section className="composer" aria-labelledby="prompt-label">
          <motion.div className="type-callout" animate={{ x: [0, 8, 0] }} transition={{ duration: 2.4, repeat: Infinity }}>
            <span>TYPE HERE</span><b>↘</b>
          </motion.div>
          <div className="composer-heading"><div><p className="eyebrow">PROMPT SIGNAL INTAKE</p><h1 id="prompt-label">Watch a team compile itself.</h1></div><span className="privacy-pill">MEMORY ONLY · CPU PREFLIGHT</span></div>
          <textarea value={prompt} rows={4} onChange={(event) => { setPrompt(event.target.value); setRevision((value) => value + 1) }} aria-label="Prompt to route" placeholder="Describe what you want to do" aria-invalid={promptTooLarge} />
          <div className="composer-footer">
            <button className="commit-button" onClick={handleCommit} disabled={busy || !status || !prompt.trim() || promptTooLarge}>{busy ? 'COMPILING…' : 'COMMIT ROUTE'}</button>
            <span>{promptBytes.toLocaleString()} / {status?.operations.maxPromptBytes.toLocaleString() ?? '…'} UTF-8 bytes</span>
            <span>{assessment ? `${(assessment.durationMicros / 1000).toFixed(3)} ms elapsed` : 'waiting for signal'}</span>
            <span>{status?.encoder.id ?? 'encoder unavailable'}</span>
            {mode === 'inspector' && <span>{status ? `${status.liveProviders} live · ${status.configuredProviders} ready to probe · ${status.fixtureProviders} fixtures` : 'provider inventory loading'}</span>}
          </div>
          {promptTooLarge && <p className="error-message" role="alert">Prompt exceeds the configured UTF-8 byte limit.</p>}
          {error && <p className="error-message" role="alert">{error}</p>}
        </section>

        <AssessmentView assessment={assessment} prompt={prompt} declared={activeSubscriptions} onToggleSubscription={toggleSubscription} />

        {!publicEmbed && mode === 'inspector' && <OwnedAssessmentPanel prompt={prompt} assessment={assessment?.revision === revision && assessmentSubscriptions === JSON.stringify(activeSubscriptions) ? assessment : null} onResult={(result) => setOwnedSnapshot(result ? { revision, result } : null)} />}
        {!publicEmbed && mode === 'inspector' && <FeedbackPanel prompt={prompt} assessmentRef={`${SESSION_ID}:${revision}`} ownedResult={ownedSnapshot?.revision === revision ? ownedSnapshot.result : null} assessment={assessment?.revision === revision && assessmentSubscriptions === JSON.stringify(activeSubscriptions) ? assessment : null} />}

        <section className="signal-strip" aria-label="Live routing tallies">
          <Metric label="Suggested agents" value={displayedTotals.predicted} detail="advisory while typing" />
          <Metric label="Committed" value={displayedTotals.committed} detail="from canonical events" />
          <Metric label="Complexity" value={assessment?.complexity.label ?? '—'} detail={assessment ? `score ${assessment.complexity.score}` : 'waiting for a request'} />
          <Metric label="Elapsed" value={assessment ? `${(assessment.durationMicros / 1000).toFixed(2)}ms` : '—'} detail="wall time, not CPU time" />
        </section>

        {(graph || mode === 'inspector') && <section className="graph-stage" aria-label={graph ? 'LFG committed routing graph' : 'LFG live category branches'}>
          {!graph && <p className="legacy-note">Older category classifier, being replaced by the assessment above. Its agent count can differ.</p>}
          {graph ? (
            <>
              <ParticleLayer pulse={graph.throughSequence + flow.nodes.length} />
              <Suspense fallback={<div className="empty-graph">Materializing committed signal graph…</div>}><GraphCanvas nodes={flow.nodes} edges={flow.edges} onSelect={setSelected} /></Suspense>
              {mode === 'guided' && <motion.aside key={narrative} className="guided-overlay" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}><span>WHAT JUST HAPPENED</span><p>{narrative}</p></motion.aside>}
            </>
          ) : (
            <PromptBranchView
              snapshot={classification}
              historyLimit={historyLimit}
              routingNeed={routingNeed}
              onRoutingNeedChange={setRoutingNeed}
              overrideTarget={overrideTarget}
              onOverride={setOverrideTarget}
            />
          )}
        </section>}

        {!publicEmbed && mode === 'inspector' && status && debounceMs && <IntakePanel
          prompt={prompt} promptTooLarge={promptTooLarge} revision={revision} classification={classification} activeSubscriptions={activeSubscriptions} debounceMs={debounceMs}
        />}
        {!publicEmbed && mode === 'inspector' && status && <BoundaryPanel maxFileBytes={status.operations.maxHttpBodyBytes} />}
        {!publicEmbed && mode === 'inspector' && <SupervisionPanel />}
        {!publicEmbed && mode === 'inspector' && status && debounceMs && historyLimit && <DevPanel
          status={status} debounceMs={debounceMs} historyLimit={historyLimit}
          onDebounce={setDebounceOverride} onHistory={setHistoryOverride}
          onReset={() => { setDebounceOverride(null); setHistoryOverride(null) }}
        />}

        {commit && <section className="replay-bar">
          <div><span>SIGNAL REPLAY</span><strong>{cursor + 1} / {commit.committedSignals}</strong></div>
          <input aria-label="Replay sequence" type="range" min="0" max={Math.max(commit.committedSignals - 1, 0)} value={cursor} onChange={(event) => void handleReplay(Number(event.target.value))} />
          <code>{commit.workflowId}</code>
        </section>}

        {!publicEmbed && <AnimatePresence>{selected && <Inspector node={selected} onClose={() => setSelected(null)} />}</AnimatePresence>}
      </main>
    </div>
  )
}

function Metric({ label, value, detail }: { label: string; value: string | number; detail: string }) {
  return <article><span>{label}</span><strong>{value}</strong><small>{detail}</small></article>
}

function Inspector({ node, onClose }: { node: GraphNode; onClose: () => void }) {
  return <motion.aside className="inspector" initial={{ opacity: 0, x: 40 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: 40 }}>
    <button onClick={onClose} aria-label="Close inspector">×</button><p className="eyebrow">GRAPH EVIDENCE</p><h2>{node.label}</h2><dl><div><dt>ID</dt><dd><code>{node.id}</code></dd></div><div><dt>State</dt><dd>{node.state}</dd></div><div><dt>Provenance</dt><dd>{node.provenance}</dd></div>{Object.entries(node.metadata).map(([key, value]) => <div key={key}><dt>{key}</dt><dd><code>{value}</code></dd></div>)}</dl>
  </motion.aside>
}

function guidedNarrative(assessment: Assessment | null, graph: GraphSnapshot | null): string {
  if (graph) return `The route is committed. ${graph.totals.committed} agent${graph.totals.committed === 1 ? '' : 's'} and every provider connection shown here were reconstructed from ${graph.throughSequence + 1} validated events.`
  if (assessment?.intent.risks.length) return 'A deterministic policy signal detected sensitive-data language. It remains advisory here, but the security gate is authoritative when the plan is committed.'
  if (assessment) return `LFG rates this ${assessment.complexity.label.toLowerCase()} and suggests ${assessment.agents.total} agent${assessment.agents.total === 1 ? '' : 's'}. Nothing has been committed or sent to a provider yet.`
  return 'Your prompt becomes CPU signals first. The graph will grow only when the Rust runtime can explain each branch.'
}
