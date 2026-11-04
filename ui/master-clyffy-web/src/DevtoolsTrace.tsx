import { useEffect, useMemo, useRef, useState } from 'react'
import { motion } from 'motion/react'

import { assess } from './api'
import { KIND_LABEL, biggestMisses } from './devtoolsData'
import type { Panel, PanelPrompt, PanelRow } from './devtoolsData'
import type { Assessment, FormulaTerm, RuntimeStatus } from './types'

const SESSION_ID = `devtools-${crypto.randomUUID()}`

const SUBSCRIPTION_LABEL: Record<string, string> = {
  'signed-in': 'signed in',
  recognized: 'found on this machine',
  declared: 'you said you have this',
  'signed-out': 'signed out',
  'not-found': 'not found'
}

export function ReadVsReal({ prompt, rows }: { prompt: PanelPrompt; rows: PanelRow[] }) {
  const ordered = [...rows].sort((left, right) => Math.abs((left.id in prompt.ranks ? prompt.ranks[left.id] ?? 2 : 2) - prompt.effortRank) - Math.abs((right.id in prompt.ranks ? prompt.ranks[right.id] ?? 2 : 2) - prompt.effortRank))
  return <div className="read-real" role="table" aria-label="Each predictor's read of this prompt against what really happened">
    <div className="read-real-axis" role="row" aria-hidden="true"><span /><span className="read-real-scale"><b>less work than most</b><b>more work than most</b></span></div>
    {ordered.map((row) => {
      const rank = prompt.ranks[row.id]
      const real = prompt.effortRank * 100
      const read = rank === null || rank === undefined ? null : rank * 100
      return <div className="read-real-row" role="row" key={row.id}>
        <span role="cell">{row.label}</span>
        <span className="read-real-track" role="cell" aria-label={read === null ? 'not scored' : `read ${Math.round(read)} of 100, really ${Math.round(real)}`}>
          <i className="read-real-line" style={{ left: `${real}%` }} />
          {read !== null && <>
            <i className="read-real-gap" style={{ left: `${Math.min(read, real)}%`, width: `${Math.abs(read - real)}%` }} />
            <i className={`read-real-dot kind-${row.kind}`} title={KIND_LABEL[row.kind]} style={{ left: `${read}%` }} />
          </>}
        </span>
      </div>
    })}
    <p className="read-real-note">The white line marks the work that really followed, compared with your other prompts in the same tool. Each dot is one predictor's read. A shorter gap is a better read.</p>
  </div>
}

function Waterfall({ terms, score }: { terms: FormulaTerm[]; score: number }) {
  const active = terms.filter((term) => term.points !== 0)
  const quiet = terms.length - active.length
  let running = 0
  const max = Math.max(1, ...active.map((term) => Math.abs(term.points)), Math.abs(score), active.reduce((sum, term) => sum + Math.max(term.points, 0), 0))
  return <div className="waterfall">
    {active.length === 0 && <p>No signals fired, so the score stays at {score}.</p>}
    {active.map((term) => {
      const start = running
      running += term.points
      return <div className="waterfall-row" key={term.id}>
        <span className="waterfall-label"><strong>{term.label}</strong>{term.detail.length > 0 && <small>{term.detail.slice(0, 8).join(', ')}{term.detail.length > 8 ? ' …' : ''}</small>}</span>
        <span className="waterfall-track" aria-hidden="true"><motion.i initial={false} animate={{ left: `${(Math.min(start, running) / max) * 100}%`, width: `${(Math.abs(term.points) / max) * 100}%` }} /></span>
        <span className="waterfall-value">{term.points > 0 ? '+' : ''}{term.points}<small>{term.value} × {term.weight}</small></span>
      </div>
    })}
    {active.length > 0 && <div className="waterfall-row is-total"><span className="waterfall-label"><strong>Complexity score</strong></span><span className="waterfall-track" aria-hidden="true"><motion.i initial={false} animate={{ left: 0, width: `${(Math.max(score, 0) / max) * 100}%` }} /></span><span className="waterfall-value">{score}</span></div>}
    {quiet > 0 && <p className="waterfall-quiet">{quiet} other {quiet === 1 ? 'signal' : 'signals'} found nothing in this prompt.</p>}
  </div>
}

function Stage({ number, title, plain, children, tone }: { number: number; title: string; plain?: string; children: React.ReactNode; tone?: string }) {
  return <motion.li className={`trace-stage ${tone ?? ''}`} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: number * .03, duration: .22 }}>
    <span className="trace-number" aria-hidden="true">{number}</span>
    <div className="trace-body"><h3>{title}</h3>{plain && <p className="trace-plain">{plain}</p>}{children}</div>
  </motion.li>
}

export function DevtoolsTrace({ status, panel, followed, onFollow }: { status: RuntimeStatus | null; panel: Panel | null; followed: PanelPrompt | null; onFollow: (prompt: PanelPrompt | null) => void }) {
  const [text, setText] = useState(followed?.text ?? '')
  const [assessment, setAssessment] = useState<Assessment | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [declared, setDeclared] = useState<string[]>([])
  const revision = useRef(0)
  const debounce = status?.operations.uiDebounceMs ?? 150
  const maxBytes = status?.operations.maxPromptBytes
  const bytes = useMemo(() => new TextEncoder().encode(text).length, [text])
  const tooLarge = maxBytes !== undefined && bytes > maxBytes

  useEffect(() => { if (followed) setText(followed.text) }, [followed])

  useEffect(() => {
    const current = ++revision.current
    if (!text.trim() || tooLarge) { setAssessment(null); setError(null); return }
    const timer = window.setTimeout(() => {
      void assess(SESSION_ID, current, text, declared).then((value) => {
        if (revision.current === current) { setAssessment(value); setError(null) }
      }).catch((value: Error) => { if (revision.current === current) setError(value.message) })
    }, debounce)
    return () => window.clearTimeout(timer)
  }, [text, debounce, tooLarge, declared])

  const examples = useMemo(() => {
    if (!panel) return []
    const usable = panel.prompts.filter((prompt) => !prompt.truncated)
    const short = biggestMisses(usable.filter((prompt) => prompt.chars < 40), 'rules', 1).underrated[0]
    const long = biggestMisses(usable.filter((prompt) => prompt.chars >= 80), 'rules', 1).underrated[0]
    const over = biggestMisses(usable.filter((prompt) => prompt.chars >= 40), 'rules', 1).overrated[0]
    return [
      short && { label: 'Short, but set off a lot of work', prompt: short },
      long && { label: 'Rules said small, it was big', prompt: long },
      over && { label: 'Rules said big, it was small', prompt: over }
    ].filter((item): item is { label: string; prompt: PanelPrompt } => Boolean(item))
  }, [panel])

  const logged = followed && followed.text === text ? followed : null
  return <div className="trace-layout">
    <section className="devtools-card trace-input">
      <label htmlFor="trace-text"><h2>Type anything, or pick one of your real prompts</h2></label>
      <textarea id="trace-text" rows={4} value={text} onChange={(event) => { setText(event.target.value); if (followed) onFollow(null) }} placeholder="Describe what you want to do" aria-invalid={tooLarge} />
      {examples.length > 0 && <div className="trace-examples" role="group" aria-label="Real prompts where the rules missed">
        {examples.map((example) => <button key={example.label} type="button" className={logged?.id === example.prompt.id ? 'is-active' : ''} onClick={() => onFollow(example.prompt)}>{example.label}</button>)}
      </div>}
      <p className="trace-meta">{assessment ? <>LFG took <strong>{(assessment.durationMicros / 1000).toFixed(2)} ms</strong> on this machine, using its rules. Nothing was sent to a provider.</> : 'LFG reads as you type. Nothing is sent to a provider.'}</p>
      {tooLarge && <p className="error-message" role="alert">This prompt is over the server's limit of {maxBytes?.toLocaleString()} bytes, so LFG can't read it.</p>}
      {error && <p className="error-message" role="alert">{error}</p>}
    </section>

    {assessment ? <ol className="trace-flow">
      <Stage number={1} title="Reads the prompt" plain={`${assessment.prompt.words} ${assessment.prompt.words === 1 ? 'word' : 'words'}, ${assessment.prompt.tasks.length} ${assessment.prompt.tasks.length === 1 ? 'task' : 'tasks'} found.`}>
        {assessment.prompt.tasks.length > 0 && <ul className="trace-chips">{assessment.prompt.tasks.slice(0, 8).map((task, index) => <li key={`${index}-${task}`}>{task}</li>)}</ul>}
        {assessment.prompt.scoringTruncated && <p role="status">Some text was past the scoring limit, so this read may be incomplete.</p>}
      </Stage>
      <Stage number={2} title="Counts signals" plain="Each signal adds points. This is the whole formula, not a summary.">
        <Waterfall terms={assessment.complexity.terms} score={assessment.complexity.score} />
      </Stage>
      <Stage number={3} title="Picks a bracket" plain={`${assessment.complexity.label}: ${assessment.complexity.plainLabel}.`}>
        <ol className="trace-ladder" aria-label="Complexity brackets">
          {assessment.complexity.ladder.map((step) => <li key={step.id} className={step.current ? 'is-current' : ''} aria-current={step.current ? 'step' : undefined}><strong>{step.label}</strong><small>from {step.minScore}</small></li>)}
        </ol>
      </Stage>
      <Stage number={4} title="Reads the intent" plain={assessment.intent.fallback ? `No clear signal, so it falls back to: ${assessment.intent.plainLabel}.` : undefined}>
        <ul className="trace-chips">
          {assessment.intent.all.map((hit) => <li key={hit.id} title={`Matched: ${hit.matched.join(', ')}`}>{hit.plainLabel}<small>{hit.matched.slice(0, 3).join(', ')}</small></li>)}
          {assessment.intent.risks.map((hit) => <li key={hit.id} className="is-risk" title={`Matched: ${hit.matched.join(', ')}`}><b aria-hidden="true">⚠</b> {hit.plainLabel}<small>{hit.matched.slice(0, 3).join(', ')}</small></li>)}
        </ul>
      </Stage>
      <Stage number={5} title="Suggests agents" plain={assessment.agents.plain}>
        <p className="trace-big">{assessment.agents.total} <small>{assessment.agents.total === 1 ? 'agent' : 'agents'}</small></p>
      </Stage>
      <Stage number={6} title="Suggests a model" plain={assessment.models.suggested ? assessment.models.plain : 'LFG suggests only from subscriptions you have. Mark yours to get a suggestion.'}>
        <div className="trace-subscriptions" role="group" aria-label="Your subscriptions">
          {assessment.subscriptions.map((subscription) => <button key={subscription.id} type="button" aria-pressed={declared.includes(subscription.id)} className={declared.includes(subscription.id) ? 'is-active' : ''} onClick={() => setDeclared((current) => current.includes(subscription.id) ? current.filter((id) => id !== subscription.id) : [...current, subscription.id])}>
            {subscription.label}<small>{SUBSCRIPTION_LABEL[subscription.status] ?? subscription.status}</small>
          </button>)}
        </div>
        {assessment.models.suggested && <div className="trace-model"><strong>{assessment.models.suggested.label}</strong>{assessment.models.suggested.effort && <small>{assessment.models.suggested.effort} effort</small>}<p>{assessment.models.suggested.why}</p></div>}
        {assessment.models.alternatives.length > 0 && <p className="trace-plain">{assessment.models.alternatives.length} other {assessment.models.alternatives.length === 1 ? 'option' : 'options'}: {assessment.models.alternatives.slice(0, 4).map((model) => model.label).join(', ')}.</p>}
      </Stage>
      {logged && panel ? <Stage number={7} tone="is-reality" title="What really happened" plain={`${logged.tools.toLocaleString()} tool ${logged.tools === 1 ? 'call' : 'calls'} and ${logged.subagents} ${logged.subagents === 1 ? 'subagent' : 'subagents'} followed, in ${logged.tool}. Rules said ${logged.bracket ?? 'nothing, the prompt was too long'}.`}>
        <ReadVsReal prompt={logged} rows={panel.rows} />
      </Stage> : <Stage number={7} tone="is-reality" title="What really happened" plain={panel ? 'Pick one of your real prompts above, or from Every prompt, to compare this read with the work that followed.' : 'Connect your scorecard results to compare a read with the work that followed.'}><></></Stage>}
    </ol> : <section className="devtools-card trace-empty"><p>Your prompt's journey appears here, step by step.</p></section>}
    <p className="assessment-note">This shows LFG's rules, since no owned model is connected. A suggestion from the rules is not a measured probability.</p>
  </div>
}
