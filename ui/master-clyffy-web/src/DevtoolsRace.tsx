import { useMemo, useState } from 'react'
import { motion } from 'motion/react'

import { KIND_LABEL, MEASURES, SPLITS, barToBeat, formatRange, formatScore, formatSize, rankedRows, scoreOf, verdict } from './devtoolsData'
import type { MeasureId, Panel, PanelRow, RowKind, SplitId } from './devtoolsData'

const AXIS_MIN = -0.5
const AXIS_MAX = 0.7
const position = (value: number) => `${((value - AXIS_MIN) / (AXIS_MAX - AXIS_MIN)) * 100}%`
const TICKS = [-0.4, -0.2, 0, 0.2, 0.4, 0.6]
const KINDS: RowKind[] = ['lfg', 'trained', 'opponent', 'baseline']

function Legend() {
  return <ul className="viz-legend" aria-label="What the colors mean">
    {KINDS.map((kind) => <li key={kind}><i className={`viz-mark kind-${kind}`} aria-hidden="true" />{KIND_LABEL[kind]}</li>)}
  </ul>
}

function ForestPlot({ panel, split, measure, selected, onSelect }: { panel: Panel; split: SplitId; measure: MeasureId; selected: string; onSelect: (id: string) => void }) {
  const rows = useMemo(() => rankedRows(panel.rows, split, measure), [panel.rows, split, measure])
  const leader = barToBeat(panel.rows, split, measure)
  const bar = scoreOf(leader, split, measure)?.value ?? 0
  return <div className="forest" role="table" aria-label="Score for each predictor, best first">
    <div className="forest-row forest-axis" role="row" aria-hidden="true">
      <span />
      <div className="forest-plot">
        {TICKS.map((tick) => <span key={tick} className={tick === 0 ? 'forest-tick is-zero' : 'forest-tick'} style={{ left: position(tick) }}><b>{formatScore(tick)}</b></span>)}
      </div>
      <span />
    </div>
    {rows.map((row) => {
      const range = scoreOf(row, split, measure)
      return <motion.button layout="position" transition={{ type: 'spring', stiffness: 260, damping: 30 }} key={row.id} type="button" role="row" className={`forest-row ${selected === row.id ? 'is-selected' : ''}`} onClick={() => onSelect(row.id)} aria-pressed={selected === row.id}>
        <span className="forest-label" role="cell"><strong>{row.label}</strong><small>{row.sizeNote}</small></span>
        <span className="forest-plot" role="cell">
          {TICKS.map((tick) => <i key={tick} className={tick === 0 ? 'forest-grid is-zero' : 'forest-grid'} style={{ left: position(tick) }} aria-hidden="true" />)}
          <i className="forest-bar-line" style={{ left: position(bar) }} aria-hidden="true" />
          {range?.value !== null && range?.value !== undefined && range.low !== null && range.high !== null && <>
            <motion.i className={`forest-range kind-${row.kind}`} initial={false} animate={{ left: position(range.low), width: `${((range.high - range.low) / (AXIS_MAX - AXIS_MIN)) * 100}%` }} aria-hidden="true" />
            <motion.i className={`forest-dot kind-${row.kind}`} initial={false} animate={{ left: position(range.value) }} aria-hidden="true" />
          </>}
        </span>
        <span className="forest-value" role="cell"><strong>{formatScore(range?.value)}</strong><small>{formatRange(range) || 'not applicable'}</small></span>
      </motion.button>
    })}
    <div className="forest-row is-target" role="row">
      <span className="forest-label" role="cell"><strong>{panel.target.label}</strong><small>{formatSize(panel.target.sizeBytes)} budget</small></span>
      <span className="forest-plot" role="cell">
        {TICKS.map((tick) => <i key={tick} className={tick === 0 ? 'forest-grid is-zero' : 'forest-grid'} style={{ left: position(tick) }} aria-hidden="true" />)}
        <i className="forest-bar-line" style={{ left: position(bar) }} aria-hidden="true" />
        <span className="forest-goal" style={{ left: position(bar) }}>has to land here</span>
      </span>
      <span className="forest-value" role="cell"><strong>not scored</strong><small>beat {formatScore(bar)}</small></span>
    </div>
  </div>
}

function SizeChart({ panel, split, measure, onSelect, selected }: { panel: Panel; split: SplitId; measure: MeasureId; onSelect: (id: string) => void; selected: string }) {
  const width = 900, height = 340, left = 56, right = 24, top = 22, bottom = 44, noWeights = 92, logStart = 240
  const xLog = (bytes: number) => logStart + ((Math.log10(bytes) - 5) / 6) * (width - right - logStart - 30)
  const y = (value: number) => top + ((0.6 - Math.max(value, -0.3)) / 0.9) * (height - top - bottom)
  const leader = barToBeat(panel.rows, split, measure)
  const bar = scoreOf(leader, split, measure)?.value ?? 0
  const budget = xLog(panel.target.sizeBytes)
  const ticks: Array<[number, string]> = [[1e5, '100 KB'], [1e6, '1 MB'], [1e7, '10 MB'], [1e8, '100 MB'], [1e9, '1 GB'], [1e10, '10 GB'], [1e11, '100 GB']]
  const points = panel.rows.flatMap((row) => {
    const value = scoreOf(row, split, measure)?.value
    return value === null || value === undefined ? [] : [{ row, value, x: row.sizeBytes ? xLog(row.sizeBytes) : noWeights, y: y(value) }]
  })
  // Each label takes the nearest free spot beside its dot, trying right and left before moving up or down.
  type Box = { left: number; right: number; top: number; bottom: number }
  const taken: Box[] = [
    ...points.map((point) => ({ left: point.x - 9, right: point.x + 9, top: point.y - 9, bottom: point.y + 9 })),
    { left: width - right - 125, right: width - right, top: y(bar) - 18, bottom: y(bar) - 3 }
  ]
  const labels = new Map<string, { x: number; y: number; anchor: 'start' | 'end'; moved: boolean }>()
  for (const point of [...points].sort((a, b) => a.y - b.y)) {
    const span = point.row.label.length * 7
    const clear = (box: Box) => box.left >= left && box.right <= width - right && !taken.some((item) => item.left < box.right && box.left < item.right && item.top < box.bottom && box.top < item.bottom)
    search: for (const shift of [0, 16, -16, 32, -32, 48, -48, 64, -64]) {
      for (const anchor of ['start', 'end'] as const) {
        const from = anchor === 'start' ? point.x + 12 : point.x - 12 - span
        const box = { left: from, right: from + span, top: point.y + shift - 8, bottom: point.y + shift + 7 }
        if (clear(box)) {
          taken.push(box)
          labels.set(point.row.id, { x: anchor === 'start' ? from : from + span, y: point.y + shift + 4, anchor, moved: shift !== 0 })
          break search
        }
      }
    }
  }
  return <svg className="size-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`Size against score. The bar to beat is ${formatScore(bar)} by ${leader.label}, and LFG's budget is ${formatSize(panel.target.sizeBytes)}.`}>
    <rect className="size-zone" x={left} y={top} width={Math.max(0, budget - left)} height={Math.max(0, y(bar) - top)} />
    <text className="size-zone-label" x={left + 8} y={top + 15}>Where LFG needs to land: this size or smaller, this score or better</text>
    {[-0.2, 0, 0.2, 0.4, 0.6].map((tick) => <g key={tick}>
      <line className={tick === 0 ? 'size-grid is-zero' : 'size-grid'} x1={left} x2={width - right} y1={y(tick)} y2={y(tick)} />
      <text className="size-axis" x={left - 8} y={y(tick) + 4} textAnchor="end">{formatScore(tick)}</text>
    </g>)}
    <text className="size-axis" x={noWeights} y={height - bottom + 18} textAnchor="middle">no weights</text>
    {ticks.map(([bytes, label]) => <text key={label} className="size-axis" x={xLog(bytes)} y={height - bottom + 18} textAnchor="middle">{label}</text>)}
    <text className="size-axis" x={(left + width - right) / 2} y={height - 6} textAnchor="middle">Size on disk, each step is 10 times larger</text>
    <line className="size-budget" x1={budget} x2={budget} y1={top} y2={height - bottom} />
    <text className="size-budget-label" x={budget - 6} y={height - bottom - 8} textAnchor="end">650 MB budget</text>
    <line className="size-bar" x1={left} x2={width - right} y1={y(bar)} y2={y(bar)} />
    <text className="size-budget-label" x={width - right} y={y(bar) - 7} textAnchor="end">bar to beat {formatScore(bar)}</text>
    {points.map(({ row, value, x, y: dotY }) => <g key={row.id} className={selected === row.id ? 'size-point is-selected' : 'size-point'} tabIndex={0} role="button" aria-label={`${row.label}, ${row.sizeNote}, ${formatScore(value)}`} onClick={() => onSelect(row.id)} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onSelect(row.id) } }}>
      <circle className="size-hit" cx={x} cy={dotY} r="16" />
      {labels.get(row.id)?.moved && <line className="size-leader" x1={x} y1={dotY} x2={labels.get(row.id)?.x} y2={(labels.get(row.id)?.y ?? dotY) - 4} />}
      <circle className={`size-dot kind-${row.kind}`} cx={x} cy={dotY} r="7" />
      <text className="size-label" x={labels.get(row.id)?.x ?? x + 12} y={labels.get(row.id)?.y ?? dotY + 4} textAnchor={labels.get(row.id)?.anchor}>{row.label}</text>
    </g>)}
  </svg>
}

function RowDetail({ row, panel, split, measure }: { row: PanelRow; panel: Panel; split: SplitId; measure: MeasureId }) {
  const entry = row.splits[split]
  const others = panel.rows.filter((other) => other.id !== row.id)
  const label = { ahead: 'clearly ahead', behind: 'clearly behind', unclear: 'not established' }
  const headline = scoreOf(row, split, measure)
  return <aside className="devtools-card race-detail" aria-label={`${row.label} details`}>
    <p className="eyebrow">{KIND_LABEL[row.kind].toUpperCase()}</p>
    <h3>{row.label}</h3>
    <p>{row.plain}</p>
    <dl className="fact-grid">
      <div><dt>This score</dt><dd>{formatScore(headline?.value)}</dd></div>
      <div><dt>Size</dt><dd>{row.sizeNote}</dd></div>
      <div><dt>Time per prompt</dt><dd>{row.latency ? `${row.latency.p50Ms.toFixed(1)} ms typical` : 'not measured here'}</dd></div>
      <div><dt>Prompts scored</dt><dd>{entry.scored.toLocaleString()}</dd></div>
    </dl>
    <p className="race-output"><strong>What it outputs:</strong> {row.output}{row.license ? ` License as read: ${row.license}.` : ''} {row.source && <a href={row.source} target="_blank" rel="noreferrer">Source</a>}</p>
    <h4>Head to head, same prompts</h4>
    <ul className="paired-list">
      {others.map((other) => {
        const difference = entry.paired[other.id]
        const call = verdict(difference)
        return <li key={other.id}>
          <span>{other.label}</span>
          <strong>{formatScore(difference?.value)}</strong>
          <small className={`call call-${call}`}><b aria-hidden="true">{call === 'ahead' ? '▲' : call === 'behind' ? '▼' : '•'}</b> {label[call]}</small>
        </li>
      })}
    </ul>
  </aside>
}

export function DevtoolsRace({ panel }: { panel: Panel }) {
  const [split, setSplit] = useState<SplitId>('long')
  const [measure, setMeasure] = useState<MeasureId>('overall')
  const [selected, setSelected] = useState('rules')
  const row = panel.rows.find((item) => item.id === selected) ?? panel.rows[0]
  const held = panel.predictions.filter((prediction) => prediction.held).length
  const auditsPassed = panel.audits.every((audit) => audit.passed)
  return <div className="devtools-stack">
    <section className="bar-tiles" aria-label="The bar to beat for each kind of prompt">
      {SPLITS.map((item) => {
        const leader = barToBeat(panel.rows, item.id, 'overall')
        const rules = panel.rows.find((candidate) => candidate.id === 'rules')
        return <button key={item.id} type="button" aria-pressed={split === item.id} className={split === item.id ? 'is-active' : ''} onClick={() => setSplit(item.id)}>
          <span>{item.label}</span>
          <strong>{formatScore(scoreOf(leader, item.id, 'overall')?.value)}</strong>
          <small>Best: {leader.label}</small>
          <small>LFG rules: {formatScore(rules ? scoreOf(rules, item.id, 'overall')?.value : null)}</small>
        </button>
      })}
      <article className="is-target-tile">
        <span>{panel.target.label}</span>
        <strong>not scored</strong>
        <small>Has to beat every bar on the left</small>
        <small>at {formatSize(panel.target.sizeBytes)} or smaller</small>
      </article>
    </section>

    <section className="devtools-card">
      <div className="card-heading">
        <div><h2>Who predicts the real work best?</h2><p>Higher is better. 0 means no relationship with the work that really followed, and 1 means perfect ordering.</p></div>
        <div className="segmented" role="group" aria-label="How to measure">
          {MEASURES.map((item) => <button key={item.id} type="button" aria-pressed={measure === item.id} className={measure === item.id ? 'is-active' : ''} onClick={() => setMeasure(item.id)}>{item.label}</button>)}
        </div>
      </div>
      <p className="measure-note">{MEASURES.find((item) => item.id === measure)?.plain} Dots further right are better. The line through each dot is its likely range. Select a row for head-to-head results.</p>
      <Legend />
      <div className="race-layout">
        <ForestPlot panel={panel} split={split} measure={measure} selected={selected} onSelect={setSelected} />
        <RowDetail row={row} panel={panel} split={split} measure={measure} />
      </div>
    </section>

    <section className="devtools-card">
      <div className="card-heading"><div><h2>Does size buy skill?</h2><p>Every predictor by size and score, on {SPLITS.find((item) => item.id === split)?.label.toLowerCase()}. Your 650 MB model has to reach the shaded corner.</p></div></div>
      <div className="chart-scroll"><SizeChart panel={panel} split={split} measure={measure} onSelect={setSelected} selected={selected} /></div>
    </section>

    <section className="verdict-grid">
      <article className="devtools-card verdict-lead">
        <p className="eyebrow">WHAT THIS MEANS</p>
        <h2>{panel.decision.title}</h2>
        <p>{panel.decision.detail}</p>
        <p className="limits"><strong>Limits:</strong> {panel.limits}</p>
      </article>
      <article className="devtools-card">
        <h3>Do short prompts mean less work?</h3>
        <p>No. Median tool calls that really followed:</p>
        <table className="plain-table"><thead><tr><th>Tool</th><th>Under 40</th><th>40+</th></tr></thead><tbody>
          {panel.work.map((item) => <tr key={item.tool}><td>{item.tool}</td><td>{item.short}</td><td>{item.long}</td></tr>)}
        </tbody></table>
      </article>
      <article className="devtools-card">
        <h3>Written before the run: {held} of {panel.predictions.length} held</h3>
        <ul className="check-list">
          {panel.predictions.map((prediction) => <li key={prediction.text} className={prediction.held ? 'is-good' : 'is-miss'}>
            <b aria-hidden="true">{prediction.held ? '✓' : '✗'}</b><span><strong>{prediction.held ? 'Held' : 'Wrong'}:</strong> {prediction.text} <small>{prediction.evidence}</small></span>
          </li>)}
        </ul>
      </article>
      <article className="devtools-card">
        <h3>Checks on the checker: {auditsPassed ? 'all passed' : 'one failed'}</h3>
        <ul className="check-list">
          {panel.audits.map((audit) => <li key={audit.id} className={audit.passed ? 'is-good' : 'is-miss'}>
            <b aria-hidden="true">{audit.passed ? '✓' : '✗'}</b><span>{audit.label} <small>{audit.low === null ? `largest gap ${audit.value?.toFixed(4)}` : `${formatScore(audit.value)} (${formatRange(audit)})`}</small></span>
          </li>)}
        </ul>
      </article>
    </section>

    <details className="devtools-card table-view">
      <summary>All the numbers as a table</summary>
      <p className="limits">{panel.metric}</p>
      <div className="dev-table-wrap"><table className="plain-table"><thead><tr><th>Predictor</th><th>Size</th>{SPLITS.map((item) => <th key={item.id}>{item.label}</th>)}<th>Inside one session, 40+</th><th>Length fixed, 40+</th><th>Typical time</th></tr></thead><tbody>
        {panel.rows.map((item) => <tr key={item.id}>
          <td>{item.label}</td><td>{item.sizeNote}</td>
          {SPLITS.map((option) => <td key={option.id}>{formatScore(item.splits[option.id].overall.value)} <small>{formatRange(item.splits[option.id].overall)}</small></td>)}
          <td>{formatScore(item.splits.long.session.value)}</td><td>{formatScore(item.splits.long.lengthFixed?.value)}</td>
          <td>{item.latency ? `${item.latency.p50Ms.toFixed(1)} ms` : 'not measured'}</td>
        </tr>)}
        {panel.secondary.map((item) => <tr key={item.label}><td>{item.label} <small>secondary, not used for decisions</small></td><td />{SPLITS.map((option) => <td key={option.id}>{formatScore(item[option.id].value)}</td>)}<td /><td /><td /></tr>)}
      </tbody></table></div>
    </details>
  </div>
}
