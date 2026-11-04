import { useMemo, useState } from 'react'

import { ReadVsReal } from './DevtoolsTrace'
import { SPLITS, biggestMisses, filterPrompts, formatScore, medianTrend, viewCorrelation } from './devtoolsData'
import type { Panel, PanelPrompt, SplitId } from './devtoolsData'

const WIDTH = 900, HEIGHT = 420, LEFT = 62, RIGHT = 20, TOP = 16, BOTTOM = 46
const TOOL_TICKS = [0, 1, 3, 10, 30, 100, 300, 1000, 3000]

const preview = (text: string, limit = 96) => {
  const flat = text.replace(/\s+/g, ' ').trim()
  return flat.length > limit ? `${flat.slice(0, limit)}…` : flat
}

function Scatter({ prompts, rowId, rowLabel, selected, onSelect }: { prompts: PanelPrompt[]; rowId: string; rowLabel: string; selected: PanelPrompt | null; onSelect: (prompt: PanelPrompt) => void }) {
  const [hovered, setHovered] = useState<{ prompt: PanelPrompt; x: number; y: number } | null>(null)
  const top = Math.log1p(Math.max(30, ...prompts.map((prompt) => prompt.tools)))
  const x = (rank: number) => LEFT + rank * (WIDTH - LEFT - RIGHT)
  const y = (tools: number) => HEIGHT - BOTTOM - (Math.log1p(tools) / top) * (HEIGHT - TOP - BOTTOM)
  const points = useMemo(() => prompts.flatMap((prompt) => {
    const rank = prompt.ranks[rowId]
    return rank === null || rank === undefined ? [] : [{ prompt, x: x(rank), y: y(prompt.tools) }]
  }), [prompts, rowId, top]) // eslint-disable-line react-hooks/exhaustive-deps
  const trend = useMemo(() => medianTrend(prompts, rowId), [prompts, rowId])

  const nearest = (event: React.MouseEvent<SVGSVGElement>) => {
    const box = event.currentTarget.getBoundingClientRect()
    const px = ((event.clientX - box.left) / box.width) * WIDTH
    const py = ((event.clientY - box.top) / box.height) * HEIGHT
    let best: (typeof points)[number] | null = null
    let distance = 18 ** 2
    for (const point of points) {
      const gap = (point.x - px) ** 2 + (point.y - py) ** 2
      if (gap < distance) { distance = gap; best = point }
    }
    return best
  }

  return <div className="scatter-wrap">
    <svg className="prompt-scatter" viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label={`Each prompt: ${rowLabel}'s rank across, and real tool calls up. ${points.length} prompts.`}
      onPointerMove={(event) => { const point = nearest(event); setHovered(point ? { prompt: point.prompt, x: point.x, y: point.y } : null) }}
      onPointerLeave={() => setHovered(null)}
      onClick={(event) => { const point = nearest(event); if (point) onSelect(point.prompt) }}>
      {TOOL_TICKS.filter((tick) => Math.log1p(tick) <= top).map((tick) => <g key={tick}>
        <line className="size-grid" x1={LEFT} x2={WIDTH - RIGHT} y1={y(tick)} y2={y(tick)} />
        <text className="size-axis" x={LEFT - 8} y={y(tick) + 4} textAnchor="end">{tick.toLocaleString()}</text>
      </g>)}
      <text className="size-axis" transform={`translate(14 ${(TOP + HEIGHT - BOTTOM) / 2}) rotate(-90)`} textAnchor="middle">Tool calls that really followed</text>
      <text className="size-axis" x={LEFT} y={HEIGHT - BOTTOM + 18}>rated smallest</text>
      <text className="size-axis" x={WIDTH - RIGHT} y={HEIGHT - BOTTOM + 18} textAnchor="end">rated biggest</text>
      <text className="size-axis" x={(LEFT + WIDTH - RIGHT) / 2} y={HEIGHT - 8} textAnchor="middle">How {rowLabel} rated the prompt, against the others in that tool</text>
      {points.map(({ prompt, x: cx, y: cy }) => <circle key={prompt.id} className={prompt.chars < 40 ? 'scatter-dot is-short' : 'scatter-dot'} cx={cx} cy={cy} r="3.5" />)}
      {trend.length > 1 && <polyline className="scatter-trend" points={trend.map((item) => `${x(item.rank)},${y(item.tools)}`).join(' ')} />}
      {selected && selected.ranks[rowId] !== null && selected.ranks[rowId] !== undefined && <circle className="scatter-selected" cx={x(selected.ranks[rowId] as number)} cy={y(selected.tools)} r="7" />}
      {hovered && <circle className="scatter-hover" cx={hovered.x} cy={hovered.y} r="6" />}
    </svg>
    {hovered && <div className="scatter-tip" style={{ left: `${(hovered.x / WIDTH) * 100}%`, top: `${(hovered.y / HEIGHT) * 100}%` }}>
      <strong>{hovered.prompt.tools.toLocaleString()} tool {hovered.prompt.tools === 1 ? 'call' : 'calls'}</strong>
      <span>{preview(hovered.prompt.text)}</span>
      <small>{hovered.prompt.tool} · {hovered.prompt.project}</small>
    </div>}
  </div>
}

export function DevtoolsPrompts({ panel, onFollow }: { panel: Panel; onFollow: (prompt: PanelPrompt) => void }) {
  const [rowId, setRowId] = useState('rules')
  const [split, setSplit] = useState<SplitId>('all')
  const [tool, setTool] = useState('all')
  const [project, setProject] = useState('all')
  const [selected, setSelected] = useState<PanelPrompt | null>(null)
  const row = panel.rows.find((item) => item.id === rowId) ?? panel.rows[0]
  const tools = useMemo(() => [...new Set(panel.prompts.map((prompt) => prompt.tool))].sort(), [panel.prompts])
  const projects = useMemo(() => {
    const counts = new Map<string, number>()
    for (const prompt of panel.prompts) counts.set(prompt.project, (counts.get(prompt.project) ?? 0) + 1)
    return [...counts].sort((left, right) => right[1] - left[1])
  }, [panel.prompts])
  const inView = useMemo(() => filterPrompts(panel.prompts, split, tool, project), [panel.prompts, split, tool, project])
  const correlation = useMemo(() => viewCorrelation(inView, rowId), [inView, rowId])
  const misses = useMemo(() => biggestMisses(inView, rowId), [inView, rowId])

  const missList = (title: string, plain: string, items: PanelPrompt[]) => <article className="miss-list">
    <h3>{title}</h3><p>{plain}</p>
    <ol>{items.map((prompt) => <li key={prompt.id}><button type="button" className={selected?.id === prompt.id ? 'is-active' : ''} onClick={() => setSelected(prompt)}>
      <strong>{prompt.tools.toLocaleString()}</strong><span>{preview(prompt.text, 72)}</span>
    </button></li>)}</ol>
  </article>

  return <div className="devtools-stack">
    <section className="devtools-card">
      <div className="card-heading"><div><h2>Every prompt, against what really happened</h2><p>If a predictor is good, dots climb from bottom left to top right. The white line is the typical real work at each rating, and it should climb too.</p></div></div>
      <div className="filter-row" role="group" aria-label="What to show">
        <label>Predictor <select value={rowId} onChange={(event) => setRowId(event.target.value)}>{panel.rows.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label>
        <label>Prompts <select value={split} onChange={(event) => setSplit(event.target.value as SplitId)}>{SPLITS.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label>
        <label>Tool <select value={tool} onChange={(event) => setTool(event.target.value)}><option value="all">Both tools</option>{tools.map((item) => <option key={item}>{item}</option>)}</select></label>
        <label>Project <select value={project} onChange={(event) => setProject(event.target.value)}><option value="all">All projects</option>{projects.map(([name, count]) => <option key={name} value={name}>{name} ({count})</option>)}</select></label>
        <p className="filter-result"><strong>{formatScore(correlation)}</strong><span>score in this view, {inView.length.toLocaleString()} prompts</span></p>
      </div>
      <ul className="viz-legend" aria-label="What the marks mean">
        <li><i className="viz-mark scatter-key" aria-hidden="true" />40+ characters</li>
        <li><i className="viz-mark scatter-key is-short" aria-hidden="true" />Under 40 characters</li>
        <li><i className="viz-line-key" aria-hidden="true" />Typical real work at each rating</li>
      </ul>
      <div className="chart-scroll"><Scatter prompts={inView} rowId={rowId} rowLabel={row.label} selected={selected} onSelect={setSelected} /></div>
    </section>

    {selected && <section className="devtools-card selected-prompt" aria-label="Selected prompt">
      <div className="card-heading">
        <div><p className="eyebrow">{selected.tool.toUpperCase()} · {selected.project.toUpperCase()} · TURN {selected.turn + 1}</p><h2>{selected.tools.toLocaleString()} tool {selected.tools === 1 ? 'call' : 'calls'}, {selected.subagents} {selected.subagents === 1 ? 'subagent' : 'subagents'}</h2></div>
        <button type="button" className="primary-action" onClick={() => onFollow(selected)}>Follow this prompt</button>
      </div>
      <blockquote>{selected.text}{selected.truncated ? ' …' : ''}</blockquote>
      <ReadVsReal prompt={selected} rows={panel.rows} />
    </section>}

    <section className="devtools-card miss-grid" aria-label={`Biggest misses for ${row.label}`}>
      {missList('Rated small, was big', `Where ${row.label} most underestimated the work. The number is real tool calls.`, misses.underrated)}
      {missList('Rated big, was small', `Where ${row.label} most overestimated the work. The number is real tool calls.`, misses.overrated)}
    </section>
    <p className="assessment-note">Prompts are redacted and served only to this machine. Select any dot or row to see all predictors side by side.</p>
  </div>
}
