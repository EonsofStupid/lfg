import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'

import { DevtoolsView } from './DevtoolsView'
import { barToBeat, biggestMisses, filterPrompts, formatScore, formatSize, medianTrend, rankedRows, readLaunchToken, verdict, viewCorrelation } from './devtoolsData'
import type { Panel, PanelPrompt, PanelRow, Range, SplitScores } from './devtoolsData'

const range = (value: number, spread = .1): Range => ({ value, low: value - spread, high: value + spread })

function split(value: number, paired: Record<string, Range> = {}): SplitScores {
  return { overall: range(value), scored: 10, session: { ...range(value / 2), groups: 3 }, project: { ...range(value / 2), groups: 2 }, lengthFixed: range(value / 3), paired }
}

function row(id: string, label: string, kind: PanelRow['kind'], long: number, short: number, sizeBytes = 0): PanelRow {
  return { id, label, kind, sizeBytes, sizeNote: sizeBytes ? formatSize(sizeBytes) : 'no weights', plain: `${label} explained`, output: `${label} output`, where: 'This PC', latency: { rows: 10, p50Ms: 2, p95Ms: 3 }, splits: { all: split((long + short) / 2), long: split(long, { rules: range(.2, .05), clef: range(-.01) }), short: split(short), later: split(long - .05) } }
}

function prompt(id: string, tool: string, chars: number, tools: number, rules: number | null, clef: number, effortRank: number, project = 'alpha'): PanelPrompt {
  return { id, text: `sample text for ${id}`, truncated: false, tool, project, session: 's', turn: 0, when: 1, chars, tools, subagents: 0, bracket: 'simple', ruleScore: null, ruleAgents: null, effortRank, ranks: { rules, clef } }
}

const PROMPTS = [
  prompt('p1', 'Codex', 10, 90, .1, .9, 1),
  prompt('p2', 'Codex', 60, 40, .4, .6, .66),
  prompt('p3', 'Codex', 70, 10, .8, .3, .33),
  prompt('p4', 'Codex', 80, 2, .9, .1, 0),
  prompt('p5', 'Claude Code', 15, 5, null, .2, .2, 'beta')
]

const PANEL: Panel = {
  schemaVersion: 'lfg.devtools-panel.v1', generatedAt: '2026-11-03T22:00:00-05:00', metric: 'Rank correlation.',
  target: { label: 'LFG 650 MB model', sizeBytes: 650000000, scored: false },
  dataset: { rows: 5, long: 3, short: 2, sessions: 1, projects: 2, first: '2026-09-22', last: '2026-11-03', redactions: { 'email-address': 2 }, sha256: { 'prompts.redacted.jsonl': 'abcdef1234567890', 'outcomes.jsonl': '1234567890abcdef' }, snapshot: 'Snapshot' },
  rows: [row('rules', 'LFG rules', 'lfg', .33, -.17), row('clef', 'Clef-flash', 'opponent', .39, .07, 19080000000)],
  secondary: [], audits: [{ id: 'random-all', label: 'A random score sits near 0', ...range(0), passed: true }],
  work: [{ tool: 'Codex', short: 42, long: 23, shortMean: 88, longMean: 135, shortCount: 1, longCount: 3 }],
  predictions: [{ text: 'Clef-flash lands below rules.', held: false, evidence: 'It did not.' }, { text: 'Short prompts are real work.', held: true, evidence: 'They are.' }],
  decision: { title: 'Nothing clearly beat the simple text model.', detail: 'Work on session signals.' },
  process: { predictionsSha256: 'fedcba0987654321', runs: { clef: { exit: 0, connections: 0, kernelBlock: true } } },
  incidents: [{ status: 'fixed', title: 'Telemetry reached Microsoft', detail: 'Rerun under a kernel block.' }],
  notMeasured: ["LFG's own model."], prompts: PROMPTS, limits: "One person's logs."
}

function serve(panel: Panel | null) {
  vi.stubGlobal('fetch', vi.fn(async (path: string, init?: RequestInit) => {
    const headers = init?.headers as Record<string, string> | undefined
    if (String(path).startsWith('/api/lfg/devtools/')) {
      if (headers?.['X-LFG-Session-Token'] !== 'launch-token') return new Response(JSON.stringify({ error: { message: 'a valid local devtools session token is required' } }), { status: 403 })
      if (!panel) return new Response(JSON.stringify({ error: { message: 'no scorecard results are connected' } }), { status: 503 })
      return new Response(JSON.stringify(String(path).endsWith('/status') ? { schemaVersion: 'lfg.devtools-status.v1', modifiedMs: 1, bytes: 5 } : panel))
    }
    return new Response(JSON.stringify({ message: 'unexpected' }), { status: 500 })
  }))
}

beforeEach(() => { window.sessionStorage.clear(); window.history.replaceState(null, '', '/') })
afterEach(() => { cleanup(); vi.unstubAllGlobals() })

describe('devtools data', () => {
  it('ranks the best row first for the chosen split and measure', () => {
    expect(rankedRows(PANEL.rows, 'long', 'overall').map((item) => item.id)).toEqual(['clef', 'rules'])
    expect(barToBeat(PANEL.rows, 'short', 'overall').id).toBe('clef')
  })

  it('formats scores with a sign and says when a value is missing', () => {
    expect(formatScore(.394)).toBe('+0.39')
    expect(formatScore(-.174)).toBe('-0.17')
    expect(formatScore(null)).toBe('n/a')
    expect(formatSize(650000000)).toBe('650 MB')
    expect(formatSize(19080000000)).toBe('19 GB')
    expect(formatSize(0)).toBe('none')
  })

  it('calls a difference only when its whole interval is on one side of zero', () => {
    expect(verdict({ value: .1, low: .01, high: .2 })).toBe('ahead')
    expect(verdict({ value: -.1, low: -.2, high: -.01 })).toBe('behind')
    expect(verdict({ value: .05, low: -.04, high: .15 })).toBe('unclear')
    expect(verdict(undefined)).toBe('unclear')
  })

  it('computes correlation within each tool and skips rows a predictor could not score', () => {
    const codexOnly = PROMPTS.filter((item) => item.tool === 'Codex')
    expect(viewCorrelation(codexOnly, 'clef')).toBeCloseTo(1)
    expect(viewCorrelation(codexOnly, 'rules')).toBeCloseTo(-1)
    expect(viewCorrelation(PROMPTS.slice(0, 2), 'rules')).toBeNull()
  })

  it('filters by prompt length, tool and project', () => {
    expect(filterPrompts(PROMPTS, 'short', 'all', 'all').map((item) => item.id)).toEqual(['p1', 'p5'])
    expect(filterPrompts(PROMPTS, 'long', 'Codex', 'all')).toHaveLength(3)
    expect(filterPrompts(PROMPTS, 'all', 'all', 'beta').map((item) => item.id)).toEqual(['p5'])
  })

  it('finds the prompts a predictor rated furthest from what happened', () => {
    const misses = biggestMisses(PROMPTS, 'rules', 1)
    expect(misses.underrated[0].id).toBe('p1')
    expect(misses.overrated[0].id).toBe('p4')
    expect(biggestMisses(PROMPTS, 'rules').underrated.map((item) => item.id)).not.toContain('p5')
  })

  it('leaves out trend bins with too few prompts to have a median', () => {
    expect(medianTrend(PROMPTS, 'clef', 10)).toEqual([])
    expect(medianTrend(PROMPTS, 'clef', 1)).toEqual([{ rank: .5, tools: 10, count: 5 }])
  })

  it('takes the launch token from the fragment and removes it from the address bar', () => {
    window.history.replaceState(null, '', '/#devtools=launch-token')
    expect(readLaunchToken()).toBe('launch-token')
    expect(window.location.hash).toBe('')
    expect(readLaunchToken()).toBe('launch-token')
  })
})

describe('DevtoolsView', () => {
  it('asks for the token before showing private results, and connects with it', async () => {
    serve(PANEL)
    render(<DevtoolsView status={null} />)
    expect(screen.getByRole('heading', { name: 'Connect your scorecard results' })).toBeInTheDocument()
    expect(screen.queryByText('sample text for p1')).not.toBeInTheDocument()
    fireEvent.change(screen.getByLabelText('Session token'), { target: { value: 'launch-token' } })
    fireEvent.click(screen.getByRole('button', { name: 'Connect' }))
    expect(await screen.findByRole('heading', { name: 'Who predicts the real work best?' })).toBeInTheDocument()
  })

  it('reports a refused token instead of showing results', async () => {
    serve(PANEL)
    window.sessionStorage.setItem('lfg.devtools.token', 'wrong')
    render(<DevtoolsView status={null} />)
    expect(await screen.findByText('a valid local devtools session token is required', { selector: '.devtools-connect p' })).toBeInTheDocument()
  })

  it('shows the race, keeps the 650 MB model marked as not scored, and follows the chosen split', async () => {
    serve(PANEL)
    window.history.replaceState(null, '', '/#devtools=launch-token')
    render(<DevtoolsView status={null} />)
    const table = await screen.findByRole('table', { name: 'Score for each predictor, best first' })
    const rows = () => within(table).getAllByRole('row').map((item) => item.textContent ?? '')
    expect(rows()[0]).toContain('Clef-flash')
    expect(rows()[0]).toContain('+0.39')
    expect(rows()[2]).toContain('LFG 650 MB model')
    expect(rows()[2]).toContain('not scored')
    fireEvent.click(screen.getByRole('button', { name: /Under 40 characters/ }))
    await waitFor(() => expect(rows()[1]).toContain('-0.17'))
    expect(screen.getByRole('heading', { name: /1 of 2 held/ })).toBeInTheDocument()
    fireEvent.click(within(table).getByRole('row', { name: /LFG rules/ }))
    expect(screen.getByLabelText('LFG rules details')).toHaveTextContent('LFG rules explained')
  })

  it('reports a dev row by its gap to the bar once one has been scored', async () => {
    const devRow = row('devtext', 'LFG dev, prompt', 'lfg', .3, .2, 643800000)
    serve({ ...PANEL, rows: [...PANEL.rows, devRow], target: { ...PANEL.target, scored: true, rows: { all: 'devtext', long: 'devtext', short: 'devtext', later: 'devtext' }, gaps: { all: -.1, long: -.09, short: .13, later: -.2 }, rival: false } })
    window.history.replaceState(null, '', '/#devtools=launch-token')
    render(<DevtoolsView status={null} />)
    const table = await screen.findByRole('table', { name: 'Score for each predictor, best first' })
    expect(within(table).queryByText('not scored')).not.toBeInTheDocument()
    expect(screen.getByText('-0.09')).toBeInTheDocument()
    expect(screen.getByText('Best dev row: LFG dev, prompt')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('tab', { name: /How it ran/ }))
    expect(screen.getByRole('heading', { name: 'Train the dev rows on your outcomes' })).toBeInTheDocument()
  })

  it('shows how the numbers were made, including what went wrong', async () => {
    serve(PANEL)
    window.history.replaceState(null, '', '/#devtools=launch-token')
    render(<DevtoolsView status={null} />)
    await screen.findByRole('heading', { name: 'Who predicts the real work best?' })
    fireEvent.click(screen.getByRole('tab', { name: /How it ran/ }))
    expect(screen.getByRole('heading', { name: 'Run each competitor blind' })).toBeInTheDocument()
    expect(screen.getByText('✓ kernel')).toBeInTheDocument()
    expect(screen.getByText('Telemetry reached Microsoft')).toBeInTheDocument()
    expect(screen.getByText("LFG's own model.")).toBeInTheDocument()
  })

  it('opens a prompt from the explorer in the live trace', async () => {
    serve(PANEL)
    window.history.replaceState(null, '', '/#devtools=launch-token')
    render(<DevtoolsView status={null} />)
    await screen.findByRole('heading', { name: 'Who predicts the real work best?' })
    fireEvent.click(screen.getByRole('tab', { name: /Every prompt/ }))
    const rated = screen.getByRole('heading', { name: 'Rated small, was big' }).parentElement as HTMLElement
    fireEvent.click(within(rated).getByRole('button', { name: /sample text for p1/ }))
    fireEvent.click(screen.getByRole('button', { name: 'Follow this prompt' }))
    expect(screen.getByRole('textbox')).toHaveValue('sample text for p1')
  })
})
