export type Range = { value: number | null; low: number | null; high: number | null }
export type SplitId = 'all' | 'long' | 'short' | 'later'
export type MeasureId = 'overall' | 'session' | 'lengthFixed'
export type RowKind = 'lfg' | 'trained' | 'opponent' | 'baseline'

export type SplitScores = {
  overall: Range
  scored: number
  session: Range & { groups: number }
  project: Range & { groups: number }
  lengthFixed?: Range
  paired: Record<string, Range>
}

export type PanelRow = {
  id: string
  label: string
  kind: RowKind
  sizeBytes: number
  sizeNote: string
  plain: string
  output: string
  where: string
  source?: string
  license?: string
  latency: { rows: number; p50Ms: number; p95Ms: number } | null
  splits: Record<SplitId, SplitScores>
}

export type PanelPrompt = {
  id: string
  text: string
  truncated: boolean
  tool: string
  project: string
  session: string
  turn: number
  when: number
  chars: number
  tools: number
  subagents: number
  bracket: string | null
  ruleScore: number | null
  ruleAgents: number | null
  effortRank: number
  ranks: Record<string, number | null>
}

export type Panel = {
  schemaVersion: string
  generatedAt: string
  metric: string
  target: { label: string; sizeBytes: number; scored: boolean; rows?: Record<SplitId, string>; gaps?: Record<SplitId, number>; rival?: boolean }
  dataset: { rows: number; long: number; short: number; sessions: number; projects: number; first: string; last: string; redactions: Record<string, number>; sha256: Record<string, string>; snapshot: string }
  rows: PanelRow[]
  secondary: Array<{ label: string } & Record<SplitId, Range>>
  audits: Array<Range & { id: string; label: string; passed: boolean }>
  work: Array<{ tool: string; short: number; long: number; shortMean: number; longMean: number; shortCount: number; longCount: number }>
  predictions: Array<{ text: string; held: boolean; evidence: string }>
  decision: { title: string; detail: string }
  process: { predictionsSha256: string; runs: Record<string, { exit: number; connections: number; kernelBlock: boolean }> }
  incidents: Array<{ status: 'fixed' | 'open' | 'noted'; title: string; detail: string }>
  notMeasured: string[]
  prompts: PanelPrompt[]
  limits: string
}

export type DevtoolsStatus = { schemaVersion: string; modifiedMs: number | null; bytes: number }

export const SPLITS: Array<{ id: SplitId; label: string; plain: string }> = [
  { id: 'all', label: 'All prompts', plain: 'Everything typed' },
  { id: 'long', label: '40+ characters', plain: 'Prompts that describe the work' },
  { id: 'short', label: 'Under 40 characters', plain: 'Short follow-ups such as "continue"' },
  { id: 'later', label: 'Later prompts', plain: 'The closest test of real use: models learn only from earlier prompts and are scored on the last 30% in each tool.' }
]

export const MEASURES: Array<{ id: MeasureId; label: string; plain: string }> = [
  { id: 'overall', label: 'Overall', plain: 'How well it orders prompts by the work that really followed.' },
  { id: 'session', label: 'Inside one session', plain: 'Compares prompts within a session, so telling kinds of work apart earns nothing.' },
  { id: 'lengthFixed', label: 'Length held fixed', plain: 'Removes prompt length, so being a stand-in for length earns nothing.' }
]

export const KIND_LABEL: Record<RowKind, string> = {
  lfg: 'LFG today',
  trained: 'Trained on your outcomes',
  opponent: 'Outside competitor',
  baseline: 'Baseline'
}

export const scoreOf = (row: PanelRow, split: SplitId, measure: MeasureId): Range | null => row.splits[split][measure] ?? null

export function rankedRows(rows: PanelRow[], split: SplitId, measure: MeasureId): PanelRow[] {
  const value = (row: PanelRow) => scoreOf(row, split, measure)?.value ?? Number.NEGATIVE_INFINITY
  return [...rows].sort((left, right) => value(right) - value(left))
}

export const barToBeat = (rows: PanelRow[], split: SplitId, measure: MeasureId): PanelRow => rankedRows(rows, split, measure)[0]

export const formatScore = (value: number | null | undefined): string => value === null || value === undefined ? 'n/a' : `${value >= 0 ? '+' : ''}${value.toFixed(2)}`

export const formatRange = (range: Range | null | undefined): string => !range || range.low === null || range.high === null ? '' : `${formatScore(range.low)} to ${formatScore(range.high)}`

export function formatSize(bytes: number): string {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(bytes >= 1e10 ? 0 : 1)} GB`
  if (bytes >= 1e6) return `${Math.round(bytes / 1e6)} MB`
  if (bytes >= 1e3) return `${Math.round(bytes / 1e3)} KB`
  return 'none'
}

// A difference counts only when its whole 95% interval is on one side of zero.
export function verdict(range: Range | undefined): 'ahead' | 'behind' | 'unclear' {
  if (!range || range.low === null || range.high === null) return 'unclear'
  if (range.low > 0) return 'ahead'
  return range.high < 0 ? 'behind' : 'unclear'
}

export function filterPrompts(prompts: PanelPrompt[], split: SplitId, tool: string, project: string): PanelPrompt[] {
  return prompts.filter((prompt) => (split === 'all' || (split === 'long') === (prompt.chars >= 40)) && (tool === 'all' || prompt.tool === tool) && (project === 'all' || prompt.project === project))
}

function averageRanks(values: number[]): number[] {
  const order = values.map((_, index) => index).sort((left, right) => values[left] - values[right])
  const ranks = new Array<number>(values.length)
  for (let start = 0; start < order.length;) {
    let end = start
    while (end + 1 < order.length && values[order[end + 1]] === values[order[start]]) end += 1
    for (let index = start; index <= end; index += 1) ranks[order[index]] = (start + end) / 2 / Math.max(order.length - 1, 1)
    start = end + 1
  }
  return ranks
}

// The scorecard's metric on whatever is in view: average ranks within each tool, then pooled.
export function viewCorrelation(prompts: PanelPrompt[], rowId: string): number | null {
  const xs: number[] = []
  const ys: number[] = []
  for (const tool of new Set(prompts.map((prompt) => prompt.tool))) {
    const scored = prompts.filter((prompt) => prompt.tool === tool && prompt.ranks[rowId] !== null && prompt.ranks[rowId] !== undefined)
    xs.push(...averageRanks(scored.map((prompt) => prompt.ranks[rowId] as number)))
    ys.push(...averageRanks(scored.map((prompt) => prompt.tools)))
  }
  if (xs.length < 3) return null
  const meanX = xs.reduce((sum, value) => sum + value, 0) / xs.length
  const meanY = ys.reduce((sum, value) => sum + value, 0) / ys.length
  let top = 0, left = 0, right = 0
  xs.forEach((value, index) => {
    top += (value - meanX) * (ys[index] - meanY)
    left += (value - meanX) ** 2
    right += (ys[index] - meanY) ** 2
  })
  return left === 0 || right === 0 ? null : top / Math.sqrt(left * right)
}

export function medianTrend(prompts: PanelPrompt[], rowId: string, bins = 10): Array<{ rank: number; tools: number; count: number }> {
  const groups = Array.from({ length: bins }, () => [] as number[])
  for (const prompt of prompts) {
    const rank = prompt.ranks[rowId]
    if (rank !== null && rank !== undefined) groups[Math.min(bins - 1, Math.floor(rank * bins))].push(prompt.tools)
  }
  return groups.flatMap((group, index) => {
    if (group.length < 3) return []
    const sorted = [...group].sort((left, right) => left - right)
    const middle = sorted.length >> 1
    return [{ rank: (index + .5) / bins, tools: sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2, count: group.length }]
  })
}

export function biggestMisses(prompts: PanelPrompt[], rowId: string, count = 6): { underrated: PanelPrompt[]; overrated: PanelPrompt[] } {
  const scored = prompts.filter((prompt) => prompt.ranks[rowId] !== null && prompt.ranks[rowId] !== undefined)
  const gap = (prompt: PanelPrompt) => (prompt.ranks[rowId] as number) - prompt.effortRank
  const sorted = [...scored].sort((left, right) => gap(left) - gap(right))
  return { underrated: sorted.slice(0, count), overrated: sorted.slice(-count).reverse() }
}

const TOKEN_KEY = 'lfg.devtools.token'

// The launcher passes the session token in the fragment, which never reaches a server, and this removes it from the address bar.
export function readLaunchToken(): string | null {
  const match = /^#devtools=([^&]+)$/.exec(window.location.hash)
  if (match) {
    window.history.replaceState(null, '', `${window.location.pathname}${window.location.search}`)
    const token = decodeURIComponent(match[1])
    rememberToken(token)
    return token
  }
  try { return window.sessionStorage.getItem(TOKEN_KEY) } catch { return null }
}

export function rememberToken(token: string) {
  try { window.sessionStorage.setItem(TOKEN_KEY, token) } catch { /* The token still works until this page closes. */ }
}

async function request<T>(path: string, token: string): Promise<T> {
  const response = await fetch(path, { cache: 'no-store', credentials: 'same-origin', headers: { Accept: 'application/json', 'X-LFG-Session-Token': token } })
  if (!response.ok) {
    const value = await response.json().catch(() => null) as { error?: { message?: string } } | null
    throw new Error(value?.error?.message ?? `${path} returned ${response.status}`)
  }
  return response.json() as Promise<T>
}

export const getDevtoolsStatus = (token: string) => request<DevtoolsStatus>('/api/lfg/devtools/status', token)
export const getDevtoolsPanel = (token: string) => request<Panel>('/api/lfg/devtools/panel', token)
