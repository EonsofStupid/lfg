import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import App from './App'

vi.mock('pixi.js', () => ({
  Application: class { canvas = document.createElement('canvas'); stage = { addChild: vi.fn() }; ticker = { add: vi.fn() }; init = vi.fn().mockResolvedValue(undefined); destroy = vi.fn() },
  Graphics: class { x = 0; y = 0; alpha = 1; circle() { return this } fill() { return this } }
}))

const preflight = {
  schemaVersion: 'lfg.preflight.v2', sessionId: 'test', revision: 1, mode: 'cpu-rule-baseline',
  model: { id: 'lfg-cpu-rule-baseline', version: '1.0.0', sha256: 'a'.repeat(64) }, durationMicros: 85,
  intents: [{ label: 'research.web', confidence: null, source: 'cpu-rule' }], capabilities: [], risks: [],
  rros: [{ id: 'rro-1', confidence: null, advisory: true, lineage: ['test'], category: { id: 'research.web', version: '1.0.0', label: 'EVIDENCE', plainLabel: 'Research current evidence', capability: 'web.access', tone: 'cyan', hash: 'b'.repeat(64) } }],
  predictedAgents: 3, predictedBranches: 2, confidence: null, abstained: true
}

const assessment = {
  schemaVersion: 'lfg.assessment.v1', sessionId: 'test', revision: 1, durationMicros: 85, configRevision: 'test', configDigest: 'f'.repeat(64),
  prompt: { characters: 34, words: 6, tasks: ['implement my feature', 'run tests'] },
  intent: { primary: 'build.software', plainLabel: 'Build or change software', fallback: false, risks: [],
    all: [{ id: 'build.software', plainLabel: 'Build or change software', matched: ['implement'] }, { id: 'verify.outcome', plainLabel: 'Prove the result', matched: ['test'] }] },
  complexity: { score: 3.65, bracket: 'moderate', label: 'Moderate', plainLabel: 'A few connected steps',
    terms: [{ id: 'tasks', label: 'Separate tasks', value: 2, weight: 1, points: 2, detail: [] }],
    ladder: [{ id: 'trivial', label: 'Trivial', minScore: 0, current: false }, { id: 'moderate', label: 'Moderate', minScore: 3, current: true }] },
  agents: { total: 7, tasks: 2, min: 1, max: 2, plain: '2 separate tasks found.' },
  models: { role: 'standard', effort: 'medium', usualModel: null, plain: 'Moderate work suits a standard model at medium effort.', alternatives: [],
    suggested: { subscriptionId: 'project-seat', model: 'example-model', label: 'Example Model', description: 'Balanced model', role: 'standard', effort: 'medium', usualEffort: 'xhigh', origin: 'provider-list', fit: 1, habit: 0.5, score: 0.85, promptsInBracket: 4, why: 'A standard model.' } },
  subscriptions: [{ id: 'project-seat', label: 'project-seat', status: 'recognized', usable: true, declared: false, plan: 'Example Plan', lastUsed: '2026-11-01', detail: 'Found on this machine; last used 2026-11-01.' }],
  knowledge: { state: 'ready', refreshedAt: '2026-11-01T00:00:00Z', promptsSeen: 1182, modelsKnown: 3 },
  rules: ['intent:build.software', 'bracket:moderate'], advisory: true, confidence: null
}

const knowledge = {
  state: 'ready', refreshedAt: '2026-11-01T00:00:00Z', promptsSeen: 1182, sources: [{ id: 'codex-sessions', files: 92, prompts: 746, status: 'read' }], subscriptions: [],
  models: [{ subscriptionId: 'codex', model: 'gpt-learned', label: 'GPT Learned', description: '', priority: 1, efforts: ['low'], origin: 'provider-list', available: true, lastUsed: '2026-11-01', role: 'fast', usage: [{ bracket: 'trivial', prompts: 12, weighted: 11, efforts: { low: 12 } }] }]
}

const classification = {
  schemaVersion: 'lfg.classification.v1', configRevision: 'test', configDigest: 'c'.repeat(64),
  sessionId: 'test', revision: 1, durationMicros: 85, tokenEstimate: 8, categories: [],
  needsBiggerModel: false, predictedSubagents: 1, routingNeed: 'balanced', project: 'test',
  recommendedEscalation: null, escalationCatalog: [{ active: false, target: {
    provider: 'example', model: 'example-model', role: 'test', rawCapability: 0.5,
    toolLoopEfficiency: 0.5, costTier: 'low', subscriptionId: 'project-seat',
    strengths: [], effortLevels: [], evidence: { source: 'fixture', basis: 'synthetic', observedAt: 'test' }
  } }]
}
let appContext: { schemaVersion: string; status: string; revision: string | null; pollIntervalMs: number | null; providers: unknown[] | null }
let ownedAssessmentBodyLimit = 3000
beforeEach(() => {
  ownedAssessmentBodyLimit = 3000
  appContext = {schemaVersion: 'lfg.app-context.v1', status: 'disabled', revision: null, pollIntervalMs: null, providers: null}
  vi.stubGlobal('crypto', { randomUUID: () => 'test-session' })
  vi.stubGlobal('matchMedia', () => ({ matches: true, addEventListener: vi.fn(), removeEventListener: vi.fn() }))
  vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const path = String(input)
    if (path.endsWith('/api/lfg/assess/owned/status')) return new Response(JSON.stringify({ enabled: true, status: 'ready', engine: { id: 'owned-test' }, limits: { maxRequestBytes: 1000, maxHttpBodyBytes: ownedAssessmentBodyLimit, maxContextBytes: 5000, maxGeneratedBytes: 512, timeoutMs: 3000, maxConcurrent: 1 } }))
    if (path.endsWith('/api/lfg/assess/owned')) {
      const body = JSON.parse(String(init?.body))
      return new Response(JSON.stringify({ status: 'ok', elapsedMs: 3, engine: { id: 'owned-test' }, projectContextSha256: body.contextSha256, output: { intent: 'research' } }))
    }
    if (path.endsWith('/status')) return new Response(JSON.stringify({ name: 'MasterClyffy LFG', phase: 'rust-authoritative-alpha', protocolVersion: 'v', workspaceId: 'clyffy.master', encoder: preflight.model, signalMode: 'hash-chain', contentPolicy: 'memory-only', liveProviders: 0, configuredProviders: 1, fixtureProviders: 4, operationsDigest: 'd'.repeat(64), operations: { schemaVersion: 'lfg.operations.v1', revision: 'test', maxPromptBytes: 120, maxHttpBodyBytes: 1000, defaultSignalPageSize: 1, maxSignalPageSize: 2, eventBufferCapacity: 8, streamKeepaliveSeconds: 2, uiDebounceMs: 10, uiTimingHistorySamples: 2, devTools: { statsEnabled: true, traceCapacity: 8, statsPollIntervalMs: 1000, debounceMinMs: 5, debounceMaxMs: 80, historyMinSamples: 1, historyMaxSamples: 8 } } }))
    if (path.endsWith('/assess')) return new Response(JSON.stringify({ ...assessment, revision: JSON.parse(String(init?.body)).revision }))
    if (path.endsWith('/knowledge')) return new Response(JSON.stringify(knowledge))
    if (path.endsWith('/classify')) return new Response(JSON.stringify({ ...classification, revision: JSON.parse(String(init?.body)).revision }))
    if (path.endsWith('/intake/app-context')) return new Response(JSON.stringify(appContext))
    if (path.endsWith('/intake/preview')) {
      const body = JSON.parse(String(init?.body))
      return new Response(JSON.stringify({schemaVersion: 'lfg.intake-preview.v1', templateRevision: 'test', templateDigest: 'e'.repeat(64), sourceApp: body.sourceApp, missingRequired: 1,
        answers: [{id: 'user_request', question: 'What is the user asking to do?', required: true, status: body.facts.request.text ? 'filled' : 'missing', value: body.facts.request.text || null}]}))
    }
    if (path.endsWith('/dev-stats')) return new Response(JSON.stringify({schemaVersion: 'lfg.dev-stats.v1', enabled: true, capacity: 8, totalRecorded: 1, evicted: 0, cpuTimeMicros: null, operationsDigest: 'd'.repeat(64), encoderDigest: 'a'.repeat(64), classifierDigest: 'c'.repeat(64), summaries: [{operation: 'preflight', samples: 1, errors: 0, p50TotalMicros: 100, p95TotalMicros: 100}], traces: [{sequence: 1, recordedAt: '2026-01-01T00:00:00Z', operation: 'assessment', outcome: 'ok', totalMicros: 100, queueMicros: 10, workMicros: 80, rules: ['bracket:moderate', 'agents:2']}]}))
    return new Response('{}', { status: 404 })
  }))
})

test('uses the live host roster and preserves qualified capabilities in intake', async () => {
  appContext = {schemaVersion: 'lfg.app-context.v1', status: 'ready', revision: 'host-test', pollIntervalMs: 5000,
    providers: [{id: 'provider-id', activated: true, capabilities: ['read-sessions']}]}
  render(<App />)
  enterPrompt()
  await screen.findByRole('button', { name: 'project-seat' })
  fireEvent.click(screen.getByRole('button', { name: 'Inspector' }))
  expect(await screen.findByText(/Activation can mean history-only access/)).toBeInTheDocument()
  await waitFor(() => expect(vi.mocked(fetch).mock.calls.some(([url, init]) => {
    if (!String(url).endsWith('/intake/preview')) return false
    const facts = JSON.parse(String(init?.body)).facts
    return facts.providers?.[0]?.id === 'provider-id' && facts.providers[0].capabilities.includes('read-sessions')
  })).toBe(true))
})

test('does not replace unavailable host facts with declared subscription availability', async () => {
  appContext = {schemaVersion: 'lfg.app-context.v1', status: 'unavailable', revision: 'host-test', pollIntervalMs: 5000, providers: null}
  render(<App />)
  enterPrompt()
  await screen.findByRole('button', { name: 'project-seat' })
  fireEvent.click(screen.getByRole('button', { name: 'Inspector' }))
  expect(await screen.findByText(/DevPulse context is unavailable/)).toBeInTheDocument()
  await waitFor(() => expect(vi.mocked(fetch).mock.calls.some(([url, init]) => {
    if (!String(url).endsWith('/intake/preview')) return false
    const facts = JSON.parse(String(init?.body)).facts
    return facts.request?.text === 'Implement my feature and run tests' && !('providers' in facts)
  })).toBe(true))
})

afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals() })

function enterPrompt(value = 'Implement my feature and run tests') {
  fireEvent.change(screen.getByRole('textbox', { name: 'Prompt to route' }), { target: { value } })
}

test('shows measured elapsed time separately from committed tallies', async () => {
  render(<App />)
  expect(screen.getByRole('heading', { name: /watch a team compile itself/i })).toBeInTheDocument()
  expect(screen.getByRole('textbox', { name: 'Prompt to route' })).toHaveValue('')
  enterPrompt()
  await waitFor(() => expect(screen.getByText('0.085 ms elapsed')).toBeInTheDocument())
  expect(screen.getByText('Committed')).toBeInTheDocument()
})

test('shows one assessment in the order a person reads it', async () => {
  render(<App />)
  enterPrompt()
  const view = await screen.findByRole('region', { name: 'LFG assessment' })
  await within(view).findByText('Prompt received')
  expect(within(view).getAllByRole('heading').map((heading) => heading.textContent)).toEqual([
    'Prompt received', 'Formula', 'Brackets', 'Agents and models suggested', 'Subscriptions recognized'
  ])
  expect(within(view).getByText('“Implement my feature and run tests”')).toBeInTheDocument()
  expect(within(view).getByText('run tests')).toBeInTheDocument()
  expect(within(view).getByText('Separate tasks')).toBeInTheDocument()
  expect(within(view).getByText('Moderate', { selector: 'li.is-current strong' })).toBeInTheDocument()
  expect(within(view).getByText('Example Model')).toBeInTheDocument()
  expect(within(view).getByText('Found on this machine')).toBeInTheDocument()
  expect(within(view).getByText('Example Plan')).toBeInTheDocument()
  expect(within(view).getByText(/Learned from 1,182 of your requests/)).toBeInTheDocument()
  expect(screen.queryByText(/ready to probe/)).not.toBeInTheDocument()
  const agents = screen.getByText('Suggested agents').closest('article')
  expect(agents).toHaveTextContent('7')
  expect(view.querySelector('.assessment-agents')).toHaveTextContent('7 agents')
  expect(vi.mocked(fetch).mock.calls.some(([url]) => String(url).endsWith('/preflight') || String(url).endsWith('/classify'))).toBe(false)
})

test('uses configured subscription IDs instead of a built-in provider roster', async () => {
  render(<App />)
  enterPrompt()
  const configured = await screen.findByRole('button', { name: 'project-seat' })
  expect(screen.queryByRole('button', { name: 'Claude' })).not.toBeInTheDocument()
  fireEvent.click(configured)
  await waitFor(() => expect(configured).toHaveClass('is-active'))
  await waitFor(() => expect(vi.mocked(fetch).mock.calls.some(([url, init]) =>
    String(url).endsWith('/assess') && JSON.parse(String(init?.body)).declaredSubscriptions.includes('project-seat')
  )).toBe(true))
})

test('uses the configured UTF-8 byte limit for multibyte input', async () => {
  render(<App />)
  enterPrompt()
  await screen.findByRole('button', { name: 'project-seat' })
  const prompt = screen.getByRole('textbox', { name: 'Prompt to route' })
  fireEvent.change(prompt, { target: { value: 'é'.repeat(61) } })
  expect(screen.getByText('122 / 120 UTF-8 bytes')).toBeInTheDocument()
  expect(screen.getByRole('alert')).toHaveTextContent('exceeds the configured UTF-8 byte limit')
  expect(screen.getByRole('button', { name: 'COMMIT ROUTE' })).toBeDisabled()
  expect(prompt).toHaveAttribute('aria-invalid', 'true')
  fireEvent.change(prompt, { target: { value: 'é'.repeat(60) } })
  expect(screen.getByText('120 / 120 UTF-8 bytes')).toBeInTheDocument()
  expect(screen.getByRole('button', { name: 'COMMIT ROUTE' })).not.toBeDisabled()
})

test('uses the owned route body cap even when it differs from the app operations cap', async () => {
  const contextDigest = '09'.repeat(32)
  vi.stubGlobal('crypto', { randomUUID: () => 'test-session', subtle: { digest: vi.fn(async () => new Uint8Array(32).fill(9).buffer) } })
  render(<App />)
  enterPrompt()
  await screen.findByRole('button', { name: 'project-seat' })
  fireEvent.click(screen.getByRole('button', { name: 'Inspector' }))
  fireEvent.change(screen.getByLabelText('Owned model token'), { target: { value: 'owned-token' } })
  fireEvent.click(screen.getByRole('button', { name: 'Connect owned model' }))
  await screen.findByText(/Model status: ready/)
  expect(screen.getByText(/Serialized request body:/)).toHaveTextContent('/ 3,000 UTF-8 bytes.')
  const context = 'owner-reviewed context '.repeat(100)
  fireEvent.change(screen.getByLabelText('Optional project context (owner-supplied, unverified)'), { target: { value: context } })
  expect(screen.getByRole('button', { name: 'Run owned model' })).not.toBeDisabled()
  expect(vi.mocked(fetch).mock.calls.some(([url]) => String(url).endsWith('/api/lfg/assess/owned'))).toBe(false)
  fireEvent.click(screen.getByRole('button', { name: 'Run owned model' }))
  expect(await screen.findByText(/server echoed the matching project context digest/i)).toBeInTheDocument()
  const ownedCall = vi.mocked(fetch).mock.calls.find(([url]) => String(url).endsWith('/api/lfg/assess/owned'))
  expect(JSON.parse(String(ownedCall?.[1]?.body))).toEqual({ text: 'Implement my feature and run tests', context, contextSha256: contextDigest })
})

test('rejects an oversized body against the owned route cap even below the app operations cap', async () => {
  ownedAssessmentBodyLimit = 500
  render(<App />)
  enterPrompt()
  await screen.findByRole('button', { name: 'project-seat' })
  fireEvent.click(screen.getByRole('button', { name: 'Inspector' }))
  fireEvent.change(screen.getByLabelText('Owned model token'), { target: { value: 'owned-token' } })
  fireEvent.click(screen.getByRole('button', { name: 'Connect owned model' }))
  await screen.findByText(/Model status: ready/)
  fireEvent.change(screen.getByLabelText('Optional project context (owner-supplied, unverified)'), { target: { value: 'owner-reviewed context '.repeat(40) } })
  expect(screen.getByRole('button', { name: 'Run owned model' })).toBeDisabled()
  expect(screen.getByRole('alert')).toHaveTextContent('configured HTTP body limit')
  expect(vi.mocked(fetch).mock.calls.some(([url]) => String(url).endsWith('/api/lfg/assess/owned'))).toBe(false)
})

test('schedules analysis using the server-configured debounce', async () => {
  const timer = vi.spyOn(window, 'setTimeout')
  render(<App />)
  enterPrompt()
  await screen.findByRole('button', { name: 'project-seat' })
  expect(timer.mock.calls.some(([, delay]) => delay === 10)).toBe(true)
})

test('does not send prompts when server configuration is incompatible', async () => {
  vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({ phase: 'old-server' }))))
  render(<App />)
  expect(await screen.findByRole('alert')).toHaveTextContent('Server operational configuration is missing or incompatible')
  expect(screen.getByRole('button', { name: 'COMMIT ROUTE' })).toBeDisabled()
  expect(vi.mocked(fetch).mock.calls.every(([url]) => String(url).endsWith('/status'))).toBe(true)
})

test('shows local traces and applies browser-only tuning', async () => {
  const timer = vi.spyOn(window, 'setTimeout')
  const { container } = render(<App />)
  enterPrompt()
  await screen.findByRole('button', { name: 'project-seat' })
  fireEvent.click(screen.getByRole('button', { name: 'Inspector' }))
  await waitFor(() => expect(container.querySelectorAll('.cpu-burst-bar')).toHaveLength(1))
  expect(screen.getByText(/being replaced by the assessment above/)).toBeInTheDocument()
  expect(await screen.findByText('Implement my feature and run tests', { selector: 'pre' })).toBeInTheDocument()
  await waitFor(() => expect(vi.mocked(fetch).mock.calls.some(([url, init]) =>
    String(url).endsWith('/intake/preview') && JSON.parse(String(init?.body)).facts.request.text === 'Implement my feature and run tests'
  )).toBe(true))
  expect(await screen.findByText(/1 traces recorded/)).toHaveTextContent('CPU time unavailable')
  expect(screen.getByText('0.010 ms')).toBeInTheDocument()
  expect(screen.getByText('bracket:moderate · agents:2')).toBeInTheDocument()
  expect(screen.getByText('gpt-learned')).toBeInTheDocument()
  expect(screen.getByText('trivial 12')).toBeInTheDocument()
  fireEvent.change(screen.getByRole('slider', { name: 'Browser debounce' }), { target: { value: '25' } })
  await waitFor(() => expect(timer.mock.calls.some(([, delay]) => delay === 25)).toBe(true))
  fireEvent.change(screen.getByRole('slider', { name: 'Displayed history' }), { target: { value: '3' } })
  expect(container.querySelectorAll('.cpu-burst-bar')).toHaveLength(1)
  expect(screen.getByText(/server configuration is unchanged/i)).toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: 'Use server defaults' }))
  expect(screen.getByRole('slider', { name: 'Browser debounce' })).toHaveValue('10')
})

test('suggested context uses the actual user-declared subscription state', async () => {
  render(<App />)
  enterPrompt()
  const toggle = await screen.findByRole('button', { name: 'project-seat' })
  fireEvent.click(toggle)
  fireEvent.click(screen.getByRole('button', { name: 'Inspector' }))
  fireEvent.change(screen.getByLabelText('Owned model token'), { target: { value: 'owned-token' } })
  fireEvent.click(screen.getByRole('button', { name: 'Connect owned model' }))
  await screen.findByText(/Model status: ready/)
  fireEvent.click(screen.getByRole('button', { name: 'Build suggested context' }))
  const draft = await screen.findByLabelText('Suggested context draft') as HTMLTextAreaElement
  expect(draft.value).toContain('["project-seat"]')
  expect(draft.value).toContain('Unknown: no qualifying')
  expect(draft.value).not.toContain('gpt-learned')
  expect(screen.getByLabelText('Optional project context (owner-supplied, unverified)')).toHaveValue('')
})
