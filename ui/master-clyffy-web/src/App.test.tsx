import { render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import App from './App'

vi.mock('pixi.js', () => ({
  Application: class { canvas = document.createElement('canvas'); stage = { addChild: vi.fn() }; ticker = { add: vi.fn() }; init = vi.fn().mockResolvedValue(undefined); destroy = vi.fn() },
  Graphics: class { x = 0; y = 0; alpha = 1; circle() { return this } fill() { return this } }
}))

const preflight = {
  schemaVersion: 'lfg.preflight.v1', sessionId: 'test', revision: 1, mode: 'cpu-rule-baseline',
  model: { id: 'lfg-cpu-rule-baseline', version: '1.0.0', sha256: 'a'.repeat(64) }, durationMicros: 85,
  intents: [{ label: 'research.web', confidence: .92, source: 'cpu-rule' }], capabilities: [], risks: [],
  rros: [{ id: 'rro-1', confidence: .92, advisory: true, lineage: ['test'], category: { id: 'research.web', version: '1.0.0', label: 'EVIDENCE', plainLabel: 'Research current evidence', capability: 'web.access', tone: 'cyan', hash: 'b'.repeat(64) } }],
  predictedAgents: 3, predictedBranches: 2, confidence: .92, abstained: false
}

beforeEach(() => {
  vi.stubGlobal('crypto', { randomUUID: () => 'test-session' })
  vi.stubGlobal('matchMedia', () => ({ matches: true, addEventListener: vi.fn(), removeEventListener: vi.fn() }))
  vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
    const path = String(input)
    if (path.endsWith('/status')) return new Response(JSON.stringify({ name: 'MasterClyffy LFG', phase: 'rust-authoritative-alpha', protocolVersion: 'v', workspaceId: 'clyffy.master', encoder: preflight.model, signalMode: 'hash-chain', contentPolicy: 'memory-only', liveProviders: 0, configuredProviders: 1, fixtureProviders: 4 }))
    if (path.endsWith('/preflight')) return new Response(JSON.stringify(preflight))
    return new Response('{}', { status: 404 })
  }))
})

afterEach(() => vi.unstubAllGlobals())

test('shows measured CPU predictions separately from committed tallies', async () => {
  render(<App />)
  expect(screen.getByRole('heading', { name: /watch a team compile itself/i })).toBeInTheDocument()
  await waitFor(() => expect(screen.getByText('0.085ms')).toBeInTheDocument())
  expect(screen.getByText('3')).toBeInTheDocument()
  expect(screen.getByText(/nothing has been committed/i)).toBeInTheDocument()
})
