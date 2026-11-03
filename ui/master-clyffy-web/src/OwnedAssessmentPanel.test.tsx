import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import { OwnedAssessmentPanel } from './OwnedAssessmentPanel'
import type { OwnedAssessmentClient, OwnedResult } from './ownedAssessmentApi'
import type { Assessment } from './types'

const deterministic = { intent: { primary: 'build' }, complexity: { bracket: 'moderate' }, agents: { total: 2 } } as Assessment
const client = (): OwnedAssessmentClient => ({
  status: vi.fn(async () => ({ enabled: true, status: 'ready' as const, engine: { id: 'owned-test', modelSha256: 'a'.repeat(64) }, limits: { maxRequestBytes: 1000, maxGeneratedBytes: 512, timeoutMs: 3000, maxConcurrent: 1 } })),
  assess: vi.fn(async () => ({ status: 'ok' as const, engine: { id: 'owned-test' }, elapsedMs: 120, stopReason: 'stop_sequence', output: { intent: 'research', complexity: 'simple', suggested_models: ['local-model'], agent_count: 1, reason: 'One task' } })),
})
afterEach(() => { cleanup(); vi.restoreAllMocks() })

async function connect() {
  fireEvent.change(screen.getByLabelText('Owned model token'), { target: { value: 'test-token' } })
  fireEvent.click(screen.getByRole('button', { name: 'Connect owned model' }))
  await screen.findByText(/Model status: ready/)
}

test('runs only on request and displays the experimental output separately from the rules', async () => {
  const api = client()
  render(<OwnedAssessmentPanel prompt="owner request" assessment={deterministic} client={api} />)
  expect(api.assess).not.toHaveBeenCalled()
  await connect()
  expect(screen.getByText(/Rules assessment: intent build/)).toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: 'Run owned model' }))
  expect(await screen.findByText('research')).toBeInTheDocument()
  expect(screen.getByText('One task')).toBeInTheDocument()
  expect(screen.getByText(/It has not been verified/)).toBeInTheDocument()
  expect(api.assess).toHaveBeenCalledWith('owner request', 'test-token')
})

test('reports malformed or truncated output without inventing assessment fields', async () => {
  const api = client()
  vi.mocked(api.assess).mockResolvedValue({ status: 'truncated', engine: { id: 'owned-test' }, elapsedMs: 50, outputText: '{partial', stopReason: 'byte_limit' })
  render(<OwnedAssessmentPanel prompt="owner request" assessment={deterministic} client={api} />)
  await connect()
  fireEvent.click(screen.getByRole('button', { name: 'Run owned model' }))
  expect(await screen.findByText(/stopped at an output or context limit/)).toBeInTheDocument()
  expect(screen.queryByText('Suggested models')).not.toBeInTheDocument()
  fireEvent.click(screen.getByText('Show raw model output'))
  expect(screen.getByText('{partial')).toBeInTheDocument()
})

test('explains when the composed request exceeds model context', async () => {
  const api = client()
  vi.mocked(api.assess).mockResolvedValue({ status: 'context_limit', engine: { id: 'owned-test' }, elapsedMs: 1 })
  render(<OwnedAssessmentPanel prompt="owner request" assessment={deterministic} client={api} />)
  await connect()
  fireEvent.click(screen.getByRole('button', { name: 'Run owned model' }))
  expect(await screen.findByText(/exceed the configured model context/)).toBeInTheDocument()
  expect(screen.queryByText('Suggested models')).not.toBeInTheDocument()
})

test('marks incomplete model output and displays missing fields as unknown', async () => {
  const api = client()
  vi.mocked(api.assess).mockResolvedValue({ status: 'incomplete_output', engine: { id: 'owned-test' }, elapsedMs: 3, output: { intent: 'research' } })
  render(<OwnedAssessmentPanel prompt="owner request" assessment={deterministic} client={api} />)
  await connect()
  fireEvent.click(screen.getByRole('button', { name: 'Run owned model' }))
  expect(await screen.findByText(/missing or invalid required fields/)).toBeInTheDocument()
  expect(screen.getByText('research')).toBeInTheDocument()
  expect(screen.getAllByText('Unknown').length).toBeGreaterThan(0)
})

test('discards a model response when the request changed during inference', async () => {
  const api = client()
  let resolve!: (value: OwnedResult) => void
  vi.mocked(api.assess).mockImplementation(() => new Promise((done) => { resolve = done }))
  const view = render(<OwnedAssessmentPanel prompt="first request" assessment={deterministic} client={api} />)
  await connect()
  fireEvent.click(screen.getByRole('button', { name: 'Run owned model' }))
  view.rerender(<OwnedAssessmentPanel prompt="changed request" assessment={null} client={api} />)
  resolve({ status: 'ok', engine: { id: 'owned-test' }, elapsedMs: 3, output: { intent: 'stale' } })
  await waitFor(() => expect(screen.getByRole('button', { name: 'Run owned model' })).not.toBeDisabled())
  expect(screen.queryByText('stale')).not.toBeInTheDocument()
})
