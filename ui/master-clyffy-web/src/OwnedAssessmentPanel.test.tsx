import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import { OwnedAssessmentPanel } from './OwnedAssessmentPanel'
import type { OwnedAssessmentClient, OwnedResult } from './ownedAssessmentApi'
import type { Assessment } from './types'

const deterministic = { intent: { primary: 'build' }, complexity: { bracket: 'moderate' }, agents: { total: 2 } } as Assessment
const client = (): OwnedAssessmentClient => ({
  status: vi.fn(async () => ({ enabled: true, status: 'ready' as const, engine: { id: 'owned-test', modelSha256: 'a'.repeat(64) }, limits: { maxRequestBytes: 1000, maxHttpBodyBytes: 10000, maxContextBytes: 5000, maxGeneratedBytes: 512, timeoutMs: 3000, maxConcurrent: 1 } })),
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

test('sends exact reviewed Unicode context bytes and accepts only the matching server digest', async () => {
  const api = client()
  const onResult = vi.fn()
  const expectedDigest = Array.from({ length: 32 }, (_, index) => index.toString(16).padStart(2, '0')).join('')
  const context = '  Café\n🦊  '
  const digest = vi.fn(async (_algorithm: string, data: BufferSource) => {
    expect(Array.from(new Uint8Array(data as ArrayBuffer))).toEqual(Array.from(new TextEncoder().encode(context)))
    return Uint8Array.from({ length: 32 }, (_, index) => index).buffer
  })
  vi.stubGlobal('crypto', { subtle: { digest } })
  vi.mocked(api.assess).mockResolvedValue({ status: 'ok', engine: { id: 'owned-test' }, elapsedMs: 4,
    projectContextSha256: expectedDigest, output: { intent: 'research', extension: { retained: true } } })
  render(<OwnedAssessmentPanel prompt="request" assessment={deterministic} client={api} onResult={onResult} />)
  await connect()
  fireEvent.change(screen.getByLabelText('Optional project context (owner-supplied, unverified)'), { target: { value: context } })
  fireEvent.click(screen.getByRole('button', { name: 'Run owned model' }))
  expect(await screen.findByText(/server echoed the matching project context digest/i)).toBeInTheDocument()
  expect(api.assess).toHaveBeenCalledWith('request', 'test-token', { context, contextSha256: expectedDigest })
  expect(onResult).toHaveBeenLastCalledWith(expect.objectContaining({ projectContextSha256: expectedDigest,
    output: { intent: 'research', extension: { retained: true } } }))
  expect(digest).toHaveBeenCalledTimes(1)
})

test.each([
  ['missing', undefined],
  ['mismatched', 'f'.repeat(64)],
])('does not expose a %s-digest response to explicit feedback', async (_kind, responseDigest) => {
  const api = client()
  const onResult = vi.fn()
  vi.stubGlobal('crypto', { subtle: { digest: vi.fn(async () => new Uint8Array(32).fill(1).buffer) } })
  vi.mocked(api.assess).mockResolvedValue({ status: 'ok', engine: { id: 'owned-test' }, elapsedMs: 4,
    ...(responseDigest ? { projectContextSha256: responseDigest } : {}), output: { intent: 'research' } })
  render(<OwnedAssessmentPanel prompt="request" assessment={deterministic} client={api} onResult={onResult} />)
  await connect()
  fireEvent.change(screen.getByLabelText('Optional project context (owner-supplied, unverified)'), { target: { value: 'private context' } })
  fireEvent.click(screen.getByRole('button', { name: 'Run owned model' }))
  expect(await screen.findByRole('alert')).toHaveTextContent(responseDigest ? 'did not match' : 'did not include')
  expect(screen.getByText('research')).toBeInTheDocument()
  expect(onResult).toHaveBeenLastCalledWith(null)
})

test('counts exact JSON UTF-8 bytes and does not send an oversized context request', async () => {
  const api = client()
  vi.mocked(api.status).mockResolvedValue({ enabled: true, status: 'ready', engine: { id: 'owned-test' },
    limits: { maxRequestBytes: 1000, maxHttpBodyBytes: 150, maxContextBytes: 5000, maxGeneratedBytes: 512, timeoutMs: 3000, maxConcurrent: 1 } })
  render(<OwnedAssessmentPanel prompt="request" assessment={deterministic} client={api} />)
  await connect()
  fireEvent.change(screen.getByLabelText('Optional project context (owner-supplied, unverified)'), { target: { value: 'é'.repeat(100) } })
  expect(screen.getByRole('alert')).toHaveTextContent('including the context digest')
  expect(screen.getByRole('button', { name: 'Run owned model' })).toBeDisabled()
  expect(api.assess).not.toHaveBeenCalled()
})

test('fails visibly without sending when context hashing is unavailable', async () => {
  const api = client()
  vi.stubGlobal('crypto', undefined)
  render(<OwnedAssessmentPanel prompt="request" assessment={deterministic} client={api} />)
  await connect()
  fireEvent.change(screen.getByLabelText('Optional project context (owner-supplied, unverified)'), { target: { value: 'private context' } })
  fireEvent.click(screen.getByRole('button', { name: 'Run owned model' }))
  expect(await screen.findByRole('alert')).toHaveTextContent(/secure local browser context/)
  expect(api.assess).not.toHaveBeenCalled()
})

test('discards a late context hash after the reviewed context changes', async () => {
  const api = client()
  let resolveDigest!: (value: ArrayBuffer) => void
  const digest = vi.fn((_algorithm: string, _data: BufferSource) => new Promise<ArrayBuffer>((resolve) => { resolveDigest = resolve }))
  vi.stubGlobal('crypto', { subtle: { digest } })
  const onResult = vi.fn()
  render(<OwnedAssessmentPanel prompt="request" assessment={deterministic} client={api} onResult={onResult} />)
  await connect()
  const contextInput = screen.getByLabelText('Optional project context (owner-supplied, unverified)')
  fireEvent.change(contextInput, { target: { value: 'first context' } })
  fireEvent.click(screen.getByRole('button', { name: 'Run owned model' }))
  await waitFor(() => expect(digest).toHaveBeenCalledTimes(1))
  fireEvent.change(contextInput, { target: { value: 'second context' } })
  resolveDigest(new Uint8Array(32).fill(1).buffer)
  await waitFor(() => expect(screen.getByRole('button', { name: 'Run owned model' })).not.toBeDisabled())
  expect(api.assess).not.toHaveBeenCalled()
  expect(onResult).toHaveBeenLastCalledWith(null)
})

test('discards a late API response after the reviewed context changes', async () => {
  const api = client()
  const digestHex = '02'.repeat(32)
  vi.stubGlobal('crypto', { subtle: { digest: vi.fn(async () => new Uint8Array(32).fill(2).buffer) } })
  let resolveResult!: (value: OwnedResult) => void
  vi.mocked(api.assess).mockImplementation(() => new Promise((resolve) => { resolveResult = resolve }))
  const onResult = vi.fn()
  render(<OwnedAssessmentPanel prompt="request" assessment={deterministic} client={api} onResult={onResult} />)
  await connect()
  const contextInput = screen.getByLabelText('Optional project context (owner-supplied, unverified)')
  fireEvent.change(contextInput, { target: { value: 'first context' } })
  fireEvent.click(screen.getByRole('button', { name: 'Run owned model' }))
  await waitFor(() => expect(api.assess).toHaveBeenCalledTimes(1))
  fireEvent.change(contextInput, { target: { value: 'second context' } })
  resolveResult({ status: 'ok', engine: { id: 'owned-test' }, elapsedMs: 3,
    projectContextSha256: digestHex, output: { intent: 'stale context result' } })
  await waitFor(() => expect(screen.getByRole('button', { name: 'Run owned model' })).not.toBeDisabled())
  expect(screen.queryByText('stale context result')).not.toBeInTheDocument()
  expect(onResult).toHaveBeenLastCalledWith(null)
})

test('does not persist supplied context in browser storage', async () => {
  const api = client()
  const setItem = vi.spyOn(Storage.prototype, 'setItem')
  render(<OwnedAssessmentPanel prompt="request" assessment={deterministic} client={api} />)
  await connect()
  fireEvent.change(screen.getByLabelText('Optional project context (owner-supplied, unverified)'), { target: { value: 'private context' } })
  expect(setItem).not.toHaveBeenCalled()
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

test('displays the model reasons array and a separate reason extension without renaming either', async () => {
  const api = client()
  vi.mocked(api.assess).mockResolvedValue({ status: 'ok', engine: { id: 'owned-test' }, elapsedMs: 3, output: { intent: 'build', suggested_models: [], agent_count: 0, reasons: ['first reason', 'second reason'], reason: 'independent field' } })
  render(<OwnedAssessmentPanel prompt="owner request" assessment={deterministic} client={api} />)
  await connect()
  fireEvent.click(screen.getByRole('button', { name: 'Run owned model' }))
  expect(await screen.findByText('first reason')).toBeInTheDocument()
  expect(screen.getByText('second reason')).toBeInTheDocument()
  expect(screen.getByText('independent field')).toBeInTheDocument()
  expect(screen.getByText('No models suggested')).toBeInTheDocument()
  expect(screen.getByText('No agents suggested')).toBeInTheDocument()
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
