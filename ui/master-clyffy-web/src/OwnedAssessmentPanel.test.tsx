import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import { OwnedAssessmentPanel } from './OwnedAssessmentPanel'
import type { OwnedAssessmentClient, OwnedResult, OwnedStatus } from './ownedAssessmentApi'
import type { Assessment } from './types'

const deterministic = { intent: { primary: 'build' }, complexity: { bracket: 'moderate' }, agents: { total: 2 } } as Assessment
const client = (): OwnedAssessmentClient => ({
  status: vi.fn(async () => ({ enabled: true, status: 'ready' as const, engine: { id: 'owned-test', modelSha256: 'a'.repeat(64) }, limits: { maxRequestBytes: 1000, maxHttpBodyBytes: 10000, maxContextBytes: 5000, maxGeneratedBytes: 512, timeoutMs: 3000, maxConcurrent: 1 } })),
  assess: vi.fn(async () => ({ status: 'ok' as const, engine: { id: 'owned-test' }, elapsedMs: 120, stopReason: 'stop_sequence', measurement: { schemaVersion: 'lfg.owned-run-measurement.v1' as const, status: 'ok', engineId: 'owned-test', stopReason: 'stop_sequence', handlerElapsedMs: 120, outputTextUtf8Bytes: 51, workerStarted: true, workerCompletedSuccessfully: true }, output: { intent: 'research', complexity: 'simple', suggested_models: ['local-model'], agent_count: 1, reason: 'One task' } })),
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
  expect(screen.getByText(/Observed handler wall time: 120 ms \(may include child startup/)).toBeInTheDocument()
  expect(screen.getByRole('region', { name: 'Current input comparison' })).toHaveTextContent(/side-by-side proposal comparison/)
  expect(screen.getByRole('region', { name: 'Current input comparison' })).toHaveTextContent(/Rules: build intent/)
  expect(api.assess).toHaveBeenCalledWith('owner request', 'test-token')
})

test('publishes connected engine identity to the Inspector scorecard and clears it on disconnect', async () => {
  const api = client()
  const onStatus = vi.fn()
  const connected = await api.status('test-token')
  render(<OwnedAssessmentPanel prompt="owner request" assessment={deterministic} client={api} onStatus={onStatus} />)
  await connect()
  expect(onStatus).toHaveBeenLastCalledWith(connected)
  fireEvent.click(screen.getByRole('button', { name: 'Disconnect' }))
  expect(onStatus).toHaveBeenLastCalledWith(null)
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

test('quarantines an unsolicited context digest when no context was submitted', async () => {
  const api = client()
  const onResult = vi.fn()
  vi.mocked(api.assess).mockResolvedValue({ status: 'ok', engine: { id: 'owned-test' }, elapsedMs: 3,
    projectContextSha256: 'f'.repeat(64), output: { intent: 'research' } })
  render(<OwnedAssessmentPanel prompt="request" assessment={deterministic} client={api} onResult={onResult} />)
  await connect()
  fireEvent.click(screen.getByRole('button', { name: 'Run owned model' }))
  expect(await screen.findByText('research')).toBeInTheDocument()
  expect(screen.queryByText(/matching project context digest/)).not.toBeInTheDocument()
  const acceptedResult = onResult.mock.calls.at(-1)?.[0]
  expect(acceptedResult).not.toHaveProperty('projectContextSha256')
  expect(api.assess).toHaveBeenCalledWith('request', 'test-token')
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

test('reports an unknown owned-route body cap instead of borrowing another status limit', async () => {
  const api = client()
  vi.mocked(api.status).mockResolvedValue({ enabled: true, status: 'ready', engine: { id: 'owned-test' },
    limits: { maxRequestBytes: 1000, maxContextBytes: 5000, maxGeneratedBytes: 512, timeoutMs: 3000, maxConcurrent: 1 } })
  render(<OwnedAssessmentPanel prompt="request" assessment={deterministic} client={api} />)
  await connect()
  expect(screen.getByText(/The owned route HTTP body limit is not advertised/)).toBeInTheDocument()
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

test('ignores an older connection response after a newer token connects', async () => {
  const api = client()
  let resolveFirst!: (value: OwnedStatus) => void
  let resolveSecond!: (value: OwnedStatus) => void
  vi.mocked(api.status)
    .mockImplementationOnce(() => new Promise((resolve) => { resolveFirst = resolve }))
    .mockImplementationOnce(() => new Promise((resolve) => { resolveSecond = resolve }))
  render(<OwnedAssessmentPanel prompt="request" assessment={deterministic} client={api} />)
  const token = screen.getByLabelText('Owned model token')
  fireEvent.change(token, { target: { value: 'first-token' } })
  fireEvent.click(screen.getByRole('button', { name: 'Connect owned model' }))
  const context = screen.getByLabelText('Optional project context (owner-supplied, unverified)')
  fireEvent.change(context, { target: { value: 'editing context does not unlock connection' } })
  expect(screen.getByRole('button', { name: 'Connecting…' })).toBeDisabled()
  fireEvent.change(token, { target: { value: 'second-token' } })
  fireEvent.click(screen.getByRole('button', { name: 'Connect owned model' }))
  const ready: OwnedStatus = { enabled: true, status: 'ready', engine: { id: 'second-engine' }, limits: { maxRequestBytes: 1000, maxGeneratedBytes: 512, timeoutMs: 3000, maxConcurrent: 1 } }
  resolveSecond(ready)
  expect(await screen.findByText(/Model status: ready/)).toBeInTheDocument()
  resolveFirst({ enabled: false, status: 'unavailable', engine: null, reason: 'stale-first-token' })
  await waitFor(() => expect(screen.queryByText(/stale-first-token/)).not.toBeInTheDocument())
  fireEvent.change(context, { target: { value: '' } })
  fireEvent.click(screen.getByRole('button', { name: 'Run owned model' }))
  expect(await screen.findByText('research')).toBeInTheDocument()
  expect(api.assess).toHaveBeenCalledWith('request', 'second-token')
})

test('prompt edits do not strand a deferred connection in the connecting state', async () => {
  const api = client()
  let resolveStatus!: (value: OwnedStatus) => void
  vi.mocked(api.status).mockImplementation(() => new Promise((resolve) => { resolveStatus = resolve }))
  const view = render(<OwnedAssessmentPanel prompt="first request" assessment={deterministic} client={api} />)
  fireEvent.change(screen.getByLabelText('Owned model token'), { target: { value: 'current-token' } })
  fireEvent.click(screen.getByRole('button', { name: 'Connect owned model' }))
  view.rerender(<OwnedAssessmentPanel prompt="edited request" assessment={deterministic} client={api} />)
  expect(screen.getByRole('button', { name: 'Connecting…' })).toBeDisabled()
  resolveStatus({ enabled: true, status: 'ready', engine: { id: 'current-engine' }, limits: { maxRequestBytes: 1000, maxGeneratedBytes: 512, timeoutMs: 3000, maxConcurrent: 1 } })
  expect(await screen.findByText(/Model status: ready/)).toBeInTheDocument()
  expect(screen.getByRole('button', { name: 'Run owned model' })).not.toBeDisabled()
})

test('unmount invalidates a deferred connection response', async () => {
  const api = client()
  let resolveStatus!: (value: OwnedStatus) => void
  vi.mocked(api.status).mockImplementation(() => new Promise((resolve) => { resolveStatus = resolve }))
  const onResult = vi.fn()
  const view = render(<OwnedAssessmentPanel prompt="request" assessment={deterministic} client={api} onResult={onResult} />)
  fireEvent.change(screen.getByLabelText('Owned model token'), { target: { value: 'late-token' } })
  fireEvent.click(screen.getByRole('button', { name: 'Connect owned model' }))
  view.unmount()
  resolveStatus({ enabled: true, status: 'ready', engine: { id: 'late-engine' }, limits: { maxRequestBytes: 1000, maxGeneratedBytes: 512, timeoutMs: 3000, maxConcurrent: 1 } })
  await waitFor(() => expect(api.status).toHaveBeenCalledTimes(1))
  expect(onResult).not.toHaveBeenCalledWith(expect.objectContaining({ engine: { id: 'late-engine' } }))
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

test('suggested context requires explicit build and append, preserving edited draft and manual context', async () => {
  const api = client()
  const knowledge = vi.fn(async () => ({ state: 'partial', refreshedAt: null, models: [{ subscriptionId: 'declared-seat', model: 'cached-model', origin: 'provider-list', available: true, description: 'not provider-proven' }] }))
  const store = vi.spyOn(Storage.prototype, 'setItem')
  render(<OwnedAssessmentPanel prompt="request" assessment={deterministic} client={api} declaredSubscriptions={['declared-seat']} loadKnowledge={knowledge} />)
  await connect()
  const context = screen.getByLabelText('Optional project context (owner-supplied, unverified)')
  fireEvent.change(context, { target: { value: 'My manual context' } })
  expect(knowledge).not.toHaveBeenCalled()
  fireEvent.click(screen.getByRole('button', { name: 'Build suggested context' }))
  const draft = await screen.findByLabelText('Suggested context draft') as HTMLTextAreaElement
  expect(draft.value).toContain('cached-model')
  expect(draft.value).not.toContain('not provider-proven')
  expect(context).toHaveValue('My manual context')
  expect(api.assess).not.toHaveBeenCalled()
  fireEvent.change(draft, { target: { value: draft.value + '\nMy draft edit' } })
  fireEvent.click(screen.getByRole('button', { name: 'Append draft to project context' }))
  expect((context as HTMLTextAreaElement).value).toContain('My manual context\n\n')
  expect((context as HTMLTextAreaElement).value).toContain('My draft edit')
  expect(api.assess).not.toHaveBeenCalled()
  expect(store).not.toHaveBeenCalled()
})

test('declaration changes preserve a stale edited draft and ignore a late knowledge response', async () => {
  const api = client()
  let resolveKnowledge!: (value: unknown) => void
  const knowledge = vi.fn(() => new Promise<unknown>(resolve => { resolveKnowledge = resolve }))
  const view = render(<OwnedAssessmentPanel prompt="request" assessment={deterministic} client={api} declaredSubscriptions={['one']} loadKnowledge={knowledge} />)
  await connect()
  fireEvent.click(screen.getByRole('button', { name: 'Build suggested context' }))
  view.rerender(<OwnedAssessmentPanel prompt="request" assessment={deterministic} client={api} declaredSubscriptions={['two']} loadKnowledge={knowledge} />)
  resolveKnowledge({ state: 'ready', refreshedAt: null, models: [{ subscriptionId: 'one', model: 'old', origin: 'provider-list', available: true }] })
  await waitFor(() => expect(screen.getByRole('button', { name: 'Build suggested context' })).not.toBeDisabled())
  expect(screen.queryByLabelText('Suggested context draft')).not.toBeInTheDocument()
})

test('changing declarations makes an applied draft stale and rejects a late model result', async () => {
  const api = client()
  const onResult = vi.fn()
  const knowledge = vi.fn(async () => ({ state: 'ready', refreshedAt: null, models: [{ subscriptionId: 'one', model: 'model', origin: 'provider-list', available: true }] }))
  vi.stubGlobal('crypto', { subtle: { digest: vi.fn(async () => new Uint8Array(32).fill(4).buffer) } })
  let resolveResult!: (result: OwnedResult) => void
  vi.mocked(api.assess).mockImplementation(() => new Promise(resolve => { resolveResult = resolve }))
  const view = render(<OwnedAssessmentPanel prompt="request" assessment={deterministic} client={api} declaredSubscriptions={['one']} loadKnowledge={knowledge} onResult={onResult} />)
  await connect()
  fireEvent.click(screen.getByRole('button', { name: 'Build suggested context' }))
  await screen.findByLabelText('Suggested context draft')
  fireEvent.click(screen.getByRole('button', { name: 'Append draft to project context' }))
  fireEvent.click(screen.getByRole('button', { name: 'Run owned model' }))
  await waitFor(() => expect(api.assess).toHaveBeenCalledTimes(1))
  view.rerender(<OwnedAssessmentPanel prompt="request" assessment={deterministic} client={api} declaredSubscriptions={['two']} loadKnowledge={knowledge} onResult={onResult} />)
  resolveResult({ status: 'ok', elapsedMs: 1, engine: null, projectContextSha256: '04'.repeat(32), output: { intent: 'stale-result' } })
  expect(await screen.findByText(/cached knowledge changed after this context was applied/)).toBeInTheDocument()
  expect(screen.getByRole('button', { name: 'Run owned model' })).toBeDisabled()
  expect(screen.queryByText('stale-result')).not.toBeInTheDocument()
  expect(onResult).toHaveBeenLastCalledWith(null)
  fireEvent.click(screen.getByRole('button', { name: 'Keep as manual context' }))
  expect(screen.getByRole('button', { name: 'Run owned model' })).not.toBeDisabled()
})

test('keeps an edited draft visible but not applicable after declarations change', async () => {
  const api = client()
  const knowledge = vi.fn(async () => ({ state: 'ready', refreshedAt: null, models: [{ subscriptionId: 'one', model: 'model', origin: 'provider-list', available: true }] }))
  const view = render(<OwnedAssessmentPanel prompt="request" assessment={deterministic} client={api} declaredSubscriptions={['one']} loadKnowledge={knowledge} />)
  await connect()
  fireEvent.click(screen.getByRole('button', { name: 'Build suggested context' }))
  const draft = await screen.findByLabelText('Suggested context draft') as HTMLTextAreaElement
  fireEvent.change(draft, { target: { value: draft.value + '\nA user edit' } })
  view.rerender(<OwnedAssessmentPanel prompt="request" assessment={deterministic} client={api} declaredSubscriptions={['two']} loadKnowledge={knowledge} />)
  expect(draft.value).toContain('A user edit')
  expect(screen.getByRole('button', { name: 'Append draft to project context' })).toBeDisabled()
  expect(screen.getByText(/Declarations changed since this draft was built/)).toBeInTheDocument()
})

test('a changed knowledge snapshot invalidates an applied context without overwriting it', async () => {
  const api = client()
  const onResult = vi.fn()
  const knowledge = vi.fn()
    .mockResolvedValueOnce({ state: 'ready', refreshedAt: 'first', models: [{ subscriptionId: 'one', model: 'old-model', origin: 'provider-list', available: true }] })
    .mockResolvedValueOnce({ state: 'ready', refreshedAt: 'second', models: [{ subscriptionId: 'one', model: 'new-model', origin: 'provider-list', available: true }] })
  render(<OwnedAssessmentPanel prompt="request" assessment={deterministic} client={api} declaredSubscriptions={['one']} loadKnowledge={knowledge} onResult={onResult} />)
  await connect()
  fireEvent.click(screen.getByRole('button', { name: 'Build suggested context' }))
  await screen.findByLabelText('Suggested context draft')
  fireEvent.click(screen.getByRole('button', { name: 'Append draft to project context' }))
  const context = screen.getByLabelText('Optional project context (owner-supplied, unverified)') as HTMLTextAreaElement
  const original = context.value
  fireEvent.click(screen.getByRole('button', { name: 'Build suggested context' }))
  const draft = await screen.findByLabelText('Suggested context draft') as HTMLTextAreaElement
  expect(draft.value).toContain('new-model')
  expect(context.value).toBe(original)
  expect(screen.getByRole('button', { name: 'Run owned model' })).toBeDisabled()
  expect(screen.getByRole('button', { name: 'Append draft to project context' })).toBeDisabled()
  expect(onResult).toHaveBeenLastCalledWith(null)
})

test('subscription changes reject a late owned result even when no context draft was applied', async () => {
  const api = client()
  const onResult = vi.fn()
  let resolveResult!: (result: OwnedResult) => void
  vi.mocked(api.assess).mockImplementation(() => new Promise(resolve => { resolveResult = resolve }))
  const view = render(<OwnedAssessmentPanel prompt="request" assessment={deterministic} client={api} declaredSubscriptions={['one']} onResult={onResult} />)
  await connect()
  fireEvent.click(screen.getByRole('button', { name: 'Run owned model' }))
  await waitFor(() => expect(api.assess).toHaveBeenCalledTimes(1))
  view.rerender(<OwnedAssessmentPanel prompt="request" assessment={deterministic} client={api} declaredSubscriptions={['two']} onResult={onResult} />)
  resolveResult({ status: 'ok', elapsedMs: 2, engine: null, output: { intent: 'stale subscription result' } })
  await waitFor(() => expect(onResult).toHaveBeenLastCalledWith(null))
  expect(screen.queryByText('stale subscription result')).not.toBeInTheDocument()
  expect(screen.queryByRole('region', { name: 'Current input comparison' })).not.toBeInTheDocument()
})
