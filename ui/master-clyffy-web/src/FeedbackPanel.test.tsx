import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, afterEach, expect, test, vi } from 'vitest'
import { FeedbackPanel, feedbackFromAssessment, feedbackFromOwned } from './FeedbackPanel'
import type { FeedbackClient, FeedbackRecord, FeedbackState } from './feedbackApi'
import type { Assessment } from './types'

const assessment = {
  sessionId: 'session-a', revision: 1, configDigest: 'config',
  prompt: { tasks: ['private prompt text'] },
  intent: { primary: 'build', all: [{ matched: ['private prompt text'] }] },
  complexity: { bracket: 'medium', score: 2 }, agents: { total: 3 },
  models: { role: 'standard', effort: 'medium', suggested: { model: 'a-model' } }, rules: ['intent:build'],
} as unknown as Assessment

beforeEach(() => { vi.stubGlobal('crypto', { subtle: { digest: vi.fn(async () => new Uint8Array(32).buffer) } }) })
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals() })

function makeClient() {
  let history: FeedbackRecord[] = []
  const record = (state: FeedbackState, revision: number): FeedbackRecord => ({ ...state, feedbackId: 'record-1', revision, createdAt: '2026-01-01', updatedAt: '2026-01-01', parentRevision: revision === 1 ? null : revision - 1, action: 'review' })
  const client: FeedbackClient = {
    status: vi.fn(async () => ({ enabled: true, available: true })),
    list: vi.fn(async () => ({ records: history.length ? [history[history.length - 1]] : [], nextCursor: null, hasMore: false })),
    read: vi.fn(async () => ({ record: history[history.length - 1], history: [...history], historyNextBefore: null, historyHasMore: false })),
    create: vi.fn(async (state) => {
      const entry = record({ ...state, extraField: { retained: true }, ownerCorrection: { note: '', custom: 'keep' } }, 1)
      history = [entry]; return entry
    }),
    revise: vi.fn(async (_id, _version, state) => {
      const entry = record(state, history.length + 1); history.push(entry); return entry
    }),
    rollback: vi.fn(async (_id, _version, restore) => {
      const old = history.find((entry) => entry.revision === restore)!
      const entry = record(old, history.length + 1); history.push(entry); return entry
    }),
  }
  return client
}

async function connect() {
  fireEvent.change(screen.getByLabelText('Local feedback token'), { target: { value: 'token' } })
  fireEvent.click(screen.getByRole('button', { name: 'Connect feedback' }))
  await screen.findByRole('button', { name: 'Save current assessment' })
}

test('captures assessment output without implicitly retaining prompt content', async () => {
  const state = await feedbackFromAssessment(assessment, 'private prompt text')
  expect(state.originalRecommendation).toMatchObject({ intent: 'build', complexity: 'medium', agentCount: 3, model: 'a-model' })
  expect(JSON.stringify(state)).not.toContain('private prompt text')
  expect(state.provenance.sourceRef).toMatch(/^sha256:[0-9a-f]{64}$/)
  expect(state.outcome).toEqual({ status: 'unknown' })
})

test('an explicitly selected owned output keeps its provenance, extensions and failure state', async () => {
  const state = await feedbackFromAssessment(assessment, 'private prompt text')
  const saved = feedbackFromOwned(state, { status: 'truncated', elapsedMs: 4, engine: { id: 'owned-test', modelSha256: 'model-hash', modelManifestSha256: 'model-manifest', trainingManifestSha256: 'training-manifest', framingSha256: 'framing-hash', runtimeConfigSha256: 'runtime-hash', serviceConfigSha256: 'service-hash' }, stopReason: 'byte_limit', output: { intent: 'research', extra: { keep: true } }, outputText: 'partial output' })
  expect(saved.provenance).toMatchObject({ engineId: 'owned-test', modelDigest: 'model-hash', modelManifestSha256: 'model-manifest', trainingManifestSha256: 'training-manifest', framingSha256: 'framing-hash', runtimeConfigSha256: 'runtime-hash', serviceConfigSha256: 'service-hash', sourceRef: expect.stringMatching(/^sha256:/) })
  expect(saved.originalRecommendation).toMatchObject({ intent: 'research', modelOutput: { intent: 'research', extra: { keep: true } }, generationStatus: 'truncated', rawOutput: 'partial output' })
  expect(saved.outcome).toEqual({ status: 'unknown' })
})

test('copies only the server context digest into explicit owned feedback', async () => {
  const digest = 'c'.repeat(64)
  const state = await feedbackFromAssessment(assessment, 'private prompt text')
  const saved = feedbackFromOwned(state, {
    status: 'ok', elapsedMs: 2, engine: { id: 'owned-test' }, projectContextSha256: digest,
    output: { intent: 'research' }, outputText: 'Proposal mentions private project context text.',
  })
  expect(saved.provenance.projectContextSha256).toBe(digest)
  expect(JSON.stringify(saved.provenance)).not.toContain('private prompt text')
  expect(JSON.stringify(saved.provenance)).not.toContain('project context text')
  expect(saved.originalRecommendation.rawOutput).toContain('private project context text')
})

test('saves owned model output only after the user selects it', async () => {
  const api = makeClient()
  const contextDigest = 'd'.repeat(64)
  const owned = { status: 'truncated' as const, engine: { id: 'owned-test', modelSha256: 'model-hash' }, elapsedMs: 5, outputText: 'partial model output', projectContextSha256: contextDigest }
  render(<FeedbackPanel prompt="private prompt text" assessment={assessment} ownedResult={owned} client={api} />)
  await connect()
  fireEvent.change(screen.getByLabelText('Assessment source'), { target: { value: 'owned' } })
  expect(screen.getByText(/actual output, including any partial text/)).toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: 'Save current assessment' }))
  await screen.findByText('Assessment saved for review.')
  expect(api.create).toHaveBeenCalledWith(expect.objectContaining({ provenance: expect.objectContaining({ engineId: 'owned-test', modelDigest: 'model-hash', projectContextSha256: contextDigest }), originalRecommendation: expect.objectContaining({ generationStatus: 'truncated', rawOutput: 'partial model output' }) }), 'token')
})

test('can save an owned result when the rules assessment is unavailable', async () => {
  const api = makeClient()
  const owned = { status: 'truncated' as const, engine: { id: 'owned-test' }, elapsedMs: 5, outputText: 'partial model output' }
  render(<FeedbackPanel prompt="private prompt text" assessmentRef="session:3" assessment={null} ownedResult={owned} client={api} />)
  await connect()
  fireEvent.change(screen.getByLabelText('Assessment source'), { target: { value: 'owned' } })
  fireEvent.click(screen.getByRole('button', { name: 'Save current assessment' }))
  await screen.findByText('Assessment saved for review.')
  expect(api.create).toHaveBeenCalledWith(expect.objectContaining({ assessmentRef: 'session:3', provenance: expect.objectContaining({ engineId: 'owned-test' }) }), 'token')
})

test('reopening owned feedback preserves its framing and training lineage', async () => {
  const api = makeClient()
  const contextDigest = 'e'.repeat(64)
  const owned = { status: 'truncated' as const, engine: { id: 'owned-test', modelSha256: 'model-hash', trainingManifestSha256: 'training-hash', framingSha256: 'frame-hash', runtimeConfigSha256: 'runtime-hash', serviceConfigSha256: 'service-hash' }, elapsedMs: 5, outputText: 'partial', projectContextSha256: contextDigest }
  const view = render(<FeedbackPanel prompt="private prompt text" assessment={assessment} ownedResult={owned} client={api} />)
  await connect()
  fireEvent.change(screen.getByLabelText('Assessment source'), { target: { value: 'owned' } })
  fireEvent.click(screen.getByRole('button', { name: 'Save current assessment' }))
  await screen.findByText('Assessment saved for review.')
  fireEvent.click(screen.getByRole('button', { name: 'Save review' }))
  await screen.findByText('Review saved as a new revision.')
  view.unmount()
  render(<FeedbackPanel prompt="private prompt text" assessment={assessment} client={api} />)
  await connect()
  fireEvent.click(screen.getByRole('button', { name: 'Load saved assessments' }))
  fireEvent.change(await screen.findByLabelText('Saved assessment'), { target: { value: 'record-1' } })
  fireEvent.click(await screen.findByText('View provenance'))
  expect(screen.getByText(/Training manifest: training-hash/)).toHaveTextContent('Framing: frame-hash')
  expect(screen.getByText(/Training manifest: training-hash/)).toHaveTextContent('Runtime config: runtime-hash')
  expect(screen.getByText(/Project context digest:/)).toHaveTextContent(contextDigest)
})

test('creates, reviews, reopens and restores feedback as separate revisions', async () => {
  const client = makeClient()
  const view = render(<FeedbackPanel prompt="private prompt text" assessment={assessment} client={client} />)
  await connect()
  fireEvent.click(screen.getByRole('button', { name: 'Save current assessment' }))
  expect(await screen.findByText('Assessment saved for review.')).toBeInTheDocument()
  expect(screen.getAllByText('Outcome: unknown').length).toBeGreaterThan(0)
  fireEvent.change(screen.getByLabelText('Owner correction'), { target: { value: 'Use a different model' } })
  fireEvent.change(screen.getByLabelText('Corrected model'), { target: { value: 'owner-choice' } })
  fireEvent.change(screen.getByLabelText('Corrected agent count'), { target: { value: '2' } })
  fireEvent.click(screen.getByLabelText('I observed the outcome'))
  fireEvent.change(screen.getByLabelText('Observed outcome'), { target: { value: 'Tests passed' } })
  fireEvent.change(screen.getByLabelText('Confirmed by'), { target: { value: 'Local reviewer' } })
  fireEvent.click(screen.getByRole('button', { name: 'Save review' }))
  expect(await screen.findByText('Review saved as a new revision.')).toBeInTheDocument()
  expect(client.revise).toHaveBeenCalledWith('record-1', 1, expect.objectContaining({
    extraField: { retained: true }, ownerCorrection: { note: 'Use a different model', custom: 'keep', model: 'owner-choice', agentCount: 2 },
    outcome: expect.objectContaining({ status: 'confirmed', observation: 'Tests passed', confirmer: 'Local reviewer' }),
  }), 'token')
  view.unmount()
  render(<FeedbackPanel prompt="private prompt text" assessment={assessment} client={client} />)
  await connect()
  fireEvent.click(screen.getByRole('button', { name: 'Load saved assessments' }))
  const saved = await screen.findByLabelText('Saved assessment')
  fireEvent.change(saved, { target: { value: 'record-1' } })
  expect(await screen.findByDisplayValue('Use a different model')).toBeInTheDocument()
  fireEvent.change(screen.getByLabelText('Earlier revision'), { target: { value: '1' } })
  fireEvent.click(screen.getByText('View selected revision'))
  expect(screen.getByText(/Revision 1 · review/)).toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: 'Restore selected revision' }))
  expect(await screen.findByText('Earlier state restored as a new revision.')).toBeInTheDocument()
  expect(client.rollback).toHaveBeenCalledWith('record-1', 2, 1, 'token')
  expect(screen.getAllByText('Outcome: unknown').length).toBeGreaterThan(0)
})

test('shows unavailable state and reports failed writes without claiming success', async () => {
  const unavailable = makeClient()
  vi.mocked(unavailable.status).mockResolvedValue({ enabled: false, available: false })
  const view = render(<FeedbackPanel prompt="private prompt text" assessment={assessment} client={unavailable} />)
  fireEvent.change(screen.getByLabelText('Local feedback token'), { target: { value: 'token' } })
  fireEvent.click(screen.getByRole('button', { name: 'Connect feedback' }))
  await waitFor(() => expect(screen.getByText(/Local feedback is unavailable/)).toBeInTheDocument())
  expect(screen.queryByRole('button', { name: 'Save current assessment' })).not.toBeInTheDocument()
  view.unmount()
  const failing = makeClient()
  vi.mocked(failing.create).mockRejectedValue(new Error('Store is full'))
  render(<FeedbackPanel prompt="private prompt text" assessment={assessment} client={failing} />)
  await connect()
  fireEvent.click(screen.getByRole('button', { name: 'Save current assessment' }))
  expect(await screen.findByRole('alert')).toHaveTextContent('Store is full')
  expect(screen.queryByText('Assessment saved for review.')).not.toBeInTheDocument()
})

test('loads additional saved assessments with the server cursor', async () => {
  const client = makeClient()
  const state = await feedbackFromAssessment(assessment, 'private prompt text')
  const first: FeedbackRecord = { ...state, feedbackId: 'a', revision: 1, createdAt: 'now', updatedAt: 'now', parentRevision: null, action: 'create' }
  const second: FeedbackRecord = { ...first, feedbackId: 'b' }
  vi.mocked(client.list).mockResolvedValueOnce({ records: [first], nextCursor: 'a', hasMore: true })
    .mockResolvedValueOnce({ records: [second], nextCursor: null, hasMore: false })
  render(<FeedbackPanel prompt="private prompt text" assessment={assessment} client={client} />)
  await connect()
  fireEvent.click(screen.getByRole('button', { name: 'Load saved assessments' }))
  fireEvent.click(await screen.findByRole('button', { name: 'Load more assessments' }))
  await waitFor(() => expect(screen.getByRole('option', { name: 'b · revision 1' })).toBeInTheDocument())
  expect(client.list).toHaveBeenLastCalledWith('token', 'a')
})

test('loads older revision history without replacing the current record', async () => {
  const client = makeClient()
  const state = await feedbackFromAssessment(assessment, 'private prompt text')
  const first = await client.create(state, 'token')
  const second = await client.revise(first.feedbackId, 1, state, 'token')
  vi.mocked(client.read).mockResolvedValueOnce({ record: second, history: [second], historyNextBefore: 2, historyHasMore: true })
    .mockResolvedValueOnce({ record: second, history: [first], historyNextBefore: null, historyHasMore: false })
  render(<FeedbackPanel prompt="private prompt text" assessment={assessment} client={client} />)
  await connect()
  fireEvent.click(screen.getByRole('button', { name: 'Load saved assessments' }))
  fireEvent.change(await screen.findByLabelText('Saved assessment'), { target: { value: 'record-1' } })
  fireEvent.click(await screen.findByRole('button', { name: 'Load older revisions' }))
  await waitFor(() => expect(screen.getByRole('option', { name: 'Revision 1' })).toBeInTheDocument())
  expect(client.read).toHaveBeenLastCalledWith('record-1', 'token', 2)
})
