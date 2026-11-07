import { afterEach, expect, test, vi } from 'vitest'
import { feedbackClient, type FeedbackState } from './feedbackApi'

afterEach(() => { vi.unstubAllGlobals() })

test('sends the session token in a header and preserves history extension fields', async () => {
  const state: FeedbackState = { assessmentRef: 's:1', provenance: { engineId: 'test' }, originalRecommendation: {}, ownerCorrection: null, outcome: { status: 'unknown' }, custom: { keep: true } }
  const record = { ...state, feedbackId: 'a/b', revision: 1, createdAt: 'now', updatedAt: 'now', parentRevision: null, action: 'create' }
  const fetcher = vi.fn(async (_path: string, init?: RequestInit) => new Response(JSON.stringify(init?.method === 'POST' ? { record } : {
    record, history: [{ revision: 1, parentRevision: null, action: 'create', createdAt: 'now', state }], historyNextBefore: null, historyHasMore: false,
  })))
  vi.stubGlobal('fetch', fetcher)
  await feedbackClient.revise('a/b', 1, state, 'secret-token')
  expect(fetcher.mock.calls[0][0]).toBe('/api/lfg/feedback/a%2Fb/revisions')
  expect(fetcher.mock.calls[0][0]).not.toContain('secret-token')
  expect(fetcher.mock.calls[0][1]?.headers).toMatchObject({ 'X-LFG-Session-Token': 'secret-token' })
  const result = await feedbackClient.read('a/b', 'secret-token')
  expect(fetcher.mock.calls[1][1]?.headers).toMatchObject({ 'X-LFG-Session-Token': 'secret-token' })
  expect(result.history[0].custom).toEqual({ keep: true })
})

test('surfaces conflict errors without converting them to a success', async () => {
  vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({ error: { code: 'conflict', message: 'Revision changed' } }), { status: 409 })))
  await expect(feedbackClient.rollback('record', 1, 1, 'token')).rejects.toThrow('Revision changed')
})
