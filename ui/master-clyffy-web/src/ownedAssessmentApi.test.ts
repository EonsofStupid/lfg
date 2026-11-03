import { afterEach, expect, test, vi } from 'vitest'
import { ownedAssessmentClient } from './ownedAssessmentApi'

afterEach(() => { vi.unstubAllGlobals() })

test('sends the token in a header and the request text only in the body', async () => {
  const fetcher = vi.fn(async (_path: string, _init?: RequestInit) => new Response(JSON.stringify({ status: 'timeout', elapsedMs: 10, engine: null })))
  vi.stubGlobal('fetch', fetcher)
  await ownedAssessmentClient.assess('private request', 'local-token')
  expect(fetcher.mock.calls[0][0]).toBe('/api/lfg/assess/owned')
  expect(fetcher.mock.calls[0][1]?.headers).toMatchObject({ 'X-LFG-Session-Token': 'local-token' })
  expect(JSON.parse(String(fetcher.mock.calls[0][1]?.body))).toEqual({ text: 'private request' })
})
