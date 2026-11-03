import { afterEach, expect, test, vi } from 'vitest'
import { ownedAssessmentClient, ownedAssessmentBody, sha256Utf8 } from './ownedAssessmentApi'

afterEach(() => { vi.unstubAllGlobals() })

test('sends the token in a header and the request text only in the body', async () => {
  const fetcher = vi.fn(async (_path: string, _init?: RequestInit) => new Response(JSON.stringify({ status: 'timeout', elapsedMs: 10, engine: null })))
  vi.stubGlobal('fetch', fetcher)
  await ownedAssessmentClient.assess('private request', 'local-token')
  expect(fetcher.mock.calls[0][0]).toBe('/api/lfg/assess/owned')
  expect(fetcher.mock.calls[0][1]?.headers).toMatchObject({ 'X-LFG-Session-Token': 'local-token' })
  expect(JSON.parse(String(fetcher.mock.calls[0][1]?.body))).toEqual({ text: 'private request' })
})

test('sends exact context bytes with their digest and preserves response extensions', async () => {
  const context = '  Projet\n🦊  '
  const digest = 'a'.repeat(64)
  const extension = { observed: true }
  const fetcher = vi.fn(async (_path: string, _init?: RequestInit) => new Response(JSON.stringify({
    status: 'ok', elapsedMs: 2, engine: null, projectContextSha256: digest, extension,
  })))
  vi.stubGlobal('fetch', fetcher)
  const result = await ownedAssessmentClient.assess('request', 'local-token', {
    context, contextSha256: digest, requestExtension: extension, text: 'extension prompt must not replace request',
  })
  expect(JSON.parse(String(fetcher.mock.calls[0][1]?.body))).toEqual({ text: 'request', context, contextSha256: digest, requestExtension: extension })
  expect(result.projectContextSha256).toBe(digest)
  expect(result.extension).toEqual(extension)
})

test('legacy request payload remains exactly text-only and SHA helper emits lowercase UTF-8 digest', async () => {
  expect(ownedAssessmentBody('request')).toEqual({ text: 'request' })
  const digest = new Uint8Array(32).fill(0xab).buffer
  const subtle = { digest: vi.fn(async (_algorithm: string, data: BufferSource) => {
    expect(Array.from(new Uint8Array(data as ArrayBuffer))).toEqual(Array.from(new TextEncoder().encode('  é\n🦊  ')))
    return digest
  }) }
  vi.stubGlobal('crypto', { subtle })
  expect(await sha256Utf8('  é\n🦊  ')).toBe('ab'.repeat(32))
  expect(subtle.digest).toHaveBeenCalledTimes(1)
  expect(subtle.digest.mock.calls[0][0]).toBe('SHA-256')
})

test('refuses context hashing when Web Crypto is unavailable', async () => {
  vi.stubGlobal('crypto', undefined)
  await expect(sha256Utf8('private context')).rejects.toThrow(/secure local browser context/)
})
