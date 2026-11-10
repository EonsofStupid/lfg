import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { waitFor } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'

const root = resolve(process.cwd(), '../../ml/lfg-encoder/node-review-web')
const markup = readFileSync(resolve(root, 'index.html'), 'utf8')
const script = readFileSync(resolve(root, 'app.js'), 'utf8')
afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks(); document.body.replaceChildren() })

async function openReview(enabled) {
  document.documentElement.innerHTML = markup
  window.location.hash = '#fixture-token'
  const item = {itemSha256: 'one', text: 'request', node: 'operation', inputSha256: 'input',
    sourceFileSha256: 'source', journalTailSha256: 'tail', observationSequence: 1, source: {status: 'not_configured'}}
  vi.stubGlobal('fetch', vi.fn(async url => {
    if (String(url) === '/api/status') return {ok: true, json: async () => ({decisions: {pending: 1, approved: 0, rejected: 0}, reviewer: 'reviewer', sourceContextEnabled: enabled})}
    if (String(url).startsWith('/api/item')) return {ok: true, json: async () => ({item, pending: 1, guideSha256: 'g', guideStatus: 'draft', guideNotice: 'draft', guide: {question: 'Which choice?', options: {}}})}
    if (String(url) === '/api/source-context') return {ok: true, json: async () => ({contextScope: 'prior_user_messages_in_session', truncated: true, messages: [{source_line: 4, text: '<script>private context</script>'}]})}
    throw new Error('unexpected request')
  }))
  window.eval(script)
  await waitFor(() => expect(document.querySelector('#review').hidden).toBe(false))
}

test('shows verified context only on request and clears it when moving to another item', async () => {
  await openReview(true)
  expect(vi.mocked(fetch).mock.calls.some(([url]) => String(url) === '/api/source-context')).toBe(false)
  document.querySelector('#show-context').click()
  await waitFor(() => expect(document.querySelector('#source-context').textContent).toContain('<script>private context</script>'))
  expect(document.querySelector('#source-context script')).toBeNull()
  expect(document.querySelector('#source-context').textContent).toContain('omitted by configured limits')
  document.querySelector('#skip').click()
  expect(document.querySelector('#source-context').textContent).toBe('')
})

test('hides context control when the server did not opt in', async () => {
  await openReview(false)
  expect(document.querySelector('#show-context').hidden).toBe(true)
})
