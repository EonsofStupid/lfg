import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { waitFor } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'

afterEach(() => { vi.unstubAllGlobals(); document.body.replaceChildren() })

test('review UI maps Unicode and Windows line endings to source spans before approval', async () => {
  const dir = resolve(process.cwd(), '../../ml/lfg-encoder/review-web')
  const parsed = new DOMParser().parseFromString(readFileSync(resolve(dir, 'index.html'), 'utf8'), 'text/html')
  document.body.innerHTML = parsed.body.innerHTML
  history.replaceState(null, '', '/#local-test-token')
  const post = vi.fn()
  const item = { id: 'fixture', text: 'one 🚀\r\nprivate data', source: {}, project: 'fixture', split: 'train',
    labels: { intents: ['build'], capabilities: [], risks: [] }, labels_pending: false,
    annotation_sha256: 'fixture-digest', pending_changed: false, interrupted_write: false }
  vi.stubGlobal('fetch', vi.fn(async (path, options = {}) => {
    if (path === '/api/decision') { post(JSON.parse(options.body)); return {ok: true, json: async () => ({saved: true})} }
    if (path === '/api/status') return {ok: true, json: async () => ({reviewer: 'reviewer', decisions: {pending: 1, approved: 0, rejected: 0}})}
    return {ok: true, json: async () => ({item, pending: 1, taxonomy: {intents: ['build'], capabilities: ['write'], risks: ['sensitive']}})}
  }))
  vi.stubGlobal('prompt', () => 'APPROVE')
  Function(readFileSync(resolve(dir, 'app.js'), 'utf8'))()
  const request = document.getElementById('request')
  await waitFor(() => expect(request.value).toBe('one 🚀\nprivate data'))
  request.setSelectionRange(7, 14)
  document.getElementById('redact').click()
  expect(request.value).toBe('one 🚀\n[REDACTED] data')
  expect(item.text).toBe('one 🚀\r\nprivate data')
  document.getElementById('redaction-replacement').value = 'X'
  request.setSelectionRange(0, 3)
  document.getElementById('redact').click()
  expect(request.value).toBe('X 🚀\n[REDACTED] data')
  document.getElementById('attest').checked = true
  document.getElementById('approve').click()
  await waitFor(() => expect(post).toHaveBeenCalledTimes(1))
  expect(post.mock.calls[0][0].redactions).toEqual([
    {start: 7, end: 14, replacement: '[REDACTED]'},
    {start: 0, end: 3, replacement: 'X'}
  ])
})

test('review UI finds a pending record by ID and skips from the returned position', async () => {
  const dir = resolve(process.cwd(), '../../ml/lfg-encoder/review-web')
  const parsed = new DOMParser().parseFromString(readFileSync(resolve(dir, 'index.html'), 'utf8'), 'text/html')
  document.body.innerHTML = parsed.body.innerHTML
  history.replaceState(null, '', '/#local-test-token')
  const calls = []
  const item = {id: 'requested', text: 'fixture', source: {}, project: 'fixture', split: 'train',
    labels: {intents: ['build'], capabilities: [], risks: []}, annotation_sha256: 'digest'}
  vi.stubGlobal('fetch', vi.fn(async path => {
    calls.push(path)
    if (path === '/api/status') return {ok: true, json: async () => ({reviewer: 'r', decisions: {pending: 3, approved: 0, rejected: 0}})}
    return {ok: true, json: async () => ({item, offset: path.includes('id=') ? 1 : 0, pending: 3,
      taxonomy: {intents: ['build'], capabilities: [], risks: []}})}
  }))
  Function(readFileSync(resolve(dir, 'app.js'), 'utf8'))()
  await waitFor(() => expect(document.getElementById('record-id').textContent).toBe('requested'))
  document.getElementById('find-id').value = 'requested'
  document.getElementById('find').click()
  await waitFor(() => expect(calls).toContain('/api/item?id=requested'))
  await waitFor(() => expect(document.getElementById('position').textContent).toBe('Pending 2 of 3'))
  document.getElementById('skip').click()
  await waitFor(() => expect(calls).toContain('/api/item?offset=2'))
})
