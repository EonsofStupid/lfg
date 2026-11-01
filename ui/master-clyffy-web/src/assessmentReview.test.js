import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { waitFor } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'

const root = resolve(process.cwd(), '../../ml/lfg-encoder/review-web')
const markup = readFileSync(resolve(root, 'index.html'), 'utf8')
const script = readFileSync(resolve(root, 'app.js'), 'utf8')

afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks(); document.body.replaceChildren() })

async function openReview() {
  document.documentElement.innerHTML = markup
  window.location.hash = '#fixture-token'
  const profile = {project: 'fixture', complexities: [{id: 'c1', label: 'First'}],
    models: [{id: 'm1', label: 'Available', provider: 'p', model: 'm', subscription_id: 's', availability: 'available', effort_levels: ['e1']},
      {id: 'm2', label: 'Unknown', provider: 'p', model: 'other', subscription_id: null, availability: 'unknown', effort_levels: []}],
    agent_range: {min: 0, max: 4}}
  const item = {id: 'one', text: 'fixture request', project: 'fixture', split: 'train', source: {},
    labels: {intents: ['build'], capabilities: [], risks: []}, labels_pending: false,
    annotation_sha256: 'b'.repeat(64), redaction_suggestions: [], assessment_targets: null}
  const itemResponse = {item, pending: 1, offset: 0, taxonomy: {intents: ['build'], capabilities: [], risks: []},
    guide: null, guide_sha256: null, assessment_profile: profile, assessment_config_sha256: 'a'.repeat(64)}
  let submitted
  vi.stubGlobal('fetch', vi.fn(async (url, options) => {
    if (String(url) === '/api/status') return {ok: true, json: async () => ({decisions: {pending: 1, approved: 0, rejected: 0}, reviewer: 'reviewer'})}
    if (String(url).startsWith('/api/item')) return {ok: true, json: async () => itemResponse}
    if (String(url) === '/api/following-source-context') return {ok: true, json: async () => ({observed_through_line: 4,
      observed_prefix_sha256: 'c'.repeat(64), truncated: false, messages: [{source_line: 4, text: '<script>later correction</script>'}]})}
    if (String(url) === '/api/decision') { submitted = JSON.parse(options.body); return {ok: true, json: async () => ({saved: true})} }
    throw new Error('unexpected request')
  }))
  vi.spyOn(window, 'prompt').mockReturnValue('APPROVE')
  window.eval(script)
  await waitFor(() => expect(document.querySelector('#assessment').hidden).toBe(false))
  return () => submitted
}

test('reviewer can submit project-configured assessment values', async () => {
  const submitted = await openReview()
  document.querySelector('#assessment-complexity').value = 'c1'
  const boxes = document.querySelectorAll('#assessment-models input[type=checkbox]')
  expect(boxes[1].disabled).toBe(true)
  boxes[0].click()
  document.querySelector('#assessment-models select').value = 'e1'
  document.querySelector('#assessment-agents').value = '0'
  document.querySelector('#attest').checked = true
  document.querySelector('#approve').click()
  await waitFor(() => expect(submitted()).toBeDefined())
  expect(submitted().assessment_values).toEqual({complexity: 'c1', models: [{id: 'm1', effort: 'e1'}], agent_count: 0})
  expect(submitted().assessment_config_sha256).toBe('a'.repeat(64))
})

test('blank assessment controls remain explicitly unknown', async () => {
  const submitted = await openReview()
  document.querySelector('#attest').checked = true
  document.querySelector('#approve').click()
  await waitFor(() => expect(submitted()).toBeDefined())
  expect(submitted().assessment_values).toEqual({complexity: null, models: null, agent_count: null})
})

test('later source context is requested explicitly and rendered as text', async () => {
  await openReview()
  expect(vi.mocked(fetch).mock.calls.some(([url]) => String(url) === '/api/following-source-context')).toBe(false)
  document.querySelector('#show-following').click()
  await waitFor(() => expect(document.querySelector('#following-context').textContent).toContain('<script>later correction</script>'))
  expect(document.querySelector('#following-context script')).toBeNull()
  const request = vi.mocked(fetch).mock.calls.find(([url]) => String(url) === '/api/following-source-context')
  expect(JSON.parse(request[1].body)).toEqual({id: 'one', annotation_sha256: 'b'.repeat(64)})
  document.querySelector('#skip').click()
  expect(document.querySelector('#following-context').textContent).toBe('')
})
