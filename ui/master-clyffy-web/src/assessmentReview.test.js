import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { waitFor } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'

const root = resolve(process.cwd(), '../../ml/lfg-encoder/review-web')
const markup = readFileSync(resolve(root, 'index.html'), 'utf8')
const script = readFileSync(resolve(root, 'app.js'), 'utf8')

afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks(); document.body.replaceChildren() })

async function openReview(partialReview = null) {
  document.documentElement.innerHTML = markup
  window.location.hash = '#fixture-token'
  const profile = {project: 'fixture', complexities: [{id: 'c1', label: 'First'}],
    models: [{id: 'm1', label: 'Available', provider: 'p', model: 'm', subscription_id: 's', availability: 'available', effort_levels: ['e1']},
      {id: 'm2', label: 'Unknown', provider: 'p', model: 'other', subscription_id: null, availability: 'unknown', effort_levels: []}],
    agent_range: {min: 0, max: 4}}
  const item = {id: 'one', text: 'fixture request', project: 'fixture', split: 'train', source: {},
    labels: {intents: ['build'], capabilities: [], risks: []}, labels_pending: false,
    annotation_sha256: 'b'.repeat(64), redaction_suggestions: [{label: 'Preview redaction', start: 0, end: 7, replacement: '[redacted]'}], assessment_targets: null,
    partial_review: partialReview || {enabled: true, state: 'not_reviewed', binding: {source_sha256: '1'.repeat(64),
      annotation_sha256: 'b'.repeat(64), taxonomy_sha256: '2'.repeat(64), text_sha256: '3'.repeat(64),
      proposal_sha256: '4'.repeat(64), guide_sha256: null}, context_checks: [], revision: null, history: []}}
  const itemResponse = {item, pending: 1, offset: 0, taxonomy: {intents: ['build'], capabilities: ['write'], risks: []},
    guide: null, guide_sha256: null, assessment_profile: profile, assessment_config_sha256: 'a'.repeat(64)}
  let submitted
  vi.stubGlobal('fetch', vi.fn(async (url, options) => {
    if (String(url) === '/api/status') return {ok: true, json: async () => ({decisions: {pending: 1, approved: 0, rejected: 0}, reviewer: 'reviewer'})}
    if (String(url).startsWith('/api/item')) return {ok: true, json: async () => itemResponse}
    if (String(url) === '/api/following-source-context') return {ok: true, json: async () => ({observed_through_line: 4,
      observed_prefix_sha256: 'c'.repeat(64), truncated: false, messages: [{source_line: 4, text: '<script>later correction</script>'}]})}
    if (String(url) === '/api/decision') { submitted = JSON.parse(options.body); return {ok: true, json: async () => ({saved: true})} }
    if (String(url) === '/api/partial-decision') { submitted = JSON.parse(options.body); return {ok: true, json: async () => ({saved: true})} }
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

test('partial mode records explicit judgments separately from rejected suggestions', async () => {
  const submitted = await openReview()
  vi.mocked(window.prompt).mockReturnValue('REVIEW PARTIAL LABELS')
  expect(document.querySelector('#partial-mode-label').hidden).toBe(false)
  document.querySelector('#partial-mode').click()
  expect(document.querySelector('#labels').hidden).toBe(true)
  const intent = document.querySelector('[aria-label="intents build judgment"]')
  expect(intent.value).toBe('')
  intent.value = 'positive'
  const proposal = document.querySelector('[aria-label="intents build proposal action"]')
  proposal.value = 'rejected'
  document.querySelector('[aria-label="capabilities write judgment"]').value = 'unknown'
  document.querySelector('#partial-attest').checked = true
  document.querySelector('#save-partial').click()
  await waitFor(() => expect(submitted()).toBeDefined())
  expect(submitted().judgments).toEqual({intents: {build: 'positive'}, capabilities: {write: 'unknown'}})
  expect(submitted().proposal_actions).toEqual({intents: {build: 'rejected'}})
  expect(submitted().operation).toBe('create')
  expect(submitted().supersedes_event_id).toBeNull()
  expect(submitted().attestation).toBe('REVIEW PARTIAL LABELS')
  expect(vi.mocked(fetch).mock.calls.find(([url]) => String(url) === '/api/partial-decision')).toBeDefined()
})

function existingPartialReview(state = 'current') {
  const binding = {source_sha256: '1'.repeat(64), annotation_sha256: 'b'.repeat(64),
    taxonomy_sha256: '2'.repeat(64), text_sha256: '3'.repeat(64), proposal_sha256: '4'.repeat(64), guide_sha256: null}
  const event = {schema_version: 'lfg.partial-label-review.v1', event_id: 'e'.repeat(32), id: 'one',
    reviewer_claim: 'reviewer', reviewed_at: '2026-11-06T00:00:00+00:00', binding, judgments: {intents: {build: 'positive'}},
    proposal_actions: {}, context_checks: [], revision: 1, event_type: 'decision', supersedes_event_id: null,
    supersedes_event_sha256: null, attestation: 'Reviewer explicitly recorded partial label judgments; unknown cells remain unknown.'}
  return {enabled: true, state, binding, event, history: [event],
    revision: {event_id: event.event_id, event_sha256: 'f'.repeat(64), revision: 1, event_type: 'decision'}, context_checks: []}
}

test('saved partial labels are visible and full approval requires explicit reconciliation', async () => {
  const submitted = await openReview(existingPartialReview())
  expect(document.querySelector('#partial-history').hidden).toBe(false)
  expect(document.querySelector('#partial-history-content').textContent).toContain('intents.build: positive')
  document.querySelector('#attest').checked = true
  document.querySelector('#approve').click()
  await waitFor(() => expect(document.querySelector('#notice').textContent).toContain('reconciliation'))
  expect(submitted()).toBeUndefined()
  document.querySelector('#partial-reconcile').checked = true
  vi.mocked(window.prompt).mockReturnValueOnce('RECONCILE PARTIAL REVIEW')
  document.querySelector('#approve').click()
  await waitFor(() => expect(submitted()).toBeDefined())
  expect(submitted().partial_reconciliation).toEqual({event_id: 'e'.repeat(32),
    event_sha256: 'f'.repeat(64), confirmation: 'RECONCILE PARTIAL REVIEW'})
})

test('existing partial review can be revised or retracted with a new event', async () => {
  const submitted = await openReview(existingPartialReview('stale'))
  expect(document.querySelector('#partial-history-status').textContent).toContain('stale and inactive')
  document.querySelector('#partial-mode').click()
  expect(document.querySelector('[aria-label="intents build judgment"]').value).toBe('')
  document.querySelector('[aria-label="intents build judgment"]').value = 'negative'
  document.querySelector('#partial-attest').checked = true
  vi.mocked(window.prompt).mockReturnValue('REVISE PARTIAL REVIEW')
  document.querySelector('#save-partial').click()
  await waitFor(() => expect(submitted()?.operation).toBe('revise'))
  expect(submitted().supersedes_event_id).toBe('e'.repeat(32))
  expect(submitted().judgments).toEqual({intents: {build: 'negative'}})
  await waitFor(() => expect(vi.mocked(fetch).mock.calls.filter(([url]) => String(url).startsWith('/api/item')).length).toBeGreaterThanOrEqual(2))
  document.querySelector('#partial-mode').click()
  expect(document.querySelector('#retract-partial').hidden).toBe(false)
  vi.mocked(window.prompt).mockReturnValue('RETRACT PARTIAL REVIEW')
  document.querySelector('#retract-partial').click()
  await waitFor(() => expect(submitted()?.operation).toBe('retract'))
  expect(submitted().judgments).toEqual({})
  expect(submitted().supersedes_event_sha256).toBe('f'.repeat(64))
})

test('partial judgment cannot be saved against a redacted preview', async () => {
  const submitted = await openReview()
  document.querySelector('#partial-mode').click()
  document.querySelector('#partial-attest').checked = true
  document.querySelector('#partial-label-controls select[data-field]').value = 'negative'
  document.querySelector('#suggestions button').click()
  document.querySelector('#save-partial').click()
  await waitFor(() => expect(document.querySelector('#notice').textContent).toContain('bound to the original request'))
  expect(submitted()).toBeUndefined()
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
