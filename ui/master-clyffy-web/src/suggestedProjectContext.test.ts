import { expect, test } from 'vitest'
import { suggestedProjectContext } from './suggestedProjectContext'

const knowledge = { state: 'partial', refreshedAt: 'fixture-refresh', models: [
  { subscriptionId: 'seat-b', model: 'model-b', origin: 'provider-list', available: true, description: 'do-not-copy', efforts: ['do-not-copy'] },
  { subscriptionId: 'seat-a', model: 'model-a', origin: 'provider-list', available: true, label: 'do-not-copy' },
  { subscriptionId: 'seat-a', model: 'model-a', origin: 'provider-list', available: true },
  { subscriptionId: 'seat-a', model: 'session-only', origin: 'session', available: true },
  { subscriptionId: 'seat-a', model: 'published-only', origin: 'published', available: true },
  { subscriptionId: 'seat-a', model: 'not-listed', origin: 'provider-list', available: false },
] }

test('projects only declared server-tagged listings and labels unknowns honestly', () => {
  const result = suggestedProjectContext(knowledge, ['seat-a'], 4096)
  expect(result.text).toContain('["seat-a","model-a"]')
  expect(result.text).not.toContain('model-b')
  expect(result.text).not.toContain('session-only')
  expect(result.text).not.toContain('published-only')
  expect(result.text).not.toContain('not-listed')
  expect(result.text).not.toContain('do-not-copy')
  expect(result.text).toContain('Entitlement, current availability, supported effort and outcomes: unknown')
  expect(result.text).toContain('Server knowledge state: "partial"')
  expect(result.text.match(/\["seat-a","model-a"\]/g)).toHaveLength(1)
  expect(suggestedProjectContext(knowledge, ['seat-b', 'seat-a'], 4096).text)
    .toBe(suggestedProjectContext(knowledge, ['seat-a', 'seat-b'], 4096).text)
})

test('missing qualifying listings stays unknown and invalid or oversized data fails visibly', () => {
  expect(suggestedProjectContext(knowledge, ['other-seat'], 4096).text).toContain('Unknown: no qualifying')
  expect(() => suggestedProjectContext({ state: 'ready', refreshedAt: null, models: [{} , null] }, ['seat'], 4096)).toThrow(/invalid model row/)
  expect(() => suggestedProjectContext(knowledge, ['seat-a'], 30)).toThrow(/exceeds/)
  expect(() => suggestedProjectContext(knowledge, ['seat-a'], 0)).toThrow(/byte limit/)
})
