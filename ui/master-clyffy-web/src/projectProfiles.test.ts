import { expect, test } from 'vitest'
import { getProfile, saveProfile } from './projectProfiles'

function storage(): Storage {
  const values = new Map<string, string>()
  return { get length() { return values.size }, clear: () => values.clear(),
    getItem: key => values.get(key) ?? null, key: index => [...values.keys()][index] ?? null,
    removeItem: key => { values.delete(key) }, setItem: (key, value) => { values.set(key, value) } }
}

test('saves independent profiles and restores old content as a new revision', () => {
  const local = storage()
  const first = saveProfile(local, 'project-a', 'catalog', '{"source":"one"}',
    { q: { value: 'app fact', status: 'proposed', origin: 'app_evidence' } }, 0, null, 10000)
  saveProfile(local, 'project-b', 'catalog', '{"source":"other"}', {}, 0, null, 10000)
  const second = saveProfile(local, 'project-a', 'catalog', '{"source":"two"}',
    { q: { value: 'reviewed answer', status: 'reviewed', origin: 'manual' } }, 1, null, 10000)
  const third = saveProfile(local, 'project-a', first[0].catalogDigest, first[0].preview, first[0].answers, 2, 1, 10000)
  expect(second).toHaveLength(2)
  expect(third).toHaveLength(3)
  expect(third[2].restoredFrom).toBe(1)
  expect(getProfile(local, 'project-a', 10000)[2].answers.q.status).toBe('proposed')
  expect(getProfile(local, 'project-b', 10000)).toHaveLength(1)
  expect(() => saveProfile(local, 'project-a', 'catalog', '{}', {}, 1, null, 10000)).toThrow(/changed/)
})

test('rejects an oversized save without modifying existing data', () => {
  const local = storage()
  saveProfile(local, 'project', 'catalog', '{}', {}, 0, null, 1000)
  const before = local.getItem('lfg.browser-project-profiles.v1')
  expect(() => saveProfile(local, 'project', 'catalog', 'x'.repeat(1000), {}, 1, null, 1000)).toThrow(/limit/)
  expect(local.getItem('lfg.browser-project-profiles.v1')).toBe(before)
})
