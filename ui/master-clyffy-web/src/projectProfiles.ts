export type DraftAnswer = { value: string; status: 'proposed' | 'reviewed'; origin: 'app_evidence' | 'manual' }
export type ProfileRevision = { revision: number; savedAt: string; restoredFrom: number | null; catalogDigest: string; preview: string; answers: Record<string, DraftAnswer> }
type Profile = { id: string; revisions: ProfileRevision[] }
type Store = { schemaVersion: 'lfg.browser-project-profiles.v1'; profiles: Profile[] }
const STORAGE_KEY = 'lfg.browser-project-profiles.v1'

function readStore(storage: Storage, maxBytes: number): Store {
  const raw = storage.getItem(STORAGE_KEY)
  if (!raw) return { schemaVersion: 'lfg.browser-project-profiles.v1', profiles: [] }
  if (new TextEncoder().encode(raw).length > maxBytes) throw new Error('Saved profiles exceed the configured browser limit')
  const value: unknown = JSON.parse(raw)
  if (!value || typeof value !== 'object') throw new Error('Invalid saved profiles')
  const data = value as Store
  if (data.schemaVersion !== 'lfg.browser-project-profiles.v1' || !Array.isArray(data.profiles)) throw new Error('Invalid saved profiles')
  const ids = new Set<string>()
  for (const profile of data.profiles) {
    if (!profile || typeof profile.id !== 'string' || !profile.id.trim() || ids.has(profile.id) || !Array.isArray(profile.revisions)) throw new Error('Invalid saved profile')
    ids.add(profile.id)
    for (const [index, revision] of profile.revisions.entries()) {
      if (!revision || revision.revision !== index + 1 || typeof revision.savedAt !== 'string' ||
          typeof revision.catalogDigest !== 'string' || typeof revision.preview !== 'string' ||
          (revision.restoredFrom !== null && (!Number.isSafeInteger(revision.restoredFrom) || revision.restoredFrom < 1 || revision.restoredFrom >= revision.revision)) ||
          !revision.answers || typeof revision.answers !== 'object' || Array.isArray(revision.answers) ||
          Object.entries(revision.answers).some(([id, answer]) => !id || !answer || typeof answer.value !== 'string' ||
            !['proposed', 'reviewed'].includes(answer.status) || !['app_evidence', 'manual'].includes(answer.origin))) throw new Error('Invalid saved profile revision')
    }
  }
  return data
}

export function getProfile(storage: Storage, id: string, maxBytes: number): ProfileRevision[] {
  return readStore(storage, maxBytes).profiles.find(profile => profile.id === id.trim())?.revisions ?? []
}

export function saveProfile(storage: Storage, id: string, catalogDigest: string, preview: string,
  answers: Record<string, DraftAnswer>, expectedHead: number, restoredFrom: number | null, maxBytes: number): ProfileRevision[] {
  id = id.trim()
  if (!id || /[\r\n]/.test(id) || !Number.isSafeInteger(maxBytes) || maxBytes <= 0) throw new Error('Enter a project ID and configure a valid browser limit')
  const data = readStore(storage, maxBytes)
  let profile = data.profiles.find(item => item.id === id)
  const head = profile?.revisions.length ?? 0
  if (head !== expectedHead) throw new Error('Profile changed in this browser; reload it before saving')
  if (restoredFrom !== null && (!Number.isSafeInteger(restoredFrom) || restoredFrom < 1 || restoredFrom > head)) throw new Error('Invalid revision to restore')
  if (!profile) { profile = { id, revisions: [] }; data.profiles.push(profile) }
  profile.revisions.push({ revision: head + 1, savedAt: new Date().toISOString(), restoredFrom, catalogDigest, preview, answers })
  const raw = JSON.stringify(data)
  if (new TextEncoder().encode(raw).length > maxBytes) throw new Error('Profile history exceeds the configured browser limit')
  storage.setItem(STORAGE_KEY, raw)
  return profile.revisions
}
