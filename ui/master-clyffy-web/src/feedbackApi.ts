export type FeedbackState = {
  assessmentRef: string
  provenance: { engineId: string; modelDigest?: string; sourceRef?: string; [key: string]: unknown }
  originalRecommendation: Record<string, unknown>
  ownerCorrection: Record<string, unknown> | null
  outcome: Record<string, unknown> | null
  extensions?: Record<string, unknown>
  [key: string]: unknown
}

export type FeedbackRecord = FeedbackState & {
  feedbackId: string
  revision: number
  createdAt: string
  updatedAt: string
  parentRevision: number | null
  action: string
  restoredRevision?: number | null
}

type FeedbackHistory = { revision: number; parentRevision: number | null; action: string; restoredRevision?: number | null; createdAt: string; state: FeedbackState }
type FeedbackStatus = { enabled: boolean; available: boolean; reason?: string | null }
type FeedbackPage = { records: FeedbackRecord[]; nextCursor: string | null; hasMore: boolean }

export type FeedbackClient = {
  status: (token: string) => Promise<FeedbackStatus>
  list: (token: string, cursor?: string) => Promise<FeedbackPage>
  read: (id: string, token: string, before?: number) => Promise<{ record: FeedbackRecord; history: FeedbackRecord[]; historyNextBefore: number | null; historyHasMore: boolean }>
  create: (state: FeedbackState, token: string) => Promise<FeedbackRecord>
  revise: (id: string, expectedRevision: number, state: FeedbackState, token: string) => Promise<FeedbackRecord>
  rollback: (id: string, expectedRevision: number, restoreRevision: number, token: string) => Promise<FeedbackRecord>
}

async function request<T>(path: string, body?: unknown, token?: string): Promise<T> {
  const response = await fetch(path, {
    method: body === undefined ? 'GET' : 'POST',
    cache: 'no-store',
    credentials: 'same-origin',
    headers: { Accept: 'application/json', ...(body === undefined ? {} : { 'Content-Type': 'application/json' }), ...(token ? { 'X-LFG-Session-Token': token } : {}) },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  })
  if (!response.ok) {
    const value = await response.json().catch(() => null) as { message?: string; error?: string | { message?: string } } | null
    const detail = typeof value?.error === 'string' ? value.error : value?.error?.message
    throw new Error(value?.message ?? detail ?? `Feedback request failed (${response.status}).`)
  }
  return response.json() as Promise<T>
}

const base = '/api/lfg/feedback'
const recordPath = (id: string) => `${base}/${encodeURIComponent(id)}`

export const feedbackClient: FeedbackClient = {
  status: (token) => request<FeedbackStatus>(`${base}/status`, undefined, token),
  list: (token, cursor) => request<FeedbackPage>(`${base}${cursor ? `?cursor=${encodeURIComponent(cursor)}` : ''}`, undefined, token),
  read: async (id, token, before) => {
    const result = await request<{ record: FeedbackRecord; history: FeedbackHistory[]; historyNextBefore: number | null; historyHasMore: boolean }>(`${recordPath(id)}${before === undefined ? '' : `?historyBefore=${before}`}`, undefined, token)
    return { record: result.record, history: result.history.map((entry) => ({
      ...entry.state, feedbackId: id, revision: entry.revision, createdAt: entry.createdAt,
      updatedAt: entry.createdAt, parentRevision: entry.parentRevision, action: entry.action,
      restoredRevision: entry.restoredRevision,
    })), historyNextBefore: result.historyNextBefore, historyHasMore: result.historyHasMore }
  },
  create: async (state, token) => (await request<{ record: FeedbackRecord }>(base, state, token)).record,
  revise: async (id, expectedRevision, state, token) => (await request<{ record: FeedbackRecord }>(`${recordPath(id)}/revisions`, { expectedRevision, state }, token)).record,
  rollback: async (id, expectedRevision, restoreRevision, token) => (await request<{ record: FeedbackRecord }>(`${recordPath(id)}/rollback`, { expectedRevision, restoreRevision }, token)).record,
}
