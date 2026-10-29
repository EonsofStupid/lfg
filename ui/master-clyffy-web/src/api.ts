import type { ClassificationSnapshot, GraphSnapshot, PlanCommit, PreflightSnapshot, RuntimeStatus } from './types'

async function requestJSON<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { Accept: 'application/json', ...(init?.body ? { 'Content-Type': 'application/json' } : {}), ...init?.headers }
  })
  if (!response.ok) {
    const value = await response.json().catch(() => null) as { message?: string } | null
    throw new Error(value?.message ?? `${path} returned ${response.status}`)
  }
  return response.json() as Promise<T>
}

export const getStatus = async () => {
  const status = await requestJSON<RuntimeStatus>('/api/shippin/intelligence/lfg/status')
  const operations = status.operations
  if (!operations || operations.schemaVersion !== 'lfg.operations.v1' ||
      !Number.isSafeInteger(operations.maxPromptBytes) || operations.maxPromptBytes <= 0 ||
      !Number.isSafeInteger(operations.uiDebounceMs) || operations.uiDebounceMs <= 0 ||
      !Number.isSafeInteger(operations.uiTimingHistorySamples) || operations.uiTimingHistorySamples <= 0) {
    throw new Error('Server operational configuration is missing or incompatible')
  }
  return status
}

export const preflight = (sessionId: string, revision: number, text: string) =>
  requestJSON<PreflightSnapshot>('/api/shippin/intelligence/lfg/preflight', {
    method: 'POST', body: JSON.stringify({ sessionId, revision, text })
  })

export const commitPlan = (sessionId: string, revision: number, text: string) =>
  requestJSON<PlanCommit>('/api/shippin/intelligence/lfg/plans', {
    method: 'POST', body: JSON.stringify({ sessionId, revision, text, includeFixtureBindings: true })
  })

export const getGraph = (workflowId: string, throughSequence?: number) =>
  requestJSON<GraphSnapshot>(`/api/shippin/intelligence/lfg/workflows/${encodeURIComponent(workflowId)}/graph${throughSequence === undefined ? '' : `?throughSequence=${throughSequence}`}`)

export const classify = (sessionId: string, revision: number, text: string, activeSubscriptions: string[], routingNeed: string | null) =>
  requestJSON<ClassificationSnapshot>('/api/lfg/classify', {
    method: 'POST', body: JSON.stringify({ sessionId, revision, text, activeSubscriptions, ...(routingNeed ? { routingNeed } : {}) })
  })
