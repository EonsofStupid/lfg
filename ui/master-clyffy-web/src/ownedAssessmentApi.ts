export type OwnedEngine = {
  id: string
  modelSha256?: string | null
  modelManifestSha256?: string | null
  trainingManifestSha256?: string | null
  framingSha256?: string | null
  runtimeConfigSha256?: string | null
  serviceConfigSha256?: string | null
  [key: string]: unknown
}

export type OwnedStatus = {
  enabled: boolean
  status: 'disabled' | 'unavailable' | 'ready'
  engine: OwnedEngine | null
  limits?: { maxRequestBytes: number; maxHttpBodyBytes?: number; maxContextBytes?: number; maxGeneratedBytes?: number; maxGeneratedTokens?: number; timeoutMs: number; maxConcurrent: number; workerThreads?: number }
  reason?: string | null
}

export type OwnedResult = {
  status: 'ok' | 'truncated' | 'timeout' | 'busy' | 'invalid_input' | 'context_limit' | 'incomplete_output' | 'malformed_output' | 'inference_error' | 'unavailable'
  engine: OwnedEngine | null
  elapsedMs: number
  stopReason?: string | null
  outputText?: string | null
  output?: Record<string, unknown> | null
  projectContextSha256?: string | null
  measurement?: {
    schemaVersion: 'lfg.owned-run-measurement.v1'
    status: string
    engineId: string | null
    stopReason: string | null
    handlerElapsedMs: number
    outputTextUtf8Bytes: number | null
    workerStarted: boolean | null
    workerCompletedSuccessfully: boolean | null
  } | null
  [key: string]: unknown
}

export type OwnedProjectContext = { context: string; contextSha256: string; [key: string]: unknown }

export type OwnedAssessmentClient = {
  status: (token: string) => Promise<OwnedStatus>
  assess: (text: string, token: string, context?: OwnedProjectContext) => Promise<OwnedResult>
}

export function ownedAssessmentBody(text: string, context?: OwnedProjectContext): Record<string, unknown> {
  if (context === undefined) return { text }
  const { context: suppliedContext, contextSha256, ...extensions } = context
  return { ...extensions, context: suppliedContext, contextSha256, text }
}

export async function sha256Utf8(text: string): Promise<string> {
  if (!globalThis.crypto?.subtle) throw new Error('A secure local browser context is required to identify project context bytes.')
  const digest = await globalThis.crypto.subtle.digest('SHA-256', new TextEncoder().encode(text))
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('')
}

async function request<T>(path: string, token: string, body?: unknown): Promise<T> {
  const response = await fetch(path, {
    method: body === undefined ? 'GET' : 'POST', cache: 'no-store', credentials: 'same-origin',
    headers: { Accept: 'application/json', 'X-LFG-Session-Token': token, ...(body === undefined ? {} : { 'Content-Type': 'application/json' }) },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  })
  if (!response.ok) {
    const value = await response.json().catch(() => null) as { message?: string; error?: string | { message?: string } } | null
    const detail = typeof value?.error === 'string' ? value.error : value?.error?.message
    throw new Error(value?.message ?? detail ?? (response.status === 403 ? 'Access denied. Check the token and local server address.' : `Owned assessment request failed (${response.status}).`))
  }
  return response.json() as Promise<T>
}

export const ownedAssessmentClient: OwnedAssessmentClient = {
  status: (token) => request('/api/lfg/assess/owned/status', token),
  assess: (text, token, context) => request('/api/lfg/assess/owned', token, ownedAssessmentBody(text, context)),
}

export const devModelClient: OwnedAssessmentClient = {
  status: (token) => request('/api/lfg/assess/dev-model/status', token),
  assess: (text, token, context) => request('/api/lfg/assess/dev-model', token, ownedAssessmentBody(text, context)),
}
