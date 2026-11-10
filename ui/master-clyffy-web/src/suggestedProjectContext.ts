export type SuggestedContext = { text: string; hasListedModels: boolean }

// This projects only fields whose provider-list origin is known. The server can
// fill other metadata from published configuration while retaining that origin.
export function suggestedProjectContext(value: unknown, declarations: string[], maxBytes: number): SuggestedContext {
  if (!Number.isSafeInteger(maxBytes) || maxBytes < 1) throw new Error('Connect the owned model to obtain a context byte limit before building a draft.')
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Provider knowledge is unavailable or invalid.')
  const knowledge = value as Record<string, unknown>
  if (typeof knowledge.state !== 'string' || !Array.isArray(knowledge.models) ||
      !(knowledge.refreshedAt === null || typeof knowledge.refreshedAt === 'string')) throw new Error('Provider knowledge is unavailable or invalid.')
  const encoder = new TextEncoder()
  const fits = (text: string) => text.length <= maxBytes && encoder.encode(text).length <= maxBytes
  if (!fits(knowledge.state) || (typeof knowledge.refreshedAt === 'string' && !fits(knowledge.refreshedAt))) throw new Error('Provider knowledge metadata exceeds the configured context limit.')
  const selected = [...new Set(declarations)].sort()
  if (selected.some(id => typeof id !== 'string' || !id.trim() || !fits(id))) throw new Error('Declared subscription data exceeds the context limit or is invalid.')
  const header = [
    'User-declared subscriptions and cached provider-list observations (unverified).',
    'Entitlement, current availability, supported effort and outcomes: unknown.',
    `Server knowledge state: ${JSON.stringify(knowledge.state)}; server refresh: ${JSON.stringify(knowledge.refreshedAt ?? 'unknown')}.`,
    'This refresh is not necessarily when a provider list was fetched.',
    `User-declared subscription IDs: ${JSON.stringify(selected)}.`,
    'Cached provider-list model IDs for those declarations:',
  ].join('\n')
  if (!fits(header)) throw new Error('The suggested context exceeds the configured byte limit. Narrow your selections or enter context manually.')
  const rows = new Map<string, string>()
  for (const item of knowledge.models) {
    if (!item || typeof item !== 'object' || Array.isArray(item)) throw new Error('Provider knowledge contains an invalid model row.')
    const row = item as Record<string, unknown>
    if (row.origin !== 'provider-list' || row.available !== true) continue
    if (typeof row.subscriptionId !== 'string' || typeof row.model !== 'string' || !row.subscriptionId.trim() || !row.model.trim()) throw new Error('Provider knowledge contains an invalid listed model ID.')
    if (!selected.includes(row.subscriptionId)) continue
    if (!fits(row.subscriptionId) || !fits(row.model)) throw new Error('A listed model ID exceeds the configured context byte limit.')
    const encoded = JSON.stringify([row.subscriptionId, row.model])
    rows.set(encoded, encoded)
    // Fail as soon as the exact representation cannot fit; do not silently omit rows.
    if (!fits(`${header}\n${[...rows.keys()].join('\n')}`)) throw new Error('The suggested context exceeds the configured byte limit. Narrow your selections or enter context manually.')
  }
  const listed = [...rows.keys()].sort()
  const tail = listed.length ? listed.join('\n') : 'Unknown: no qualifying cached provider-list rows were found for these declarations.'
  const text = `${header}\n${tail}`
  if (!fits(text)) throw new Error('The suggested context exceeds the configured byte limit. Narrow your selections or enter context manually.')
  return { text, hasListedModels: listed.length > 0 }
}
