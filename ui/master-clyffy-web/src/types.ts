export type ManifestRef = { id: string; version: string; sha256: string }

export type Prediction = { label: string; confidence: number | null; source: string }

export type RroCategory = {
  id: string
  version: string
  label: string
  plainLabel: string
  capability: string
  tone: string
  hash: string
}

export type ReasonReadyObject = {
  id: string
  category: RroCategory
  confidence: number | null
  advisory: boolean
  lineage: string[]
}

export type PreflightSnapshot = {
  schemaVersion: string
  sessionId: string
  revision: number
  mode: string
  model: ManifestRef
  durationMicros: number
  intents: Prediction[]
  capabilities: Prediction[]
  risks: Prediction[]
  rros: ReasonReadyObject[]
  predictedAgents: number
  predictedBranches: number
  confidence: number | null
  abstained: boolean
}

export type GraphTotals = {
  predicted: number
  committed: number
  queued: number
  active: number
  blocked: number
  completed: number
  failed: number
}

export type GraphNode = {
  id: string
  kind: 'prompt' | 'rro' | 'agent' | 'task' | 'provider' | string
  label: string
  plainLabel: string
  state: string
  tone: string
  provenance: string
  metadata: Record<string, string>
}

export type GraphEdge = {
  id: string
  source: string
  target: string
  relation: string
  state: string
}

export type GraphSnapshot = {
  workflowId: string
  throughSequence: number
  nodes: GraphNode[]
  edges: GraphEdge[]
  totals: GraphTotals
}

export type PlanCommit = {
  workflowId: string
  plan: { id: string; hash: string; maxParallelTasks: number }
  graph: GraphSnapshot
  committedSignals: number
}

export type CategoryScore = { id: string; label: string; percent: number; matchedKeywords: string[] }

export type EscalationTarget = {
  provider: string
  model: string
  role: string
  rawCapability: number
  toolLoopEfficiency: number
  costTier: 'low' | 'medium' | 'high' | string
  subscriptionId: string
  strengths: string[]
  effortLevels: string[]
  evidence: { source: string; basis: string; observedAt: string }
}

export type EscalationCandidate = { target: EscalationTarget; active: boolean }

export type RoutingNeed = 'balanced' | 'capability' | 'efficiency' | 'cost'

export type ClassificationSnapshot = {
  schemaVersion: string
  configRevision: string
  configDigest: string
  sessionId: string
  revision: number
  durationMicros: number
  tokenEstimate: number
  categories: CategoryScore[]
  needsBiggerModel: boolean
  predictedSubagents: number
  escalationCatalog: EscalationCandidate[]
  recommendedEscalation: EscalationTarget | null
  routingNeed: string
  project: string
}

export type RuntimeStatus = {
  name: string
  phase: string
  protocolVersion: string
  workspaceId: string
  encoder: ManifestRef
  signalMode: string
  contentPolicy: string
  liveProviders: number
  configuredProviders: number
  fixtureProviders: number
  operationsDigest: string
  operations: {
    schemaVersion: string
    revision: string
    maxPromptBytes: number
    maxHttpBodyBytes: number
    defaultSignalPageSize: number
    maxSignalPageSize: number
    eventBufferCapacity: number
    streamKeepaliveSeconds: number
    uiDebounceMs: number
    uiTimingHistorySamples: number
  }
}
