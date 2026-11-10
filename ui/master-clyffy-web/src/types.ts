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

export type WordHit = { id: string; plainLabel: string; matched: string[] }

export type FormulaTerm = { id: string; label: string; value: number; weight: number; points: number; detail: string[] }

export type ModelSuggestion = {
  subscriptionId: string
  model: string
  label: string
  description: string
  role: string | null
  effort: string | null
  usualEffort: string | null
  origin: string
  fit: number
  habit: number
  score: number
  promptsInBracket: number
  why: string
}

export type SubscriptionStatus = {
  id: string
  label: string
  status: 'verified' | 'recognized' | 'declared' | 'signed-out' | 'not-found' | string
  usable: boolean
  declared: boolean
  plan: string | null
  lastUsed: string | null
  detail: string
}

export type Assessment = {
  schemaVersion: string
  sessionId: string
  revision: number
  durationMicros: number
  configRevision: string
  configDigest: string
  prompt: { characters: number; words: number; tasks: string[]; scoringTruncated?: boolean }
  intent: { primary: string; plainLabel: string; all: WordHit[]; risks: WordHit[]; fallback: boolean }
  complexity: {
    score: number
    bracket: string
    label: string
    plainLabel: string
    terms: FormulaTerm[]
    ladder: Array<{ id: string; label: string; minScore: number; current: boolean }>
  }
  agents: { total: number; tasks: number; min: number; max: number; plain: string }
  models: { role: string; effort: string; suggested: ModelSuggestion | null; alternatives: ModelSuggestion[]; usualModel: string | null; plain: string }
  subscriptions: SubscriptionStatus[]
  knowledge: { state: string; refreshedAt: string | null; promptsSeen: number; modelsKnown: number }
  rules: string[]
  advisory: boolean
  confidence: number | null
}

export type KnownModel = {
  subscriptionId: string
  model: string
  label: string
  description: string
  priority: number | null
  efforts: string[]
  origin: string
  available: boolean
  lastUsed: string | null
  usage: Array<{ bracket: string; prompts: number; weighted: number; efforts: Record<string, number> }>
  role: string | null
}

export type Knowledge = {
  state: string
  refreshedAt: string | null
  promptsSeen: number
  sources: Array<{ id: string; files: number; prompts: number; status: string }>
  subscriptions: Array<{ id: string; plan: string | null; listedAt: string | null; lastUsed: string | null; daysSinceUsed: number | null }>
  models: KnownModel[]
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
    scorecardMaxFeedbackPages: number
    scorecardMaxFeedbackPageSize: number
    eventBufferCapacity: number
    streamKeepaliveSeconds: number
    uiDebounceMs: number
    uiTimingHistorySamples: number
    devTools?: {
      statsEnabled: boolean
      traceCapacity: number
      statsPollIntervalMs: number
      debounceMinMs: number
      debounceMaxMs: number
      historyMinSamples: number
      historyMaxSamples: number
    }
  }
}

export type DevStats = {
  schemaVersion: string
  enabled: boolean
  capacity: number
  totalRecorded: number
  evicted: number
  cpuTimeMicros: number | null
  operationsDigest: string
  encoderDigest: string
  classifierDigest: string
  assessmentDigest?: string
  summaries: Array<{ operation: 'preflight' | 'classification' | 'assessment'; samples: number; errors: number; p50TotalMicros: number | null; p95TotalMicros: number | null }>
  traces: Array<{ sequence: number; recordedAt: string; operation: string; outcome: string; totalMicros: number; queueMicros: number | null; workMicros: number | null; rules?: string[] }>
}

export type IntakePreview = {
  schemaVersion: 'lfg.intake-preview.v1'
  templateRevision: string
  templateDigest: string
  sourceApp: string
  missingRequired: number
  answers: Array<{ id: string; question: string; required: boolean; status: 'filled' | 'missing' | 'invalid'; value: unknown }>
}

export type AppContext = {
  schemaVersion: 'lfg.app-context.v1'
  status: 'disabled' | 'ready' | 'unavailable'
  revision: string | null
  configDigest?: string | null
  pollIntervalMs: number | null
  providers: Array<{ id: string; activated: boolean; capabilities: string[] }> | null
}

export type BoundaryQuestion = {
  id: string; section: string; number: number; question: string; sourceTerm: string
  domain?: string | null; binding: string | null
}
export type BoundaryCatalog = {
  catalog: { schemaVersion: string; revision: string; sourceNotice: string; banks: Array<{
    id: string; title: string; status: string; questions: BoundaryQuestion[]
  }> }; digest: string
}
