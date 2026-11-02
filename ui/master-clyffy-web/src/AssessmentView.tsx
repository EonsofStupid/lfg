import type { Assessment, ModelSuggestion, SubscriptionStatus } from './types'

const STATUS_LABEL: Record<string, string> = {
  'signed-in': 'Signed in; subscription unverified',
  recognized: 'Found on this machine',
  declared: 'You said you have this',
  'signed-out': 'Signed out',
  'not-found': 'Not found'
}

const ORIGIN_LABEL: Record<string, string> = {
  'provider-list': 'From the provider’s own model list',
  session: 'Seen in your sessions',
  published: 'From the published line-up'
}

function knowledgeNote(knowledge: Assessment['knowledge']): string {
  if (knowledge.state === 'reading') return 'Reading your sessions. Suggestions improve when this finishes.'
  if (knowledge.state === 'ready') return `Learned from ${knowledge.promptsSeen.toLocaleString()} of your requests. Only counts are kept, on this machine.`
  return 'Not learning from your sessions, so suggestions use the published line-up.'
}

function ModelCard({ model, lead }: { model: ModelSuggestion; lead?: boolean }) {
  return <article className={lead ? 'model-card is-lead' : 'model-card'}>
    <header>
      <strong>{model.label}</strong>
      {model.effort && <span className="pill">{model.effort} effort</span>}
      {model.role && <span className="pill">{model.role}</span>}
    </header>
    {lead && model.description && <p>{model.description}</p>}
    <p>{model.why}{model.usualEffort && model.effort && model.usualEffort !== model.effort ? ` LFG suggests ${model.effort} here.` : ''}</p>
    {lead && <small>{ORIGIN_LABEL[model.origin] ?? model.origin} · fit {model.fit} · your use {model.habit} · score {model.score}</small>}
  </article>
}

function SubscriptionRow({ subscription, declared, onToggle }: { subscription: SubscriptionStatus; declared: boolean; onToggle: (id: string) => void }) {
  return <li className={`subscription-status status-${subscription.status}`}>
    <button
      type="button"
      className={declared ? 'is-active' : ''}
      aria-pressed={declared}
      title={declared ? 'You marked this as yours. Click to undo.' : 'Mark this subscription as yours'}
      onClick={() => onToggle(subscription.id)}
    >{subscription.label}</button>
    <span className="pill">{STATUS_LABEL[subscription.status] ?? subscription.status}</span>
    {subscription.plan && <span className="pill">{subscription.plan}</span>}
    <small>{subscription.detail}</small>
  </li>
}

export function AssessmentView({ assessment, prompt, declared, onToggleSubscription }: {
  assessment: Assessment | null
  prompt: string
  declared: string[]
  onToggleSubscription: (id: string) => void
}) {
  if (!assessment) return <section className="assessment" aria-label="LFG assessment"><p className="assessment-empty">Type a request above to see how LFG reads it.</p></section>
  const { complexity, intent, agents, models } = assessment
  return <section className="assessment" aria-label="LFG assessment">
    <ol className="assessment-steps">
      <li>
        <h2>Prompt received</h2>
        <p className="assessment-prompt">{`“${prompt}”`}</p>
        <p>{assessment.prompt.words} words · {assessment.prompt.tasks.length} {assessment.prompt.tasks.length === 1 ? 'task' : 'tasks'} found</p>
        {assessment.prompt.scoringTruncated && <p role="status">Some text was outside the configured scoring limit; the assessment may be incomplete.</p>}
        {assessment.prompt.tasks.length > 0 && <ul className="assessment-tasks">{assessment.prompt.tasks.map((task, index) => <li key={`${index}-${task}`}>{task}</li>)}</ul>}
      </li>
      <li>
        <h2>Formula</h2>
        <table className="formula">
          <thead><tr><th>What LFG counted</th><th>Found</th><th>Worth each</th><th>Points</th></tr></thead>
          <tbody>
            {complexity.terms.map((term) => <tr key={term.id}>
              <td>{term.label}{term.detail.length > 0 && <small>{term.detail.join(', ')}</small>}</td>
              <td>{term.value}</td><td>{term.weight}</td><td>{term.points}</td>
            </tr>)}
          </tbody>
          <tfoot><tr><td colSpan={3}>Complexity score</td><td>{complexity.score}</td></tr></tfoot>
        </table>
      </li>
      <li>
        <h2>Brackets</h2>
        <ol className="ladder" aria-label="Complexity brackets">
          {complexity.ladder.map((step) => <li key={step.id} className={step.current ? 'is-current' : ''} aria-current={step.current ? 'step' : undefined}><strong>{step.label}</strong><small>from {step.minScore}</small></li>)}
        </ol>
        <p><strong>{complexity.label}:</strong> {complexity.plainLabel}.</p>
        <p className="assessment-intents">
          {intent.all.map((hit) => <span className="pill" key={hit.id} title={`Matched: ${hit.matched.join(', ')}`}>{hit.plainLabel}</span>)}
          {intent.fallback && <span className="pill">{intent.plainLabel}</span>}
          {intent.risks.map((hit) => <span className="pill is-risk" key={hit.id} title={`Matched: ${hit.matched.join(', ')}`}>{hit.plainLabel}</span>)}
        </p>
      </li>
      <li>
        <h2>Agents and models suggested</h2>
        <p className="assessment-agents"><strong>{agents.total}</strong> {agents.total === 1 ? 'agent' : 'agents'}</p>
        <p>{agents.plain}</p>
        <p>{models.plain}</p>
        {models.suggested && <ModelCard model={models.suggested} lead />}
        {models.alternatives.length > 0 && <details className="model-alternatives"><summary>Other options ({models.alternatives.length})</summary>
          {models.alternatives.map((model) => <ModelCard key={`${model.subscriptionId}/${model.model}`} model={model} />)}
        </details>}
      </li>
      <li>
        <h2>Subscriptions recognized</h2>
        <ul className="subscription-list">
          {assessment.subscriptions.map((subscription) => <SubscriptionRow key={subscription.id} subscription={subscription} declared={declared.includes(subscription.id)} onToggle={onToggleSubscription} />)}
        </ul>
        <p className="assessment-note">{knowledgeNote(assessment.knowledge)}</p>
      </li>
    </ol>
    <p className="assessment-note">A suggestion from LFG’s rules, not a measured probability. Nothing has been sent to a provider.</p>
  </section>
}
