import type { Panel } from './devtoolsData'

const STATUS_LABEL = { fixed: 'Fixed', open: 'Still open', noted: 'Noted' }
const STATUS_ICON = { fixed: '✓', open: '!', noted: '•' }

const short = (digest: string) => digest.slice(0, 12)

function Step({ number, lane, title, children }: { number: number; lane: 'pc' | 'spark'; title: string; children: React.ReactNode }) {
  return <li className={`process-step lane-${lane}`}>
    <span className="process-number" aria-hidden="true">{number}</span>
    <div><small>{lane === 'pc' ? 'THIS PC' : 'SPARK'}</small><h3>{title}</h3>{children}</div>
  </li>
}

export function DevtoolsProcess({ panel }: { panel: Panel }) {
  const { dataset, process, rows, audits, predictions } = panel
  const redacted = Object.entries(dataset.redactions)
  const runs = Object.entries(process.runs).filter(([id]) => rows.some((item) => item.id === id && item.kind === 'opponent'))
  const held = predictions.filter((prediction) => prediction.held).length
  const devRows = rows.filter((row) => row.kind === 'lfg' && row.sizeBytes > 0)
  const next = devRows.length > 0 ? 7 : 6
  return <div className="devtools-stack">
    <section className="devtools-card">
      <div className="card-heading"><div><h2>How these numbers were made</h2><p>Each step, where it ran, and what it produced. The Spark that ran the competitors never held outcomes, so no competitor could see the answers.</p></div></div>
      <div className="lane-key" aria-hidden="true"><span>This PC</span><span>Spark</span></div>
      <ol className="process-flow">
        <Step number={1} lane="pc" title="Read your session logs"><p>{dataset.sessions} sessions across {dataset.projects} projects, {dataset.first} to {dataset.last}. {dataset.snapshot}.</p></Step>
        <Step number={2} lane="pc" title="Screen for secrets first"><p>{redacted.length ? `Removed before anything else used the text: ${redacted.map(([name, count]) => `${name.replace(/-/g, ' ')} (${count})`).join(', ')}.` : 'Nothing matched before anything else used the text.'} This is a screening aid, not proof that every secret is gone.</p></Step>
        <Step number={3} lane="pc" title="Freeze the test"><p>{dataset.rows.toLocaleString()} prompts: {dataset.long} of 40+ characters and {dataset.short} shorter. Text and outcomes go in separate files.</p><p><code>{short(dataset.sha256['prompts.redacted.jsonl'] ?? '')}</code> prompts · <code>{short(dataset.sha256['outcomes.jsonl'] ?? '')}</code> outcomes</p></Step>
        <Step number={4} lane="pc" title="Write down predictions"><p>{predictions.length} predictions and one fixed output per competitor, before any score existed.</p><p><code>{short(process.predictionsSha256)}</code> predictions file</p></Step>
        <Step number={5} lane="spark" title="Run each competitor blind">
          <p>The Spark received prompts and ids only. The kernel refused internet connections for each run.</p>
          <div className="dev-table-wrap"><table className="plain-table"><thead><tr><th>Competitor</th><th>Scored</th><th>Typical</th><th>Slow case</th><th>Network block</th><th>Connections seen</th></tr></thead><tbody>
            {runs.map(([id, run]) => {
              const row = rows.find((item) => item.id === id)
              return <tr key={id}><td>{row?.label ?? id}</td><td>{row?.latency?.rows.toLocaleString() ?? 'n/a'}</td><td>{row?.latency ? `${row.latency.p50Ms.toFixed(1)} ms` : 'n/a'}</td><td>{row?.latency ? `${row.latency.p95Ms.toFixed(1)} ms` : 'n/a'}</td><td>{run.kernelBlock ? '✓ kernel' : '✗ none'}</td><td>{run.connections}</td></tr>
            })}
          </tbody></table></div>
        </Step>
        {devRows.length > 0 && <Step number={6} lane="spark" title="Train the dev rows on your outcomes">
          <p>One model per project group, each scored only on projects it never saw. These Sparks hold prompts and outcome levels, because training needs them. The kernel refused internet connections here too.</p>
          <div className="dev-table-wrap"><table className="plain-table"><thead><tr><th>Dev row</th><th>Ran on</th><th>Size</th><th>Scored</th><th>Typical</th></tr></thead><tbody>
            {devRows.map((row) => <tr key={row.id}><td>{row.label}</td><td>{row.where}</td><td>{row.sizeNote}</td><td>{row.latency?.rows.toLocaleString() ?? 'n/a'}</td><td>{row.latency ? `${row.latency.p50Ms.toFixed(1)} ms` : 'n/a'}</td></tr>)}
          </tbody></table></div>
        </Step>}
        <Step number={next} lane="pc" title="Join scores to real outcomes"><p>All outcomes are joined here, so the final scoring happens on this PC. Models trained on your outcomes are always tested on a project they never saw.</p></Step>
        <Step number={next + 1} lane="pc" title="Check the checker"><p>{audits.filter((audit) => audit.passed).length} of {audits.length} audits passed: random scores and shuffled outcomes sit near 0, and a second implementation of the metric agrees.</p></Step>
        <Step number={next + 2} lane="pc" title="Read the result"><p><strong>{panel.decision.title}</strong> {panel.decision.detail} {held} of {predictions.length} predictions held.</p></Step>
      </ol>
    </section>

    <section className="verdict-grid">
      <article className="devtools-card">
        <h2>What went wrong along the way</h2>
        <ul className="incident-list">{panel.incidents.map((incident) => <li key={incident.title} className={`incident-${incident.status}`}>
          <b aria-hidden="true">{STATUS_ICON[incident.status]}</b><div><small>{STATUS_LABEL[incident.status]}</small><strong>{incident.title}</strong><p>{incident.detail}</p></div>
        </li>)}</ul>
      </article>
      <article className="devtools-card">
        <h2>Not measured yet</h2>
        <ul className="plain-list">{panel.notMeasured.map((item) => <li key={item}>{item}</li>)}</ul>
      </article>
    </section>
  </div>
}
