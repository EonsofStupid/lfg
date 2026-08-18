import { useEffect, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import type { ClassificationSnapshot, EscalationTarget, RoutingNeed } from './types'

const TONES = ['cyan', 'lime', 'magenta', 'amber', 'cyan', 'lime', 'magenta', 'amber']
const NEEDS: RoutingNeed[] = ['balanced', 'capability', 'efficiency', 'cost']
const sameTarget = (a: EscalationTarget | null, b: EscalationTarget) => a !== null && a.provider === b.provider && a.model === b.model

export function PromptBranchView({
  snapshot,
  routingNeed,
  onRoutingNeedChange,
  overrideTarget,
  onOverride
}: {
  snapshot: ClassificationSnapshot | null
  routingNeed: RoutingNeed
  onRoutingNeedChange: (need: RoutingNeed) => void
  overrideTarget: EscalationTarget | null
  onOverride: (target: EscalationTarget | null) => void
}) {
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [burst, setBurst] = useState<number[]>([])

  useEffect(() => {
    if (!snapshot) return
    setBurst((current) => [...current, snapshot.durationMicros].slice(-14))
  }, [snapshot?.revision, snapshot?.durationMicros])

  if (!snapshot) return <div className="empty-graph">Begin typing to activate the CPU signal field.</div>

  const rows = snapshot.categories.slice().sort((a, b) => b.percent - a.percent)
  const selected = rows.find((row) => row.id === selectedId) ?? null
  const maxBurst = Math.max(1, ...burst)
  const activeChoice = overrideTarget ?? snapshot.recommendedEscalation

  return (
    <div className="branch-view">
      <div className="cpu-burst" title="Real measured CPU time per classification pass, most recent on the right">
        <span className="eyebrow">CPU · live poll</span>
        <div className="cpu-burst-bars">
          {burst.map((value, index) => (
            <motion.span
              key={index}
              className="cpu-burst-bar"
              initial={{ height: 2, opacity: 0.3 }}
              animate={{ height: Math.max(3, (value / maxBurst) * 26), opacity: index === burst.length - 1 ? 1 : 0.55 }}
              transition={{ duration: 0.2 }}
            />
          ))}
        </div>
        <span className="cpu-burst-value">{(snapshot.durationMicros / 1000).toFixed(2)} ms · rev {snapshot.revision} · {snapshot.tokenEstimate} tok</span>
      </div>

      <div className="branch-stage">
        <div className="branch-trunk">
          <span className="brand-mark">P</span>
          <span>PROMPT</span>
        </div>

        <ul className="branch-list">
          <AnimatePresence initial={false}>
            {rows.map((category, index) => (
              <motion.li
                key={category.id}
                layout
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.25 }}
                className={`tone-${TONES[index % TONES.length]} ${category.id === selectedId ? 'is-selected' : ''} ${category.percent >= 50 ? 'is-dominant' : ''}`}
              >
                <motion.span
                  className="branch-connector"
                  initial={{ width: 0 }}
                  animate={{ width: `${Math.max(6, category.percent)}%` }}
                  transition={{ duration: 0.45, ease: 'easeOut' }}
                />
                <button
                  className="branch-node"
                  title={`${category.label} · ${category.percent.toFixed(0)}%${category.matchedKeywords.length ? ' · matched: ' + category.matchedKeywords.join(', ') : ' · no keyword evidence, base prior only'}`}
                  onClick={() => setSelectedId((current) => (current === category.id ? null : category.id))}
                >
                  <span className="branch-label">{category.label}</span>
                  <span className="branch-percent">{category.percent.toFixed(0)}%</span>
                </button>
              </motion.li>
            ))}
          </AnimatePresence>
        </ul>

        <div className="branch-subagents" title="Estimated concurrent sub-agents this task would fan out to">
          <span>Predicted sub-agents</span>
          <strong>{snapshot.predictedSubagents}</strong>
        </div>

        <AnimatePresence>
          {snapshot.needsBiggerModel && (
            <motion.div
              className="branch-escalation"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 8 }}
            >
              <div className="escalation-header">
                <span className="eyebrow">Needs a bigger model</span>
                <div className="routing-need-switch" role="group" aria-label="Routing priority">
                  {NEEDS.map((need) => (
                    <button
                      key={need}
                      className={routingNeed === need ? 'is-active' : ''}
                      onClick={() => onRoutingNeedChange(need)}
                      title={need === 'balanced' ? 'weighted happy medium of capability and tool-loop efficiency' : `optimize purely for ${need}`}
                    >
                      {need}
                    </button>
                  ))}
                </div>
              </div>

              <p className="escalation-active-choice">
                {overrideTarget
                  ? <>Zero-shot override: <code>{overrideTarget.provider}/{overrideTarget.model}</code><button className="clear-override" onClick={() => onOverride(null)}>use recommendation</button></>
                  : snapshot.recommendedEscalation
                    ? <>Recommended ({routingNeed}): <code>{snapshot.recommendedEscalation.provider}/{snapshot.recommendedEscalation.model}</code></>
                    : 'No active subscription covers this task yet — activate one above to get a recommendation.'}
              </p>

              <div className="branch-escalation-targets">
                {snapshot.escalationCatalog.map((candidate, index) => {
                  const isChosen = sameTarget(activeChoice, candidate.target)
                  const isOverride = sameTarget(overrideTarget, candidate.target)
                  return (
                    <motion.button
                      key={`${candidate.target.provider}-${candidate.target.model}`}
                      type="button"
                      className={`branch-escalation-chip ${candidate.active ? '' : 'is-locked'} ${isChosen ? 'is-chosen' : ''}`}
                      initial={{ opacity: 0, scale: 0.9 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ delay: index * 0.05 }}
                      disabled={!candidate.active}
                      title={
                        candidate.active
                          ? `capability ${(candidate.target.rawCapability * 100).toFixed(0)}% · efficiency ${(candidate.target.toolLoopEfficiency * 100).toFixed(0)}% · ${candidate.target.costTier} cost — click to zero-shot override`
                          : `activate ${candidate.target.subscriptionId} to unlock this model`
                      }
                      onClick={() => onOverride(isOverride ? null : candidate.target)}
                    >
                      <code>{candidate.target.provider}/{candidate.target.model}</code>
                      <small>{candidate.target.role}</small>
                      {isOverride ? <span className="chip-tag">your override</span>
                        : isChosen ? <span className="chip-tag">recommended</span>
                        : !candidate.active ? <span className="chip-tag chip-tag-locked">locked</span>
                        : null}
                    </motion.button>
                  )
                })}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <AnimatePresence>
        {selected && (
          <motion.aside
            className="inspector branch-inspector"
            initial={{ opacity: 0, x: 40 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 40 }}
          >
            <button onClick={() => setSelectedId(null)} aria-label="Close category detail">×</button>
            <p className="eyebrow">CATEGORY EVIDENCE</p>
            <h2>{selected.label}</h2>
            <dl>
              <div><dt>Score</dt><dd>{selected.percent.toFixed(2)}%</dd></div>
              <div>
                <dt>Matched</dt>
                <dd>
                  {selected.matchedKeywords.length
                    ? selected.matchedKeywords.map((keyword) => <code key={keyword} className="keyword-chip">{keyword}</code>)
                    : 'no literal keyword hit — base prior only, this category was not detected in the text'}
                </dd>
              </div>
              <div><dt>Category id</dt><dd><code>{selected.id}</code></dd></div>
            </dl>
          </motion.aside>
        )}
      </AnimatePresence>
    </div>
  )
}
