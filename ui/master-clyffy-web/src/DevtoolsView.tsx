import { useEffect, useRef, useState } from 'react'
import { MotionConfig } from 'motion/react'

import './devtools.css'
import { DevtoolsProcess } from './DevtoolsProcess'
import { DevtoolsPrompts } from './DevtoolsPrompts'
import { DevtoolsRace } from './DevtoolsRace'
import { DevtoolsTrace } from './DevtoolsTrace'
import { getDevtoolsPanel, getDevtoolsStatus, readLaunchToken, rememberToken } from './devtoolsData'
import type { Panel, PanelPrompt } from './devtoolsData'
import type { RuntimeStatus } from './types'

type Tab = 'trace' | 'race' | 'prompts' | 'process'

const TABS: Array<{ id: Tab; label: string; plain: string; needsResults: boolean }> = [
  { id: 'trace', label: 'Follow a prompt', plain: 'Every step LFG takes, live as you type', needsResults: false },
  { id: 'race', label: 'The race', plain: 'LFG against every competitor', needsResults: true },
  { id: 'prompts', label: 'Every prompt', plain: 'Each prediction against what really happened', needsResults: true },
  { id: 'process', label: 'How it ran', plain: 'Where the numbers came from', needsResults: true }
]

const POLL_MS = 4000

export function DevtoolsView({ status }: { status: RuntimeStatus | null }) {
  const [tab, setTab] = useState<Tab>('race')
  const [token, setToken] = useState<string | null>(() => readLaunchToken())
  const [tokenInput, setTokenInput] = useState('')
  const [panel, setPanel] = useState<Panel | null>(null)
  const [problem, setProblem] = useState<string | null>(null)
  const [followed, setFollowed] = useState<PanelPrompt | null>(null)
  const seen = useRef<number | null | undefined>(undefined)

  useEffect(() => {
    if (!token) return
    let stopped = false
    const refresh = async () => {
      try {
        const current = await getDevtoolsStatus(token)
        if (current.modifiedMs !== seen.current || !panel) {
          const next = await getDevtoolsPanel(token)
          if (stopped) return
          seen.current = current.modifiedMs
          setPanel(next)
        }
        if (!stopped) setProblem(null)
      } catch (value) {
        if (!stopped) setProblem(value instanceof Error ? value.message : 'Could not load the scorecard results')
      }
    }
    void refresh()
    const timer = window.setInterval(() => void refresh(), POLL_MS)
    return () => { stopped = true; window.clearInterval(timer) }
  }, [token, panel])

  const follow = (prompt: PanelPrompt) => { setFollowed(prompt); setTab('trace') }
  const active = TABS.find((item) => item.id === tab) ?? TABS[0]

  return <MotionConfig reducedMotion="user"><section className="devtools" aria-label="LFG devtools">
    <header className="devtools-head">
      <div>
        <p className="eyebrow">LFG DEVTOOLS · LOCAL ONLY</p>
        <h1>See how LFG reads a prompt, and whether it was right.</h1>
      </div>
      <p className="devtools-connection" role="status">{panel
        ? <><i className="status-dot" aria-hidden="true" />Results from {new Date(panel.generatedAt).toLocaleString()} · refreshes on its own</>
        : problem ?? (token ? 'Loading results…' : 'Results not connected')}</p>
    </header>

    <div className="devtools-tabs" role="tablist" aria-label="Devtools views">
      {TABS.map((item) => <button key={item.id} role="tab" type="button" aria-selected={tab === item.id} className={tab === item.id ? 'is-active' : ''} onClick={() => setTab(item.id)}>
        <strong>{item.label}</strong><small>{item.plain}</small>
      </button>)}
    </div>

    {active.needsResults && !panel
      ? <section className="devtools-card devtools-connect">
          <h2>Connect your scorecard results</h2>
          <p>{problem ?? 'These views show private prompts and outcomes, so they need the session token from start-devtools.ps1. Follow a prompt works without it.'}</p>
          <form onSubmit={(event) => { event.preventDefault(); const value = tokenInput.trim(); if (value) { rememberToken(value); setToken(value); setTokenInput('') } }}>
            <label>Session token <input type="password" autoComplete="off" value={tokenInput} onChange={(event) => setTokenInput(event.target.value)} /></label>
            <button type="submit">Connect</button>
          </form>
        </section>
      : <>
          {tab === 'trace' && <DevtoolsTrace status={status} panel={panel} followed={followed} onFollow={setFollowed} />}
          {tab === 'race' && panel && <DevtoolsRace panel={panel} />}
          {tab === 'prompts' && panel && <DevtoolsPrompts panel={panel} onFollow={follow} />}
          {tab === 'process' && panel && <DevtoolsProcess panel={panel} />}
        </>}
  </section></MotionConfig>
}
