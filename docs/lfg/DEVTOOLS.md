# LFG: devtools

The Devtools view in LFG's web UI shows how LFG reads a prompt and how it scored
against real outcomes. It has four tabs:

- **Follow a prompt:** every step of the assessment, live as you type, then what
  really happened if the prompt came from your logs.
- **The race:** each predictor on the [scorecard](EVIDENCE-ANCHORS.md#scorecard), by
  kind of prompt and by measure, with size against score, written predictions and
  audits.
- **Every prompt:** each prompt's rating against the tool calls that followed, with
  the biggest misses.
- **How it ran:** where each step ran, what it produced, what went wrong and what
  hasn't been measured.

Follow a prompt works with no setup. Open `/?view=devtools`. It shows LFG's rules,
because no owned model is connected to it yet.

## Connecting results

The other tabs read one results file, which can hold redacted prompts and outcomes.
It stays out of Git, in the private data folder.

| Setting | Meaning |
| --- | --- |
| `LFG_DEVTOOLS_PANEL_FILE` | Path to the results file, schema `lfg.devtools-panel.v1` (see `ui/master-clyffy-web/src/devtoolsData.ts`) |
| `LFG_DEVTOOLS_TOKEN` | Session token of at least 32 bytes. Use a new random value for each run |
| `LFG_DEVTOOLS_ALLOWED_HOSTS`, `LFG_DEVTOOLS_ALLOWED_ORIGINS` | Extra loopback hosts and origins, for a development server |

The server reads the file on each request, and the page checks for changes every few
seconds, so a new scorecard run appears on its own. Results are served only on a
loopback listener, to loopback peers, with the shared Host, Origin and token checks,
and are never cached. A launcher can pass the token as `#devtools=<token>`: the
fragment never reaches a server, and the page removes it from the address bar.

The tool that writes the results file is private for now and needs to move into the
repository with its redaction.
