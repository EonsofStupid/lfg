# Private Codex turn evidence

`ml/lfg-encoder/codex_turn_evidence.py` links selected, inventoried Codex user
requests to preceding `turn_context` records with the same recorded turn ID.
It recovers the configured model and effort, not a recommendation, proof of
user preference, successful outcome, agent count, reviewed assessment, or
training label. Provider settings and session histories may themselves contain
private information. Keep the output local and out of Git, and review and redact
it before using it in training. Pattern screening or a hash is not proof that
secrets are absent.

The operator selects an existing development plan and configures the input,
prefix, line, record, context and metadata bounds. The full plan is checked for
test splits before any source is opened. This tool reads selected plan sources
and, for Codex records, verifies the stored original session prefix and selected
message hashes. It matches only the exact recorded turn ID; an absent ID or
conflicting model/effort settings yields unknown context. Repeated contexts with
the same turn ID and identical model/effort values are represented by their
latest matching line and a count. The source file is checked for changes while
read, but is not locked afterward.

The output contains candidate IDs, declared project and family, split, original
text and session-prefix hashes, hashed turn IDs, model/effort settings and the
context line and hash. The extractor does not copy prompt text or source-file
paths, but input-provided IDs, project names and model metadata may themselves
contain private information. Each
row records plan, configuration, implementation and verifier hashes, Python
version, `outcome: unknown`, `agent_count: null`, `review_status: unreviewed`,
and `promotion_eligible: false`. These hashes are sensitive and are not
anonymization; predictable text can be guessed. Windows permissions follow the
parent directory's ACL. An output inside a Git worktree must be ignored and not
already tracked, and an existing file is never overwritten.

Project and project-group split declarations, and family and normalized-text
crossings for the selected Codex candidates, are checked. This does not verify
the true project identity, all cross-provider duplicates, shared context,
semantic leakage, ownership, or evaluation independence. Claude candidates are
not processed. A source changing during extraction causes the run to fail; use
a new output path after the session stabilizes. The tool has no background
collection, and does not modify session files or the review workspace.

```powershell
python ml/lfg-encoder/codex_turn_evidence.py `
  --plan PRIVATE_DEVELOPMENT_PLAN.json `
  --config PRIVATE_REVIEWED_CONFIGURATION.json `
  --output NEW_PRIVATE_EVIDENCE.jsonl
```

The [example configuration](../../ml/lfg-encoder/codex-turn-evidence.example.json)
is a starting point; adapt the limits to the source and machine. The program
prints aggregate counts and hashes, not the model names or requests.

On 2026-11-02, the local development plan's 555 selected Codex candidates (520
training and 35 validation) all had an exact turn-ID match with model and effort
present. The remaining 90 plan candidates are Claude messages and were not
processed. Repeated but consistent context records were present. The held-out
Makers-Impulse plan was not supplied or opened. The output remains private,
unreviewed usage evidence; no assessment heads or owned LFG model were trained
from it. Exact matching describes recorded execution settings only, and does not
establish which settings were optimal or that a task succeeded.
Two runs with the same inputs, code and Python version produced identical
output bytes. That local check does not establish cross-platform reproducibility.
