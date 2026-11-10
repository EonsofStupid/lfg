# Comparison harness

The harness answers two questions with the owner's own prompts:
- Does LFG rival Claude Code and Codex at judging a prompt?
- Which of LFG's jobs fit in about 650 MB, and which are asking too much of
  one small model?

Every contestant judges the same prompts. Each answer is scored against what
actually happened next, not against another model's opinion.

## Jobs and ground truth

| Job | Ground truth | Score |
| --- | --- | --- |
| Work level | Tool calls the agent actually made, in five levels (0–2, 3–9, 10–30, 31–100, over 100) | Rank correlation with tool calls (ranks within each tool, then pooled), exact and within-one level accuracy |
| Intent | What the agent actually did, read from its tools | Accuracy and macro-F1 |
| Clarification risk | Whether the person's next prompt was frustrated | AUC |
| CO-STAR-A recipe | The person's next prompt | Share of the new words the person had to add that the recipe already raised |

The intent labels are:
- `delegate`: the agent spawned sub-agents;
- otherwise `change`: it edited files;
- otherwise `run`: it ran commands other than reads;
- otherwise `investigate`: it only read or searched;
- otherwise `answer`: it used no tools.

Commands are classified by a pattern, so intent labels are claims. The recipe
score is a rough proxy; the real test of a recipe is blind review by the owner.

Each score is reported for all prompts and for later prompts, with 95%
intervals that resample whole projects. Later prompts are the last 30% in each
tool, and they are the fairest test of real use.

## Contestants

Language models get the same instructions,
[`harness_instructions.v1.md`](../../ml/lfg-encoder/harness_instructions.v1.md),
and must reply with one JSON object. Malformed fields count as missing, not as
guesses.

| Kind | What runs |
| --- | --- |
| `lfg` | A running LFG's `/api/lfg/assess`. `useModel: false` takes the rules' own bracket. |
| `precomputed` | Scores made elsewhere, such as LFG's cross-validated model scores. |
| `claude` | `claude -p` on the owner's subscription, with no tools, hooks, MCP servers or saved session, in an empty folder. |
| `codex` | `codex exec` on the owner's subscription: read-only, ephemeral, hooks off, in an empty folder. |
| `openrouter`, `openai-compatible` | Any OpenAI-compatible endpoint, such as OpenRouter or a local llama.cpp or vLLM server. The key is read from the environment variable `apiKeyEnv` and never stored. |

**Fair scoring:** a model trained on these prompts must not be scored on them.
LFG's work-level model is scored with project-fold predictions, made by models
that never saw the project, and with later-prompt predictions from models
trained only on earlier prompts. It is never scored live on prompts it trained
on.

## Running it

```text
python ml/lfg-encoder/harness_run.py --config PRIVATE/harness.json [--limit 5] [--only claude]
python ml/lfg-encoder/harness_score.py --config PRIVATE/harness.json --json PRIVATE/report.json
```

Answers append to one private JSONL file per contestant, so a run can be
stopped and resumed. See [`harness.example.json`](../../ml/lfg-encoder/harness.example.json).

The ground truth comes from the owner's private logs and stays local, and so do
the answers.

**Where prompts go:** Claude and Codex receive the redacted prompts through the
owner's subscriptions, which is where the prompts came from. OpenRouter passes
prompts on to third-party hosts. Use it on the owner's prompts only with their
explicit agreement; otherwise give it public test prompts.

## Is 650 MB enough, or is it too many jobs?

Compare scores across sizes for each job separately, then compare one model
doing all three jobs against one model per job, at the same size:

1. **Size ladder per job:** train the same recipe at several sizes, such as
   about 150 MB, 650 MB and 1.3 GB, plus a 4B-class teacher. Plot each job's
   score against size, with Claude and Codex as the ceiling. A job whose curve
   is still climbing at 650 MB needs more capacity; a job that is flat by then
   fits.
2. **One model or several:** at 650 MB, train one shared model with three heads
   (work level, intent, clarification risk) and three single-job models. If the
   shared model trails the single-job models clearly, 650 MB is being asked to
   do too much.
3. **Generation is a separate question:** writing a recipe needs a small
   generative model, not an encoder. Compare 0.5–4B generative models' recipes
   with Claude's and Codex's by blind review and by the anticipation proxy.
