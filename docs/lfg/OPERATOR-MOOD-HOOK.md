# Operator mood hook

When a prompt to Claude Code or Codex reads as strongly frustrated, LFG adds
context for the agent before it answers. The context says what the owner has
asked for at those moments:
- grounded, transparent answers rather than reassurance;
- what has actually been verified and how, and what is still assumed;
- exactly what information is needed from them;
- one concrete next step.

It also says that replies to the tone, long apologies and claims that earlier
work was useful are not wanted. Calm and pleased prompts get nothing added.

The reading comes from configured patterns, so it is a claim, not a verified
label. Frustration takes precedence over praise, because praise words are often
sarcastic in an angry message.

## How it runs

`lfg-server prompt-hook --client claude|codex|gemini|kimi` answers a
prompt-submit hook. Claude Code and Codex (`UserPromptSubmit`) and Gemini CLI
(`BeforeAgent`) take `hookSpecificOutput.additionalContext` as JSON on stdout;
Kimi Code appends plain stdout.

- **Never blocks:** the hook never stops a prompt. A malformed payload, an
  unknown client or any error prints nothing and exits 0. A leading UTF-8
  byte-order mark, which some shells add when piping, is ignored.
- **Fast:** it loads no model. On an Intel Core Ultra 9 275HX, a call took
  12–18 ms including process start.

| Setting | Bundled file | Override |
| --- | --- | --- |
| Patterns, all-caps rule and the guidance text | [`operator-mood.v1.json`](../../crates/lfg-core/config/operator-mood.v1.json) | `LFG_OPERATOR_MOOD_CONFIG` |
| Local journal of readings | none | `--journal PATH` or `LFG_MOOD_JOURNAL` |

**Journal:** each reading is one JSON line holding the pattern IDs that fired,
the state, the prompt's length, the configuration's revision and digest, and
SHA-256 digests of the prompt, session and turn. It never holds the prompt's
words. These readings, joined to transcripts by the session digest, are the
labels for learning what leads to frustration.

**Guidance wording:** the guidance is written as statements of the owner's
stated preference, not as commands, as Claude Code's hook documentation advises
for added context. It is capped at 10,000 characters.

## Measured on the owner's prompts

On the 1,270 prompts in the private scorecard (10 November 2026):

- **How often it fires:** the bundled patterns fire on 26%.
- **What drives it:**
  - swearing: 226 prompts, 90 of them on swearing alone;
  - all-caps bursts: 205 prompts, 75 of them on caps alone;
  - 145 prompts matched two or more patterns.
- **Caps threshold:** raising the all-caps rule from two words to three changes
  the rate only to 25%.
- **Persistence:** after a frustrated prompt, the next prompt in the same
  session is also frustrated 56% of the time; after a calm one, 18%.

**Revision `baseline-2` (13 November 2026).** In use, the all-caps rule fired on
the instruction blocks that apps insert into a prompt, such as the browser
extension's text with words like `IMPORTANT` and `ALWAYS`. Every such prompt of
about 4,200 characters in four sessions read as frustrated, while the
operator's own words read as calm. Two changes:

- **Inserted blocks:** the contents of tags listed in `insertedBlocks`, such as
  `browser_instruction` and `system-reminder`, are removed before reading. A
  block cut off by the length limit runs to the end of the text.
- **Share of words:** all-caps words count as shouting only when they are also
  at least `minShare` (20%) of the prompt's words, so a few project names in a
  long message do not count.

On the same 1,270 prompts, 273 read as frustrated instead of 325, and 52
readings changed: 51 to calm and 1 to positive. After a frustrated reading the
next prompt is frustrated 59% of the time, against 55% for `baseline-1`, and
18% after a calm one.

Lowering that persistence rate is the measure of whether the guidance works. Use the journal
and the transcripts to compare sessions before and after the hook is installed.

## Install

**Claude Code** (`~/.claude/settings.json`):

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {"hooks": [{"type": "command", "command": "\"C:\\path\\to\\lfg-server.exe\" prompt-hook --client claude --journal \"C:\\private\\mood.jsonl\"", "timeout": 5}]}
    ]
  }
}
```

**Codex** (`~/.codex/hooks.json`; run `/hooks` in Codex to review and trust it):

```json
{"type": "command", "command": "\"C:\\path\\to\\lfg-server.exe\" prompt-hook --client codex --journal \"C:\\private\\mood.jsonl\"", "timeout": 5}
```

Add the Codex entry to the existing `UserPromptSubmit` hooks list.

**Gemini CLI** (`~/.gemini/settings.json`; `BeforeAgent` takes JSON only, and the
timeout is in milliseconds):

```json
{"hooks": {"BeforeAgent": [{"hooks": [{"type": "command", "name": "lfg-mood", "command": "C:/path/to/lfg-server.exe prompt-hook --client gemini --journal C:/private/mood.jsonl", "timeout": 5000}]}]}}
```

**Kimi Code** (`~/.kimi-code/config.toml`; plain stdout is added as context, and
the timeout is in seconds):

```toml
[[hooks]]
event = "UserPromptSubmit"
command = "C:/path/to/lfg-server.exe prompt-hook --client kimi --journal C:/private/mood.jsonl"
timeout = 5
```

Kimi's documentation does not name the field that carries the prompt, so the
hook reads `prompt`, `user_prompt`, `message` or `text`, whichever is present.

## Not yet established

- **Anticipating frustration:** predicting whether the next prompt will be
  frustrated is only weakly possible so far. On later prompts, the simple
  baselines reach an AUC of 0.55–0.60, with intervals that include chance. A
  trained risk score must beat them before LFG reports one.
- **Accuracy of the patterns:** nobody has checked the patterns' precision
  against the owner's own judgement.
