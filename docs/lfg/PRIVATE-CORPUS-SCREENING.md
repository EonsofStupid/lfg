# Private corpus screening for an owned LFG model

This tool prepares **unreviewed text-only development candidates** from an
explicitly selected owner-authorized project/conversation plan. It applies
operator-configured pattern replacements, retains content digests and declared
project/family/split provenance, and does not invent assessment labels, teacher
responses, outcomes or human approvals. It does not train a model. The current
output is an intermediate for future owned-model training, not input to the
existing supervised classifier or assessment-head trainer.

## Privacy and limits

The [example policy](../../ml/lfg-encoder/private-corpus-screen.example.json)
is a starting point, not a complete secret detector. GitHub describes how
[secret-scanning coverage depends on patterns and context](https://docs.github.com/en/code-security/reference/secret-security/secret-scanning-scope);
[OWASP also notes](https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html)
that human-created passwords may not match detection rules. Review and extend
the policy for the selected sources. A match can be a false positive; absence of
a match does not establish that a record has no secret or private information.
Potentially exposed credentials require the owner's ordinary credential
handling and rotation decisions; this tool neither tests nor transmits them.

Inputs, outputs, identifiers, hashes and reports remain private. The program
reads original candidate text into local process memory and writes only the
screened text and minimal provenance into a new output directory. The original
source and review workspace remain unchanged. A source record hash can be
matched against the retained candidate file and original inventory; those
references are claims, not a fresh verification of the underlying session.
SHA-256 hashes of short or predictable text can reveal information by guessing.
Do not publish the output or put it in Git; Git ignore is not encryption, and
Windows access follows the directory's inherited ACL.

Every output record says `review_status: unreviewed` and
`redaction_complete: false`; the manifest separately says original source and
independent split are unverified, with no promotion eligibility. Under the
owner's authorization, local exploratory training may use appropriately
protected data while preserving these limitations. Do not describe it as human
reviewed, fully redacted, independently evaluated or production-qualified.
Assessment supervision still requires actual task targets and outcomes, not
labels inferred from an arbitrary request or fabricated examples.

The rules are trusted operator-supplied Python regular expressions. Byte,
record, and match limits constrain the processed material and output, but Python
`re` does not provide a hard per-match time limit: a pathological expression
can consume excessive CPU even on bounded input. Use reviewed expressions and
an externally resource-bounded process when running untrusted or experimental
policies. The tool checks inputs again before writing but does not lock source
files; concurrent changes between checks are not impossible. A failed write may
leave a partial new output directory without a manifest; treat that as incomplete
and use a different new directory after examining the failure.

## Operation

1. Copy and review the policy in a private location. Set the rules, replacements
   and all byte/record/match limits for the source and machine. Empty patterns,
   zero-length matches, duplicate rule IDs and oversized inputs fail closed.
2. Select a development plan with only `train` and `validation` sources and no
   review ledger. The tool checks the full split list before opening any source;
   `test` is rejected. Project, group, family and post-screening normalized-text
   crossings are rejected. These checks cannot find every paraphrase, shared
   conversation, source overlap, mislabeled project or data-rights issue.
   The final screened text is also checked against all configured patterns;
   overlapping replacements that reintroduce a match fail instead of being saved.
3. Run to a **new** private output location. The `--policy-read-limit` bounds
   reading the policy before its own limits are known. Existing outputs are never
   overwritten. Within a Git working tree, every possible output file must be
   ignored and must not already be tracked, even as a deleted file. Paths below
   are placeholders, not source or target defaults:

```powershell
python ml/lfg-encoder/screen_private_corpus.py `
  --plan PRIVATE_DEVELOPMENT_PLAN.json `
  --policy PRIVATE_SCREEN_POLICY.json `
  --policy-read-limit 65536 `
  --output NEW_PRIVATE_OUTPUT_DIRECTORY
```

`train.jsonl` and, when present, `validation.jsonl` contain screened candidate
text and per-record original/screened hashes and replacement counts. The
`manifest.json` records source-file, plan, policy, implementation, helper and
output hashes, Python and Unicode versions, counts and limitations. It contains
no matched secret or prompt text, but its hashes and source counts remain
sensitive. Exact replay assumes the same source bytes, policy, implementation,
helper code, runtime and Unicode behavior; these identities do not authenticate
the operator or prove privacy. The existing supervised review UI continues to
review its original workspace; this text-only output is not yet a review-UI
workspace or an inference-ready model.

## Local evidence, 2026-11-02

The tool screened the existing private development plan into
`ml/lfg-encoder/dataset/private/screened-text-candidates-20261102-v4` in the
original checkout. It processed **598 train and 47 validation** requests; two
records matched the example bearer-credential rule and had those matches
replaced. The result does not establish whether the matches were valid secrets
or whether other secrets remain. The held-out Makers-Impulse source was not in
the development plan and was not opened by this run. Prior raw input and the three
earlier screening outputs are retained separately.

The screened text totals **313,858 UTF-8 bytes for training** and **17,705 for
validation**, before JSON/provenance overhead. These counts do not establish
token count, training sufficiency or evaluation independence. They are a small
starting sample for an owned language model, not evidence that the desired
general capabilities can be learned from it.

A separate metadata-only inventory of selected tracked code/document extensions
in the current authorized LFG, RRFlow, DevPulseLabs and SWAG checkouts counted
224, 1,650, 710 and 179 files respectively, about 31.3 MB combined. It excluded
common build/vendor/generated/fixture directories and did not read file content,
audit licensing or secrets, verify revision cleanliness, deduplicate history, or
prepare these files for training. Treat the numbers as an opportunity for
additional source selection, not clean or independent training data. The
Makers-Impulse holdout was not inventoried by this check.
