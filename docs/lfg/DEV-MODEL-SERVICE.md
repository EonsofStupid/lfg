# Dev model service

The dev model route lets the owner's dev version of LFG call an open model running on
their own hardware, so it can be used on real work and corrected. It sits next to the
[owned assessment service](OWNED-ASSESSMENT-SERVICE.md) and does not replace the rules
in `POST /api/lfg/assess` or fall back to them.

**The model behind this route is not LFG's own.** Its engine reports `kind: "open_model"`
and the configured label, the Inspector names it as an open model, and saved feedback
records its engine id. Nothing released to others may present it as LFG's model.

## Setup

1. Serve the model on a loopback port of the machine running LFG. If the model runs on
   another machine, forward its port over SSH. The service refuses any other address.
2. Copy `crates/lfg-server/config/dev-model-service.example.json` outside the repository
   and set it up. Set `enabled` to true.
3. Set `LFG_DEV_MODEL_CONFIG` to that file, `LFG_DEV_MODEL_TOKEN` to a random secret of at
   least 32 characters, and the variable named by `apiKeyEnv` to the model server's key.
   `lfg-server` needs no extra build feature for this route.
4. In the Inspector, set **Model to compare** to the dev model and enter the token.

The instruction lives in a private file pinned by `expectedInstructionSha256`, so a changed
file stops the service from starting rather than silently changing its answers.
`requestExtensions` are passed to the model server, except for fields the service sets itself.

## Which models it can suggest

Put `{{available_models}}` in the instruction and the service fills it on each request with
the models LFG has learned are available, with each model's role and provider description.
No model names need to be kept in the instruction, and the result's `modelListSha256`
identifies the list that was used.

Small models sometimes invent a name. A name that isn't on the list is moved from
`suggested_models` to `unlisted_models`, and the raw answer stays in `outputText`. If LFG
knows of no models, the list reads "unknown" and no name counts as a suggestion. Without the
slot in the instruction, this checking doesn't apply.

## API

`GET /api/lfg/assess/dev-model/status` and `POST /api/lfg/assess/dev-model` use the same
access rules as the owned route: a loopback listener and peer, a matching Host and Origin,
and the `X-LFG-Session-Token` header. Requests take `text`, with optional `context` and
`contextSha256`, and unknown fields are ignored.

Results use the owned route's statuses. `ok` means the model finished and its JSON meets
`requiredOutputFields`; it does not mean the answer is right. `truncated` means it reached
`maxGeneratedTokens`, `incomplete_output` and `malformed_output` mean the answer doesn't meet
the contract, and `unavailable` means the model server couldn't be reached.

## Limits

- **Identity is stated, not checked.** The service can't see which weights the model server
  loaded, so `weightsSha256` is the operator's claim.
- **Ready means configured.** A `ready` status doesn't mean the model server is running.
- **The list isn't limited to declared subscriptions.** It holds every model marked available,
  and a listing doesn't prove the user can use that model. A model seen only in session
  history isn't marked available, so the list can miss models the user does have.
- **Answers are proposals.** Prompts and answers aren't logged or saved by this route, and
  an answer enters the feedback journal only when the owner saves it.
