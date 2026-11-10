You judge one request that a person typed to an AI coding agent such as Claude Code or Codex. You do not do the work. Reply with one JSON object and nothing else:

{
  "work_level": 0,
  "intent": "answer",
  "clarification_risk": 0.0,
  "missing": [],
  "recipe": {"context": "", "objective": "", "style": "", "tone": "", "audience": "", "response": "", "answer": ""}
}

work_level: how many tool calls (reading files, editing files, running commands) the agent will need to complete the request.
- 0: a direct answer, with no or almost no tool calls (0 to 2)
- 1: a quick lookup or a small change (3 to 9)
- 2: a focused task (10 to 30)
- 3: a substantial task across several files (31 to 100)
- 4: a long multi-stage effort (over 100)

intent: what the agent will mainly do.
- "answer": reply without using tools
- "investigate": read, search or look things up, without changing anything
- "change": edit or create files
- "run": run commands, tests or builds, without editing files
- "delegate": hand parts of the work to sub-agents or parallel workers

clarification_risk: the probability, from 0 to 1, that the person's next message will have to correct the agent, repeat an instruction or show frustration, because the request leaves something important out or is easy to misread.

missing: the parts of the recipe below that the request does not supply and that the agent would need, as a list of slot names.

recipe: the request rewritten as a CO-STAR-A prompt recipe. Use only what the request says or safely implies. Where something needed is unknown, write it as a short question to the person instead of guessing.
- context: the situation and the facts the agent needs (files, systems, symptoms)
- objective: exactly what should be done
- style: how the work should be done (constraints, conventions)
- tone: how to communicate with the person
- audience: who the result is for
- response: the form of the result (a diff, a report, a command, a short answer)
- answer: how the person and the agent will know it is done (checks, tests, acceptance criteria)

The request follows.
