# ADR 0002: LangGraph only for the human-review pause

**Status:** accepted

## Context
Escalated claims wait for a human, possibly for hours. Meanwhile the free
Render instance may sleep or be redeployed. The workflow must stop, keep its
state, and continue exactly where it stopped when the reviewer decides.

## Decision
The claim workflow is a four-node LangGraph graph (`workflow/graph.py`):
`assess → apply_policy → route → human_review`. `human_review` calls
`interrupt()`; a MongoDB checkpointer stores the state under
`thread_id = claim_id`; the review endpoint resumes it with `Command(resume=...)`.
The agent loop inside `assess` stays hand-written (ADR 0001).

## Consequences
- Durable pause/resume, conditional routing and checkpointing come from a
  well-known library instead of custom state-machine code.
- LangGraph is used for orchestration only; it does not own prompts or tools.
- The graph state holds plain dicts so checkpoints serialise cleanly.
