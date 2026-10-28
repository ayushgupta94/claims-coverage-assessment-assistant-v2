# ADR 0001: A hand-written agent loop instead of an agent framework

**Status:** accepted

## Context
The agent needs one model, four tools and a bounded loop: call tools until it
can answer, then return a structured decision. Agent frameworks add their own
abstractions for prompts, tools, memory and control flow.

## Decision
The ReAct loop is about 60 lines of plain Python in `agent/orchestrator.py`:
call the LLM with strict tool schemas, execute requested tools over MCP, feed
results back, stop at a structured answer or after six iterations.

## Consequences
- Every step is visible and testable with a scripted fake LLM (`tests/fakes.py`).
- Guardrails (tool-argument checks), timing, logging and fallbacks sit exactly
  where they are needed, without framework hooks.
- Features a framework would give for free (streaming, memory, multi-agent
  routing) would have to be built if ever needed. None are needed today.
