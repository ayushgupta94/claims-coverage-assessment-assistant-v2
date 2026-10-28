# ADR 0006: Asynchronous claim submission, polled by the client

**Status:** accepted

## Context
An assessment takes 8-16 s (several LLM calls and tool calls). Holding the
`POST /claims` connection open for that long is fragile: load balancers and
API gateways commonly cut requests at 30-60 s, a client retry would submit
again, and the UI can show nothing until the end.

## Decision
`POST /claims` validates the claim synchronously (404, 409, 422 and 429 are
still immediate), stores it as `submitted`, starts the LangGraph workflow as a
background `asyncio` task and returns **202 Accepted** with a `Location:
/claims/{id}` header. Clients poll that URL until the status leaves
`submitted`. While the agent runs, each tool step is appended to the claim as
it finishes, so the poll shows progress.

- A run that raises marks the claim `failed`, with the reason in the audit trail.
- On shutdown, running assessments get a few seconds to finish; any that are
  cut off stay `submitted`. On startup, `submitted` claims are run again from
  the start (once the server accepts connections, since the agent calls its
  own `/mcp`), with a `requeued` audit entry.

Polling was chosen over Server-Sent Events: the state lives in MongoDB, so it
needs no new endpoint, no in-process pub/sub, works through any proxy and keeps
working across restarts or several instances. One-second updates are
enough for a run of a few seconds.

## Consequences
- `POST /claims` returns 202 instead of 201 with a final decision; the UI, the
  end-to-end test and the eval runner poll.
- Background tasks live in the web process. That suits the single free
  instance this runs on. With several instances, startup recovery could pick up
  a claim another instance is still assessing, and load would not be spread;
  the next step is a job queue (e.g. Arq on Redis, or Azure Service Bus) with
  workers, leaving this API contract unchanged.
- A re-run after an interrupted assessment repeats its LLM calls (a small cost,
  only after a restart mid-run); the daily cap is not charged twice.
