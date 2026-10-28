# ADR 0005: One deployable serving both REST and MCP

**Status:** accepted

## Context
The tools must be callable by this app's agent and by external MCP clients
(VS Code, Claude Desktop). They could live in a separate MCP service.

## Decision
The MCP server is mounted at `/mcp` inside the FastAPI app, behind the same
access-code check. The agent uses a real MCP client against that endpoint over
loopback HTTP (not a direct function call), forwarding its request id.

## Consequences
- One container, one free Render service, one set of secrets.
- The agent exercises exactly the protocol path external clients use, so MCP
  wiring is covered by the end-to-end test.
- Loopback HTTP adds a few milliseconds per tool call; the session is reused
  for the whole run. If the tools needed independent scaling or ownership, the
  mount could become its own service without changing the agent.
