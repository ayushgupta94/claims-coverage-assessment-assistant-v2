"""Access-code authentication for the public demo.

Codes are configured as a comma-separated ACCESS_CODES env var (e.g. one per
recruiter) and sent as `X-Access-Code: <code>` -- or `Authorization: Bearer
<code>`, which is what MCP clients such as VS Code / Claude Desktop send.
The same check guards the REST API (a FastAPI dependency) and the mounted
MCP endpoint (an ASGI middleware). Fails closed: no codes configured means
every request is rejected.
"""

from __future__ import annotations

import json
import secrets

from fastapi import Depends, Request

from app.api.deps import get_app_settings
from app.config import Settings
from app.core.exceptions import AuthenticationError


def is_valid_access_code(provided: str | None, settings: Settings) -> bool:
    if not provided:
        return False
    # compare_digest against every code: constant-time, no early exit on a match.
    matches = [secrets.compare_digest(provided, code) for code in settings.access_code_list]
    return any(matches)


def extract_access_code(headers) -> str | None:
    if code := headers.get("x-access-code"):
        return code
    authorization = headers.get("authorization", "")
    return authorization[7:].strip() if authorization.lower().startswith("bearer ") else None


def verify_access_code(request: Request, settings: Settings = Depends(get_app_settings)) -> None:
    if not is_valid_access_code(extract_access_code(request.headers), settings):
        raise AuthenticationError("Missing or invalid access code.")


class AccessCodeMiddleware:
    """Wraps the mounted MCP ASGI app so /mcp needs the same access code."""

    def __init__(self, app, settings: Settings) -> None:
        self._app = app
        self._settings = settings

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "http":
            headers = {key.decode().lower(): value.decode() for key, value in scope["headers"]}
            if not is_valid_access_code(extract_access_code(headers), self._settings):
                body = json.dumps({"error_code": "unauthorized", "message": "Missing or invalid access code."})
                await send(
                    {"type": "http.response.start", "status": 401, "headers": [(b"content-type", b"application/json")]}
                )
                await send({"type": "http.response.body", "body": body.encode()})
                return
        await self._app(scope, receive, send)
