"""OpenAI chat client used by the agent loop and the injection classifier.

- Tool calling: tools are strict JSON-schema function definitions.
- Structured Outputs: when `response_format` is a Pydantic model, the final
  answer is constrained to that schema and returned already parsed -- no
  hand-written JSON parsing.
- Reliability: request timeout + SDK retries with exponential backoff.
- Determinism: temperature 0.
- Cost: every call returns its token usage and USD cost, and logs one
  `llm_call` event (model, tokens, latency).
"""

from __future__ import annotations

import json
import time

import openai
from pydantic import BaseModel

from app.config import Settings
from app.core.exceptions import LLMProviderError
from app.core.logging import get_logger
from app.core.tracing import observe
from app.domain.models import TokenUsage

logger = get_logger(__name__)


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict


class ChatMessage(BaseModel):
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


class LLMResponse(BaseModel):
    content: str | None = None
    tool_calls: list[ToolCall] = []
    parsed: BaseModel | None = None  # set when response_format was given and the model answered
    refusal: str | None = None
    usage: TokenUsage = TokenUsage()


class OpenAIChatClient:
    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise LLMProviderError("OPENAI_API_KEY is required.")
        self._client = openai.AsyncOpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
        self._settings = settings
        self.model = settings.openai_model

    async def generate(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict] | None = None,
        response_format: type[BaseModel] | None = None,
        purpose: str = "agent",
    ) -> LLMResponse:
        request: dict = {
            "model": self.model,
            "messages": [_to_openai_message(m) for m in messages],
            "temperature": 0,
            "max_tokens": self._settings.llm_max_output_tokens,
        }
        if tools:
            request["tools"] = tools
        if response_format:
            request["response_format"] = response_format

        with observe(f"llm_{purpose}", as_type="generation", model=self.model) as obs:
            started = time.perf_counter()
            try:
                response = await self._client.chat.completions.parse(**request)
            except (openai.OpenAIError, ValueError) as exc:
                logger.error("llm_call_failed", extra={"ctx_purpose": purpose, "ctx_error": str(exc)})
                raise LLMProviderError(f"LLM call failed: {exc}") from exc
            latency_ms = round((time.perf_counter() - started) * 1000)

            message = response.choices[0].message
            usage = self._usage(response.usage)
            tool_calls = [
                ToolCall(id=tc.id, name=tc.function.name, arguments=json.loads(tc.function.arguments))
                for tc in (message.tool_calls or [])
            ]
            obs.update(
                output={"tool_calls": [tc.name for tc in tool_calls]} if tool_calls else message.content,
                usage_details={"input": usage.prompt_tokens, "output": usage.completion_tokens},
                cost_details={"total": usage.cost_usd},
            )

        logger.info(
            "llm_call",
            extra={
                "ctx_purpose": purpose,
                "ctx_model": self.model,
                "ctx_prompt_tokens": usage.prompt_tokens,
                "ctx_completion_tokens": usage.completion_tokens,
                "ctx_cost_usd": usage.cost_usd,
                "ctx_latency_ms": latency_ms,
                "ctx_tool_calls": [tc.name for tc in tool_calls],
                "ctx_finish_reason": response.choices[0].finish_reason,
            },
        )
        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            parsed=message.parsed if not tool_calls else None,
            refusal=message.refusal,
            usage=usage,
        )

    def _usage(self, raw) -> TokenUsage:
        prompt, completion = (raw.prompt_tokens, raw.completion_tokens) if raw else (0, 0)
        cost = (
            prompt * self._settings.llm_input_price_per_million
            + completion * self._settings.llm_output_price_per_million
        ) / 1_000_000
        return TokenUsage(llm_calls=1, prompt_tokens=prompt, completion_tokens=completion, cost_usd=round(cost, 6))


def _to_openai_message(message: ChatMessage) -> dict:
    entry: dict = {"role": message.role}
    if message.content is not None:
        entry["content"] = message.content
    if message.tool_calls:
        entry["tool_calls"] = [
            {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
            for tc in message.tool_calls
        ]
    if message.tool_call_id:
        entry["tool_call_id"] = message.tool_call_id
    return entry
