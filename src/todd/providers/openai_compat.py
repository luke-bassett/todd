from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from todd.provider import (
    AssistantResponse,
    Message,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
)


class OpenAICompatibleProvider:
    """Provider for any server speaking the OpenAI Chat Completions API.

    Works with OpenAI, Ollama (base_url='http://localhost:11434/v1'),
    LM Studio, vLLM, OpenRouter, Together, etc.
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 4096,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.client = OpenAI(base_url=base_url, api_key=api_key or "not-needed")
        self.model = model
        self.max_tokens = max_tokens
        self.extra_body = extra_body or {}

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system: str | None = None,
    ) -> AssistantResponse:
        oai_messages = _to_openai_messages(messages)
        if system:
            oai_messages = [{"role": "system", "content": system}, *oai_messages]
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=oai_messages,
            tools=[_to_openai_tool(t) for t in tools] or None,
            extra_body=self.extra_body or None,
        )

        choice = response.choices[0].message
        blocks: list[TextBlock | ToolUseBlock] = []
        if choice.content:
            blocks.append(TextBlock(text=choice.content))
        for call in choice.tool_calls or []:
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            blocks.append(
                ToolUseBlock(id=call.id, name=call.function.name, input=args)
            )
        return AssistantResponse(blocks=blocks)


def _to_openai_tool(tool: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }


def _to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Flatten our block-based messages into OpenAI's per-role messages.

    OpenAI splits the conversation differently than Anthropic:
      - assistant text + tool_calls collapse into one message
      - each tool_result becomes its own role='tool' message
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "assistant":
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for block in msg.content:
                if isinstance(block, TextBlock):
                    text_parts.append(block.text)
                elif isinstance(block, ToolUseBlock):
                    tool_calls.append(
                        {
                            "id": block.id,
                            "type": "function",
                            "function": {
                                "name": block.name,
                                "arguments": json.dumps(block.input),
                            },
                        }
                    )
            assistant: dict[str, Any] = {
                "role": "assistant",
                "content": "\n".join(text_parts),
            }
            if tool_calls:
                assistant["tool_calls"] = tool_calls
            out.append(assistant)
            continue

        # user role: split tool_results from plain text
        text_parts = []
        for block in msg.content:
            if isinstance(block, TextBlock):
                text_parts.append(block.text)
            elif isinstance(block, ToolResultBlock):
                content = (
                    f"Error: {block.content}" if block.is_error else block.content
                )
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.tool_use_id,
                        "content": content,
                    }
                )
        if text_parts:
            out.append({"role": "user", "content": "\n".join(text_parts)})
    return out
