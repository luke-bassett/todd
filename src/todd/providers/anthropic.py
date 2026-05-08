from __future__ import annotations

from typing import Any

from anthropic import Anthropic

from todd.provider import (
    AssistantResponse,
    Message,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
)


class AnthropicProvider:
    def __init__(
        self,
        model: str = "claude-haiku-4-5",
        max_tokens: int = 4096,
        client: Anthropic | None = None,
    ) -> None:
        self.client = client or Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system: str | None = None,
    ) -> AssistantResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [_to_anthropic_message(m) for m in messages],
            "tools": [_to_anthropic_tool(t) for t in tools],
        }
        if system:
            kwargs["system"] = system
        response = self.client.messages.create(**kwargs)

        blocks: list[TextBlock | ToolUseBlock] = []
        for block in response.content:
            if block.type == "text":
                blocks.append(TextBlock(text=block.text))
            elif block.type == "tool_use":
                blocks.append(
                    ToolUseBlock(
                        id=block.id,
                        name=block.name,
                        input=dict(block.input) if block.input else {},
                    )
                )
        return AssistantResponse(blocks=blocks)


def _to_anthropic_tool(tool: ToolSpec) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema,
    }


def _to_anthropic_message(message: Message) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    for block in message.content:
        if isinstance(block, TextBlock):
            content.append({"type": "text", "text": block.text})
        elif isinstance(block, ToolUseBlock):
            content.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                }
            )
        elif isinstance(block, ToolResultBlock):
            content.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.tool_use_id,
                    "content": block.content,
                    "is_error": block.is_error,
                }
            )
    return {"role": message.role, "content": content}
