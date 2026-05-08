"""Message-shape conversion for both provider backends.

Pure functions on our neutral types → SDK shapes. No network. The OpenAI
encoder is the riskier one (different conversation model than Anthropic),
so it gets the regression tests.
"""
from __future__ import annotations

import json

from todd.provider import Message, TextBlock, ToolResultBlock, ToolUseBlock
from todd.providers.anthropic import _to_anthropic_message
from todd.providers.openai_compat import _to_openai_messages


def _sample_conversation() -> list[Message]:
    return [
        Message(role="user", content=[TextBlock(text="hi")]),
        Message(
            role="assistant",
            content=[
                TextBlock(text="ok"),
                ToolUseBlock(id="t1", name="read_file", input={"path": "x"}),
            ],
        ),
        Message(
            role="user",
            content=[ToolResultBlock(tool_use_id="t1", content="file contents")],
        ),
    ]


def test_anthropic_message_encoding():
    encoded = [_to_anthropic_message(m) for m in _sample_conversation()]

    assert encoded[0] == {
        "role": "user",
        "content": [{"type": "text", "text": "hi"}],
    }
    assert encoded[1]["role"] == "assistant"
    assert encoded[1]["content"][1] == {
        "type": "tool_use",
        "id": "t1",
        "name": "read_file",
        "input": {"path": "x"},
    }
    assert encoded[2]["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "t1",
        "content": "file contents",
        "is_error": False,
    }


def test_openai_message_encoding_splits_results_into_tool_role():
    out = _to_openai_messages(_sample_conversation())

    assert out[0] == {"role": "user", "content": "hi"}

    # assistant: text + tool_calls collapse into one message
    assert out[1]["role"] == "assistant"
    assert out[1]["content"] == "ok"
    assert out[1]["tool_calls"][0]["id"] == "t1"
    assert out[1]["tool_calls"][0]["function"]["name"] == "read_file"
    assert json.loads(out[1]["tool_calls"][0]["function"]["arguments"]) == {
        "path": "x"
    }

    # tool_result becomes its own role='tool' message
    assert out[2] == {
        "role": "tool",
        "tool_call_id": "t1",
        "content": "file contents",
    }


def test_openai_assistant_with_only_tool_calls_uses_empty_string_not_null():
    """Regression: Ollama rejects {"role": "assistant", "content": null}.
    When the assistant turn has only tool calls and no text, content must
    serialize to "" — not None.
    """
    msgs = [
        Message(
            role="assistant",
            content=[ToolUseBlock(id="t1", name="x", input={})],
        ),
    ]
    out = _to_openai_messages(msgs)
    assert out[0]["content"] == ""
    assert "tool_calls" in out[0]
