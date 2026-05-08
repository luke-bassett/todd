"""Agent loop tests, using a scripted FakeProvider so no API calls happen.

Covers the loop's behavioral contract: tool round-trips feed back into the
next inference, empty-text-block cleaning is non-poisoning, loop detection
hands control back, and the turn cap eventually stops a runaway agent.
"""
from __future__ import annotations

from todd.main import (
    LIST_FILES,
    MAX_TOOL_RESULT_CHARS,
    MAX_TURNS_PER_INPUT,
    READ_FILE,
    Agent,
    _truncate,
)
from todd.provider import (
    AssistantResponse,
    Message,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)


class FakeProvider:
    def __init__(self, responses):
        self.queue = list(responses)
        self.calls: list[list[Message]] = []

    def complete(self, messages, tools, system=None):
        self.calls.append(list(messages))
        if self.queue:
            return self.queue.pop(0)
        return AssistantResponse(blocks=[TextBlock(text="done")])


def _scripted_input(*lines):
    """Returns a get_user_message callable that yields then EOFs."""
    queue: list = list(lines) + [None]
    return lambda: queue.pop(0) if queue else None


def test_tool_round_trip_feeds_results_back(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "hello.py").write_text("print('hi')\n")

    provider = FakeProvider(
        [
            AssistantResponse(
                blocks=[
                    ToolUseBlock(id="t1", name="read_file", input={"path": "hello.py"})
                ]
            ),
            AssistantResponse(blocks=[TextBlock(text="ok")]),
        ]
    )
    Agent(
        provider=provider,
        get_user_message=_scripted_input("read it"),
        tools=[READ_FILE],
    ).run()

    # Two inferences: the second one must include the tool result.
    assert len(provider.calls) == 2
    last_msg = provider.calls[1][-1]
    assert last_msg.role == "user"
    assert any(
        isinstance(b, ToolResultBlock) and "print('hi')" in b.content
        for b in last_msg.content
    )


def test_empty_text_blocks_do_not_poison_conversation(tmp_path, monkeypatch):
    """Regression: stray model tokens (e.g. <channel|>) clean to "" and
    must not be persisted as empty text blocks — Anthropic 400s on those.
    """
    monkeypatch.chdir(tmp_path)

    provider = FakeProvider(
        [
            AssistantResponse(
                blocks=[
                    TextBlock(text="<channel|>"),  # cleans to ""
                    ToolUseBlock(id="t1", name="list_files", input={}),
                ]
            ),
            AssistantResponse(blocks=[TextBlock(text="ok")]),
        ]
    )
    Agent(
        provider=provider,
        get_user_message=_scripted_input("go"),
        tools=[LIST_FILES],
    ).run()

    # Inspect the conversation snapshot the second inference saw.
    second = provider.calls[1]
    assistant_msg = next(m for m in second if m.role == "assistant")
    # No empty text blocks survived the cleaner.
    assert all(
        not (isinstance(b, TextBlock) and b.text == "")
        for b in assistant_msg.content
    )


def test_loop_detection_breaks_back_to_user(tmp_path, monkeypatch):
    """Two identical tool-call turns in a row should hand control back."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.txt").write_text("a")

    same = ToolUseBlock(id="x", name="read_file", input={"path": "a.txt"})
    provider = FakeProvider(
        [
            AssistantResponse(blocks=[same]),
            AssistantResponse(blocks=[same]),  # repeat → loop trip
            AssistantResponse(blocks=[TextBlock(text="should not run")]),
        ]
    )
    Agent(
        provider=provider,
        get_user_message=_scripted_input("go"),
        tools=[READ_FILE],
    ).run()

    # Two inferences happened, then loop detection bailed and the EOF on
    # stdin terminated the run before a third.
    assert len(provider.calls) == 2


def test_truncate_caps_oversized_tool_results():
    short = "x" * 50
    assert _truncate(short) == short  # under limit, untouched

    huge = "x" * (MAX_TOOL_RESULT_CHARS + 500)
    out = _truncate(huge)
    assert out.startswith("x" * MAX_TOOL_RESULT_CHARS)
    assert "[... truncated 500 more characters ...]" in out
    # The model needs to see the marker, but the head has to be intact.
    assert len(out) < len(huge)


def test_turn_cap_stops_runaway_agent(tmp_path, monkeypatch):
    """If every turn picks a *different* tool call, loop detection won't
    fire — the hard turn cap is the only thing that saves us.
    """
    monkeypatch.chdir(tmp_path)

    responses = [
        AssistantResponse(
            blocks=[
                ToolUseBlock(
                    id=f"t{i}", name="list_files", input={"path": f"sub{i}"}
                )
            ]
        )
        for i in range(MAX_TURNS_PER_INPUT + 5)
    ]
    provider = FakeProvider(responses)
    Agent(
        provider=provider,
        get_user_message=_scripted_input("go"),
        tools=[LIST_FILES],
    ).run()

    assert len(provider.calls) == MAX_TURNS_PER_INPUT
