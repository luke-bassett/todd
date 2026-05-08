from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

MAX_TURNS_PER_INPUT = 25

from todd.provider import (
    Message,
    Provider,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
)
from todd.providers.anthropic import AnthropicProvider
from todd.providers.openai_compat import OpenAICompatibleProvider
from todd.tools.bash import BASH_SPEC, bash as _bash

MAX_TOOL_RESULT_CHARS = 8000


SYSTEM_PROMPT = """\
You are Todd, a coding agent operating in the user's working directory.

You have these tools: list_files, read_file, create_file, write_file, \
edit_file, bash. When the user asks you to do something, use the tools \
— do not describe what you would do, do not simulate, do not narrate plans.

Rules:
- Use list_files and read_file to gather facts before acting. Do not guess \
at file contents.
- create_file makes a new file (errors if it exists). edit_file patches an \
existing file. write_file unconditionally writes the full contents of a \
file (use it to recover from a stuck state, or to replace a file wholesale).
- For edit_file, old_str must appear exactly once in the file. Pick a \
distinctive snippet with enough surrounding context to be unique.
- Use bash to verify your work: run tests, type-checkers, linters, sync \
deps. Read the output and fix what you broke before declaring done.
- If a tool returns an error, read it and adjust. Do not repeat the same \
failing call.
- Reply with tool calls or short, direct text. No meta-commentary.
"""


MODELS: dict[str, Callable[[], Provider]] = {
    "gemma4": lambda: OpenAICompatibleProvider(
        model="gemma4", base_url="http://localhost:11434/v1"
    ),
    "gemma4-31b": lambda: OpenAICompatibleProvider(
        model="gemma4:31b",
        base_url="http://localhost:11434/v1",
        extra_body={"think": False, "options": {"num_ctx": 16384}},
    ),
    "claude-haiku": lambda: AnthropicProvider(model="claude-haiku-4-5"),
    "claude-sonnet": lambda: AnthropicProvider(model="claude-sonnet-4-6"),
    "claude-opus": lambda: AnthropicProvider(model="claude-opus-4-7"),
}
DEFAULT_MODEL = "gemma4"


@dataclass(frozen=True)
class ToolDefinition:
    spec: ToolSpec
    function: Callable[[dict[str, Any]], str]


def _read_file(input: dict[str, Any]) -> str:
    path = input.get("path") or ""
    if not path:
        raise ValueError("path is required")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"file does not exist: {path}")
    if p.is_dir():
        raise IsADirectoryError(
            f"{path} is a directory; use list_files to see its contents"
        )
    return p.read_text()


READ_FILE = ToolDefinition(
    spec=ToolSpec(
        name="read_file",
        description=(
            "Read the contents of a file at a relative path.\n\n"
            'Example: read_file({"path": "src/todd/main.py"})'
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path of a file in the working directory.",
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    ),
    function=_read_file,
)


_LIST_SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    "target",
}


def _list_files(input: dict[str, Any]) -> str:
    root = Path(input.get("path") or ".")
    entries: list[str] = []

    def walk(dir: Path) -> None:
        for child in sorted(dir.iterdir()):
            if child.name.startswith(".") or child.name in _LIST_SKIP_DIRS:
                continue
            rel = child.relative_to(root).as_posix()
            if child.is_dir():
                entries.append(rel + "/")
                walk(child)
            else:
                entries.append(rel)

    walk(root)
    return json.dumps(entries)


LIST_FILES = ToolDefinition(
    spec=ToolSpec(
        name="list_files",
        description=(
            "List files and directories under a path, recursively. Skips "
            "hidden and build/cache directories. If no path is provided, "
            "lists from the current directory.\n\n"
            'Example: list_files({"path": "src/"})'
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Optional relative path to list files from. Defaults to current directory if not provided.",
                },
            },
            "additionalProperties": False,
        },
    ),
    function=_list_files,
)


def _create_file(input: dict[str, Any]) -> str:
    path = input.get("path") or ""
    content = input.get("content", "")
    if not path:
        raise ValueError("path is required")
    p = Path(path)
    if p.exists():
        raise FileExistsError(
            f"file already exists: {path}; use edit_file to modify it"
        )
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"Created {path}"


CREATE_FILE = ToolDefinition(
    spec=ToolSpec(
        name="create_file",
        description=(
            "Create a new file with the given content. Fails if the file "
            "already exists. Parent directories are created automatically.\n\n"
            'Example: create_file({"path": "src/foo.py", "content": "print(1)\\n"})'
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path for the new file.",
                },
                "content": {
                    "type": "string",
                    "description": "Full text content of the new file.",
                },
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    ),
    function=_create_file,
)


def _write_file(input: dict[str, Any]) -> str:
    path = input.get("path") or ""
    content = input.get("content", "")
    if not path:
        raise ValueError("path is required")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    existed = p.exists()
    p.write_text(content)
    return f"Wrote {path}" + (" (overwrote)" if existed else " (created)")


WRITE_FILE = ToolDefinition(
    spec=ToolSpec(
        name="write_file",
        description=(
            "Write the full contents of a file, overwriting if it exists. "
            "Use this for full-file replacement or to recover when "
            "create_file/edit_file is stuck. Parent directories are "
            "created automatically.\n\n"
            'Example: write_file({"path": "src/foo.py", "content": "x = 1\\n"})'
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write.",
                },
                "content": {
                    "type": "string",
                    "description": "Full text content to write.",
                },
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    ),
    function=_write_file,
)


def _edit_file(input: dict[str, Any]) -> str:
    path = input.get("path") or ""
    old_str = input.get("old_str", "")
    new_str = input.get("new_str", "")

    if not path:
        raise ValueError("path is required")
    if not old_str:
        raise ValueError(
            "old_str must be non-empty; use create_file to make a new file"
        )
    if old_str == new_str:
        raise ValueError("old_str and new_str must differ")

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"file does not exist: {path}; use create_file to make it"
        )

    content = p.read_text()
    count = content.count(old_str)
    if count == 0:
        snippet = content[:200]
        raise ValueError(
            f"old_str not found in {path}. File starts with: {snippet!r}"
        )
    if count > 1:
        raise ValueError(
            f"old_str appears {count} times in {path}; "
            "include more surrounding context so it matches exactly once"
        )

    p.write_text(content.replace(old_str, new_str, 1))
    return f"Edited {path}"


EDIT_FILE = ToolDefinition(
    spec=ToolSpec(
        name="edit_file",
        description=(
            "Edit an existing file by replacing old_str with new_str. "
            "old_str must appear exactly once in the file. To create a new "
            "file, use create_file instead.\n\n"
            'Example: edit_file({"path": "main.py", "old_str": "x = 1", '
            '"new_str": "x = 2"})'
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file."},
                "old_str": {
                    "type": "string",
                    "description": "Exact text to find. Must appear exactly once.",
                },
                "new_str": {
                    "type": "string",
                    "description": "Text to replace old_str with.",
                },
            },
            "required": ["path", "old_str", "new_str"],
            "additionalProperties": False,
        },
    ),
    function=_edit_file,
)


BASH = ToolDefinition(spec=BASH_SPEC, function=_bash)


_TOKEN_LEAK = re.compile(r"<\|?[a-zA-Z_][\w/-]*\|?>")


def _clean_text(text: str) -> str:
    return _TOKEN_LEAK.sub("", text).strip()


def _truncate(content: str, limit: int = MAX_TOOL_RESULT_CHARS) -> str:
    if len(content) <= limit:
        return content
    dropped = len(content) - limit
    return content[:limit] + f"\n\n[... truncated {dropped} more characters ...]"


class Agent:
    def __init__(
        self,
        provider: Provider,
        get_user_message: Callable[[], str | None],
        tools: list[ToolDefinition],
        system: str | None = None,
    ) -> None:
        self.provider = provider
        self.get_user_message = get_user_message
        self.tools = tools
        self.system = system
        self._tools_by_name = {t.spec.name: t for t in tools}

    def run(self) -> None:
        conversation: list[Message] = []
        print("Chat with Todd (use 'ctrl-c' to quit)")

        read_user_input = True
        turns = 0
        prev_signatures: list[tuple[str, str]] = []

        while True:
            if read_user_input:
                print("\033[94mYou\033[0m: ", end="", flush=True)
                user_input = self.get_user_message()
                if user_input is None:
                    return
                conversation.append(
                    Message(role="user", content=[TextBlock(text=user_input)])
                )
                turns = 0
                prev_signatures = []

            if turns >= MAX_TURNS_PER_INPUT:
                print(
                    f"\033[91m[turn cap reached ({MAX_TURNS_PER_INPUT}); "
                    "returning control]\033[0m"
                )
                read_user_input = True
                continue
            turns += 1

            response = self.provider.complete(
                conversation,
                [t.spec for t in self.tools],
                system=self.system,
            )

            cleaned: list[TextBlock | ToolUseBlock] = []
            for block in response.blocks:
                if isinstance(block, TextBlock):
                    text = _clean_text(block.text)
                    if text:
                        cleaned.append(TextBlock(text=text))
                else:
                    cleaned.append(block)

            if not cleaned:
                read_user_input = True
                prev_signatures = []
                continue

            conversation.append(Message(role="assistant", content=list(cleaned)))

            tool_results: list[ToolResultBlock] = []
            for block in cleaned:
                if isinstance(block, TextBlock):
                    print(f"\033[93mTodd\033[0m: {block.text}")
                elif isinstance(block, ToolUseBlock):
                    tool_results.append(self._execute_tool(block))

            signatures = sorted(
                (b.name, json.dumps(b.input, sort_keys=True))
                for b in cleaned
                if isinstance(b, ToolUseBlock)
            )
            if signatures and signatures == prev_signatures:
                print(
                    "\033[91m[loop detected: same tool calls as previous turn; "
                    "returning control]\033[0m"
                )
                if tool_results:
                    conversation.append(
                        Message(role="user", content=list(tool_results))
                    )
                read_user_input = True
                prev_signatures = []
                continue
            prev_signatures = signatures

            if not tool_results:
                read_user_input = True
                continue

            read_user_input = False
            conversation.append(Message(role="user", content=list(tool_results)))

    def _execute_tool(self, call: ToolUseBlock) -> ToolResultBlock:
        tool = self._tools_by_name.get(call.name)
        if tool is None:
            return ToolResultBlock(
                tool_use_id=call.id, content="tool not found", is_error=True
            )

        print(f"\033[92mtool \033[0m: {call.name}({json.dumps(call.input)})")
        try:
            content = tool.function(call.input)
        except Exception as e:
            return ToolResultBlock(
                tool_use_id=call.id, content=_truncate(str(e)), is_error=True
            )
        return ToolResultBlock(tool_use_id=call.id, content=_truncate(content))


def _stdin_reader() -> Callable[[], str | None]:
    def read() -> str | None:
        try:
            return input()
        except EOFError:
            return None

    return read


def main() -> None:
    parser = argparse.ArgumentParser(prog="todd")
    parser.add_argument(
        "--model",
        choices=sorted(MODELS),
        default=DEFAULT_MODEL,
        help=f"LLM to use (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()

    print(f"\033[95mmodel\033[0m: {args.model}")
    provider = MODELS[args.model]()
    agent = Agent(
        provider=provider,
        get_user_message=_stdin_reader(),
        tools=[LIST_FILES, READ_FILE, CREATE_FILE, WRITE_FILE, EDIT_FILE, BASH],
        system=SYSTEM_PROMPT,
    )
    try:
        agent.run()
    except KeyboardInterrupt:
        print()
        sys.exit(0)


if __name__ == "__main__":
    main()
