from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from anthropic import Anthropic
from anthropic.types import MessageParam, ToolParam

MODEL = "claude-haiku-4-5"
MAX_TOKENS = 4096


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    function: Callable[[dict[str, Any]], str]


def read_file(input: dict[str, Any]) -> str:
    return Path(input["path"]).read_text()


READ_FILE = ToolDefinition(
    name="read_file",
    description=(
        "Read the contents of a given relative file path. Use this when you "
        "want to see what's inside a file. Do not use this with directory names."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The relative path of a file in the working directory.",
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    },
    function=read_file,
)


def list_files(input: dict[str, Any]) -> str:
    root = Path(input.get("path") or ".")
    entries: list[str] = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        entries.append(rel + "/" if p.is_dir() else rel)
    return json.dumps(entries)


LIST_FILES = ToolDefinition(
    name="list_files",
    description=(
        "List files and directories at a given path. If no path is provided, "
        "lists files in the current directory."
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
    function=list_files,
)


def edit_file(input: dict[str, Any]) -> str:
    path = input.get("path") or ""
    old_str = input.get("old_str", "")
    new_str = input.get("new_str", "")

    if not path or old_str == new_str:
        raise ValueError("invalid input parameters")

    file_path = Path(path)
    if not file_path.exists():
        if old_str == "":
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(new_str)
            return f"Successfully created file {path}"
        raise FileNotFoundError(path)

    content = file_path.read_text()
    if old_str and old_str not in content:
        raise ValueError("old_str not found in file")

    file_path.write_text(content.replace(old_str, new_str))
    return "OK"


EDIT_FILE = ToolDefinition(
    name="edit_file",
    description=(
        "Make edits to a text file.\n\n"
        "Replaces 'old_str' with 'new_str' in the given file. 'old_str' and "
        "'new_str' MUST be different from each other.\n\n"
        "If the file specified with path doesn't exist, it will be created."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "The path to the file"},
            "old_str": {
                "type": "string",
                "description": "Text to search for - must match exactly and must have exactly one match",
            },
            "new_str": {
                "type": "string",
                "description": "Text to replace old_str with",
            },
        },
        "required": ["path", "old_str", "new_str"],
        "additionalProperties": False,
    },
    function=edit_file,
)


class Agent:
    def __init__(
        self,
        client: Anthropic,
        get_user_message: Callable[[], str | None],
        tools: list[ToolDefinition],
    ) -> None:
        self.client = client
        self.get_user_message = get_user_message
        self.tools = tools
        self._tools_by_name = {t.name: t for t in tools}

    def run(self) -> None:
        conversation: list[MessageParam] = []
        print("Chat with Todd (use 'ctrl-c' to quit)")

        read_user_input = True
        while True:
            if read_user_input:
                print("\033[94mYou\033[0m: ", end="", flush=True)
                user_input = self.get_user_message()
                if user_input is None:
                    return
                conversation.append({"role": "user", "content": user_input})

            message = self._run_inference(conversation)
            conversation.append({"role": "assistant", "content": message.content})

            tool_results: list[dict[str, Any]] = []
            for block in message.content:
                if block.type == "text":
                    print(f"\033[93mTodd\033[0m: {block.text}")
                elif block.type == "tool_use":
                    tool_results.append(
                        self._execute_tool(block.id, block.name, block.input)
                    )

            if not tool_results:
                read_user_input = True
                continue

            read_user_input = False
            conversation.append({"role": "user", "content": tool_results})

    def _execute_tool(
        self, tool_use_id: str, name: str, input: dict[str, Any]
    ) -> dict[str, Any]:
        tool = self._tools_by_name.get(name)
        if tool is None:
            return {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": "tool not found",
                "is_error": True,
            }

        print(f"\033[92mtool \033[0m: {name}({json.dumps(input)})")
        try:
            content = tool.function(input)
        except Exception as e:
            return {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": str(e),
                "is_error": True,
            }
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": content,
        }

    def _run_inference(self, conversation: list[MessageParam]):
        tools: list[ToolParam] = [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in self.tools
        ]
        return self.client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=conversation,
            tools=tools,
        )


def _stdin_reader() -> Callable[[], str | None]:
    def read() -> str | None:
        try:
            return input()
        except EOFError:
            return None

    return read


def main() -> None:
    client = Anthropic()
    agent = Agent(
        client=client,
        get_user_message=_stdin_reader(),
        tools=[READ_FILE, LIST_FILES, EDIT_FILE],
    )
    try:
        agent.run()
    except KeyboardInterrupt:
        print()
        sys.exit(0)


if __name__ == "__main__":
    main()
