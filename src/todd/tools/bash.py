from __future__ import annotations

import shlex
import subprocess
from typing import Any

from todd.provider import ToolSpec


ALLOWED_PREFIXES: tuple[tuple[str, ...], ...] = (
    ("uv", "run"),
    ("uv", "sync"),
    ("uv", "lock"),
    ("uv", "add"),
    ("uv", "tree"),
    ("python", "-m"),
    ("python3", "-m"),
    ("pytest",),
    ("ruff",),
    ("mypy",),
    ("pyright",),
    ("git", "status"),
    ("git", "diff"),
    ("git", "log"),
    ("git", "show"),
    ("git", "branch"),
    ("ls",),
    ("cat",),
    ("head",),
    ("tail",),
    ("wc",),
    ("rg",),
    ("grep",),
    ("find",),
    ("echo",),
    ("pwd",),
    ("which",),
    ("file",),
)

# Shell metacharacters disqualify a command from auto-allow even if its prefix
# matches. The model can still run them — they just trigger a confirmation.
SHELL_METACHARS = (";", "&&", "||", "|", "`", "$(", ">", "<", ">>")

DEFAULT_TIMEOUT = 30
MAX_TIMEOUT = 300


def _is_allowed(cmd: str) -> bool:
    if any(meta in cmd for meta in SHELL_METACHARS):
        return False
    try:
        tokens = shlex.split(cmd)
    except ValueError:
        return False
    if not tokens:
        return False
    for prefix in ALLOWED_PREFIXES:
        if tuple(tokens[: len(prefix)]) == prefix:
            return True
    return False


def _confirm(cmd: str) -> bool:
    print(
        f"\033[93m[bash] command not on allow-list:\033[0m {cmd}\n"
        "Run it? [y/N] ",
        end="",
        flush=True,
    )
    try:
        answer = input().strip().lower()
    except EOFError:
        return False
    return answer in ("y", "yes")


def bash(input: dict[str, Any]) -> str:
    cmd = input.get("command") or ""
    if not cmd:
        raise ValueError("command is required")
    timeout = min(int(input.get("timeout") or DEFAULT_TIMEOUT), MAX_TIMEOUT)

    if not _is_allowed(cmd) and not _confirm(cmd):
        raise PermissionError("user declined to run this command")

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise TimeoutError(f"command exceeded {timeout}s timeout: {cmd}")

    parts = [f"exit_code: {result.returncode}"]
    if result.stdout:
        parts.append(f"--- stdout ---\n{result.stdout.rstrip()}")
    if result.stderr:
        parts.append(f"--- stderr ---\n{result.stderr.rstrip()}")
    return "\n".join(parts) if len(parts) > 1 else parts[0] + "\n(no output)"


BASH_SPEC = ToolSpec(
    name="bash",
    description=(
        "Run a shell command in the working directory and return its "
        "stdout, stderr, and exit code. Use this to verify your work — "
        "run tests, type-checkers, linters, sync dependencies, view "
        "git status. Common commands (uv, pytest, ruff, mypy, git status, "
        "ls, cat, grep, etc.) run automatically. Other commands prompt "
        "the user. Shell metacharacters (|, &&, ;, etc.) always prompt.\n\n"
        'Example: bash({"command": "uv run pytest -x"})'
    ),
    input_schema={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to run.",
            },
            "timeout": {
                "type": "integer",
                "description": (
                    f"Optional timeout in seconds (default {DEFAULT_TIMEOUT}, "
                    f"max {MAX_TIMEOUT})."
                ),
            },
        },
        "required": ["command"],
        "additionalProperties": False,
    },
)


