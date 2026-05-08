# todd

A small terminal coding agent. Talks to Claude or to local models via Ollama.

Started from [How to Build an Agent](https://ampcode.com/how-to-build-an-agent), now
extended with multiple backends, a few extra tools, and tests.

## Install

```sh
uv sync
```

For Claude models, set your API key:

```sh
export ANTHROPIC_API_KEY=sk-...
```

For local models, install [Ollama](https://ollama.com) and pull a model
that supports tool calls:

```sh
ollama pull gemma4:31b
```

## Run

```sh
uv run todd                       # default: gemma4 via Ollama
uv run todd --model claude-haiku
```

Available models:

| `--model` value | backend                                |
|-----------------|----------------------------------------|
| `gemma4`        | Ollama, tag `gemma4`                   |
| `gemma4-31b`    | Ollama, tag `gemma4:31b`               |
| `claude-haiku`  | Anthropic API, `claude-haiku-4-5`      |
| `claude-sonnet` | Anthropic API, `claude-sonnet-4-6`     |
| `claude-opus`   | Anthropic API, `claude-opus-4-7`       |

Adding another model is a one-liner in `MODELS` in `src/todd/main.py`.
Any OpenAI-compatible endpoint works (vLLM, LM Studio, OpenRouter,
Together, etc.) — pass `base_url` to `OpenAICompatibleProvider`.

## Tools

The agent has six tools:

- `list_files` — recursive listing, skips `.git`, `.venv`, `node_modules`, etc.
- `read_file` — read a file's contents.
- `create_file` — create a new file. Errors if it already exists.
- `write_file` — overwrite a file unconditionally.
- `edit_file` — replace one occurrence of `old_str` with `new_str`.
- `bash` — run a shell command. Common dev commands (`uv`, `pytest`,
  `git status`, `ls`, etc.) run automatically; anything else, or
  anything containing shell metacharacters, prompts you first.

## Tests

```sh
uv run pytest
```
