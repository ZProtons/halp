# halp

<img width="1254" height="1254" alt="c1c336dd-c631-4fef-9ea6-3ab0ce8e8f62" src="https://github.com/user-attachments/assets/7d773eba-6f03-44b3-b29e-6da9094a1993" />


`halp` turns a natural-language task into a shell command grounded in the selected tool's help or man page. It talks to an OpenAI-compatible LLM endpoint and does not execute suggestions unless `-a` is enabled and the user confirms.

## Install

Install the published package with:

```sh
python -m pip install halp-cli
```

To install the current checkout instead:

```sh
python -m pip install .
```

For local development, use `python -m pip install -e .`. The install provides the `halp` command.

## Usage

```sh
halp [OPTIONS] <tool> "<prompt>"
```

Example for a local Gemma server:

```sh
halp -l gemma4:e4b -ip 127.0.0.1:8001 nmap "Scan the top 10 TCP ports on 8.8.8.8"
```

By default, halp prints the suggested command and exits. It checks that the requested tool is installed, gathers `-h` and `--help` output, then tries no-argument usage and a man page. Help collection is time-limited and capped; the captured context is trimmed to fit `-c` with room reserved for the prompt and model response.

## Options

| Option | Purpose |
| --- | --- |
| `-l`, `--llm MODEL` | Model/backend name passed as the request's `model` (default `gemma4:e4b`) |
| `-ip`, `--server HOST:PORT` | Server address or base URL (default `127.0.0.1:8001`; can also use `HALP_SERVER`) |
| `-api`, `--api-key KEY` | Optional Bearer token (can also use `HALP_API_KEY`) |
| `-c`, `--context-size TOKENS` | Context budget (default `4000`) |
| `-extra-args ARGS` | Extra request fields as JSON or comma-separated `key=value` pairs |
| `--request-timeout SECONDS` | Per-request HTTP timeout (default `120`) |
| `-p`, `--persist` | Open the interactive terminal workbench |
| `-a`, `--allow-execution` | Enable command execution actions; each run always requires y/n confirmation |
| `--no-color` | Disable terminal colors |
| `-v`, `--verbose` | Print the exact final LLM prompt to stderr |
| `--version` | Print the halp version |

Extra parameter examples:

```sh
halp -extra-args 'temperature=0.2,top_p=0.9' nmap "Scan localhost"
halp -extra-args '{"temperature":0.2,"top_p":0.9}' nmap "Scan localhost"
```

halp manages the `model`, `messages`, `prompt`, `max_tokens`, and `stream` fields, so extra arguments cannot override them. The default endpoint probe tries standard `/v1/chat/completions` first, then common chat and completions route variants.

## Interactive workbench

Start with `-p`. Enter a refinement directly or use an action:

- `/refine TEXT` — refine the current command
- `/task TEXT` — start a new task
- `/retry` — request another suggestion for the same task
- `/history` and `/use N` — review suggestions and select a prior candidate
- `/copy` — copy the current command using `wl-copy`, `xclip`, `xsel`, or `pbcopy` when available
- `/run` — run the current candidate (available only with `-a`, always asks for confirmation)
- `/clear`, `/help`, `/quit`

When `-a` is active, halp asks whether to execute every newly generated command. In the workbench, `/run` can also execute the current candidate after another confirmation. Output from accepted commands streams directly to the terminal.
