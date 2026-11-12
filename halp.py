#!/usr/bin/env python3
"""Generate grounded shell commands with an OpenAI-compatible language model."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import textwrap
import threading
from typing import Any
from urllib.parse import urlsplit, urlunsplit

__version__ = "0.2.0"
SYSTEM_PROMPT = (
    "You are a command-line syntax assistant. Given a tool’s help/man output and a task, "
    "output the exact shell command to accomplish the task. Output only the command in a "
    "bash code block. No explanation."
)
RESERVED_REQUEST_KEYS = {"model", "messages", "prompt", "max_tokens", "stream"}
MAX_CAPTURE_BYTES = 256 * 1024
UNRECOGNIZED_OPTION = re.compile(
    r"(?:unrecognized|unknown|invalid|illegal|unsupported)\s+(?:command[- ]line\s+)?option|"
    r"option\s+(?:not recognized|not supported)",
    re.IGNORECASE,
)
BASH_BLOCK = re.compile(r"```(?:bash|sh)?\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)


class TerminalUI:
    """Small dependency-free terminal presentation layer."""

    def __init__(self, *, color: bool = True, interactive: bool = False) -> None:
        self.interactive = interactive
        self.color_enabled = (
            color
            and sys.stdout.isatty()
            and not os.environ.get("NO_COLOR")
            and os.environ.get("TERM", "") != "dumb"
        )

    def paint(self, value: str, code: str) -> str:
        if not self.color_enabled:
            return value
        return f"\033[{code}m{value}\033[0m"

    def _panel(
        self,
        title: str,
        lines: list[str],
        *,
        color: str = "36",
        body_color: str | None = None,
    ) -> None:
        terminal_width = shutil.get_terminal_size((80, 24)).columns
        width = max(38, min(96, terminal_width - 2))
        inner = width - 4
        expanded: list[str] = []
        for line in lines:
            safe_line = _safe_display_text(line)
            if not safe_line:
                expanded.append("")
                continue
            expanded.extend(
                textwrap.wrap(
                    safe_line,
                    width=inner,
                    break_long_words=True,
                    break_on_hyphens=False,
                    replace_whitespace=False,
                    drop_whitespace=False,
                ) or [""]
            )
        title_text = f" {title} "
        top = "╭─" + title_text + "─" * max(1, width - len(title_text) - 3) + "╮"
        bottom = "╰" + "─" * (width - 2) + "╯"
        print(self.paint(top, color))
        for line in expanded:
            body = f" {line:<{inner}} "
            if body_color:
                body = self.paint(body, body_color)
            print(self.paint("│", "90") + body + self.paint("│", "90"))
        print(self.paint(bottom, color))

    def banner(self, tool: str, model: str, server: str, task: str) -> None:
        self._panel(
            "HALP · command workbench",
            [f"Tool: {tool}    Model: {model}", f"Server: {server}", f"Task: {task}"],
            color="36",
        )

    def status(self, message: str, *, error: bool = False) -> None:
        marker = self.paint("!" if error else "•", "31;1" if error else "34;1")
        print(f"{marker} {message}", file=sys.stdout if self.interactive else sys.stderr, flush=True)

    def command(self, command: str, index: int | None = None) -> None:
        title = f"Suggested command #{index}" if index is not None else "Suggested command"
        if sys.stdout.isatty():
            self._panel(title, command.splitlines() or [command], color="32", body_color="32")
        else:
            print(command, flush=True)

    def prompt(self, label: str = "halp › ") -> str:
        colored = self.paint(label, "36;1")
        return input(colored)

    def info(self, message: str) -> None:
        print(self.paint(message, "90"))

    def error(self, message: str) -> None:
        print(self.paint(f"Error: {message}", "31;1"), file=sys.stderr)

    def help(self, allow_execution: bool) -> None:
        rows = [
            "/refine TEXT   refine the current command; plain text works too",
            "/task TEXT     start over with a different task",
            "/retry         ask again with the same task and context",
            "/history       show commands generated in this session",
            "/use N         select a previous command as the current candidate",
            "/copy          copy the current command if a clipboard tool is installed",
        ]
        rows.append("/run           confirm and run the current command" if allow_execution
                    else "/run           disabled; restart halp with -a to enable execution")
        rows.extend(["/clear         clear and redraw the workbench", "/help          show this list", "/quit          exit halp"])
        self._panel("Actions", rows, color="35")
        self.info("Or enter a refinement directly. Commands are never executed without -a and a y/n confirmation.")


def _safe_display_text(text: str) -> str:
    """Remove terminal control bytes from displayed model output, leaving the command intact."""
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text).replace("\t", "    ")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="halp",
        description=(
            "Use an OpenAI-compatible LLM to turn a natural-language task into a shell "
            "command, grounded in the selected tool's help or man page."
        ),
        epilog=(
            "Examples:\n"
            '  halp -l gemma4:e4b -ip 127.0.0.1:8001 nmap "Scan top 10 ports on 8.8.8.8"\n'
            '  halp -p -a curl "Fetch the response headers from https://example.com"\n\n'
            "By default halp only prints the proposed command. With -a/--allow-execution it "
            "always asks for y/n confirmation before running it. In -p mode use /help for "
            "interactive actions."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"halp {__version__}")
    parser.add_argument(
        "-l", "--llm", default="gemma4:e4b", metavar="MODEL",
        help="LLM model/backend name sent as the OpenAI-compatible 'model' (default: %(default)s)",
    )
    parser.add_argument(
        "-ip", "--server", default=os.environ.get("HALP_SERVER", "127.0.0.1:8001"), metavar="HOST:PORT",
        help="LLM server host:port or base URL (default: %(default)s; may use HALP_SERVER)",
    )
    parser.add_argument(
        "-api", "--api-key", default=os.environ.get("HALP_API_KEY", ""), metavar="KEY",
        help="optional API key (or HALP_API_KEY); sent as a Bearer token when non-empty",
    )
    parser.add_argument(
        "-c", "--context-size", type=int, default=4000, metavar="TOKENS",
        help="model context size used to budget tool help and the response (default: %(default)s)",
    )
    parser.add_argument(
        "-extra-args", "--extra-args", default="", metavar="ARGS",
        help=(
            "extra LLM request parameters as a JSON object or comma-separated key=value pairs; "
            "for example: 'temperature=0.2,top_p=0.9'"
        ),
    )
    parser.add_argument(
        "--request-timeout", type=float, default=120.0, metavar="SECONDS",
        help="timeout for each LLM HTTP request (default: %(default)s)",
    )
    parser.add_argument(
        "-p", "--persist", action="store_true",
        help="open the interactive terminal workbench for iterative refinement",
    )
    parser.add_argument(
        "-a", "--allow-execution", action="store_true",
        help="enable execution actions; halp always asks for y/n confirmation before running",
    )
    parser.add_argument("--no-color", action="store_true", help="disable terminal colors")
    parser.add_argument("-v", "--verbose", action="store_true", help="print the final LLM prompt to stderr")
    parser.add_argument("tool", help="installed shell tool to generate a command for (for example nmap)")
    parser.add_argument("prompt", help="natural-language description of the task")
    return parser


def _run_capture(
    argv: list[str],
    *,
    timeout: float = 8.0,
    input_text: str | None = None,
    env: dict[str, str] | None = None,
) -> tuple[str, int | None]:
    """Capture up to MAX_CAPTURE_BYTES while draining excess output to avoid pipe stalls."""
    process: subprocess.Popen[bytes] | None = None
    captured = bytearray()
    was_truncated = [False]

    try:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            env=env,
        )
        assert process.stdout is not None

        def drain_output() -> None:
            while True:
                chunk = process.stdout.read(16 * 1024)
                if not chunk:
                    return
                remaining = MAX_CAPTURE_BYTES - len(captured)
                if remaining > 0:
                    captured.extend(chunk[:remaining])
                if len(chunk) > remaining:
                    was_truncated[0] = True

        reader = threading.Thread(target=drain_output, daemon=True)
        reader.start()
        if input_text is not None and process.stdin is not None:
            try:
                process.stdin.write(input_text.encode("utf-8", errors="replace"))
                process.stdin.close()
            except BrokenPipeError:
                pass
        try:
            return_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            return_code = None
        reader.join(timeout=2.0)
        output = captured.decode("utf-8", errors="replace")
        if was_truncated[0]:
            output += "\n[tool output truncated after 256 KiB]"
        if return_code is None:
            output += "\n[command timed out]"
        return output.strip(), return_code
    except OSError as exc:
        return str(exc), None
    finally:
        if process is not None:
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass


def _useful_help(text: str) -> bool:
    normalized = text.replace("[command timed out]", "").strip()
    return bool(normalized) and not bool(UNRECOGNIZED_OPTION.search(normalized))


def _strip_terminal_formatting(text: str) -> str:
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    # man pages rendered without a pager can contain backspace overstrikes (bold/underline).
    text = re.sub(r".\x08", "", text)
    return text.strip()


def gather_tool_context(tool: str) -> str:
    """Collect usage text by trying help flags, no arguments, then the man page."""
    help_results: list[str] = []
    for flag in ("-h", "--help"):
        output, _ = _run_capture([tool, flag])
        if _useful_help(output):
            help_results.append(f"$ {tool} {flag}\n{output}")
    if help_results:
        return "\n\n".join(help_results)

    output, _ = _run_capture([tool])
    if _useful_help(output):
        return f"$ {tool}\n{output}"

    man_env = os.environ.copy()
    man_env.update({"MANPAGER": "cat", "PAGER": "cat", "GROFF_NO_SGR": "1"})
    man_output = ""
    if shutil.which("man"):
        man_output, _ = _run_capture(["man", "-P", "cat", tool], timeout=12.0, env=man_env)
    man_output = _strip_terminal_formatting(man_output)
    if man_output and shutil.which("col"):
        stripped, _ = _run_capture(["col", "-b"], timeout=5.0, input_text=man_output)
        if stripped:
            man_output = stripped
    if man_output and not re.search(r"no manual entry|no entry for", man_output, re.IGNORECASE):
        return f"$ man {tool}\n{man_output}"
    return "No usable help text or man page was available for this tool."


def parse_extra_args(raw: str) -> dict[str, Any]:
    """Parse request parameters from a JSON object or key=value,key=value string."""
    if not raw.strip():
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = None
    if value is not None:
        if not isinstance(value, dict):
            raise ValueError("-extra-args JSON must be an object, such as '{\"temperature\":0.2}'")
        params = value
    else:
        params: dict[str, Any] = {}
        for item in raw.split(","):
            key, separator, raw_value = item.partition("=")
            key = key.strip()
            if not separator or not key:
                raise ValueError("-extra-args must be a JSON object or comma-separated key=value pairs")
            raw_value = raw_value.strip()
            try:
                parsed_value = json.loads(raw_value)
            except json.JSONDecodeError:
                parsed_value = raw_value
            params[key] = parsed_value
    conflicts = RESERVED_REQUEST_KEYS.intersection(params)
    if conflicts:
        names = ", ".join(sorted(conflicts))
        raise ValueError(f"-extra-args cannot override halp-managed request fields: {names}")
    return params


def _token_estimate(text: str) -> int:
    return (len(text) + 3) // 4


def make_user_message(tool: str, goal: str, help_text: str, context_size: int) -> str:
    core = f"Tool: {tool}\nTask: {goal}\n\nTool help/man output:\n"
    response_reserve = max(256, min(1024, context_size // 5))
    fixed_tokens = _token_estimate(SYSTEM_PROMPT + core) + 32
    context_tokens = context_size - response_reserve - fixed_tokens
    if context_tokens < 8:
        raise ValueError(
            "The context size is too small for the prompt and reserved response space; "
            "increase -c or shorten the task."
        )
    max_chars = context_tokens * 4
    if len(help_text) > max_chars:
        suffix = "\n[tool output truncated to fit the configured context size]"
        help_text = help_text[: max(0, max_chars - len(suffix))] + suffix[:max_chars]
    return core + help_text


def _base_and_candidates(server: str) -> list[tuple[str, str]]:
    """Return likely OpenAI endpoint URLs, preferring the standard /v1 chat route."""
    value = server.strip()
    if not value:
        raise ValueError("LLM server address cannot be blank")
    if "://" not in value:
        value = "http://" + value
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("LLM server must be host:port or an http(s) URL")
    path = parsed.path.rstrip("/")
    if path.endswith("/chat/completions"):
        return [(urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, "")), "chat")]
    if path.endswith("/completions"):
        return [(urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, "")), "completion")]

    if path.endswith("/v1"):
        base = path[:-3].rstrip("/")
        route_sets = [
            (base + "/v1/chat/completions", "chat"),
            (base + "/chat/completions", "chat"),
            (base + "/v1/completions", "completion"),
            (base + "/completions", "completion"),
        ]
    elif path:
        route_sets = [
            (path + "/v1/chat/completions", "chat"),
            (path + "/chat/completions", "chat"),
            (path + "/v1/completions", "completion"),
            (path + "/completions", "completion"),
        ]
    else:
        route_sets = [
            ("/v1/chat/completions", "chat"),
            ("/chat/completions", "chat"),
            ("/v1/completions", "completion"),
            ("/completions", "completion"),
        ]
    candidates: list[tuple[str, str]] = []
    for route, kind in route_sets:
        candidate = urlunsplit((parsed.scheme, parsed.netloc, route, parsed.query, ""))
        if all(candidate != existing[0] for existing in candidates):
            candidates.append((candidate, kind))
    return candidates


def _extract_response(data: Any, style: str) -> str:
    try:
        choice = data["choices"][0]
        result = choice["message"]["content"] if style == "chat" else choice["text"]
        if isinstance(result, list):
            result = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in result
            )
        if not isinstance(result, str) or not result.strip():
            raise ValueError("the response contained no command text")
        return result
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"LLM returned an unexpected response format: {data!r}") from exc


def call_llm(
    server: str,
    model: str,
    api_key: str,
    context_size: int,
    extra_params: dict[str, Any],
    tool: str,
    goal: str,
    help_text: str,
    *,
    verbose: bool = False,
    request_timeout: float = 120.0,
) -> str:
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - occurs only in an uninstalled checkout
        raise RuntimeError("The 'requests' package is required. Install halp with: pip install .") from exc
    user_message = make_user_message(tool, goal, help_text, context_size)
    if verbose:
        print("--- LLM prompt ---", file=sys.stderr)
        print(f"SYSTEM:\n{SYSTEM_PROMPT}\n\nUSER:\n{user_message}", file=sys.stderr)
        print("--- end LLM prompt ---", file=sys.stderr)

    response_reserve = max(256, min(1024, context_size // 5))
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    errors: list[str] = []
    for endpoint, style in _base_and_candidates(server):
        if style == "chat":
            payload: dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                "max_tokens": response_reserve,
                "stream": False,
            }
        else:
            payload = {
                "model": model,
                "prompt": f"System: {SYSTEM_PROMPT}\n\nUser: {user_message}\n\nAssistant:",
                "max_tokens": response_reserve,
                "stream": False,
            }
        payload.update(extra_params)
        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=request_timeout)
        except requests.Timeout as exc:
            raise RuntimeError(f"LLM request to {endpoint} timed out after {request_timeout:g}s") from exc
        except requests.ConnectionError as exc:
            raise RuntimeError(f"Could not connect to LLM server at {server}: {exc}") from exc
        except requests.RequestException as exc:
            errors.append(f"{endpoint}: {exc}")
            continue
        if response.status_code in (404, 405):
            errors.append(f"{endpoint}: HTTP {response.status_code}")
            continue
        if not response.ok:
            detail = response.text.strip()
            raise RuntimeError(f"LLM request failed at {endpoint} (HTTP {response.status_code}): {detail}")
        try:
            return _extract_response(response.json(), style)
        except (ValueError, requests.RequestException) as exc:
            raise RuntimeError(f"Could not read LLM response from {endpoint}: {exc}") from exc
    raise RuntimeError(
        "Could not reach a compatible chat/completions endpoint. Tried these URL variants:\n"
        + "\n".join(errors)
    )


def extract_command(response: str) -> str:
    match = BASH_BLOCK.search(response)
    if not match:
        raise ValueError("The LLM response did not contain a bash code block; retry or refine the task.")
    command = match.group(1).strip()
    if not command:
        raise ValueError("The LLM returned an empty command.")
    return command


def copy_to_clipboard(command: str) -> str | None:
    """Use an installed clipboard helper, if one is available."""
    if platform.system() == "Darwin":
        candidates = [["pbcopy"]]
    elif os.name == "nt":
        candidates = [["clip"]]
    else:
        candidates = [
            ["wl-copy"],
            ["xclip", "-selection", "clipboard"],
            ["xsel", "--clipboard", "--input"],
        ]
    errors: list[str] = []
    for candidate in candidates:
        if not shutil.which(candidate[0]):
            continue
        try:
            subprocess.run(
                candidate,
                input=command,
                text=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=4,
                check=True,
            )
            return None
        except (OSError, subprocess.SubprocessError) as exc:
            errors.append(f"{candidate[0]}: {exc}")
    if errors:
        return "Clipboard helpers failed: " + "; ".join(errors)
    return "No clipboard helper found (install wl-copy, xclip, or xsel)."


def confirm_and_run(command: str, ui: TerminalUI) -> int | None:
    try:
        answer = ui.prompt("Run this command? [y/N] ").strip().lower()
    except EOFError:
        ui.info("No confirmation received; command not run.")
        return None
    if answer not in {"y", "yes"}:
        ui.info("Not executed.")
        return None
    print(ui.paint("Running command…", "33;1"), flush=True)
    try:
        completed = subprocess.run(command, shell=True, check=False)
    except OSError as exc:
        ui.error(f"Could not execute command: {exc}")
        return 1
    if completed.returncode == 0:
        ui.status("Command finished successfully.")
    else:
        ui.status(f"Command exited with status {completed.returncode}.", error=True)
    return completed.returncode


def _compose_goal(original: str, refinements: list[str], candidate: str | None) -> str:
    parts = [f"Original task: {original}"]
    if refinements:
        parts.append("Refinement requests:\n" + "\n".join(f"- {item}" for item in refinements[-8:]))
    if candidate:
        parts.append(f"Current candidate command:\n{candidate}")
    return "\n\n".join(parts)


def _show_history(ui: TerminalUI, history: list[tuple[str, str]]) -> None:
    if not history:
        ui.info("No generated commands in this session yet.")
        return
    for index, (goal, command) in enumerate(history, start=1):
        ui._panel(f"History #{index}", [f"Task: {goal}", command], color="90")


def _handle_refinement(
    feedback: str,
    original: str,
    refinements: list[str],
    latest_command: str | None,
) -> tuple[list[str], str]:
    refinements.append(feedback)
    return refinements, _compose_goal(original, refinements, latest_command)


def run_persistent(args: argparse.Namespace, extra_params: dict[str, Any],
                   help_text: str, ui: TerminalUI) -> int:
    original = args.prompt
    refinements: list[str] = []
    latest_command: str | None = None
    request_goal = _compose_goal(original, refinements, latest_command)
    history: list[tuple[str, str]] = []
    need_request = True

    ui.banner(args.tool, args.llm, args.server, original)
    ui.info("Type /help for actions. Enter a refinement directly or use /refine TEXT.")
    while True:
        if need_request:
            ui.status("Consulting the model…")
            try:
                raw_response = call_llm(
                    args.server, args.llm, args.api_key, args.context_size, extra_params,
                    args.tool, request_goal, help_text,
                    verbose=args.verbose, request_timeout=args.request_timeout,
                )
                latest_command = extract_command(raw_response)
                history.append((request_goal, latest_command))
                ui.command(latest_command, len(history))
                if args.allow_execution:
                    confirm_and_run(latest_command, ui)
            except (ValueError, RuntimeError) as exc:
                ui.error(str(exc))
            need_request = False

        try:
            action = ui.prompt("halp › ").strip()
        except EOFError:
            print()
            return 0
        if not action:
            continue
        lowered = action.lower()
        if lowered in {"/quit", "/exit", "quit", "exit", "q"}:
            return 0
        if lowered in {"/help", "help", "?"}:
            ui.help(args.allow_execution)
            continue
        if lowered in {"/retry", "retry"}:
            need_request = True
            continue
        if lowered == "/history":
            _show_history(ui, history)
            continue
        if lowered.startswith("/use "):
            try:
                selected = int(action.split(maxsplit=1)[1])
                if selected < 1:
                    raise ValueError
                latest_command = history[selected - 1][1]
                ui.command(latest_command, selected)
            except (ValueError, IndexError):
                ui.error("Use /use N with a command number shown by /history.")
            continue
        if lowered == "/copy":
            if latest_command is None:
                ui.error("There is no command to copy yet.")
            else:
                error = copy_to_clipboard(latest_command)
                ui.status("Copied current command to clipboard." if error is None else error, error=error is not None)
            continue
        if lowered == "/run":
            if not args.allow_execution:
                ui.error("Execution is disabled. Restart halp with -a to enable it.")
            elif latest_command is None:
                ui.error("There is no command to run yet.")
            else:
                confirm_and_run(latest_command, ui)
            continue
        if lowered == "/clear":
            if sys.stdout.isatty():
                print("\033[2J\033[H", end="")
            ui.banner(args.tool, args.llm, args.server, original)
            if latest_command:
                ui.command(latest_command, len(history) or None)
            continue
        if lowered.startswith("/task "):
            original = action.split(maxsplit=1)[1].strip()
            if not original:
                ui.error("Usage: /task <new task description>")
                continue
            refinements.clear()
            latest_command = None
            request_goal = _compose_goal(original, refinements, latest_command)
            ui.banner(args.tool, args.llm, args.server, original)
            need_request = True
            continue
        if lowered == "/task":
            ui.error("Usage: /task <new task description>")
            continue
        if lowered == "/refine" or lowered.startswith("/refine "):
            feedback = action[len("/refine"):].strip()
            if not feedback:
                try:
                    feedback = ui.prompt("Refinement › ").strip()
                except EOFError:
                    return 0
            if not feedback:
                ui.info("No refinement entered.")
                continue
        elif action.startswith("/"):
            ui.error(f"Unknown action: {action.split()[0]}. Type /help to see available actions.")
            continue
        else:
            feedback = action
        refinements, request_goal = _handle_refinement(feedback, original, refinements, latest_command)
        need_request = True


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.context_size < 512:
        parser.error("-c/--context-size must be at least 512 tokens")
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be greater than zero")
    executable = shutil.which(args.tool)
    if executable is None:
        print(f"halp: tool '{args.tool}' was not found on PATH; install it or check its name.", file=sys.stderr)
        return 127
    try:
        extra_params = parse_extra_args(args.extra_args)
        help_text = gather_tool_context(executable)
        ui = TerminalUI(color=not args.no_color, interactive=args.persist)
        if args.persist:
            return run_persistent(args, extra_params, help_text, ui)

        raw_response = call_llm(
            args.server, args.llm, args.api_key, args.context_size, extra_params,
            args.tool, args.prompt, help_text,
            verbose=args.verbose, request_timeout=args.request_timeout,
        )
        command = extract_command(raw_response)
        ui.command(command)
        if args.allow_execution:
            result = confirm_and_run(command, ui)
            return result if result not in (None, 0) else 0
        return 0
    except (ValueError, RuntimeError) as exc:
        print(f"halp: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
