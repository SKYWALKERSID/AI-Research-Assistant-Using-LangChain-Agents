"""
app.py
------
AI Research Assistant — LangChain Agent with Ollama (LLaMA 3.1)

Features
--------
• Autonomous tool selection via ReAct reasoning (no hardcoding)
• DuckDuckGo web search
• Wikipedia lookup
• Calculator / Python REPL
• Conversation memory
• Colored console output
• Execution timing
• Verbose agent logs
• Graceful error handling
• Loading animation

Usage
-----
    python app.py
    python app.py --verbose      # show full chain-of-thought
    python app.py --demo         # run the 4 required demo questions
    python app.py --model mistral  # override Ollama model
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import threading
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import requests

# ── LangChain imports ─────────────────────────────────────────────────────────
from langchain_ollama import OllamaLLM
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate

# ── Local tool registry ───────────────────────────────────────────────────────
from utils.tools import get_all_tools

DEMO_QUESTIONS = [
    "What is Agentic AI?",
    "Who is the CEO of OpenAI?",
    "Calculate: 125 × 48 + 900",
    "Explain LangChain in 5 lines.",
]

OLLAMA_BASE_URL = "http://localhost:11434"

# ══════════════════════════════════════════════════════════════════════════════
# ANSI colour helpers
# ══════════════════════════════════════════════════════════════════════════════

class Color:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"

def cprint(text: str, color: str = Color.WHITE, bold: bool = False) -> None:
    """Print colored text to stdout."""
    prefix = Color.BOLD if bold else ""
    print(f"{prefix}{color}{text}{Color.RESET}")

def print_banner() -> None:
    """Display a styled startup banner."""
    banner = r"""
  ╔══════════════════════════════════════════════════════════╗
  ║        AI Research Assistant — LangChain Agent           ║
  ║            Powered by Ollama + LLaMA 3.1                 ║
  ║     Tools: DuckDuckGo · Wikipedia · Calculator · REPL    ║
  ╚══════════════════════════════════════════════════════════╝
    """
    cprint(banner, Color.CYAN, bold=True)

def print_separator() -> None:
    cprint("─" * 62, Color.BLUE)

# ══════════════════════════════════════════════════════════════════════════════
# Loading animation (runs in a daemon thread)
# ══════════════════════════════════════════════════════════════════════════════

class Spinner:
    """Simple CLI spinner shown while the agent thinks."""

    _frames = ["|", "/", "-", "\\"]

    def __init__(self, message: str = "Thinking") -> None:
        self._message = message
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)

    def _spin(self) -> None:
        idx = 0
        while not self._stop_event.is_set():
            frame = self._frames[idx % len(self._frames)]
            sys.stdout.write(f"\r{Color.YELLOW}{frame} {self._message}...{Color.RESET}")
            sys.stdout.flush()
            time.sleep(0.1)
            idx += 1

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join()
        sys.stdout.write("\r" + " " * 40 + "\r")
        sys.stdout.flush()

# ══════════════════════════════════════════════════════════════════════════════
# ReAct prompt template
# ══════════════════════════════════════════════════════════════════════════════

REACT_PROMPT_TEMPLATE = """You are a helpful AI Research Assistant with access to multiple tools.
Use the tools to gather accurate information before answering.

You have access to the following tools:

{tools}

Use the following format strictly:

Question: the input question you must answer
Thought: think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Previous conversation history:
{chat_history}

Begin!

Question: {input}
Thought:{agent_scratchpad}"""


def build_prompt() -> PromptTemplate:
    """Build the ReAct prompt template."""
    return PromptTemplate.from_template(REACT_PROMPT_TEMPLATE)


def resolve_ollama_model(model_name: str) -> str:
    """Return the exact Ollama tag for a requested model name."""
    response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
    response.raise_for_status()
    models = [m.get("name", "") for m in response.json().get("models", []) if m.get("name")]

    if model_name in models:
        return model_name

    for name in models:
        if name.split(":")[0] == model_name:
            return name

    for name in models:
        if name.startswith(f"{model_name}:"):
            return name

    raise RuntimeError(
        f"Model '{model_name}' not found. Pull it with: ollama pull {model_name}"
    )


def check_ollama(model_name: str) -> str:
    """Verify Ollama is running and return the resolved model tag."""
    try:
        return resolve_ollama_model(model_name)
    except requests.RequestException as exc:
        raise RuntimeError(
            "Ollama is not reachable at http://localhost:11434. "
            "Start Ollama and try again."
        ) from exc

# ══════════════════════════════════════════════════════════════════════════════
# Agent factory
# ══════════════════════════════════════════════════════════════════════════════

def create_agent(model_name: str = "llama3.1", verbose: bool = False) -> AgentExecutor:
    """
    Initialise the LLM, tools, memory, and AgentExecutor.

    The LLM reads each tool's *description* and uses ReAct reasoning to
    autonomously decide which tool is appropriate — no if-else, no keyword
    matching, no hardcoded routing.
    """
    resolved_model = check_ollama(model_name)

    cprint(f"\n  Loading Ollama model: {resolved_model} ...", Color.MAGENTA)
    llm = OllamaLLM(
        model=resolved_model,
        temperature=0.1,
        num_predict=1024,
    )

    tools = get_all_tools()
    tool_names = ", ".join(t.name for t in tools)
    cprint(f"  Tools registered: {tool_names}", Color.GREEN)

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=False,
        k=10,
    )

    prompt = build_prompt()
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=verbose,
        handle_parsing_errors=True,
        max_iterations=8,
        return_intermediate_steps=True,
    )

    cprint("  Agent ready!\n", Color.GREEN, bold=True)
    return executor

# ══════════════════════════════════════════════════════════════════════════════
# Query runner
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class QueryResult:
    output: str
    tools_used: list[str]
    elapsed: float
    error: Optional[str] = None


def run_query(executor: AgentExecutor, query: str, verbose: bool) -> QueryResult:
    """
    Send a query to the agent and return structured results.
    Shows a spinner while waiting (suppressed in verbose mode).
    """
    spinner: Optional[Spinner] = None
    if not verbose:
        spinner = Spinner("Agent thinking")
        spinner.start()

    start_ts = time.perf_counter()
    output = "No answer returned."
    tools_used: list[str] = []
    error: Optional[str] = None

    try:
        response = executor.invoke({"input": query})
        output = response.get("output", "No answer returned.")
        steps = response.get("intermediate_steps", [])
        tools_used = [step[0].tool for step in steps]
        if tools_used and not verbose:
            cprint(f"\n  Tools used: {' -> '.join(tools_used)}", Color.MAGENTA)
    except Exception as exc:
        error = str(exc)
        output = f"[Error] {exc}"
        cprint(f"\n  Agent error: {exc}", Color.RED)
    finally:
        if spinner:
            spinner.stop()

    elapsed = time.perf_counter() - start_ts
    cprint(f"  Completed in {elapsed:.2f}s", Color.YELLOW)
    return QueryResult(output=output, tools_used=tools_used, elapsed=elapsed, error=error)

# ══════════════════════════════════════════════════════════════════════════════
# Demo mode
# ══════════════════════════════════════════════════════════════════════════════

def run_demo(executor: AgentExecutor, verbose: bool, output_file: Optional[str]) -> None:
    """Run the four required demo questions and optionally save output."""
    buffer = io.StringIO()
    original_stdout = sys.stdout

    class DualWriter:
        def write(self, data: str) -> None:
            original_stdout.write(data)
            buffer.write(data)
        def flush(self) -> None:
            original_stdout.flush()
            buffer.flush()

    sys.stdout = DualWriter()  # type: ignore
    try:
        print()
        for idx, question in enumerate(DEMO_QUESTIONS, start=1):
            print("-" * 62)
            print(f"\nQuestion {idx}: {question}\n")
            result = run_query(executor, question, verbose=verbose)
            print()
            print("AI:")
            print(result.output)
            print()

        print("-" * 62)
        print("\nDemo complete. Tool selection was autonomous (no hardcoded routing).")
    finally:
        sys.stdout = original_stdout

    if output_file:
        import re
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        clean_text = ansi_escape.sub('', buffer.getvalue())
        with open(output_file, "w", encoding="utf-8") as fh:
            fh.write(clean_text)
        cprint(f"\n  Demo output saved to {output_file}", Color.GREEN)

# ══════════════════════════════════════════════════════════════════════════════
# Main REPL loop
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="AI Research Assistant")
    parser.add_argument("--model", default="llama3.1", help="Ollama model name")
    parser.add_argument("--verbose", action="store_true", help="Show agent chain-of-thought")
    parser.add_argument("--demo", action="store_true", help="Run the 4 required demo questions")
    parser.add_argument(
        "--output",
        default="output.txt",
        help="File to save demo output (used with --demo)",
    )
    args = parser.parse_args()

    print_banner()
    cprint(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", Color.WHITE)
    if args.demo:
        cprint("  Running demo mode (4 questions).\n", Color.WHITE)
    else:
        cprint("  Type 'exit' or 'quit' to stop.\n", Color.WHITE)
    print_separator()

    try:
        executor = create_agent(model_name=args.model, verbose=args.verbose)
    except Exception as exc:
        cprint(f"\n  Failed to initialise agent: {exc}", Color.RED, bold=True)
        cprint("   Make sure Ollama is running: ollama serve", Color.YELLOW)
        cprint(f"   And the model is pulled:   ollama pull {args.model}", Color.YELLOW)
        sys.exit(1)

    if args.demo:
        run_demo(executor, verbose=args.verbose, output_file=args.output)
        return

    while True:
        print_separator()
        try:
            user_input = input(f"{Color.CYAN}{Color.BOLD}You: {Color.RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            cprint("\n\nGoodbye!", Color.GREEN, bold=True)
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit", "bye"}:
            cprint("\nGoodbye! Thanks for using the AI Research Assistant.", Color.GREEN, bold=True)
            break

        cprint(f"\n  Processing: {user_input}", Color.BLUE)
        result = run_query(executor, user_input, verbose=args.verbose)

        print()
        cprint("AI:", Color.GREEN, bold=True)
        cprint(result.output or "(no response)", Color.WHITE)
        print()


if __name__ == "__main__":
    main()
