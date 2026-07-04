"""
utils/tools.py
--------------
Tool definitions for the LangChain AI Research Assistant.
All tools are registered here and imported by app.py.
The LLM — NOT the developer — decides which tool to call at runtime.
"""

from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_classic.tools import Tool
from langchain_experimental.tools import PythonREPLTool


def safe_calc(expression: str) -> str:
    """Evaluate a math expression with a restricted character set."""
    allowed = set("0123456789+-*/().% ")
    cleaned = expression.strip()
    if not cleaned:
        return "Error: empty expression"
    if not all(c in allowed for c in cleaned):
        return "Error: only math expressions allowed"
    try:
        return str(eval(cleaned, {"__builtins__": {}}, {}))
    except Exception as exc:
        return f"Error: {exc}"


# ── 1. DuckDuckGo Web Search ──────────────────────────────────────────────────
def get_search_tool() -> Tool:
    """Return a DuckDuckGo search tool."""
    ddg = DuckDuckGoSearchRun()
    return Tool(
        name="web_search",
        func=ddg.run,
        description=(
            "Search the web using DuckDuckGo. "
            "Use this for recent events, news, or any question that needs "
            "up-to-date information from the internet. "
            "Input should be a natural language search query."
        ),
    )


# ── 2. Wikipedia Search ───────────────────────────────────────────────────────
def get_wikipedia_tool() -> Tool:
    """Return a Wikipedia search tool."""
    wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2))
    return Tool(
        name="wikipedia_search",
        func=wiki.run,
        description=(
            "Search Wikipedia for factual, encyclopedic information. "
            "Ideal for explaining concepts, definitions, historical facts, "
            "and technology overviews. "
            "Input should be a topic or concept name."
        ),
    )


# ── 3. Calculator ─────────────────────────────────────────────────────────────
def get_calculator_tool() -> Tool:
    """Return a safe math calculator tool."""
    return Tool(
        name="calculator",
        func=safe_calc,
        description=(
            "Evaluate mathematical expressions. "
            "Use this for any arithmetic, algebra, or numerical calculations. "
            "Input should be a valid math expression like '125 * 48 + 900'."
        ),
    )


# ── 4. Python REPL ────────────────────────────────────────────────────────────
def get_python_repl_tool() -> Tool:
    """Return a full Python REPL tool for code execution."""
    repl = PythonREPLTool()
    return Tool(
        name="python_repl",
        func=repl.run,
        description=(
            "Execute arbitrary Python code snippets. "
            "Use this when the user asks to run code, generate data, "
            "or perform tasks that require programming logic beyond simple math. "
            "Input must be valid Python code."
        ),
    )


# ── Aggregate all tools ───────────────────────────────────────────────────────
def get_all_tools() -> list[Tool]:
    """
    Return the full list of tools available to the agent.
    The LLM reads each tool's 'description' to decide which to use.
    No hardcoding, no if-else — pure LLM reasoning.
    """
    return [
        get_search_tool(),
        get_wikipedia_tool(),
        get_calculator_tool(),
        get_python_repl_tool(),
    ]
