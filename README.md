# AI Research Assistant — LangChain Agent

> **College Assessment Project** — SAARTHILMS  
> Build an AI Research Assistant Using LangChain Agents

---

## Project Overview

This project implements a fully autonomous **AI Research Assistant** using
**LangChain's ReAct (Reasoning + Acting) Agent** framework powered by a local
**LLaMA 3.1** model running via **Ollama**.

The assistant accepts any natural language query and **automatically decides**
which tool (web search, Wikipedia, calculator, Python REPL) to use — without
any hardcoded routing, keyword matching, or if-else logic.

---

## Features

| Feature | Details |
|---|---|
| Local LLM | Ollama + LLaMA 3.1 (no API key needed) |
| Web Search | DuckDuckGo (real-time results) |
| Encyclopedia | Wikipedia lookup |
| Calculator | Safe math expression evaluator |
| Code Executor | Full Python REPL for logic tasks |
| Memory | Conversation history (last 10 turns) |
| Demo Mode | Auto-run the 4 required assessment questions |
| Verbose Mode | Optional chain-of-thought logging |
| Ollama Health Check | Validates model availability on startup |

---

## Folder Structure

```
LangChain-Agent/
├── app.py                 Main application & REPL loop
├── demo.py                Standalone demo runner
├── requirements.txt       Python dependencies
├── README.md              This file
├── output.txt             Sample execution output
├── screenshots/           Submission evidence screenshots
└── utils/
    ├── __init__.py
    └── tools.py           Tool definitions (search, wiki, calc, REPL)
```

---

## Installation

### Step 1 — Install Python 3.11+

Windows: download from https://www.python.org/downloads/ (check "Add Python to PATH")

### Step 2 — Install Ollama

Windows: download from https://ollama.com/download

### Step 3 — Pull LLaMA 3.1

```powershell
ollama pull llama3.1
```

### Step 4 — Create virtual environment and install dependencies

```powershell
cd LangChain-Agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## Running the Project

### Interactive mode

```powershell
python app.py
```

### Demo mode (4 required questions)

```powershell
python app.py --demo
python demo.py
```

### Verbose mode (shows ReAct chain-of-thought)

```powershell
python app.py --verbose
python app.py --demo --verbose
```

### Use a different Ollama model

```powershell
python app.py --model mistral
```

---

## Sample Questions

| # | Question | Expected Tool |
|---|---|---|
| 1 | What is Agentic AI? | `web_search` |
| 2 | Who is the CEO of OpenAI? | `web_search` |
| 3 | Calculate 125 * 48 + 900 | `calculator` |
| 4 | Explain LangChain in 5 lines. | `wikipedia_search` |

---

## Tool Selection (No Hardcoding)

Each tool in `utils/tools.py` has a plain-English **description**. The LangChain
`create_react_agent` passes all descriptions to the LLM inside the ReAct prompt.
The LLM reads the query and autonomously emits `Action: <tool_name>` — there is
no Python if/else routing logic.

Run with `--verbose` to see the full Thought/Action/Observation trace.

---

*Submitted for SAARTHILMS College Assessment — AI Research Assistant Project*
