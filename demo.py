"""
demo.py
-------
Run the four required demo questions and save output to output.txt.

Usage:
    python demo.py
    python demo.py --verbose
    python demo.py --model llama3.1 --output output.txt
"""

from app import DEMO_QUESTIONS, create_agent, print_banner, print_separator, run_demo, cprint, Color
import argparse
import sys
from datetime import datetime


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LangChain agent demo questions")
    parser.add_argument("--model", default="llama3.1", help="Ollama model name")
    parser.add_argument("--verbose", action="store_true", help="Show chain-of-thought")
    parser.add_argument("--output", default="output.txt", help="Output file path")
    args = parser.parse_args()

    print_banner()
    cprint(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", Color.WHITE)
    cprint(f"  Questions: {len(DEMO_QUESTIONS)}\n", Color.WHITE)
    print_separator()

    try:
        executor = create_agent(model_name=args.model, verbose=args.verbose)
    except Exception as exc:
        cprint(f"\n  Failed to initialise agent: {exc}", Color.RED, bold=True)
        sys.exit(1)

    run_demo(executor, verbose=args.verbose, output_file=args.output)


if __name__ == "__main__":
    main()
