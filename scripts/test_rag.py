"""Quick CLI test for the RAG pipeline.

Usage:
    python scripts/test_rag.py
    python scripts/test_rag.py --question "your custom question"
    python scripts/test_rag.py --top-k 8
"""

import argparse
import sys
from pathlib import Path

# Make the project's src/ importable when running from anywhere
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--question", "-q",
        default="What are recent advances in vision transformers?",
        help="Question to ask",
    )
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Papers to retrieve")
    args = parser.parse_args()

    print(f"Question: {args.question}")
    print(f"Retrieving top {args.top_k} papers...\n")

    from src.arxiv_intel.rag import RAG
    rag = RAG()
    result = rag.answer(args.question, top_k=args.top_k)

    print("─" * 72)
    print("ANSWER")
    print("─" * 72)
    print(result.answer)
    print()
    print("─" * 72)
    print("SOURCES")
    print("─" * 72)
    for src in result.sources:
        print(f"[#{src['rank']}] {src['title']}")
        print(f"     arXiv:{src['id']}  |  {src['first_cat']}  |  {src['year']}  |  sim {src['similarity']:.3f}")
    print()
    print("─" * 72)
    print("USAGE")
    print("─" * 72)
    print(f"Input tokens:  {result.input_tokens:,}")
    print(f"Output tokens: {result.output_tokens:,}")
    print(f"Cost: ${result.cost_usd:.5f}")


if __name__ == "__main__":
    main()