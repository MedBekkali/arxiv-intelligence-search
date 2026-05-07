"""RAG pipeline: retrieve relevant papers + ground LLM answer in them.

Architecture:
    user question
        │
        ▼
    SPECTER2 encode        (reuses Recommender's model)
        │
        ▼
    FAISS top-K retrieve   (reuses Recommender's index)
        │
        ▼
    format as context      (numbered paper list)
        │
        ▼
    Anthropic API call     (claude-haiku-4-5 by default)
        │
        ▼
    answer + citations

Usage:
    from arxiv_intel.rag import RAG

    rag = RAG()                                   # loads model+index once
    result = rag.answer("What is contrastive learning?", top_k=5)
    print(result["answer"])
    print(result["sources"])     # list of dicts with title, id, abs_url, etc.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import anthropic
import pandas as pd
from dotenv import load_dotenv

from .recommender import Recommender


# Load .env from project root once at module import
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")


# Token-cost rates for cost tracking (per 1M tokens, Claude Haiku 4.5)
_INPUT_COST_PER_M = 1.00
_OUTPUT_COST_PER_M = 5.00


SYSTEM_PROMPT = """You are a research assistant for an academic search engine over a corpus of \
902,645 arXiv computer-science papers.

You will be given a user question and a numbered list of paper excerpts retrieved from the corpus. \
Your job is to answer the question using ONLY information from the provided excerpts.

Rules:
1. Ground every claim in a specific source paper. Cite using [#1], [#2], etc., matching the numbered list.
2. If the excerpts don't contain enough information to answer, say so plainly. Do not invent facts \
or use general knowledge beyond the excerpts.
3. Be concise. Aim for 3-6 sentences unless the question genuinely needs more.
4. Prefer plain language over jargon when possible. The user is technical but may not be a specialist \
in this exact subfield.
5. End with a one-line summary of which papers were most relevant.
"""


@dataclass
class RAGResult:
    """Result of a single RAG query."""
    question: str
    answer: str
    sources: list[dict]            # the retrieved papers (with metadata)
    input_tokens: int
    output_tokens: int
    cost_usd: float


class RAG:
    """Retrieval-Augmented Generation pipeline."""

    def __init__(
        self,
        recommender: Recommender | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        # Reuse a Recommender (same SPECTER2 + FAISS) or build one
        self.recommender = recommender or Recommender()

        # Model: from arg, or env, or default
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")

        # API key: from arg or env (env loaded at module import)
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not found. Add it to .env at project root."
            )
        self.client = anthropic.Anthropic(api_key=api_key)

    def _format_context(self, papers: pd.DataFrame) -> str:
        """Format retrieved papers as a numbered context block."""
        lines = []
        for i, (_, row) in enumerate(papers.iterrows(), start=1):
            title = row.get("title", "").strip()
            authors = row.get("authors", "")
            if isinstance(authors, list):
                authors = ", ".join(authors[:3])
                if len(row.get("authors", [])) > 3:
                    authors += " et al."
            year = row.get("year", "")
            cat = row.get("first_cat", "")
            arxiv_id = row.get("id", "")
            abstract = row.get("abstract", "")

            lines.append(
                f"[#{i}] {title}\n"
                f"    Authors: {authors}\n"
                f"    Year: {year}  |  Category: {cat}  |  arXiv:{arxiv_id}\n"
                f"    Abstract: {abstract}\n"
            )
        return "\n".join(lines)

    def answer(
        self,
        question: str,
        top_k: int = 5,
        max_tokens: int = 800,
    ) -> RAGResult:
        """Answer a user question using retrieval + grounded generation.

        Args:
            question: The user's natural-language question.
            top_k: How many papers to retrieve as context.
            max_tokens: LLM response length cap.

        Returns:
            RAGResult with answer, sources, and cost info.
        """
        if not question or not question.strip():
            raise ValueError("Empty question.")

        # 1. Retrieve top-K papers
        papers = self.recommender.recommend(question, top_k=top_k)

        # We need the abstract column for context, which Recommender doesn't currently return.
        # Pull abstracts from the recommender's metadata using the row indices.
        # (We'll patch Recommender to optionally include abstracts — see updated recommender.)
        if "abstract" not in papers.columns:
            # Fallback: load abstracts from data file
            abstracts_path = _PROJECT_ROOT / "data" / "processed" / "arxiv_cs_clean.parquet"
            if abstracts_path.exists():
                full = pd.read_parquet(abstracts_path, columns=["id", "abstract"])
                papers = papers.merge(full, on="id", how="left")
            else:
                papers["abstract"] = ""

        # 2. Build the context block
        context = self._format_context(papers)

        user_message = (
            f"Question: {question}\n\n"
            f"Retrieved papers:\n\n{context}\n"
            f"Answer the question using only these papers. Cite as [#1], [#2], etc."
        )

        # 3. Call the LLM
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        answer_text = response.content[0].text
        in_tok = response.usage.input_tokens
        out_tok = response.usage.output_tokens
        cost = (in_tok * _INPUT_COST_PER_M + out_tok * _OUTPUT_COST_PER_M) / 1_000_000

        # 4. Build the source list (without the heavy abstract field for the UI)
        sources = []
        for i, (_, row) in enumerate(papers.iterrows(), start=1):
            sources.append({
                "rank": i,
                "id": row.get("id", ""),
                "title": row.get("title", ""),
                "authors": row.get("authors", ""),
                "year": row.get("year", ""),
                "first_cat": row.get("first_cat", ""),
                "similarity": float(row.get("similarity", 0.0)),
                "abs_url": row.get("abs_url", ""),
                "pdf_url": row.get("pdf_url", ""),
            })

        return RAGResult(
            question=question,
            answer=answer_text,
            sources=sources,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
        )