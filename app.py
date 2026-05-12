"""arXiv Intelligence Search — Streamlit app (V3).

Tabs:
  🏷️  Classify        — multi-label prediction (SPECTER2 + LoRA, per-class thresholds)
  🔎  Find Similar    — semantic search (SPECTER2 + FAISS HNSW)
  💬  Ask             — RAG Q&A (SPECTER2 + FAISS + Claude Haiku 4.5)
"""

from pathlib import Path

import pandas as pd
import streamlit as st

# ── Page configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="arXiv Intelligence Search",
    page_icon="🔬",
    layout="centered",
)

# ── Custom styling ────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* Tighter spacing for result cards */
    div[data-testid="stMarkdownContainer"] h3 {
        margin-top: 1.2rem;
    }
    /* Subtle badge styling for category chips */
    .cat-chip {
        display: inline-block;
        padding: 0.2em 0.65em;
        margin: 0.15em 0.1em;
        border-radius: 6px;
        font-size: 0.88em;
        font-weight: 600;
        background: rgba(99, 102, 241, 0.12);
        color: rgb(99, 102, 241);
    }
    .cat-chip-muted {
        display: inline-block;
        padding: 0.2em 0.65em;
        margin: 0.15em 0.1em;
        border-radius: 6px;
        font-size: 0.82em;
        background: rgba(148, 163, 184, 0.10);
        color: rgba(148, 163, 184, 0.8);
    }
    /* Confidence bar backgrounds */
    .conf-bar {
        height: 6px;
        border-radius: 3px;
        margin-top: 2px;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Cached resource loaders ───────────────────────────────────────────────────

CHECKPOINT_DIR = (
    "models/v3_lora_classifier_r32_alpha64_len384_batch64_expI/best_micro"
)


@st.cache_resource(show_spinner=False)
def load_classifier():
    """V3 multi-label classifier — SPECTER2 + LoRA with per-class thresholds."""
    from src.arxiv_intel.classifier import LoRAClassifier

    return LoRAClassifier(checkpoint_dir=CHECKPOINT_DIR, device="cpu")


@st.cache_resource(show_spinner=False)
def load_recommender():
    """V3 semantic recommender — SPECTER2 + FAISS HNSW (~3.4 GB)."""
    from src.arxiv_intel.recommender import Recommender

    return Recommender()


@st.cache_resource(show_spinner=False)
def load_rag():
    """V3 RAG pipeline — retriever + Claude Haiku 4.5."""
    from src.arxiv_intel.rag import RAG

    return RAG(recommender=load_recommender())


# ── Header ────────────────────────────────────────────────────────────────────

st.title("🔬 arXiv Intelligence Search")
st.caption(
    "Multi-label classification · Semantic search · RAG Q&A — "
    "built on 902,645 arXiv CS papers."
)

tab_classify, tab_similar, tab_ask = st.tabs(
    ["🏷️ Classify", "🔎 Find Similar", "💬 Ask"]
)


# ── Tab 1 — Classify ─────────────────────────────────────────────────────────

with tab_classify:
    st.markdown(
        "Predict one or more CS categories for a paper. "
        "The model uses **optimized per-class confidence thresholds** "
        "tuned on 90K validation papers."
    )

    col_title, _ = st.columns([4, 1])
    with col_title:
        classify_title = st.text_input(
            "Title",
            key="cls_title",
            placeholder="e.g. Attention Is All You Need",
        )

    classify_abstract = st.text_area(
        "Abstract",
        height=160,
        key="cls_abstract",
        placeholder="Paste the paper abstract here…",
    )

    if st.button("Classify", key="btn_cls", type="primary", use_container_width=True):
        if not classify_abstract.strip():
            st.warning("Paste an abstract to classify.")
        else:
            title = classify_title.strip() or "Untitled"

            with st.spinner("Loading classifier…"):
                clf = load_classifier()

            with st.spinner("Running inference…"):
                predictions = clf.predict(title, classify_abstract)
                all_probs = clf.predict_all_probs(title, classify_abstract)

            # Predicted categories
            st.markdown("### Predicted categories")

            is_fallback = (
                len(predictions) == 1
                and predictions[0][1] < clf.per_class_thresholds.get(
                    predictions[0][0], 0.90
                )
            )
            if is_fallback:
                st.info(
                    "No category exceeded its threshold — "
                    "showing the highest-scoring one as a fallback."
                )

            for cat, prob in predictions:
                pct = f"{prob:.1%}"
                color = (
                    "rgba(34, 197, 94, 0.8)"
                    if prob >= 0.90
                    else "rgba(99, 102, 241, 0.7)"
                    if prob >= 0.70
                    else "rgba(234, 179, 8, 0.7)"
                )
                st.markdown(
                    f'<span class="cat-chip">{cat}</span> &nbsp; '
                    f"**{pct}**"
                    f'<div class="conf-bar" style="width:{prob*100:.0f}%; '
                    f'background:{color};"></div>',
                    unsafe_allow_html=True,
                )

            st.caption(
                f"{len(predictions)} "
                f"categor{'y' if len(predictions) == 1 else 'ies'} predicted "
                f"· per-class thresholds applied"
            )

            # Top-10 probability chart
            with st.expander("Full probability breakdown (top 10)", expanded=False):
                top10 = all_probs[:10]
                chart_df = pd.DataFrame(top10, columns=["Category", "Probability"])
                chart_df = chart_df.set_index("Category")
                st.bar_chart(chart_df)

    with st.expander("About this model"):
        st.markdown(
            "**Architecture:** SPECTER2 (110M params, frozen) + LoRA "
            "(r=32, α=64, Q/K/V/Dense, 5.4M trainable)  \n"
            "**Training:** 722K papers, BCE loss with capped pos_weight  \n"
            "**Thresholds:** Per-class, optimized on validation set — "
            "broad categories like cs.AI and cs.LG use lower thresholds, "
            "narrow categories use higher ones  \n"
            "**Test Macro F1:** 0.624 · **Micro F1:** 0.714"
        )


# ── Tab 2 — Find Similar ─────────────────────────────────────────────────────

with tab_similar:
    st.markdown(
        "Find semantically similar papers using dense embeddings. "
        "Searches by **meaning**, not keywords."
    )

    similar_text = st.text_area(
        "Abstract",
        height=160,
        key="sim_input",
        placeholder="Paste an abstract to find similar work…",
    )
    top_k_similar = st.slider(
        "Results", 5, 25, 10, key="topk_sim", label_visibility="collapsed"
    )
    st.caption(f"Returning top {top_k_similar} results")

    if st.button(
        "Search", key="btn_sim", type="primary", use_container_width=True
    ):
        if not similar_text.strip():
            st.warning("Paste an abstract to search.")
        else:
            with st.spinner("Loading index…"):
                recommender = load_recommender()
            with st.spinner("Searching 902K papers…"):
                df = recommender.recommend(similar_text, top_k=top_k_similar)

            st.markdown(f"### Top {top_k_similar} results")

            for rank, (_, row) in enumerate(df.iterrows(), 1):
                title = row["title"]
                abs_link = row["abs_url"]
                pdf_link = row["pdf_url"]
                arxiv_id = row["id"]
                cat = row.get("first_cat", "")
                year = row.get("year", "")
                sim = row["similarity"]

                st.markdown(
                    f"**{rank}.** [{title}]({abs_link}) "
                    f"&nbsp;·&nbsp; [PDF]({pdf_link})"
                )
                meta_parts = [f"`{cat}`" if cat else None, str(year) if year else None]
                meta = " · ".join(p for p in meta_parts if p)
                st.caption(
                    f"`arXiv:{arxiv_id}` · {meta} · similarity **{sim:.3f}**"
                )

                authors = row.get("authors", "")
                if pd.notna(authors) and authors:
                    if isinstance(authors, list):
                        authors = ", ".join(authors)
                    st.caption(authors)
                st.divider()

    with st.expander("About this model"):
        st.markdown(
            "**Embeddings:** SPECTER2 base (768-dim, contrastive-trained for paper similarity)  \n"
            "**Index:** FAISS HNSW (M=32, efConstruction=200) over 902,645 papers  \n"
            "**Note:** Uses the *base* SPECTER2 model (not the fine-tuned classifier). "
            "The base model's contrastive training is better suited for similarity search."
        )


# ── Tab 3 — Ask ──────────────────────────────────────────────────────────────

with tab_ask:
    st.markdown(
        "Ask a question about CS research. "
        "Answers are **grounded in real papers** from the corpus — "
        "every claim cites its source."
    )

    with st.expander("Example questions"):
        examples = [
            "What are recent advances in vision transformers?",
            "How do contrastive learning methods compare to autoencoders?",
            "What is mixture-of-experts and when is it useful?",
            "Explain how diffusion models work for image generation.",
            "What are the trade-offs between LoRA and full fine-tuning?",
        ]
        for ex in examples:
            st.markdown(f"- *{ex}*")

    question = st.text_area(
        "Question",
        height=100,
        key="rag_input",
        placeholder="Ask anything answerable from CS research…",
    )

    top_k_rag = st.slider(
        "Papers to retrieve", 3, 10, 5, key="topk_rag", label_visibility="collapsed"
    )
    st.caption(f"Retrieving {top_k_rag} papers for context")

    if st.button("Ask", key="btn_ask", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            with st.spinner("Loading models…"):
                rag = load_rag()

            with st.spinner("Retrieving papers and generating answer…"):
                try:
                    result = rag.answer(question, top_k=top_k_rag)
                except Exception as e:
                    st.error(f"LLM error: {e}")
                    st.stop()

            st.markdown("### Answer")
            st.markdown(result.answer)

            st.caption(
                f"Tokens: {result.input_tokens:,} in · "
                f"{result.output_tokens:,} out · "
                f"Cost: ${result.cost_usd:.4f}"
            )

            st.markdown("### Sources")
            for src in result.sources:
                rank = src["rank"]
                title = src.get("title", "")
                abs_link = src.get("abs_url", "")
                pdf_link = src.get("pdf_url", "")
                arxiv_id = src.get("id", "")
                cat = src.get("first_cat", "")
                year = src.get("year", "")
                sim = src["similarity"]

                st.markdown(
                    f"**[{rank}]** [{title}]({abs_link}) "
                    f"&nbsp;·&nbsp; [PDF]({pdf_link})"
                )
                meta_parts = [f"`{cat}`" if cat else None, str(year) if year else None]
                meta = " · ".join(p for p in meta_parts if p)
                st.caption(
                    f"`arXiv:{arxiv_id}` · {meta} · similarity **{sim:.3f}**"
                )

                authors = src.get("authors", "")
                if isinstance(authors, list):
                    authors = ", ".join(authors)
                if authors:
                    st.caption(authors)
                st.divider()

    with st.expander("About this pipeline"):
        st.markdown(
            "**Retrieval:** SPECTER2 + FAISS (same as Find Similar tab)  \n"
            "**Generation:** Claude Haiku 4.5 via Anthropic API  \n"
            "**Grounding:** The prompt includes retrieved paper titles and abstracts. "
            "Claude is instructed to cite only from provided sources and to say "
            "\"I don't know\" when the context is insufficient."
        )


# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Built by [Mohamed Bekkali](https://github.com/MedBekkali) · "
    "SPECTER2 + LoRA + FAISS + Claude · "
    "[GitHub](https://github.com/MedBekkali/arxiv-intelligence-search)"
)
