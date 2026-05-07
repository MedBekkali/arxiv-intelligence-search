"""arXiv Intelligence Search — Streamlit app (V3 complete).

Tabs:
  🏷️  Classify        — V1 single-label classifier (TF-IDF + LogReg)
  🔎  Find similar    — V3 semantic recommender (SPECTER2 + FAISS)
  💬  Ask a question  — V3 RAG (SPECTER2 + FAISS + Claude Haiku)
"""

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="arXiv Intelligence Search",
    page_icon="🔬",
    layout="centered",
)


# ── Cached resource loaders ───────────────────────────────────────────────────

@st.cache_resource
def load_classifier():
    """V1 classifier — runs on CPU, fast to load."""
    clf = pickle.load(open("models/v1_logreg.pkl", "rb"))
    vec = pickle.load(open("models/v1_tfidf_vectorizer.pkl", "rb"))
    le = pickle.load(open("models/v1_label_encoder.pkl", "rb"))
    return clf, vec, le


@st.cache_resource
def load_recommender():
    """V3 semantic recommender — loads SPECTER2 (~440 MB) + FAISS (~3 GB)."""
    from src.arxiv_intel.recommender import Recommender
    return Recommender()


@st.cache_resource
def load_rag():
    """V3 RAG pipeline — reuses the recommender, adds Claude API client."""
    from src.arxiv_intel.rag import RAG
    return RAG(recommender=load_recommender())


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔬 arXiv Intelligence Search")
st.caption(
    "Classify papers by category, find semantically similar work, "
    "or ask natural-language questions grounded in 902,645 arXiv CS papers."
)


tab_classify, tab_similar, tab_ask = st.tabs(
    ["🏷️ Classify", "🔎 Find similar papers", "💬 Ask a question"]
)


# ── Tab 1: Classify (V1) ──────────────────────────────────────────────────────
with tab_classify:
    st.markdown("**Single-label category prediction across 39 CS sub-fields.**")
    st.caption("Model: TF-IDF + Logistic Regression. Trained on 902k papers. Test accuracy 73.5%.")

    classify_text = st.text_area(
        "Paste an abstract",
        height=180,
        key="classify_input",
        placeholder="Paste a paper abstract here…",
    )

    if st.button("Classify", key="btn_classify", type="primary"):
        if not classify_text.strip():
            st.warning("Please paste an abstract first.")
        else:
            clf, vec, le = load_classifier()
            X = vec.transform([classify_text])
            proba = clf.predict_proba(X)[0]
            top_idx = proba.argsort()[::-1][:5]
            results = pd.DataFrame({
                "Category": [le.classes_[i] for i in top_idx],
                "Confidence": [f"{proba[i]:.1%}" for i in top_idx],
            })
            st.subheader("Top 5 predicted categories")
            st.table(results)


# ── Tab 2: Find similar (V3 — semantic) ───────────────────────────────────────
with tab_similar:
    st.markdown("**Semantic similarity search across 902k papers.**")
    st.caption(
        "Model: SPECTER2 (transformer trained on scientific papers) + FAISS HNSW index. "
        "Finds papers by *meaning*, not just shared words."
    )

    similar_text = st.text_area(
        "Paste an abstract",
        height=180,
        key="similar_input",
        placeholder="Paste an abstract here…",
    )
    top_k_similar = st.slider("Number of results", 5, 25, 10, key="topk_similar")

    if st.button("Find similar papers", key="btn_similar", type="primary"):
        if not similar_text.strip():
            st.warning("Please paste an abstract first.")
        else:
            with st.spinner("Loading model and index (first call ~10s)…"):
                recommender = load_recommender()
            with st.spinner("Searching…"):
                df = recommender.recommend(similar_text, top_k=top_k_similar)

            st.subheader(f"Top {top_k_similar} semantically similar papers")
            for _, row in df.iterrows():
                title = row["title"]
                abs_link = row["abs_url"]
                pdf_link = row["pdf_url"]
                arxiv_id = row["id"]
                cat = row.get("first_cat", "")
                year = row.get("year", "")
                sim = f"{row['similarity']:.3f}"

                st.markdown(
                    f"**[{title}]({abs_link})**  "
                    f"[[PDF]]({pdf_link})  ·  "
                    f"`arXiv:{arxiv_id}`  ·  "
                    f"`{cat}`  ·  "
                    f"{year}  ·  "
                    f"sim **{sim}**"
                )
                authors = row.get("authors", "")
                if pd.notna(authors) and authors is not None:
                    if isinstance(authors, list):
                        authors = ", ".join(authors)
                    if authors:
                        st.caption(authors)
                st.divider()


# ── Tab 3: Ask a question (V3 RAG) ────────────────────────────────────────────
with tab_ask:
    st.markdown("**Natural-language Q&A grounded in 902,645 arXiv papers.**")
    st.caption(
        "Architecture: SPECTER2 retrieval + Claude Haiku 4.5 generation. "
        "Every claim cites a real paper from the corpus."
    )

    # Example questions to spark ideas
    with st.expander("💡 Example questions"):
        st.markdown("""
        - *What are recent advances in vision transformers?*
        - *How do contrastive learning methods compare to autoencoders for representation learning?*
        - *What is mixture-of-experts and when is it useful?*
        - *Explain how diffusion models work for text-to-image generation.*
        - *What are the trade-offs between LoRA and full fine-tuning?*
        """)

    question = st.text_area(
        "Your question",
        height=100,
        key="rag_input",
        placeholder="Ask anything answerable from CS research…",
    )

    col_a, col_b = st.columns([1, 2])
    with col_a:
        top_k_rag = st.slider("Papers to retrieve", 3, 10, 5, key="topk_rag")

    if st.button("Ask", key="btn_ask", type="primary"):
        if not question.strip():
            st.warning("Please enter a question first.")
        else:
            with st.spinner("Loading model and index (first call ~10s)…"):
                rag = load_rag()

            with st.spinner("Retrieving relevant papers and generating answer…"):
                try:
                    result = rag.answer(question, top_k=top_k_rag)
                except Exception as e:
                    st.error(f"Error calling the LLM: {e}")
                    st.stop()

            st.subheader("Answer")
            st.markdown(result.answer)

            # Cost meter
            st.caption(
                f"Tokens: {result.input_tokens:,} in · {result.output_tokens:,} out  ·  "
                f"Cost: ${result.cost_usd:.5f}"
            )

            st.subheader("Sources")
            for src in result.sources:
                cat = src.get("first_cat", "")
                year = src.get("year", "")
                arxiv_id = src.get("id", "")
                title = src.get("title", "")
                abs_link = src.get("abs_url", "")
                pdf_link = src.get("pdf_url", "")
                sim = f"{src['similarity']:.3f}"
                rank = src["rank"]

                st.markdown(
                    f"**[#{rank}] [{title}]({abs_link})**  "
                    f"[[PDF]]({pdf_link})  ·  "
                    f"`arXiv:{arxiv_id}`  ·  "
                    f"`{cat}`  ·  "
                    f"{year}  ·  "
                    f"sim **{sim}**"
                )
                authors = src.get("authors", "")
                if isinstance(authors, list):
                    authors = ", ".join(authors)
                if authors:
                    st.caption(authors)
                st.divider()