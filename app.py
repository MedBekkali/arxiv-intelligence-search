"""
app.py — Streamlit web interface for the arXiv CS paper classifier.

Run locally with:
    streamlit run app.py

Opens at http://localhost:8501
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# ---------- Configuration ----------

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / 'models'

VECTORIZER_PATH = MODELS_DIR / 'v1_tfidf_vectorizer.pkl'
MODEL_PATH      = MODELS_DIR / 'v1_logreg.pkl'
ENCODER_PATH    = MODELS_DIR / 'v1_label_encoder.pkl'

# Human-readable names for the cs.* category codes
CATEGORY_NAMES = {
    'cs.AI': 'Artificial Intelligence',
    'cs.AR': 'Hardware Architecture',
    'cs.CC': 'Computational Complexity',
    'cs.CE': 'Computational Engineering',
    'cs.CG': 'Computational Geometry',
    'cs.CL': 'Computation and Language (NLP)',
    'cs.CR': 'Cryptography and Security',
    'cs.CV': 'Computer Vision',
    'cs.CY': 'Computers and Society',
    'cs.DB': 'Databases',
    'cs.DC': 'Distributed, Parallel, Cluster Computing',
    'cs.DL': 'Digital Libraries',
    'cs.DM': 'Discrete Mathematics',
    'cs.DS': 'Data Structures and Algorithms',
    'cs.ET': 'Emerging Technologies',
    'cs.FL': 'Formal Languages and Automata',
    'cs.GR': 'Graphics',
    'cs.GT': 'Computer Science and Game Theory',
    'cs.HC': 'Human-Computer Interaction',
    'cs.IR': 'Information Retrieval',
    'cs.IT': 'Information Theory',
    'cs.LG': 'Machine Learning',
    'cs.LO': 'Logic in Computer Science',
    'cs.MA': 'Multiagent Systems',
    'cs.MM': 'Multimedia',
    'cs.MS': 'Mathematical Software',
    'cs.NA': 'Numerical Analysis',
    'cs.NE': 'Neural and Evolutionary Computing',
    'cs.NI': 'Networking and Internet Architecture',
    'cs.OH': 'Other Computer Science',
    'cs.OS': 'Operating Systems',
    'cs.PF': 'Performance',
    'cs.PL': 'Programming Languages',
    'cs.RO': 'Robotics',
    'cs.SC': 'Symbolic Computation',
    'cs.SD': 'Sound',
    'cs.SE': 'Software Engineering',
    'cs.SI': 'Social and Information Networks',
    'cs.SY': 'Systems and Control',
}

EXAMPLE_ABSTRACTS = {
    'Computer vision (cs.CV)': (
        "We propose a novel deep learning architecture for semantic image segmentation "
        "based on a transformer encoder and a hierarchical decoder. Our model achieves "
        "state-of-the-art results on Cityscapes and ADE20K benchmarks while reducing "
        "inference latency by 40% compared to prior methods."
    ),
    'NLP (cs.CL)': (
        "We present a method for cross-lingual sentence embeddings that aligns "
        "representations across 50 languages without parallel training data. The approach "
        "uses contrastive learning over multilingual masked language model outputs and "
        "demonstrates strong zero-shot transfer on cross-lingual retrieval and "
        "classification benchmarks."
    ),
    'Distributed systems (cs.DC)': (
        "We introduce a Byzantine fault-tolerant consensus protocol that achieves linear "
        "communication complexity in the partial synchrony model. Our algorithm tolerates "
        "up to f < n/3 Byzantine failures and provides deterministic safety with "
        "probabilistic liveness, suitable for permissioned blockchain deployments."
    ),
    'Robotics (cs.RO)': (
        "This paper presents a sim-to-real reinforcement learning framework for "
        "quadruped locomotion over uneven terrain. By combining domain randomization "
        "with proprioceptive feedback, our policy transfers from simulation to a "
        "physical robot without fine-tuning and maintains stable gait at speeds up to 2 m/s."
    ),
}


# ---------- Model loading (cached) ----------

@st.cache_resource
def load_model():
    """Load the three pickled artifacts. Cached so it only runs once per session."""
    missing = [p for p in [VECTORIZER_PATH, MODEL_PATH, ENCODER_PATH] if not p.exists()]
    if missing:
        st.error(
            "Model files not found. Expected:\n"
            + "\n".join(f"  - {p}" for p in missing)
            + "\n\nRun `notebooks/03_baseline_classifier.ipynb` to train and save the model."
        )
        st.stop()
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(MODEL_PATH, 'rb') as f:
        clf = pickle.load(f)
    with open(ENCODER_PATH, 'rb') as f:
        le = pickle.load(f)
    return vectorizer, clf, le


def predict(abstract, vectorizer, clf, le, top_k=5):
    """Returns a DataFrame of top-k (category, full_name, probability)."""
    vec = vectorizer.transform([abstract])
    proba = clf.predict_proba(vec)[0]
    top_idx = np.argsort(proba)[::-1][:top_k]
    return pd.DataFrame({
        'category': [le.classes_[i] for i in top_idx],
        'name': [CATEGORY_NAMES.get(le.classes_[i], '') for i in top_idx],
        'probability': [float(proba[i]) for i in top_idx],
    })


# ---------- Page configuration ----------

st.set_page_config(
    page_title='arXiv CS Classifier',
    page_icon='📄',
    layout='centered',
)


# ---------- Header ----------

st.title('arXiv CS Paper Classifier')
st.caption(
    'Paste a paper abstract; the model predicts which of 39 Computer Science '
    'categories it most likely belongs to.'
)


# ---------- Sidebar: model info & controls ----------

with st.sidebar:
    st.header('About the model')
    st.markdown(
        "**Model:** TF-IDF + Logistic Regression  \n"
        "**Trained on:** 722,116 arXiv CS papers  \n"
        "**Categories:** 39 CS sub-fields  \n"
        "**Test accuracy:** 73.5%  \n"
        "**Test F1 (macro):** 0.607"
    )
    st.divider()

    st.subheader('Settings')
    top_k = st.slider('Number of predictions to show', min_value=3, max_value=10, value=5)

    st.divider()
    st.caption(
        'V1 MVP — single-label classification on abstract text only. '
        'Multi-label and semantic embeddings are planned for V2.'
    )


# ---------- Main interface ----------

# Initialize session state for the textarea
if 'abstract_text' not in st.session_state:
    st.session_state.abstract_text = ''

# Example chips
st.markdown('**Try an example:**')
example_cols = st.columns(len(EXAMPLE_ABSTRACTS))
for col, (label, text) in zip(example_cols, EXAMPLE_ABSTRACTS.items()):
    if col.button(label, use_container_width=True):
        st.session_state.abstract_text = text

# Text area
abstract = st.text_area(
    'Abstract',
    value=st.session_state.abstract_text,
    height=200,
    placeholder='Paste a paper abstract here...',
    label_visibility='collapsed',
)

# Predict button
predict_clicked = st.button('Classify', type='primary', use_container_width=True)


# ---------- Run prediction ----------

if predict_clicked:
    if not abstract or not abstract.strip():
        st.warning('Please enter an abstract or pick an example.')
    else:
        vectorizer, clf, le = load_model()
        results = predict(abstract, vectorizer, clf, le, top_k=top_k)

        # Top prediction summary
        top_row = results.iloc[0]
        st.success(
            f"**Top prediction:** `{top_row['category']}` — {top_row['name']}  \n"
            f"**Confidence:** {top_row['probability']:.1%}"
        )

        # Bar chart
        st.subheader('Top predictions')
        chart_df = results.copy()
        chart_df['label'] = chart_df['category'] + ' — ' + chart_df['name']
        chart_df = chart_df.set_index('label')[['probability']]
        st.bar_chart(chart_df, horizontal=True, height=max(200, top_k * 45))

        # Detail table
        with st.expander('Show full prediction table'):
            display_df = results.copy()
            display_df['probability'] = display_df['probability'].apply(lambda p: f'{p:.4f}')
            display_df.columns = ['Category', 'Name', 'Probability']
            st.dataframe(display_df, hide_index=True, use_container_width=True)

        # Word count caveat for short input
        wc = len(abstract.split())
        if wc < 20:
            st.info(
                f'Heads up: this abstract is only {wc} words. '
                f'The model was trained on abstracts of 20–500 words, '
                f'so very short inputs may be less reliable.'
            )


# ---------- Footer ----------

st.divider()
st.caption(
    'Built for the LA SALLE final ML project • '
    'Dataset: arXiv via Kaggle • '
    'Single-label classification of `first_cat` (39 classes)'
)