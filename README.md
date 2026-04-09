# ArXiv Semantic Intelligence Hub

An end-to-end Machine Learning product for research paper discovery, classification, temporal analysis, and explainability.

## Features
- **Paper Recommendation**: Semantic similarity search using Sentence-Transformer embeddings
- **Category Classification**: Multi-class text classification (NB, SVM, RF, XGBoost)
- **Year Prediction**: Temporal language drift analysis via regression
- **Research Map**: UMAP/t-SNE clustering visualization with anomaly detection
- **Explainability**: SHAP-based decision explanations

## Tech Stack
Python, Scikit-learn, XGBoost, Sentence-Transformers, SHAP, UMAP, Streamlit, Plotly

## Project Structure
```
arxiv-intelligence/
├── data/               # Raw and processed data
├── notebooks/          # EDA and experimentation
├── src/                # Core pipeline modules
├── app/                # Streamlit application
├── models/             # Saved trained models
├── reports/            # Figures and report
└── README.md
```

## Author
Mohammed BEKKALI — ESAIP Angers, 4th Year Engineering (AI Specialization)
