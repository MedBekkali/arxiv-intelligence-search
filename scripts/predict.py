"""
predict.py — Predict the CS category of an arXiv paper abstract.

Usage:
    python predict.py "Your abstract text here..."
    python predict.py --file path/to/abstract.txt
    python predict.py --top-k 10 "Your abstract..."
    cat abstract.txt | python predict.py

Loads the V1 model trained in `03_baseline_classifier.ipynb` and prints
the top-K predicted categories with confidence scores.
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np


# Resolve model paths relative to this script's location, so it runs from anywhere.
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR.parent / 'models'

VECTORIZER_PATH = MODELS_DIR / 'v1_tfidf_vectorizer.pkl'
MODEL_PATH      = MODELS_DIR / 'v1_logreg.pkl'
ENCODER_PATH    = MODELS_DIR / 'v1_label_encoder.pkl'


def load_model():
    """Load the three pickled artifacts. Exits with a clear error if any are missing."""
    for path in [VECTORIZER_PATH, MODEL_PATH, ENCODER_PATH]:
        if not path.exists():
            sys.exit(
                f'ERROR: Missing model file: {path}\n'
                f'Run notebooks/03_baseline_classifier.ipynb first to train and save the model.'
            )
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(MODEL_PATH, 'rb') as f:
        clf = pickle.load(f)
    with open(ENCODER_PATH, 'rb') as f:
        le = pickle.load(f)
    return vectorizer, clf, le


def predict(abstract, vectorizer, clf, le, top_k=5):
    """Return a list of (category, probability) tuples, sorted descending."""
    if not abstract or not abstract.strip():
        raise ValueError('Abstract is empty.')
    vec = vectorizer.transform([abstract])
    proba = clf.predict_proba(vec)[0]
    top_idx = np.argsort(proba)[::-1][:top_k]
    return [(le.classes_[i], float(proba[i])) for i in top_idx]


def get_abstract_from_args(args):
    """Resolve the abstract text from CLI args, file, or stdin (in that order)."""
    if args.text:
        return args.text
    if args.file:
        return Path(args.file).read_text(encoding='utf-8')
    if not sys.stdin.isatty():
        return sys.stdin.read()
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Predict the CS category of an arXiv paper abstract.'
    )
    parser.add_argument('text', nargs='?', help='Abstract text (in quotes).')
    parser.add_argument('--file', '-f', help='Path to a text file with the abstract.')
    parser.add_argument('--top-k', '-k', type=int, default=5,
                        help='Number of top predictions to show (default: 5).')
    args = parser.parse_args()

    abstract = get_abstract_from_args(args)
    if abstract is None:
        parser.print_help()
        sys.exit(1)

    vectorizer, clf, le = load_model()
    predictions = predict(abstract, vectorizer, clf, le, top_k=args.top_k)

    preview = abstract.strip().replace('\n', ' ')[:200]
    print(f'\nAbstract preview: {preview}{"..." if len(abstract) > 200 else ""}\n')
    print(f'Top {args.top_k} predicted categories:')
    for category, probability in predictions:
        bar = '#' * int(probability * 40)
        print(f'  {category:8s}  {probability:.4f}  {bar}')


if __name__ == '__main__':
    main()
