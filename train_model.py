"""
Train a Logistic Regression model on the phishing URL dataset
and save it as a .pkl file for use by the Flask backend.
"""

import os
import sys
import re
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR   = os.path.dirname(SCRIPT_DIR)
CSV_PATH     = os.path.join(SCRIPT_DIR, "phishing_site_urls.csv")
MODEL_PATH   = os.path.join(SCRIPT_DIR, "model.pkl")
VEC_PATH     = os.path.join(SCRIPT_DIR, "vectorizer.pkl")

# ── Feature extraction helpers ─────────────────────────────────────────────────
def extract_url_features(url: str) -> str:
    """Return a string of URL tokens for TF-IDF."""
    # Split on non-alphanumeric characters so TF-IDF treats each token separately
    tokens = re.split(r'[^a-zA-Z0-9]', url.lower())
    tokens = [t for t in tokens if len(t) > 1]
    return " ".join(tokens)


def main():
    print("[1/4] Loading dataset …")
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: dataset not found at {CSV_PATH}")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    df = df.drop_duplicates()
    df = df.dropna(subset=["URL", "Label"])
    print(f"      Rows after dedup: {len(df):,}")

    # Binary encode labels
    df["label_bin"] = (df["Label"].str.strip().str.lower() == "bad").astype(int)

    print("[2/4] Tokenising URLs …")
    df["url_tokens"] = df["URL"].apply(extract_url_features)

    X = df["url_tokens"]
    y = df["label_bin"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("[3/4] Fitting TF-IDF vectoriser …")
    vectorizer = TfidfVectorizer(max_features=30_000, ngram_range=(1, 2),
                                  sublinear_tf=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    print("[4/4] Training Logistic Regression …")
    model = LogisticRegression(
        C=1.0,
        solver="saga",
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅  Test Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Scam"]))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VEC_PATH)
    print(f"\nModel saved   → {MODEL_PATH}")
    print(f"Vectoriser saved → {VEC_PATH}")


if __name__ == "__main__":
    main()
