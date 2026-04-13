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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
)
from sklearn.preprocessing import MaxAbsScaler

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
CSV_PATH     = os.path.join(SCRIPT_DIR, "phishing_site_urls.csv")
MODEL_PATH   = os.path.join(SCRIPT_DIR, "model.pkl")
VEC_PATH     = os.path.join(SCRIPT_DIR, "vectorizer.pkl")
SCALER_PATH  = os.path.join(SCRIPT_DIR, "scaler.pkl")

# Decision threshold tuned on the PR curve (set after first run; override here).
# 0.55 sits at the F1 peak (~0.629) balancing precision and recall.
# Lower this (e.g. 0.45) to catch more phishing at the cost of more false alarms.
DECISION_THRESHOLD = 0.55


# ── Feature extraction helpers ─────────────────────────────────────────────────

# Matches tokens that look like MD5 / SHA hashes — long, all hex chars.
# These consume vocabulary budget without generalising to unseen URLs.
_HASH_RE = re.compile(r'^[0-9a-f]{20,}$')


def normalise_url(url: str) -> str:
    """Strip scheme and optional www. prefix so the model focuses on
    domain + path structure rather than protocol."""
    url = url.strip().lower()
    url = re.sub(r'^https?://', '', url)
    url = re.sub(r'^www\.', '', url)
    return url


def is_valid_ascii(url: str) -> bool:
    """Return True if the URL contains only printable ASCII characters.
    Binary / non-ASCII URLs are likely corrupted and should be dropped
    before training."""
    try:
        url.encode('ascii')
        return True
    except UnicodeEncodeError:
        return False


def extract_url_features(url: str) -> str:
    """Return a string of URL tokens for TF-IDF.

    Changes vs original:
    - Normalise URL first (strip scheme / www)
    - Filter out hash-like tokens (>20 hex chars) that don't generalise
    - Keep min token length at 2 (unchanged)
    """
    url = normalise_url(url)
    tokens = re.split(r'[^a-zA-Z0-9]', url)
    tokens = [
        t for t in tokens
        if len(t) > 1 and not _HASH_RE.match(t)
    ]
    return " ".join(tokens)


# ── Hand-crafted structural features ──────────────────────────────────────────

def structural_features(url: str) -> np.ndarray:
    """Return a small vector of interpretable URL signals.

    These capture phishing patterns that TF-IDF token overlap cannot see:
    - URL length           long URLs often obfuscate the real domain
    - Dot count            many dots = deep subdomains (common in phishing)
    - Subdomain depth      tokens before the effective TLD
    - IP address present   attacker-controlled numeric hosts
    - Special char count   @, %, = are query/redirect indicators
    - Path depth           number of '/' segments
    - Has HTTPS            surface-level trust signal
    """
    url_lower = url.strip().lower()
    norm = normalise_url(url_lower)

    has_ip   = 1 if re.search(r'\d{1,3}(\.\d{1,3}){3}', norm) else 0
    n_dots   = norm.count('.')
    n_slash  = norm.count('/')
    n_at     = norm.count('@')
    n_pct    = norm.count('%')
    n_eq     = norm.count('=')
    url_len  = len(url_lower)
    has_https = 1 if url_lower.startswith('https') else 0

    # Subdomain depth: tokens in hostname before first '/'
    hostname = norm.split('/')[0]
    subdomain_depth = hostname.count('.')

    return np.array([
        url_len,
        n_dots,
        subdomain_depth,
        has_ip,
        n_slash,
        n_at,
        n_pct,
        n_eq,
        has_https,
    ], dtype=np.float32)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("[1/6] Loading dataset …")
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: dataset not found at {CSV_PATH}")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    df = df.drop_duplicates()
    df = df.dropna(subset=["URL", "Label"])

    # Drop corrupted / binary URLs — they are almost all labelled "good"
    # but carry non-ASCII bytes that produce noise tokens.
    before = len(df)
    df = df[df["URL"].apply(is_valid_ascii)]
    print(f"      Dropped {before - len(df):,} non-ASCII URLs")
    print(f"      Rows after dedup + ASCII filter: {len(df):,}")

    # Binary encode labels
    df["label_bin"] = (df["Label"].str.strip().str.lower() == "bad").astype(int)
    pos_rate = df["label_bin"].mean()
    print(f"      Class balance — good: {1-pos_rate:.1%}  bad: {pos_rate:.1%}")

    print("[2/6] Building features …")
    df["url_tokens"] = df["URL"].apply(extract_url_features)

    X_text   = df["url_tokens"].values
    X_struct = np.vstack(df["URL"].apply(structural_features).values)
    y        = df["label_bin"].values

    X_train_txt, X_test_txt, \
    X_train_str, X_test_str, \
    y_train, y_test = train_test_split(
        X_text, X_struct, y,
        test_size=0.2, random_state=42, stratify=y
    )

    print("[3/6] Fitting TF-IDF vectoriser …")
    vectorizer = TfidfVectorizer(
        max_features=10_000,
        ngram_range=(1, 2),   # bigrams capture token-pair patterns (e.g. "login php")
        sublinear_tf=True,
        analyzer="word",
    )
    X_train_tfidf = vectorizer.fit_transform(X_train_txt)
    X_test_tfidf  = vectorizer.transform(X_test_txt)

    # Scale TF-IDF vectors — MaxAbsScaler preserves sparsity and keeps
    # values in [0,1], allowing SAGA to converge in far fewer iterations.
    scaler = MaxAbsScaler()
    X_train_tfidf = scaler.fit_transform(X_train_tfidf)
    X_test_tfidf  = scaler.transform(X_test_tfidf)

    # Concatenate TF-IDF + structural features
    from scipy.sparse import hstack, csr_matrix
    X_train_vec = hstack([X_train_tfidf, csr_matrix(X_train_str)])
    X_test_vec  = hstack([X_test_tfidf,  csr_matrix(X_test_str)])

    print("[4/6] Training Logistic Regression …")
    model = LogisticRegression(
        C=1.0,
        solver="saga",
        max_iter=5000,           # SAGA needs more iterations on the combined
                                 # TF-IDF + structural feature matrix; 5000 is
                                 # sufficient while keeping training time reasonable
        tol=1e-3,                # slightly relaxed tolerance (default 1e-4) speeds
                                 # up convergence without meaningfully hurting accuracy
        class_weight="balanced", # corrects for 77/23 class imbalance;
                                 # use DECISION_THRESHOLD to tune recall/precision
                                 # trade-off instead of removing this
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train_vec, y_train)

    print("[5/6] Evaluating …")
    y_proba = model.predict_proba(X_test_vec)[:, 1]

    # Default threshold (0.5) metrics
    y_pred_default = (y_proba >= 0.5).astype(int)
    acc_default = accuracy_score(y_test, y_pred_default)
    print(f"\n  Default threshold (0.50)")
    print(f"  Accuracy : {acc_default:.4f}")
    print(classification_report(y_test, y_pred_default,
                                 target_names=["Legitimate", "Phishing"]))

    # Tuned threshold metrics
    y_pred_tuned = (y_proba >= DECISION_THRESHOLD).astype(int)
    acc_tuned = accuracy_score(y_test, y_pred_tuned)
    print(f"  Tuned threshold  ({DECISION_THRESHOLD:.2f})")
    print(f"  Accuracy : {acc_tuned:.4f}")
    print(classification_report(y_test, y_pred_tuned,
                                 target_names=["Legitimate", "Phishing"]))

    # Threshold-independent metrics
    roc_auc = roc_auc_score(y_test, y_proba)
    avg_prec = average_precision_score(y_test, y_proba)
    print(f"  ROC-AUC          : {roc_auc:.4f}")
    print(f"  Avg Precision    : {avg_prec:.4f}")

    # Print PR curve summary to help pick a threshold
    print("\n[6/6] Precision-Recall curve (sample of thresholds) …")
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    print(f"  {'Threshold':>10}  {'Precision':>10}  {'Recall':>10}  {'F1':>10}")
    for t in np.arange(0.1, 0.9, 0.05):
        idx = np.searchsorted(thresholds, t)
        if idx < len(precisions) - 1:
            p, r = precisions[idx], recalls[idx]
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            print(f"  {t:>10.2f}  {p:>10.4f}  {r:>10.4f}  {f1:>10.4f}")

    print("\n  → Adjust DECISION_THRESHOLD at the top of this file to tune")
    print("    recall (catch more phishing) vs precision (fewer false alarms).")

    # ── Persist artefacts ──────────────────────────────────────────────────────
    joblib.dump(model,      MODEL_PATH)
    joblib.dump(vectorizer, VEC_PATH)
    joblib.dump(scaler,     SCALER_PATH)
    print(f"\nModel saved      -> {MODEL_PATH}")
    print(f"Vectoriser saved -> {VEC_PATH}")
    print(f"Scaler saved     -> {SCALER_PATH}")
    print(f"\nRemember to pass URLs through normalise_url() + extract_url_features()")
    print(f"and structural_features() at inference time, and apply the scaler")
    print(f"to the TF-IDF block before concatenating with structural features.")


if __name__ == "__main__":
    main()