"""
Flask backend for the Scam URL Detector.
Serves the trained Logistic Regression model via a JSON API and
serves the static UI files.

IMPORTANT: the feature pipeline here must stay in sync with train_model.py.
           Any change to extract_url_features(), structural_features(), or
           normalise_url() must be mirrored in both files.
"""

import os, re, logging
import joblib
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from scipy.sparse import hstack, csr_matrix

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "model.pkl")
VEC_PATH     = os.path.join(BASE_DIR, "vectorizer.pkl")
SCALER_PATH  = os.path.join(BASE_DIR, "scaler.pkl")
STATIC_DIR   = os.path.join(BASE_DIR, "static")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__, static_folder=STATIC_DIR)
CORS(app)

# ── Decision threshold ─────────────────────────────────────────────────────────
# Must match the value used / evaluated in train_model.py.
# 0.55 sits at the F1 peak; lower it to prioritise recall over precision.
DECISION_THRESHOLD = 0.55

# ── Load artefacts ─────────────────────────────────────────────────────────────
try:
    model      = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)
    scaler     = joblib.load(SCALER_PATH)
    log.info("Model, vectoriser, and scaler loaded successfully.")
    MODEL_READY = True
except Exception as e:
    log.error(f"Could not load model artefacts: {e}")
    MODEL_READY = False
    model = vectorizer = scaler = None


# ── Feature pipeline (mirrors train_model.py exactly) ─────────────────────────

_HASH_RE = re.compile(r'^[0-9a-f]{20,}$')


def normalise_url(url: str) -> str:
    """Strip scheme and optional www. prefix."""
    url = url.strip().lower()
    url = re.sub(r'^https?://', '', url)
    url = re.sub(r'^www\.', '', url)
    return url


def extract_url_features(url: str) -> str:
    """Tokenise a URL for TF-IDF, stripping hash-like path segments."""
    url = normalise_url(url)
    tokens = re.split(r'[^a-zA-Z0-9]', url)
    tokens = [t for t in tokens if len(t) > 1 and not _HASH_RE.match(t)]
    return " ".join(tokens)


def structural_features(url: str) -> np.ndarray:
    """9-element hand-crafted feature vector (matches training)."""
    url_lower = url.strip().lower()
    norm      = normalise_url(url_lower)

    has_ip         = 1 if re.search(r'\d{1,3}(\.\d{1,3}){3}', norm) else 0
    n_dots         = norm.count('.')
    n_slash        = norm.count('/')
    n_at           = norm.count('@')
    n_pct          = norm.count('%')
    n_eq           = norm.count('=')
    url_len        = len(url_lower)
    has_https      = 1 if url_lower.startswith('https') else 0
    hostname       = norm.split('/')[0]
    subdomain_depth = hostname.count('.')

    return np.array(
        [url_len, n_dots, subdomain_depth, has_ip,
         n_slash, n_at, n_pct, n_eq, has_https],
        dtype=np.float32
    )


def build_feature_matrix(urls: list[str]):
    """Transform a list of URLs into the combined feature matrix used at
    training time: scaled TF-IDF block concatenated with structural features."""
    tokens  = [extract_url_features(u) for u in urls]
    structs = np.vstack([structural_features(u) for u in urls])

    tfidf_mat   = vectorizer.transform(tokens)
    scaled_mat  = scaler.transform(tfidf_mat)          # MaxAbsScaler, fit on train
    feature_mat = hstack([scaled_mat, csr_matrix(structs)])
    return feature_mat


# ── Human-readable URL stats for the UI ───────────────────────────────────────

def url_stats(url: str) -> dict:
    """Return simple statistics about a URL for display in the frontend."""
    norm          = normalise_url(url)
    hostname      = norm.split('/')[0]
    path_parts    = norm.split('/')
    has_ip        = bool(re.search(r'\d{1,3}(\.\d{1,3}){3}', hostname))
    has_at        = '@' in url
    subdomain_count = max(hostname.count('.') - 1, 0)
    has_https     = url.strip().lower().startswith('https')

    suspicious_kws = [
        'login', 'signin', 'bank', 'update', 'secure', 'verify',
        'account', 'password', 'paypal', 'ebay', 'amazon',
    ]
    kws_found = [kw for kw in suspicious_kws if kw in url.lower()]

    return {
        "url_length":         len(url),
        "domain":             hostname,
        "subdomain_count":    subdomain_count,
        "path_depth":         max(len(path_parts) - 1, 0),
        "has_https":          has_https,
        "has_ip_address":     has_ip,
        "has_at_symbol":      has_at,
        "suspicious_kws":     kws_found,
        "special_char_count": len(re.findall(r'[^a-zA-Z0-9./:-]', url)),
    }


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_file(os.path.join(STATIC_DIR, "index.html"))


@app.route("/api/predict", methods=["POST"])
def predict():
    if not MODEL_READY:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503

    data = request.get_json(force=True)
    urls = data.get("urls", [])
    if isinstance(urls, str):
        urls = [urls]
    urls = [u.strip() for u in urls if u.strip()]
    if not urls:
        return jsonify({"error": "No URL provided"}), 400

    try:
        feature_mat = build_feature_matrix(urls)
        probas      = model.predict_proba(feature_mat)   # shape (n, 2)
    except Exception as e:
        log.exception("Feature extraction / inference failed")
        return jsonify({"error": f"Inference error: {e}"}), 500

    results = []
    for url, prob in zip(urls, probas):
        scam_prob = float(prob[1])
        is_scam   = scam_prob >= DECISION_THRESHOLD

        results.append({
            "url":        url,
            "verdict":    "SCAM" if is_scam else "LEGITIMATE",
            "scam_prob":  round(scam_prob * 100, 2),
            "legit_prob": round((1 - scam_prob) * 100, 2),
            # confidence = how far the probability is from the decision boundary
            "confidence": round(abs(scam_prob - DECISION_THRESHOLD) / max(DECISION_THRESHOLD, 1 - DECISION_THRESHOLD) * 100, 2),
            "stats":      url_stats(url),
        })

    return jsonify({"results": results})


@app.route("/api/status")
def status():
    return jsonify({
        "model_ready":       MODEL_READY,
        "model_type":        type(model).__name__ if MODEL_READY else None,
        "decision_threshold": DECISION_THRESHOLD,
    })


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(STATIC_DIR, filename)


if __name__ == "__main__":
    log.info("Starting Scam URL Detector on http://localhost:5000")
    app.run(debug=False, host="0.0.0.0", port=5000)