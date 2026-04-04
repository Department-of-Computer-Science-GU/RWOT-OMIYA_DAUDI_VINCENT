"""
Flask backend for the Scam URL Detector.
Serves the trained Logistic Regression model via a JSON API and
serves the static UI files.
"""

import os, re, json, logging
import joblib
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VEC_PATH   = os.path.join(BASE_DIR, "vectorizer.pkl")
STATIC_DIR = os.path.join(BASE_DIR, "static")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__, static_folder=STATIC_DIR)
CORS(app)

# ── Load artefacts ─────────────────────────────────────────────────────────────
try:
    model      = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)
    log.info("Model and vectoriser loaded successfully.")
    MODEL_READY = True
except Exception as e:
    log.error(f"Could not load model: {e}")
    MODEL_READY = False
    model = vectorizer = None


def extract_url_features(url: str) -> str:
    tokens = re.split(r'[^a-zA-Z0-9]', url.lower())
    tokens = [t for t in tokens if len(t) > 1]
    return " ".join(tokens)


def url_stats(url: str) -> dict:
    """Return simple, human-readable statistics about a URL."""
    domain_match = re.match(r'(?:https?://)?([^/]+)', url)
    domain = domain_match.group(1) if domain_match else url

    path_parts = url.split('/')
    depth = max(len(path_parts) - 1, 0)

    has_ip = bool(re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', domain))
    has_at = '@' in url
    subdomain_count = max(domain.count('.') - 1, 0)
    has_https = url.lower().startswith('https')
    url_len = len(url)

    suspicious_kws = ['login', 'signin', 'bank', 'update', 'secure', 'verify',
                      'account', 'password', 'paypal', 'ebay', 'amazon']
    kws_found = [kw for kw in suspicious_kws if kw in url.lower()]

    return {
        "url_length":       url_len,
        "domain":           domain,
        "subdomain_count":  subdomain_count,
        "path_depth":       depth,
        "has_https":        has_https,
        "has_ip_address":   has_ip,
        "has_at_symbol":    has_at,
        "suspicious_kws":   kws_found,
        "special_char_count": len(re.findall(r'[^a-zA-Z0-9./:-]', url)),
    }


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_file(os.path.join(STATIC_DIR, "index.html"))


@app.route("/api/predict", methods=["POST"])
def predict():
    if not MODEL_READY:
        return jsonify({"error": "Model not loaded. Please train first."}), 503

    data = request.get_json(force=True)
    urls = data.get("urls", [])
    if isinstance(urls, str):
        urls = [urls]
    if not urls:
        return jsonify({"error": "No URL provided"}), 400

    results = []
    for url in urls:
        url = url.strip()
        if not url:
            continue
        tokens  = extract_url_features(url)
        vec     = vectorizer.transform([tokens])
        prob    = model.predict_proba(vec)[0]       # [P(legit), P(scam)]
        label   = model.predict(vec)[0]             # 0=legit 1=scam
        confidence = float(np.max(prob)) * 100
        scam_prob  = float(prob[1]) * 100

        results.append({
            "url":          url,
            "verdict":      "SCAM" if label == 1 else "LEGITIMATE",
            "scam_prob":    round(scam_prob, 2),
            "legit_prob":   round(100 - scam_prob, 2),
            "confidence":   round(confidence, 2),
            "stats":        url_stats(url),
        })

    return jsonify({"results": results})


@app.route("/api/status")
def status():
    return jsonify({
        "model_ready": MODEL_READY,
        "model_type":  type(model).__name__ if MODEL_READY else None,
    })


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(STATIC_DIR, filename)


if __name__ == "__main__":
    log.info("Starting Scam URL Detector on http://localhost:5000")
    app.run(debug=False, host="0.0.0.0", port=5000)
