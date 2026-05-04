"""
features.py
-----------
URL feature extraction for the ensemble-methods scam-detection pipeline.

Each URL is converted into a fixed-length numeric vector of 13 features
covering length, character composition, entropy, TLD trustworthiness,
suspicious keywords, and structural properties.
"""

import re
import math
from urllib.parse import urlparse
from collections import Counter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUSPICIOUS_KEYWORDS = [
    'login', 'verify', 'secure', 'update', 'bank',
    'paypal', 'free', 'click', 'win', 'account',
    'signin', 'password', 'confirm', 'billing',
]

TRUSTED_TLDS = {'.com', '.org', '.edu', '.gov', '.net', '.ac'}

FEATURE_NAMES = [
    'url_length',
    'domain_length',
    'digit_ratio',
    'special_chars',
    'dot_count',
    'hyphen_count',
    'entropy',
    'is_trusted_tld',
    'has_suspicious_keyword',
    'subdomain_count',
    'has_ip_address',
    'path_depth',
    'has_port',
]


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def _shannon_entropy(text: str) -> float:
    """Shannon entropy of a string — higher values indicate more randomness."""
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def extract_features(url: str) -> list:
    """Return a list of 13 numeric features for *url*.

    Parameters
    ----------
    url : str
        Raw URL string (with or without scheme).

    Returns
    -------
    list of float/int
        Feature vector of length ``len(FEATURE_NAMES)`` == 13.
    """
    parsed = urlparse(url)
    domain = parsed.netloc or url.split('/')[0]
    path   = parsed.path

    # --- Length-based ---
    url_length    = len(url)
    domain_length = len(domain)

    # --- Character composition ---
    digit_ratio   = sum(c.isdigit() for c in url) / max(len(url), 1)
    special_chars = len(re.findall(r'[@\-_~]', url))
    dot_count     = url.count('.')
    hyphen_count  = domain.count('-')

    # --- Entropy (randomness indicator) ---
    entropy = _shannon_entropy(url)

    # --- TLD features ---
    tld = ('.' + domain.split('.')[-1]) if '.' in domain else ''
    is_trusted_tld = int(tld.lower() in TRUSTED_TLDS)

    # --- Keyword features ---
    url_lower = url.lower()
    has_suspicious_keyword = int(any(kw in url_lower for kw in SUSPICIOUS_KEYWORDS))

    # --- Structural ---
    subdomain_count = max(len(domain.split('.')) - 2, 0)
    has_ip_address  = int(bool(re.match(r'\d{1,3}(\.\d{1,3}){3}', domain)))
    path_depth      = path.count('/')
    has_port        = int(bool(parsed.port))

    return [
        url_length, domain_length, digit_ratio, special_chars,
        dot_count, hyphen_count, entropy, is_trusted_tld,
        has_suspicious_keyword, subdomain_count, has_ip_address,
        path_depth, has_port,
    ]
