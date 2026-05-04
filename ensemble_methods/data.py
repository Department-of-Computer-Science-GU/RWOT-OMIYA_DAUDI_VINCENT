"""
data.py
-------
Dataset loading and train/test splitting utilities.

Reads a CSV file with 'url' and 'label' columns (label = 'bad' | 'good')
and converts each row into a numeric feature vector via features.extract_features.
"""

import csv
import random

from features import extract_features


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dataset(filepath: str):
    """Load URLs from *filepath* and return (X, y).

    The CSV must have at minimum two columns:
      - ``url``   : the raw URL string
      - ``label`` : 'bad' (scam) or 'good' (legitimate)

    Returns
    -------
    X : list of list[float]   — feature vectors
    y : list of int           — 1 = scam, 0 = legitimate
    """
    X, y = [], []
    with open(filepath, encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        # Support both 'url'/'URL' and 'label'/'Label' header styles
        fieldnames_lower = {k.lower(): k for k in reader.fieldnames or []}
        url_col   = fieldnames_lower.get('url')
        label_col = fieldnames_lower.get('label')

        if url_col is None or label_col is None:
            raise ValueError(
                f"CSV must have 'url' and 'label' columns. "
                f"Found: {list(reader.fieldnames)}"
            )

        for row in reader:
            raw_url = row[url_col].strip()
            if not raw_url:
                continue
            features = extract_features(raw_url)
            label = 1 if row[label_col].strip().lower() == 'bad' else 0
            X.append(features)
            y.append(label)

    return X, y


def train_test_split(X, y, test_ratio: float = 0.2, seed: int = 42):
    """Shuffle and split (X, y) into train and test sets.

    Parameters
    ----------
    X          : feature matrix (list of lists)
    y          : labels (list of int)
    test_ratio : fraction of data to hold out as test set
    seed       : random seed for reproducibility

    Returns
    -------
    X_train, y_train, X_test, y_test
    """
    random.seed(seed)
    data = list(zip(X, y))
    random.shuffle(data)
    split = int(len(data) * (1 - test_ratio))
    train, test = data[:split], data[split:]
    X_tr, y_tr = zip(*train)
    X_te, y_te = zip(*test)
    return list(X_tr), list(y_tr), list(X_te), list(y_te)
