"""
main.py
-------
Entry point for the ensemble-methods URL scam detection pipeline.

Usage
-----
    # From inside ensemble_methods/ (or from the coursework root):
    python ensemble_methods/main.py

    # Override the dataset path:
    python ensemble_methods/main.py --dataset /path/to/urls.csv

    # Quick smoke-test on a small sample (faster iteration):
    python ensemble_methods/main.py --sample 5000

Pipeline
--------
1. Load & featurise URLs from the CSV dataset.
2. Split 80 / 20 into train and test sets.
3. Further split train in half for Stacking's validation partition.
4. Train three ensemble methods:
       • Random Forest  (bagging)
       • AdaBoost       (boosting)
       • Stacking       (meta-learning: RF + AdaBoost → Logistic Regression)
5. Evaluate each on the held-out test set and print a results table.
"""

import argparse
import os
import sys
import time

# ---------------------------------------------------------------------------
# Allow running from either the coursework root or the ensemble_methods dir
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from data      import load_dataset, train_test_split
from models    import RandomForest, AdaBoost, StackingEnsemble, LogisticRegression
from evaluate  import accuracy, precision_recall_f1, confusion_matrix, print_report


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

_COURSEWORK_ROOT = os.path.dirname(_THIS_DIR)
DEFAULT_CSV      = os.path.join(_COURSEWORK_ROOT, 'phishing_site_urls.csv')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Ensemble methods for URL scam detection (from scratch).'
    )
    parser.add_argument(
        '--dataset', default=DEFAULT_CSV,
        help='Path to the CSV file (must have "url" and "label" columns).'
    )
    parser.add_argument(
        '--sample', type=int, default=None,
        help='If set, use only the first N rows (useful for quick testing).'
    )
    parser.add_argument(
        '--test-ratio', type=float, default=0.2,
        help='Fraction of data to hold out as the test set (default: 0.2).'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility (default: 42).'
    )
    # Hyper-parameters
    parser.add_argument('--rf-trees',    type=int, default=15,
                        help='Number of trees in Random Forest (default: 15).')
    parser.add_argument('--rf-depth',    type=int, default=6,
                        help='Max depth of RF trees (default: 6).')
    parser.add_argument('--ada-rounds',  type=int, default=50,
                        help='AdaBoost boosting rounds (default: 50).')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # -- 1. Load dataset ----------------------------------------------------
    print(f"\n[1/5] Loading dataset from: {args.dataset}")
    if not os.path.exists(args.dataset):
        print(f"  ERROR: file not found — {args.dataset}")
        sys.exit(1)

    t0 = time.time()
    X, y = load_dataset(args.dataset)
    print(f"  Loaded {len(X):,} samples in {time.time()-t0:.1f}s")

    # Optional subsampling for quick runs
    if args.sample and args.sample < len(X):
        X, y = X[:args.sample], y[:args.sample]
        print(f"  Subsampled to {len(X):,} rows (--sample {args.sample})")

    pos_rate = sum(y) / len(y)
    print(f"  Class balance — legitimate: {1-pos_rate:.1%}  scam: {pos_rate:.1%}")

    # -- 2. Train / test split ----------------------------------------------
    print(f"\n[2/5] Splitting data (test={args.test_ratio:.0%}, seed={args.seed})")
    X_tr, y_tr, X_te, y_te = train_test_split(
        X, y, test_ratio=args.test_ratio, seed=args.seed
    )
    print(f"  Train: {len(X_tr):,}  Test: {len(X_te):,}")

    # Stacking needs a further split of the training set into
    # a base-model training partition and a meta-feature validation set.
    mid   = len(X_tr) // 2
    X_tr1, y_tr1 = X_tr[:mid], y_tr[:mid]   # base-model training
    X_val, y_val = X_tr[mid:], y_tr[mid:]   # stacking validation

    # -- 3. Define models ---------------------------------------------------
    models = {
        'Random Forest': RandomForest(
            n_trees=args.rf_trees,
            max_depth=args.rf_depth,
            seed=args.seed,
            verbose=True,
        ),
        'AdaBoost': AdaBoost(
            n_estimators=args.ada_rounds,
            verbose=True,
        ),
        'Stacking': StackingEnsemble(
            base_models=[
                RandomForest(n_trees=5,  max_depth=5, seed=args.seed, verbose=True),
                AdaBoost(n_estimators=20, verbose=True),
            ],
            meta_model=LogisticRegression(lr=0.01, epochs=200, verbose=True),
        ),
    }

    # -- 4. Train -----------------------------------------------------------
    print(f"\n[3/5] Training models ...")
    for name, model in models.items():
        t0 = time.time()
        print(f"  -- {name} {'-'*(40-len(name))}")
        if name == 'Stacking':
            model.fit(X_tr1, y_tr1, X_val, y_val, verbose=True)
        else:
            model.fit(X_tr, y_tr)
        print(f"  Done in {time.time()-t0:.1f}s")

    # -- 5. Evaluate --------------------------------------------------------
    print(f"\n[4/5] Evaluating on {len(X_te):,} test samples ...")
    results = {}
    for name, model in models.items():
        t0    = time.time()
        preds = model.predict(X_te)
        elapsed = time.time() - t0

        acc       = accuracy(y_te, preds)
        p, r, f1  = precision_recall_f1(y_te, preds)
        cm        = confusion_matrix(y_te, preds)

        results[name] = {
            'accuracy':  acc,
            'precision': p,
            'recall':    r,
            'f1':        f1,
            'cm':        cm,
        }
        print(f"  {name:<18} predicted in {elapsed:.2f}s")

    # -- 6. Report ----------------------------------------------------------
    print(f"\n[5/5] Results")
    print_report(results)

    print("Key: F1 is the primary metric — it balances precision and recall")
    print("     for the minority 'scam' class.\n")


if __name__ == '__main__':
    main()
