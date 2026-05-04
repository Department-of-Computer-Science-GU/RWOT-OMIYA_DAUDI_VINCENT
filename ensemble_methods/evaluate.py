"""
evaluate.py
-----------
Evaluation utilities for the ensemble-methods pipeline.

Functions
---------
accuracy             -- simple accuracy score
precision_recall_f1  -- P / R / F1 for a given positive class
confusion_matrix     -- TP / FP / FN / TN counts
print_report         -- pretty-print a full evaluation table
"""


def accuracy(y_true: list, y_pred: list) -> float:
    """Fraction of correctly classified samples."""
    if not y_true:
        return 0.0
    return sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)


def precision_recall_f1(y_true: list, y_pred: list, pos: int = 1):
    """Precision, Recall, and F1 for the *pos* class.

    Returns
    -------
    (precision, recall, f1)  all rounded to 4 decimal places
    """
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == pos and b == pos)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a != pos and b == pos)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == pos and b != pos)

    p  = tp / (tp + fp) if (tp + fp) else 0.0
    r  = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0

    return round(p, 4), round(r, 4), round(f1, 4)


def confusion_matrix(y_true: list, y_pred: list, pos: int = 1) -> dict:
    """Return a dict with TP, FP, FN, TN counts."""
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == pos and b == pos)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a != pos and b == pos)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == pos and b != pos)
    tn = sum(1 for a, b in zip(y_true, y_pred) if a != pos and b != pos)
    return {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn}


def print_report(results: dict):
    """Pretty-print evaluation results produced by main.py.

    Parameters
    ----------
    results : dict
        Mapping of model_name -> {'accuracy', 'precision', 'recall', 'f1', 'cm'}
    """
    header = f"\n{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}"
    sep    = '-' * len(header)
    print(header)
    print(sep)

    for name, m in results.items():
        print(
            f"{name:<20} "
            f"{m['accuracy']:>10.4f} "
            f"{m['precision']:>10.4f} "
            f"{m['recall']:>10.4f} "
            f"{m['f1']:>10.4f}"
        )

    print(sep)
    print("\nConfusion matrices (scam = positive class):")
    for name, m in results.items():
        cm = m['cm']
        print(
            f"  {name:<18} | "
            f"TP={cm['TP']:>5}  FP={cm['FP']:>5}  "
            f"FN={cm['FN']:>5}  TN={cm['TN']:>5}"
        )
    print()
