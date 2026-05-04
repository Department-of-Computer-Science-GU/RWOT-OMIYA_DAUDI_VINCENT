"""
models.py
---------
All base classifiers and ensemble methods — implemented from scratch
using only the Python standard library.

Classes
-------
DecisionTree          -- Gini-based CART tree (used as base learner)
DecisionStump         -- Depth-1 tree (weak learner for AdaBoost)
LogisticRegression    -- Gradient-descent logistic regression (meta-learner)
RandomForest          -- Bagging ensemble of DecisionTrees
AdaBoost              -- Boosting ensemble of DecisionStumps
StackingEnsemble      -- Stacked generalisation with any meta-model
"""

import math
import random
import sys
import time
from collections import Counter


# ===========================================================================
# Progress-bar utility (no external dependencies)
# ===========================================================================

def _bar(current: int, total: int, width: int = 28) -> str:
    """Return an ASCII progress bar string, e.g.  [=======>    ] 7/10."""
    filled = int(width * current / max(total, 1))
    bar    = '=' * filled
    if filled < width:
        bar += '>'
    bar    = bar.ljust(width)
    return f'[{bar}] {current}/{total}'


def _print_progress(prefix: str, current: int, total: int, extra: str = '') -> None:
    """Overwrite the current terminal line with a progress indicator."""
    line = f'  {prefix}  {_bar(current, total)}  {extra}'
    # Pad to avoid leftover chars from a longer previous line
    sys.stdout.write('\r' + line.ljust(78))
    sys.stdout.flush()


def _print_done(prefix: str, total: int, elapsed: float, extra: str = '') -> None:
    """Print a final 'done' line, then move to the next line."""
    bar  = '[' + '=' * 28 + ']'
    line = f'  {prefix}  {bar} {total}/{total}  {elapsed:.1f}s  {extra}'
    sys.stdout.write('\r' + line.ljust(78) + '\n')
    sys.stdout.flush()


# ===========================================================================
# Base Learners
# ===========================================================================

class DecisionTree:
    """CART-style binary decision tree trained with Gini impurity.

    Parameters
    ----------
    max_depth    : int  -- maximum tree depth
    min_samples  : int  -- minimum samples required to split a node
    """

    def __init__(self, max_depth: int = 5, min_samples: int = 2):
        self.max_depth   = max_depth
        self.min_samples = min_samples
        self.tree        = None

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _gini(labels: list) -> float:
        """Gini impurity of a label list."""
        n = len(labels)
        if n == 0:
            return 0.0
        counts = Counter(labels)
        return 1.0 - sum((c / n) ** 2 for c in counts.values())

    def _best_split(self, X: list, y: list):
        """Find the feature index and threshold that give the best Gini gain."""
        best_gain, best_feat, best_thresh = -1.0, None, None
        parent_gini = self._gini(y)
        n = len(y)

        for feat_idx in range(len(X[0])):
            # Candidate thresholds: midpoints between consecutive sorted values
            values = sorted(set(row[feat_idx] for row in X))
            thresholds = [
                (values[i] + values[i + 1]) / 2.0
                for i in range(len(values) - 1)
            ]

            for thresh in thresholds:
                left_y  = [y[i] for i in range(n) if X[i][feat_idx] <= thresh]
                right_y = [y[i] for i in range(n) if X[i][feat_idx] >  thresh]
                if not left_y or not right_y:
                    continue

                gain = parent_gini - (
                    len(left_y)  / n * self._gini(left_y) +
                    len(right_y) / n * self._gini(right_y)
                )
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat_idx, thresh

        return best_feat, best_thresh

    def _build(self, X: list, y: list, depth: int) -> dict:
        """Recursively build the tree and return a node dictionary."""
        # Stopping conditions
        if (depth >= self.max_depth or
                len(y) < self.min_samples or
                len(set(y)) == 1):
            return {'leaf': True, 'label': Counter(y).most_common(1)[0][0]}

        feat, thresh = self._best_split(X, y)
        if feat is None:
            return {'leaf': True, 'label': Counter(y).most_common(1)[0][0]}

        mask_l = [X[i][feat] <= thresh for i in range(len(X))]
        X_l = [X[i] for i, m in enumerate(mask_l) if m]
        y_l = [y[i] for i, m in enumerate(mask_l) if m]
        X_r = [X[i] for i, m in enumerate(mask_l) if not m]
        y_r = [y[i] for i, m in enumerate(mask_l) if not m]

        return {
            'leaf':   False,
            'feat':   feat,
            'thresh': thresh,
            'left':   self._build(X_l, y_l, depth + 1),
            'right':  self._build(X_r, y_r, depth + 1),
        }

    def _predict_one(self, node: dict, x: list) -> int:
        if node['leaf']:
            return node['label']
        branch = node['left'] if x[node['feat']] <= node['thresh'] else node['right']
        return self._predict_one(branch, x)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def fit(self, X: list, y: list):
        self.tree = self._build(X, y, depth=0)

    def predict(self, X: list) -> list:
        return [self._predict_one(self.tree, x) for x in X]


# ---------------------------------------------------------------------------

class DecisionStump:
    """A single-split decision tree (depth=1) — canonical AdaBoost weak learner.

    Uses weighted Gini / weighted error to find the best split.
    Supports polarity flipping so the stump can model both label directions.
    """

    def __init__(self):
        self.feat     = None
        self.thresh   = None
        self.polarity = 1   # +1 or -1

    def fit(self, X: list, y: list, weights: list) -> float:
        """Find the best weighted split; return the weighted training error."""
        n      = len(X)
        n_feats = len(X[0])
        best = {'error': float('inf')}

        for feat in range(n_feats):
            vals    = sorted(set(row[feat] for row in X))
            threshs = [(vals[i] + vals[i + 1]) / 2.0 for i in range(len(vals) - 1)]

            for t in threshs:
                for polarity in (1, -1):
                    preds = [polarity if row[feat] <= t else -polarity for row in X]
                    err   = sum(
                        weights[i] for i in range(n) if preds[i] != y[i]
                    )
                    if err < best['error']:
                        best = {
                            'error':    err,
                            'feat':     feat,
                            'thresh':   t,
                            'polarity': polarity,
                        }

        self.feat, self.thresh, self.polarity = (
            best['feat'], best['thresh'], best['polarity']
        )
        return best['error']

    def predict(self, X: list) -> list:
        """Return predictions in {-1, +1}."""
        return [
            self.polarity if row[self.feat] <= self.thresh else -self.polarity
            for row in X
        ]


# ---------------------------------------------------------------------------

class LogisticRegression:
    """Mini batch-free logistic regression trained with stochastic gradient descent.

    Used as the meta-learner in StackingEnsemble.

    Parameters
    ----------
    lr      : float -- learning rate
    epochs  : int   -- number of passes over the training set
    verbose : bool  -- print epoch-level progress bar
    """

    def __init__(self, lr: float = 0.01, epochs: int = 200, verbose: bool = False):
        self.lr      = lr
        self.epochs  = epochs
        self.verbose = verbose
        self.weights = None
        self.bias    = 0.0

    @staticmethod
    def _sigmoid(z: float) -> float:
        z = max(-500.0, min(500.0, z))          # numerical stability
        return 1.0 / (1.0 + math.exp(-z))

    def fit(self, X: list, y: list):
        n_features   = len(X[0])
        self.weights = [0.0] * n_features
        self.bias    = 0.0
        t0           = time.time()
        update_every = max(1, self.epochs // 20)   # refresh ~20 times total

        for epoch in range(self.epochs):
            for i in range(len(X)):
                z    = sum(w * x for w, x in zip(self.weights, X[i])) + self.bias
                pred = self._sigmoid(z)
                err  = pred - y[i]
                self.weights = [
                    w - self.lr * err * X[i][j]
                    for j, w in enumerate(self.weights)
                ]
                self.bias -= self.lr * err

            if self.verbose and (epoch + 1) % update_every == 0:
                _print_progress(
                    'LogReg  epochs',
                    epoch + 1,
                    self.epochs,
                    f'lr={self.lr}',
                )

        if self.verbose:
            _print_done('LogReg  epochs', self.epochs, time.time() - t0)

    def predict(self, X: list) -> list:
        return [
            1 if self._sigmoid(
                sum(w * x for w, x in zip(self.weights, row)) + self.bias
            ) > 0.5 else 0
            for row in X
        ]


# ===========================================================================
# Ensemble Methods
# ===========================================================================

class RandomForest:
    """Bagging ensemble of DecisionTrees with random feature subsets.

    Parameters
    ----------
    n_trees      : int   -- number of trees in the forest
    max_depth    : int   -- max depth of each individual tree
    max_features : int | None -- features sampled per tree (None → sqrt)
    sample_ratio : float -- fraction of training data bootstrapped per tree
    seed         : int   -- random seed
    """

    def __init__(
        self,
        n_trees:      int   = 10,
        max_depth:    int   = 5,
        max_features: int   = None,
        sample_ratio: float = 0.8,
        seed:         int   = 42,
        verbose:      bool  = False,
    ):
        self.n_trees      = n_trees
        self.max_depth    = max_depth
        self.max_features = max_features
        self.sample_ratio = sample_ratio
        self.seed         = seed
        self.verbose      = verbose
        self.trees_           = []    # trained DecisionTree objects
        self.feature_subsets_ = []   # list of feature index lists

    # -----------------------------------------------------------------------

    def _bootstrap(self, X: list, y: list):
        """Draw a bootstrap sample of size ~ sample_ratio * N."""
        n = int(len(X) * self.sample_ratio)
        idxs = [random.randint(0, len(X) - 1) for _ in range(n)]
        return [X[i] for i in idxs], [y[i] for i in idxs]

    # -----------------------------------------------------------------------

    def fit(self, X: list, y: list):
        random.seed(self.seed)
        n_feats = len(X[0])
        max_f   = self.max_features or max(1, int(n_feats ** 0.5))

        self.trees_           = []
        self.feature_subsets_ = []
        t0 = time.time()

        for i in range(self.n_trees):
            # Random feature subset
            feat_idxs = random.sample(range(n_feats), min(max_f, n_feats))
            X_sub     = [[row[i] for i in feat_idxs] for row in X]

            # Bootstrap sample
            X_b, y_b = self._bootstrap(X_sub, y)

            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_b, y_b)

            self.trees_.append(tree)
            self.feature_subsets_.append(feat_idxs)

            if self.verbose:
                elapsed = time.time() - t0
                _print_progress(
                    'RF  trees    ',
                    i + 1,
                    self.n_trees,
                    f'{elapsed:.1f}s  depth={self.max_depth}',
                )

        if self.verbose:
            _print_done('RF  trees    ', self.n_trees, time.time() - t0)

    def predict(self, X: list) -> list:
        """Majority vote across all trees."""
        all_preds = []
        for tree, feat_idxs in zip(self.trees_, self.feature_subsets_):
            X_sub = [[row[i] for i in feat_idxs] for row in X]
            all_preds.append(tree.predict(X_sub))

        # Transpose and take majority vote column-wise
        return [
            Counter(col).most_common(1)[0][0]
            for col in zip(*all_preds)
        ]


# ---------------------------------------------------------------------------

class AdaBoost:
    """Boosting ensemble of DecisionStumps (Freund & Schapire 1997).

    Iteratively trains weak learners by upweighting misclassified examples.

    Parameters
    ----------
    n_estimators : int -- number of boosting rounds
    """

    def __init__(self, n_estimators: int = 50, verbose: bool = False):
        self.n_estimators = n_estimators
        self.verbose      = verbose
        self.stumps_: list = []
        self.alphas_: list = []

    def fit(self, X: list, y: list):
        # AdaBoost works with labels ∈ {-1, +1}
        y_sign  = [1 if label == 1 else -1 for label in y]
        n       = len(X)
        weights = [1.0 / n] * n
        t0      = time.time()

        self.stumps_ = []
        self.alphas_ = []

        for i in range(self.n_estimators):
            stump = DecisionStump()
            err   = stump.fit(X, y_sign, weights)
            err   = max(err, 1e-10)                          # avoid log(0)

            alpha = 0.5 * math.log((1.0 - err) / err)
            preds = stump.predict(X)

            # Boost misclassified examples, shrink correctly classified ones
            weights = [
                w * math.exp(-alpha * y_sign[i] * preds[i])
                for i, w in enumerate(weights)
            ]
            total   = sum(weights)
            weights = [w / total for w in weights]

            self.stumps_.append(stump)
            self.alphas_.append(alpha)

            if self.verbose:
                _print_progress(
                    'AdaBoost rounds',
                    i + 1,
                    self.n_estimators,
                    f'err={err:.4f}  a={alpha:.3f}',
                )

        if self.verbose:
            _print_done('AdaBoost rounds', self.n_estimators, time.time() - t0)

    def predict(self, X: list) -> list:
        """Weighted majority vote; returns 0/1 labels."""
        scores = [0.0] * len(X)
        for stump, alpha in zip(self.stumps_, self.alphas_):
            preds = stump.predict(X)
            for i, p in enumerate(preds):
                scores[i] += alpha * p
        return [1 if s > 0 else 0 for s in scores]


# ---------------------------------------------------------------------------

class StackingEnsemble:
    """Two-level stacked generalisation.

    Base models are trained on X_train; their predictions on a held-out
    validation set form the meta-features that the meta-learner is trained on.

    Parameters
    ----------
    base_models : list  -- instances of any classifier with fit/predict API
    meta_model  : obj   -- classifier that learns from base-model outputs
    """

    def __init__(self, base_models: list, meta_model):
        self.base_models = base_models
        self.meta_model  = meta_model

    def fit(
        self,
        X_train: list,
        y_train: list,
        X_val:   list,
        y_val:   list,
        verbose: bool = False,
    ):
        n_base = len(self.base_models)

        # 1. Train all base models on the training partition
        for idx, m in enumerate(self.base_models):
            name = type(m).__name__
            if verbose:
                print(f'  Stacking  base model {idx+1}/{n_base}: {name} …')
            m.fit(X_train, y_train)

        # 2. Generate meta-features: each column is one base model's prediction
        if verbose:
            print(f'  Stacking  building meta-features from {len(X_val):,} validation samples …')
        meta_X = [
            [m.predict([x])[0] for m in self.base_models]
            for x in X_val
        ]

        # 3. Train the meta-learner on those meta-features
        meta_name = type(self.meta_model).__name__
        if verbose:
            print(f'  Stacking  training meta-learner: {meta_name} …')
        self.meta_model.fit(meta_X, y_val)

        if verbose:
            print(f'  Stacking  done.')

    def predict(self, X: list) -> list:
        meta_X = [
            [m.predict([x])[0] for m in self.base_models]
            for x in X
        ]
        return self.meta_model.predict(meta_X)
