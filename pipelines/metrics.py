import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def evaluate(y_true: np.ndarray, scores: np.ndarray, max_fpr: float = 0.01) -> dict:
    """Fraud metrics. The operating point is 'catch as much fraud as possible at 1% FPR'."""
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    idx = np.searchsorted(fpr, max_fpr, side="right") - 1
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "recall_at_1pct_fpr": float(tpr[idx]),
        "threshold_at_1pct_fpr": float(min(thresholds[idx], 1.0)),
        "eval_rows": float(len(y_true)),
        "eval_frauds": float(np.sum(y_true)),
    }
