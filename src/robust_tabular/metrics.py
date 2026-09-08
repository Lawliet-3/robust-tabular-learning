from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def evaluate(
    task: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> dict[str, float]:
    if task == "regression":
        return {
            "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if y_score is not None and len(np.unique(y_true)) > 1:
        try:
            if y_score.ndim == 2 and y_score.shape[1] == 2:
                result["roc_auc"] = float(roc_auc_score(y_true, y_score[:, 1]))
            elif y_score.ndim == 2:
                result["roc_auc"] = float(
                    roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
                )
        except ValueError:
            pass
    return result

