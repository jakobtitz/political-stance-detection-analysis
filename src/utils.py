from sklearn.metrics import accuracy_score, f1_score


def evaluate_predictions(y_true, y_pred, labels=None):
    """Compute the main classification metrics."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro-F1": f1_score(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        ),
    }