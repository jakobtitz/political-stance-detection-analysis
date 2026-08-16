import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

#Compute overall and class-specific classification metrics
def evaluate_predictions(y_true, y_pred, labels=None, label_names=None):
    

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        ),
    }

    if labels is not None and label_names is not None:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0,
        )

        for i, label_name in enumerate(label_names):
            name = label_name.lower()

            metrics[f"precision_{name}"] = precision[i]
            metrics[f"recall_{name}"] = recall[i]
            metrics[f"f1_{name}"] = f1[i]

    return metrics

#Adapter for Hugging Face Trainer evaluation
def compute_trainer_metrics(eval_pred, labels=None, label_names=None):

    logits, y_true = eval_pred
    y_pred = np.argmax(logits, axis=-1)

    return evaluate_predictions(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        label_names=label_names,
    )
