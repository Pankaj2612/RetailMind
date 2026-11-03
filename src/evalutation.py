from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import pandas as pd

def evaluate_classifier(model_name, fitted_pipeline, features, target, show_details=True):
    """Calculate consistent metrics. Churn=Yes (1) is the positive class."""
    predictions = fitted_pipeline.predict(features)

    if hasattr(fitted_pipeline, "predict_proba"):
        continuous_scores = fitted_pipeline.predict_proba(features)[:, 1]
    else:
        continuous_scores = fitted_pipeline.decision_function(features)

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(target, predictions),
        "Precision": precision_score(target, predictions, zero_division=0),
        "Recall": recall_score(target, predictions, zero_division=0),
        "F1": f1_score(target, predictions, zero_division=0),
        "ROC_AUC": roc_auc_score(target, continuous_scores),
    }

    if show_details:
        # display(pd.DataFrame([metrics]).set_index("Model"))
        print("Confusion matrix [[TN, FP], [FN, TP]]:")
        print(confusion_matrix(target, predictions))
        print("\nClassification report:")
        print(classification_report(
            target,
            predictions,
            target_names=["No Churn", "Churn"],
            zero_division=0,
        ))

    return metrics

