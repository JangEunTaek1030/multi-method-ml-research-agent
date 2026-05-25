from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report


def evaluate_classification_model(y_true, y_pred):
    """
    Evaluate classification predictions.

    Returns:
        A dictionary containing accuracy, weighted precision,
        weighted recall, and weighted F1 score.
    """

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
    }


def get_classification_report_text(y_true, y_pred, target_names):
    """
    Generate a detailed classification report as text.
    """

    return classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        zero_division=0,
    )
