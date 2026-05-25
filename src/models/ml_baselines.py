from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier


def get_baseline_models(random_state=42):
    """
    Define traditional machine learning baseline models.

    Logistic Regression:
        Strong linear baseline for sparse TF-IDF features.

    Naive Bayes:
        Simple and fast probabilistic baseline for text classification.

    Random Forest:
        Non-linear tree ensemble baseline.
        It is included for comparison, although it may not always perform best
        on high-dimensional sparse text features.
    """

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
        ),
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    return models
