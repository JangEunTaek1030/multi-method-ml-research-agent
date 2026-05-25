from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split


def load_toy_text_data(test_size=0.25, random_state=42):
    """
    Load a small toy text classification dataset.

    This fallback dataset is used when the online 20 Newsgroups dataset
    cannot be downloaded due to network issues.
    """

    texts = [
        "the car engine is powerful",
        "this vehicle has great speed",
        "baseball game was exciting",
        "the team won the baseball match",
        "medicine can improve patient health",
        "the doctor gave medical advice",
        "the election debate was intense",
        "government policy affects voters",
        "new car model has better fuel efficiency",
        "the baseball player hit a home run",
        "the hospital introduced a new treatment",
        "political parties discussed the reform",
    ]

    labels = [
        0, 0,   # autos
        1, 1,   # baseball
        2, 2,   # medicine
        3, 3,   # politics
        0, 1, 2, 3,
    ]

    target_names = ["autos", "baseball", "medicine", "politics"]

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    return X_train, X_test, y_train, y_test, target_names


def load_20newsgroups_data(test_size=0.2, random_state=42):
    """
    Load a subset of the 20 Newsgroups dataset.

    The dataset is a classic text classification benchmark.
    We use four categories to keep the first baseline simple.
    """

    categories = [
        "rec.autos",
        "rec.sport.baseball",
        "sci.med",
        "talk.politics.misc",
    ]

    dataset = fetch_20newsgroups(
        subset="all",
        categories=categories,
        remove=("headers", "footers", "quotes"),
    )

    X = dataset.data
    y = dataset.target
    target_names = dataset.target_names

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test, target_names


def load_text_classification_data(use_toy_fallback=True):
    """
    Load text classification data.

    First try to load 20 Newsgroups.
    If downloading fails, use the small toy dataset.
    """

    try:
        return load_20newsgroups_data()
    except Exception as error:
        if not use_toy_fallback:
            raise error

        print("Warning: failed to load 20 Newsgroups dataset.")
        print(f"Reason: {error}")
        print("Using toy text dataset instead.")

        return load_toy_text_data()
