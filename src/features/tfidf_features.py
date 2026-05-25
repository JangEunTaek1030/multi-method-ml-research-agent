from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_features(
    X_train,
    X_test,
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
):
    """
    Convert raw text into TF-IDF feature matrices.

    Important:
    - fit_transform is only used on training data.
    - transform is used on test data.
    - This avoids data leakage.
    """

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        stop_words="english",
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf, vectorizer
