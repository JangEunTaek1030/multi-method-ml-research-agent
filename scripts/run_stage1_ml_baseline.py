import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


from src.data.load_text_data import load_text_classification_data
from src.features.tfidf_features import build_tfidf_features
from src.models.ml_baselines import get_baseline_models
from src.evaluation.classification_metrics import (
    evaluate_classification_model,
    get_classification_report_text,
)


def main():
    print("Stage 1: Traditional Machine Learning Baseline")
    print("=" * 60)

    print("Loading text classification data...")
    X_train, X_test, y_train, y_test, target_names = load_text_classification_data()

    print(f"Train size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")
    print(f"Classes: {target_names}")

    print("\nBuilding TF-IDF features...")
    X_train_tfidf, X_test_tfidf, vectorizer = build_tfidf_features(
        X_train,
        X_test,
    )

    print(f"TF-IDF train shape: {X_train_tfidf.shape}")
    print(f"TF-IDF test shape: {X_test_tfidf.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

    models = get_baseline_models()

    results = []

    report_dir = PROJECT_ROOT / "reports" / "stage1"
    report_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model in models.items():
        print("\n" + "-" * 60)
        print(f"Training model: {model_name}")

        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)

        metrics = evaluate_classification_model(y_test, y_pred)
        metrics["model"] = model_name
        results.append(metrics)

        report_text = get_classification_report_text(
            y_test,
            y_pred,
            target_names,
        )

        safe_model_name = model_name.lower().replace(" ", "_")
        report_path = report_dir / f"{safe_model_name}_classification_report.txt"

        with open(report_path, "w", encoding="utf-8") as file:
            file.write(report_text)

        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Weighted F1: {metrics['f1_weighted']:.4f}")
        print(f"Report saved to: {report_path}")

    results_df = pd.DataFrame(results)

    results_df = results_df[
        [
            "model",
            "accuracy",
            "precision_weighted",
            "recall_weighted",
            "f1_weighted",
        ]
    ]

    results_path = report_dir / "model_results.csv"
    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 60)
    print("Model comparison:")
    print(results_df)
    print(f"\nModel results saved to: {results_path}")


if __name__ == "__main__":
    main()
