import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.load_text_data import load_text_classification_data, load_toy_text_data
from src.evaluation.classification_metrics import (
    evaluate_classification_model,
    get_classification_report_text,
)
from src.features.tfidf_features import build_tfidf_features
from src.models.ml_baselines import get_baseline_models


def parse_args():
    """Parse command-line arguments for the Stage 1 baseline runner."""
    parser = argparse.ArgumentParser(
        description="Run Stage 1 traditional ML text classification baselines."
    )
    parser.add_argument("--data", type=Path, default=None, help="Path to CSV dataset.")
    parser.add_argument("--text_col", type=str, default="text", help="Text column name.")
    parser.add_argument("--label_col", type=str, default="label", help="Label column name.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split size.")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("reports/stage1"),
        help="Output directory for reports and artifacts.",
    )
    parser.add_argument(
        "--max_features",
        type=int,
        default=5000,
        help="Maximum TF-IDF features.",
    )
    parser.add_argument(
        "--ngram_max",
        type=int,
        default=2,
        help="Maximum n-gram size for TF-IDF.",
    )
    return parser.parse_args()


def load_csv_dataset(data_path, text_col, label_col):
    """Load and clean CSV dataset for text classification."""
    if not data_path.exists():
        raise FileNotFoundError(f"CSV file not found: {data_path}")

    df = pd.read_csv(data_path)
    print(f"Loaded CSV: {data_path}")
    print(f"Original dataset shape: {df.shape}")

    missing_cols = [col for col in [text_col, label_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. Available columns: {list(df.columns)}"
        )

    df = df[[text_col, label_col]].copy()
    df = df.dropna(subset=[text_col, label_col])
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col] != ""]

    print(f"Cleaned dataset shape: {df.shape}")
    print("Label distribution:")
    print(df[label_col].value_counts(dropna=False))

    return df[text_col].tolist(), df[label_col].tolist()


def split_dataset(texts, labels, test_size, random_state):
    """Split dataset, using stratification when possible."""
    stratify_labels = None
    label_counts = pd.Series(labels).value_counts()

    if len(label_counts) > 1 and label_counts.min() >= 2:
        stratify_labels = labels
        print("Using stratified train/test split.")
    else:
        print(
            "Warning: stratified split is not possible (a class has fewer than 2 samples)."
        )
        print("Falling back to non-stratified train/test split.")

    try:
        return train_test_split(
            texts,
            labels,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_labels,
        )
    except ValueError as error:
        print(f"Warning: stratified split failed: {error}")
        print("Falling back to non-stratified train/test split.")
        return train_test_split(
            texts,
            labels,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )


def train_and_evaluate_models(X_train_tfidf, X_test_tfidf, y_train, y_test, model_map):
    """Train baseline models and compute metrics/reports."""
    results = []
    reports = {}
    trained_models = {}

    target_names = sorted(pd.Series(y_train).astype(str).unique().tolist())

    for model_name, model in model_map.items():
        print("\n" + "-" * 60)
        print(f"Training model: {model_name}")

        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)

        metrics = evaluate_classification_model(y_test, y_pred)
        metrics["f1_macro"] = f1_score(y_test, y_pred, average="macro", zero_division=0)
        metrics["model"] = model_name
        results.append(metrics)

        try:
            reports[model_name] = get_classification_report_text(
                y_test,
                y_pred,
                target_names=target_names,
            )
        except ValueError:
            labels_for_report = sorted(set(y_test) | set(y_pred))
            reports[model_name] = classification_report(
                y_test,
                y_pred,
                labels=labels_for_report,
                target_names=[str(label) for label in labels_for_report],
                zero_division=0,
            )
        trained_models[model_name] = model

        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 weighted: {metrics['f1_weighted']:.4f}")
        print(f"F1 macro: {metrics['f1_macro']:.4f}")

    results_df = pd.DataFrame(results)[
        [
            "model",
            "accuracy",
            "precision_weighted",
            "recall_weighted",
            "f1_weighted",
            "f1_macro",
        ]
    ]

    return results_df, reports, trained_models


def save_outputs(output_dir, results_df, reports, best_model_name, best_model, vectorizer):
    """Save model comparison, reports, summary, and optional artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "model_results.csv"
    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")

    filename_map = {
        "Logistic Regression": "logistic_regression_classification_report.txt",
        "Naive Bayes": "naive_bayes_classification_report.txt",
        "Random Forest": "random_forest_classification_report.txt",
    }

    for model_name, report_text in reports.items():
        report_path = output_dir / filename_map.get(
            model_name, f"{model_name.lower().replace(' ', '_')}_classification_report.txt"
        )
        report_path.write_text(report_text, encoding="utf-8")

    summary_path = output_dir / "best_model_summary.txt"
    summary_text = (
        f"Best model (by f1_macro): {best_model_name}\n"
        f"Best f1_macro: {results_df.loc[results_df['model'] == best_model_name, 'f1_macro'].iloc[0]:.6f}\n"
    )
    summary_path.write_text(summary_text, encoding="utf-8")

    try:
        import joblib

        joblib.dump(best_model, output_dir / "best_model.joblib")
        joblib.dump(vectorizer, output_dir / "tfidf_vectorizer.joblib")
        print("Saved best model and TF-IDF vectorizer with joblib.")
    except Exception as error:
        print(f"Warning: could not save .joblib artifacts: {error}")

    print(f"Saved outputs to: {output_dir}")


def main():
    args = parse_args()

    print("Stage 1: Traditional Machine Learning Baseline")
    print("=" * 60)

    if args.data is None:
        print("No --data provided. Using built-in dataset loader for smoke testing.")
        try:
            X_train, X_test, y_train, y_test, target_names = load_text_classification_data()
        except Exception as error:
            print(f"Warning: built-in loader failed: {error}")
            print("Falling back to toy dataset with compatible split.")
            X_train, X_test, y_train, y_test, target_names = load_toy_text_data(test_size=0.34)
        print(f"Train size: {len(X_train)}")
        print(f"Test size: {len(X_test)}")
        print(f"Classes: {target_names}")
    else:
        texts, labels = load_csv_dataset(args.data, args.text_col, args.label_col)
        X_train, X_test, y_train, y_test = split_dataset(
            texts, labels, args.test_size, args.random_state
        )
        print(f"Train size: {len(X_train)}")
        print(f"Test size: {len(X_test)}")

    print("\nBuilding TF-IDF features...")
    X_train_tfidf, X_test_tfidf, vectorizer = build_tfidf_features(
        X_train,
        X_test,
        max_features=args.max_features,
        ngram_range=(1, args.ngram_max),
        min_df=1,
    )

    print(f"TF-IDF train shape: {X_train_tfidf.shape}")
    print(f"TF-IDF test shape: {X_test_tfidf.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

    models = get_baseline_models(random_state=args.random_state)
    results_df, reports, trained_models = train_and_evaluate_models(
        X_train_tfidf, X_test_tfidf, y_train, y_test, models
    )

    best_row = results_df.sort_values("f1_macro", ascending=False).iloc[0]
    best_model_name = best_row["model"]
    best_model = trained_models[best_model_name]

    save_outputs(
        output_dir=args.output_dir,
        results_df=results_df,
        reports=reports,
        best_model_name=best_model_name,
        best_model=best_model,
        vectorizer=vectorizer,
    )

    print("\n" + "=" * 60)
    print("Model comparison:")
    print(results_df)
    print(f"\nBest model by macro F1: {best_model_name}")


if __name__ == "__main__":
    main()
