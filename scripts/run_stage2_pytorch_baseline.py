import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.torch_text_dataset import ResearchTextDataset, build_label_mapping, build_vocab
from src.models.torch_text_classifier import MeanPoolingTextClassifier
from src.training.torch_trainer import (
    compute_metrics,
    evaluate,
    save_stage2_outputs,
    set_seed,
    train_one_epoch,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Stage 2 PyTorch neural text classification baseline.")
    parser.add_argument("--data", type=Path, default=Path("data/sample/research_queries_sample.csv"))
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--max_len", type=int, default=24)
    parser.add_argument("--min_freq", type=int, default=1)
    parser.add_argument("--max_vocab_size", type=int, default=5000)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output_dir", type=Path, default=Path("reports/stage2"))
    return parser.parse_args()


def pick_device(device_arg):
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    args = parse_args()
    set_seed(args.random_state)
    device = pick_device(args.device)

    print("Stage 2: PyTorch Neural Baseline")
    print("=" * 60)
    print(f"Using device: {device}")

    if not args.data.exists():
        raise FileNotFoundError(f"CSV file not found: {args.data}")

    df = pd.read_csv(args.data)
    missing_cols = [c for c in [args.text_col, args.label_col] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}. Available columns: {list(df.columns)}")

    df = df[[args.text_col, args.label_col]].copy()
    df = df.dropna(subset=[args.text_col, args.label_col])
    df[args.text_col] = df[args.text_col].astype(str).str.strip()
    df = df[df[args.text_col] != ""]

    print(f"Cleaned dataset shape: {df.shape}")
    print("Label distribution:")
    print(df[args.label_col].value_counts(dropna=False))

    texts = df[args.text_col].tolist()
    labels = df[args.label_col].astype(str).tolist()

    label_counts = df[args.label_col].value_counts()
    stratify_labels = labels if (len(label_counts) > 1 and label_counts.min() >= 2) else None
    if stratify_labels is None:
        print("Warning: stratified split not possible, using non-stratified split.")

    try:
        X_train, X_val, y_train, y_val = train_test_split(
            texts,
            labels,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=stratify_labels,
        )
    except ValueError as error:
        print(f"Warning: stratified split failed ({error}), retrying without stratify.")
        X_train, X_val, y_train, y_val = train_test_split(
            texts, labels, test_size=args.test_size, random_state=args.random_state, stratify=None
        )

    label_to_id, id_to_label = build_label_mapping(labels)
    vocab = build_vocab(X_train, min_freq=args.min_freq, max_vocab_size=args.max_vocab_size)

    print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")
    print(f"Num classes: {len(label_to_id)}, Vocab size: {len(vocab)}")

    train_ds = ResearchTextDataset(X_train, y_train, vocab, label_to_id, args.max_len)
    val_ds = ResearchTextDataset(X_val, y_val, vocab, label_to_id, args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = MeanPoolingTextClassifier(
        vocab_size=len(vocab),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_classes=len(label_to_id),
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    training_log = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, y_true, y_pred = evaluate(model, val_loader, criterion, device)
        metrics = compute_metrics(y_true, y_pred)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": metrics["accuracy"],
            "val_f1_macro": metrics["f1_macro"],
        }
        training_log.append(row)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_acc={metrics['accuracy']:.4f} | val_f1_macro={metrics['f1_macro']:.4f}"
        )

    final_val_loss, y_true, y_pred = evaluate(model, val_loader, criterion, device)
    final_metrics = compute_metrics(y_true, y_pred)
    final_metrics["model"] = "PyTorch MeanPooling MLP"
    final_metrics["val_loss"] = final_val_loss

    save_stage2_outputs(args.output_dir, final_metrics, y_true, y_pred, id_to_label, training_log)

    print("\nFinal validation metrics:")
    for k in ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "f1_macro", "val_loss"]:
        print(f"- {k}: {final_metrics[k]:.4f}")

    print("\nThis is a small PyTorch neural baseline on a tiny reproducible demo dataset. "
          "It is not expected to necessarily outperform Stage 1 Logistic Regression.")
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
