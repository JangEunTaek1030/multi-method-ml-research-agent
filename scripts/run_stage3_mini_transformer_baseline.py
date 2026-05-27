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
from src.models.mini_transformer_classifier import MiniTransformerTextClassifier
from src.training.torch_trainer import (
    compute_metrics,
    evaluate,
    save_stage3_outputs,
    set_seed,
    train_one_epoch,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Stage 3 Mini Transformer text classification baseline.")
    parser.add_argument("--data", type=Path, default=Path("data/sample/research_queries_sample.csv"))
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--max_len", type=int, default=24)
    parser.add_argument("--min_freq", type=int, default=1)
    parser.add_argument("--max_vocab_size", type=int, default=5000)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--ff_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output_dir", type=Path, default=Path("reports/stage3"))
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

    print("Stage 3B: Mini Transformer Text Classification Baseline")
    print("=" * 60)
    print(f"Using device: {device}")

    df = pd.read_csv(args.data)
    missing_cols = [c for c in [args.text_col, args.label_col] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}. Available columns: {list(df.columns)}")

    df = df[[args.text_col, args.label_col]].copy()
    df = df.dropna(subset=[args.text_col, args.label_col])
    df[args.text_col] = df[args.text_col].astype(str).str.strip()
    df = df[df[args.text_col] != ""]

    texts = df[args.text_col].tolist()
    labels = df[args.label_col].astype(str).tolist()

    label_counts = df[args.label_col].value_counts()
    stratify_labels = labels if (len(label_counts) > 1 and label_counts.min() >= 2) else None

    try:
        X_train, X_val, y_train, y_val = train_test_split(
            texts,
            labels,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=stratify_labels,
        )
    except ValueError:
        X_train, X_val, y_train, y_val = train_test_split(
            texts, labels, test_size=args.test_size, random_state=args.random_state, stratify=None
        )

    label_to_id, id_to_label = build_label_mapping(labels)
    vocab = build_vocab(X_train, min_freq=args.min_freq, max_vocab_size=args.max_vocab_size)

    train_ds = ResearchTextDataset(X_train, y_train, vocab, label_to_id, args.max_len)
    val_ds = ResearchTextDataset(X_val, y_val, vocab, label_to_id, args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = MiniTransformerTextClassifier(
        vocab_size=len(vocab),
        num_classes=len(label_to_id),
        max_len=args.max_len,
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    training_log = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, y_true, y_pred = evaluate(model, val_loader, criterion, device)
        metrics = compute_metrics(y_true, y_pred)

        training_log.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": metrics["accuracy"],
                "val_f1_macro": metrics["f1_macro"],
            }
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={metrics['accuracy']:.4f} | "
            f"val_f1_macro={metrics['f1_macro']:.4f}"
        )

    final_val_loss, y_true, y_pred = evaluate(model, val_loader, criterion, device)
    final_metrics = compute_metrics(y_true, y_pred)
    final_metrics["model"] = "MiniTransformerTextClassifier"
    final_metrics["val_loss"] = final_val_loss

    save_stage3_outputs(args.output_dir, final_metrics, y_true, y_pred, id_to_label, training_log)
    print(f"Saved Stage 3 outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
