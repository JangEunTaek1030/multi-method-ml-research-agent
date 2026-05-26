import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(dataloader))


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    avg_loss = total_loss / max(1, len(dataloader))
    return avg_loss, y_true, y_pred


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def save_stage2_outputs(output_dir, metrics, y_true, y_pred, id_to_label, training_log):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame([metrics])
    results_df.to_csv(output_dir / "model_results.csv", index=False, encoding="utf-8-sig")

    target_names = [id_to_label[i] for i in sorted(id_to_label.keys())]
    report_text = classification_report(
        y_true,
        y_pred,
        labels=sorted(id_to_label.keys()),
        target_names=target_names,
        zero_division=0,
    )
    (output_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")

    pd.DataFrame(training_log).to_csv(output_dir / "training_log.csv", index=False, encoding="utf-8-sig")

    summary_text = (
        "Stage 2 PyTorch Baseline Summary\n"
        "================================\n\n"
        "Task:\n"
        "Research task classification using the same sample dataset as Stage 1.\n\n"
        "Model:\n"
        "PyTorch MeanPoolingTextClassifier\n"
        "Embedding + masked mean pooling + MLP classifier\n\n"
        "Dataset:\n"
        "data/sample/research_queries_sample.csv\n"
        "70 rows, 7 balanced labels\n"
        "text,label columns\n\n"
        "Metrics:\n"
        f"Accuracy: {metrics['accuracy']:.4f}\n"
        f"F1 weighted: {metrics['f1_weighted']:.4f}\n"
        f"F1 macro: {metrics['f1_macro']:.4f}\n\n"
        "Outputs:\n"
        "reports/stage2/model_results.csv\n"
        "reports/stage2/classification_report.txt\n"
        "reports/stage2/training_log.csv\n"
        "reports/stage2/stage2_summary.txt\n\n"
        "Note:\n"
        "This is a small PyTorch neural baseline on a tiny reproducible demo dataset.\n"
        "It is intended for learning and comparison, not as a production benchmark.\n"
        "It is not expected to necessarily outperform Stage 1 Logistic Regression.\n"
    )
    (output_dir / "stage2_summary.txt").write_text(summary_text, encoding="utf-8")

