import torch
import torch.nn as nn


class MeanPoolingTextClassifier(nn.Module):
    """Embedding + masked mean pooling + small MLP classifier."""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, input_ids):
        # input_ids: (batch_size, max_len)
        embeddings = self.embedding(input_ids)  # (batch_size, max_len, embedding_dim)

        # Mask PAD tokens (id=0) so PAD does not contribute to average.
        mask = (input_ids != 0).unsqueeze(-1).float()  # (batch_size, max_len, 1)
        masked_embeddings = embeddings * mask

        sum_embeddings = masked_embeddings.sum(dim=1)
        valid_counts = mask.sum(dim=1).clamp(min=1.0)
        mean_pooled = sum_embeddings / valid_counts

        features = self.dropout(mean_pooled)
        logits = self.classifier(features)
        return logits
