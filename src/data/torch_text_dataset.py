import re
from collections import Counter

import torch
from torch.utils.data import Dataset


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_ID = 0
UNK_ID = 1


def simple_tokenize(text: str) -> list[str]:
    """Lowercase and keep only English letters/digits as tokens."""
    text = str(text).lower()
    return re.findall(r"[a-z0-9]+", text)


def build_vocab(texts, min_freq=1, max_vocab_size=None):
    """Build vocabulary from training texts only."""
    counter = Counter()
    for text in texts:
        counter.update(simple_tokenize(text))

    # Filter and sort for deterministic ids.
    items = [(tok, freq) for tok, freq in counter.items() if freq >= min_freq]
    items.sort(key=lambda x: (-x[1], x[0]))

    if max_vocab_size is not None:
        max_tokens = max(0, int(max_vocab_size) - 2)  # reserve PAD/UNK
        items = items[:max_tokens]

    word_to_id = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
    for token, _ in items:
        word_to_id[token] = len(word_to_id)

    return word_to_id


def encode_text(text, vocab):
    """Convert text to token ids using vocab with UNK fallback."""
    tokens = simple_tokenize(text)
    return [vocab.get(token, UNK_ID) for token in tokens]


def pad_or_truncate(ids, max_len):
    """Pad with PAD_ID or truncate to fixed max_len."""
    ids = ids[:max_len]
    if len(ids) < max_len:
        ids = ids + [PAD_ID] * (max_len - len(ids))
    return ids


def build_label_mapping(labels):
    """Build deterministic label mappings."""
    sorted_labels = sorted({str(label) for label in labels})
    label_to_id = {label: idx for idx, label in enumerate(sorted_labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return label_to_id, id_to_label


class ResearchTextDataset(Dataset):
    """PyTorch dataset for research query text classification."""

    def __init__(self, texts, labels, vocab, label_to_id, max_len):
        self.texts = list(texts)
        self.labels = [str(label) for label in labels]
        self.vocab = vocab
        self.label_to_id = label_to_id
        self.max_len = int(max_len)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        input_ids = encode_text(text, self.vocab)
        input_ids = pad_or_truncate(input_ids, self.max_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "label": torch.tensor(self.label_to_id[label], dtype=torch.long),
        }
