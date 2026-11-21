"""Assistant-only token-normalized supervised learning primitives."""

import math
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class Rows(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


class MaskedCollator:
    def __init__(self, tokenizer):
        self.pad = tokenizer.pad_token_id

    def __call__(self, features):
        length = math.ceil(max(len(f["input_ids"]) for f in features) / 8) * 8
        batch = {k: [] for k in ("input_ids", "attention_mask", "labels")}
        for f in features:
            extra = length - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [self.pad] * extra)
            batch["attention_mask"].append([1] * len(f["input_ids"]) + [0] * extra)
            batch["labels"].append(f["labels"] + [-100] * extra)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


def supervised_loss(outputs, labels, num_items_in_batch=None):
    # Sum over actual target tokens, including im_end, then normalize across the
    # entire accumulation window supplied by Trainer (not per-microbatch means).
    targets = labels[:, 1:].contiguous()
    logits = outputs.logits[:, :-1, :].contiguous().float()
    loss = F.cross_entropy(
        logits.view(-1, logits.shape[-1]),
        targets.view(-1),
        ignore_index=-100,
        reduction="sum",
    )
    denom = targets.ne(-100).sum() if num_items_in_batch is None else num_items_in_batch
    return loss / denom
