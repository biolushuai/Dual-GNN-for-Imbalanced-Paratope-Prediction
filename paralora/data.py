"""Sequence dataset construction for ParaLoRA.

The released notebook reads the data from a CSV (``sequence / label / mask``),
batches sequences with ``" "`` between residues for the ProtT5 tokenizer,
and uses ``DataCollatorForTokenClassification`` to align labels with the
sub-token output. ``create_dataset`` reproduces this workflow in a
re-usable function that supports both CSV and pickle inputs.

* ``mask`` is the CDR indicator (``1`` for residues inside a CDR). It is
  used only during training (per Section III-B-2 of the manuscript) to
  bias the loss toward paratope predictions.
* For evaluation, the mask is set to all-ones so every residue
  contributes to the metric.
"""

from __future__ import annotations

import ast
import json
import os
import pickle
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset

NON_STANDARD_AA = ("O", "B", "U", "Z")


def _parse_int_list(cell: str) -> List[int]:
    """Parse a list-of-ints cell that can be either a Python repr (``"[0, 1, 0]"``)
    produced by ``pandas.to_csv`` or a compact digit string (``"010"``).
    Falls back to per-character conversion when neither pattern matches.
    """
    if not isinstance(cell, str) or not cell:
        return [int(c) for c in str(cell)]
    head = cell[0]
    if head in "[{":
        try:
            parsed = ast.literal_eval(cell)
            if isinstance(parsed, (list, tuple)):
                return [int(v) for v in parsed]
        except (ValueError, SyntaxError):
            pass
    if all(c in "01" for c in cell):
        return [int(c) for c in cell]
    # Last-resort: split by comma
    return [int(c) for c in cell.split(",")]


@dataclass
class Split:
    sequences: List[str]
    labels: List[List[int]]
    masks: List[List[int]]


def _sanitise_sequence(seq: str) -> str:
    """Replace non-standard residues with ``X`` and add spaces between AAs."""
    seq = re.sub("|".join(NON_STANDARD_AA), "X", seq)
    return " ".join(list(seq))


def _from_csv(path: str) -> Split:
    df = pd.read_csv(path, dtype={"label": object, "mask": object})
    sequences = df["sequence"].astype(str).tolist()
    labels = [_parse_int_list(l) for l in df["label"].tolist()]
    masks = [_parse_int_list(m) for m in df["mask"].tolist()]
    return Split(sequences, labels, masks)


def _from_paraperd_csv(path: str) -> Split:
    df = pd.read_csv(path, dtype={"paratope": str})
    sequences = df["sequence"].astype(str).tolist()
    labels = [[int(c) for c in str(l)] for l in df["paratope"].tolist()]
    cdrs = [json.loads(c) for c in df["cdrs"].tolist()]
    masks: List[List[int]] = []
    for label, cdr in zip(labels, cdrs):
        mask = [0] * len(label)
        for idx in cdr:
            mask[int(idx)] = 1
        masks.append(mask)
    return Split(sequences, labels, masks)


def _from_pickle(path: str, sequence_key: str = "ab_sequence", label_key: str = "ab_label") -> Split:
    with open(path, "rb") as handle:
        raw = pickle.load(handle)
    sequences: List[str] = []
    labels: List[List[int]] = []
    masks: List[List[int]] = []
    for entry in raw:
        sequences.append(entry[sequence_key])
        labels.append(entry[label_key])
        masks.append([1] * len(entry[sequence_key]))
    return Split(sequences, labels, masks)


def load_split(
    path: str,
    format: Optional[str] = None,
    sequence_key: str = "ab_sequence",
    label_key: str = "ab_label",
) -> Split:
    """Load sequences/labels/masks from a CSV, a paraperd CSV or a pickle."""
    format = format or os.path.splitext(path)[1].lstrip(".")
    if format == "csv":
        return _from_csv(path)
    if format == "paraperd":
        return _from_paraperd_csv(path)
    if format == "pkl":
        return _from_pickle(path, sequence_key, label_key)
    raise ValueError(f"Unsupported split format: {format}")


def prepare_split(
    split: Split,
    cdr_mask: bool,
) -> Split:
    """Apply the CDR-mask policy described in Section III-B-2.

    During training the mask is forced to all-zeros so the loss ignores
    non-CDR positions; during evaluation the mask is forced to all-ones
    so every residue is scored. CDR-only paratope training improves
    robustness because non-CDR paratope residues are rare and noisy.
    """
    if cdr_mask:
        masks = [[1 if m == 1 else 0 for m in seq_mask] for seq_mask in split.masks]
    else:
        masks = [[1] * len(seq_mask) for seq_mask in split.masks]
    return Split(split.sequences, split.labels, masks)


def create_dataset(
    tokenizer,
    sequences: Sequence[str],
    labels: Optional[Sequence[Sequence[int]]] = None,
    max_length: int = 1024,
) -> Dataset:
    """Tokenise sequences into a :class:`datasets.Dataset`.

    Labels are trimmed to ``max_length - 1`` so the data collator can
    append the special token produced by
    :class:`transformers.DataCollatorForTokenClassification` without
    indexing past the end of the label array.
    """
    sanitised = [_sanitise_sequence(seq) for seq in sequences]
    tokenized = tokenizer(sanitised, max_length=max_length, padding=False, truncation=True)
    dataset = Dataset.from_dict(tokenized)

    if labels is not None:
        trimmed = [list(l[: max_length - 1]) for l in labels]
        dataset = dataset.add_column("labels", trimmed)

    return dataset


def label_stats(split: Split) -> Dict[str, float]:
    """Return simple statistics of a split (useful for notebooks/CLI)."""
    num_residues = sum(len(seq) for seq in split.sequences)
    num_positives = sum(int(c) for seq in split.labels for c in seq)
    return {
        "num_sequences": len(split.sequences),
        "num_residues": num_residues,
        "num_positives": num_positives,
        "positive_ratio": num_positives / max(num_residues, 1),
    }