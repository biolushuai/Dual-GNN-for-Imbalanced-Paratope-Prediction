from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorForTokenClassification

from .data import create_dataset, load_split
from .model import build_pt5_classifier, load_trainable_parameters


@dataclass
class EvalResult:
    metrics: Dict[str, float]
    scores: np.ndarray
    labels: np.ndarray

    def to_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "metrics": self.metrics,
            "num_residues": int(self.labels.size),
        }
        with open(path, "w") as handle:
            json.dump(payload, handle, indent=2)


def _flatten_valid(
    logits_or_probs: Sequence[Sequence[float]],
    padded_labels: Sequence[Sequence[int]],
    score_dim: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Drop padded tokens and convert to flat numpy arrays."""
    flat_scores: List[List[float]] = []
    flat_labels: List[int] = []
    for batch_scores, batch_labels in zip(logits_or_probs, padded_labels):
        for score, label in zip(batch_scores, batch_labels):
            if label == -100:
                continue
            if score_dim > 1:
                flat_scores.append(score)
            else:
                flat_scores.append([score])
            flat_labels.append(int(label))
    if not flat_labels:
        return np.zeros((0, score_dim)), np.zeros((0,), dtype=int)
    return np.asarray(flat_scores), np.asarray(flat_labels, dtype=int)


def _select_scores(scores: np.ndarray, num_labels: int) -> np.ndarray:
    """Pick the positive-class column from a (N, num_labels) array."""
    if num_labels == 2:
        return scores[:, 1]
    return scores[:, -1]


def _compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Mirrors the metric set reported in Section III-C of the manuscript."""
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        roc_auc_score,
    )

    y_true = y_true.astype(int)
    y_pred = (y_score >= threshold).astype(int)

    metrics: Dict[str, float] = {
        "auc_roc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan"),
        "auc_pr": float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan"),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan"),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "threshold": float(threshold),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) else float("nan")
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) else float("nan")
    return metrics


def evaluate_paratope(
    config: Dict,
    checkpoint_path: Optional[str],
    split_path: str,
    split_format: Optional[str] = None,
    batch_size: int = 16,
    threshold: float = 0.5,
    device: Optional[str] = None,
    max_length: Optional[int] = None,
    output_path: Optional[str] = None,
) -> EvalResult:
    """Score one split with the ParaLoRA model and return the metrics."""
    split = load_split(split_path, format=split_format)

    model, tokenizer, _, _ = build_pt5_classifier(config=config)
    if checkpoint_path is not None:
        load_trainable_parameters(model, checkpoint_path)

    if max_length is None:
        max_length = int(config["data"]["max_length"])
    dataset = create_dataset(tokenizer, split.sequences, split.labels, max_length=max_length)

    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(target_device)
    model.eval()

    dataset = dataset.with_format("torch", device=target_device)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DataCollatorForTokenClassification(tokenizer),
    )

    num_labels = int(config["num_labels"])
    raw_scores: List[List[List[float]]] = []
    raw_labels: List[List[int]] = []
    autocast_ctx = torch.cuda.amp.autocast if target_device.startswith("cuda") else torch.cpu.amp.autocast

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(target_device)
            attention_mask = batch["attention_mask"].to(target_device)
            with autocast_ctx():
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits.float(), dim=-1).cpu().tolist()
            raw_scores.extend(probs)
            raw_labels.extend(batch["labels"].cpu().tolist())

    scores, labels = _flatten_valid(raw_scores, raw_labels, score_dim=num_labels)
    positive_scores = _select_scores(scores, num_labels)
    metrics = _compute_metrics(labels, positive_scores, threshold=threshold)
    result = EvalResult(metrics=metrics, scores=positive_scores, labels=labels)
    if output_path is not None:
        result.to_json(output_path)
        np.savez(os.path.join(os.path.dirname(output_path) or ".", "predictions.npz"),
                  scores=positive_scores, labels=labels)
    return result


def extract_embeddings(
    config: Dict,
    checkpoint_path: Optional[str],
    split_path: str,
    split_format: Optional[str] = None,
    batch_size: int = 8,
    device: Optional[str] = None,
    max_length: Optional[int] = None,
    layer: int = -1,
) -> np.ndarray:
    """Return per-residue embeddings (sequence_length, hidden_size).

    The embeddings come from the encoder hidden state at ``layer`` (default:
    last layer) and are extracted at the *unmasked* positions (the data
    collator pads to the longest item in the batch).
    """
    split = load_split(split_path, format=split_format)

    model, tokenizer, _, _ = build_pt5_classifier(config=config)
    if checkpoint_path is not None:
        load_trainable_parameters(model, checkpoint_path)

    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(target_device)
    model.eval()

    if max_length is None:
        max_length = int(config["data"]["max_length"])

    dataset = create_dataset(tokenizer, split.sequences, labels=None, max_length=max_length)
    dataset = dataset.with_format("torch", device=target_device)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DataCollatorForTokenClassification(tokenizer),
    )

    outputs: List[np.ndarray] = []
    autocast_ctx = torch.cuda.amp.autocast if target_device.startswith("cuda") else torch.cpu.amp.autocast

    with torch.no_grad():
        for batch in tqdm(loader, desc="Embedding"):
            input_ids = batch["input_ids"].to(target_device)
            attention_mask = batch["attention_mask"].to(target_device)
            with autocast_ctx():
                encoder_out = model.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
            hidden = encoder_out.hidden_states[layer].float().cpu().numpy()
            mask = attention_mask.cpu().numpy()
            for emb, m in zip(hidden, mask):
                outputs.append(emb[: int(m.sum())])

    return outputs


def main(argv: Optional[list] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a ParaLoRA checkpoint")
    parser.add_argument("--config", default="configs/paralora.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--format", choices=["csv", "paraperd", "pkl"], default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="./runs/paralora/eval.json")
    parser.add_argument("--mode", choices=["metrics", "embeddings"], default="metrics")
    args = parser.parse_args(argv)

    with open(args.config) as handle:
        config = json.load(handle)

    if args.mode == "metrics":
        result = evaluate_paratope(
            config=config,
            checkpoint_path=args.checkpoint,
            split_path=args.data,
            split_format=args.format,
            batch_size=args.batch_size,
            threshold=args.threshold,
            device=args.device,
            max_length=args.max_length,
            output_path=args.output,
        )
        print(json.dumps(result.metrics, indent=2))
    else:
        embeddings = extract_embeddings(
            config=config,
            checkpoint_path=args.checkpoint,
            split_path=args.data,
            split_format=args.format,
            batch_size=args.batch_size,
            device=args.device,
            max_length=args.max_length,
        )
        out_dir = os.path.dirname(args.output) or "."
        os.makedirs(out_dir, exist_ok=True)
        np.savez(args.output.replace(".json", ".npz"),
                 names=np.array(list(range(len(embeddings)))),
                 lengths=np.array([e.shape[0] for e in embeddings]),
                 embeddings=np.concatenate(embeddings, axis=0))
        print(f"Saved embeddings for {len(embeddings)} sequences to {args.output.replace('.json', '.npz')}")


if __name__ == "__main__":
    main()
