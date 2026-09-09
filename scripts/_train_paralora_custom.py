#!/usr/bin/env python
"""ParaLoRA 自定义训练循环（绕开 transformers Trainer / accelerate GradScaler）

设计目标：
- 完全控制 optimizer + scheduler + loss + AMP（强制 fp32 cast）
- 兼容 transformers 4.57 + accelerate 1.10 + peft 0.17 + torch 2.3 的当前 env
- 复用 ``build_pt5_classifier`` + ``_to_dataset``，与原 Trainer 共享模型/数据装配
- 复现论文 §III-B-2 训练协议：30 epoch, lr=3e-4, weight_decay=0, 手动 linear warmup

用法（从 ``ParaLoRA/`` 目录执行）：
    python -m scripts._train_paralora_custom \\
        --config configs/paralora.json \\
        --train-data /path/to/train.csv \\
        --valid-data /path/to/valid.csv \\
        --train-format paraperd --valid-format paraperd \\
        --output-dir ../results/paralora_seed42 \\
        --epochs 30 --seed 42

或（pkl 格式）：
    python -m scripts._train_paralora_custom \\
        --config configs/paralora.json \\
        --train-data data/paralora/train.pkl --train-format pkl \\
        --valid-data data/paralora/valid.pkl --valid-format pkl \\
        --output-dir ../results/paralora_seed42 \\
        --epochs 30 --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from transformers import DataCollatorForTokenClassification, set_seed

from paralora.data import (
    Split,
    create_dataset,
    load_split,
    prepare_split,
)
from paralora.model import (
    build_pt5_classifier,
    save_trainable_parameters,
)


# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------

def set_all_seeds(seed: int) -> None:
    """Seed every random source touched by the trainer."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


# ----------------------------------------------------------------------------
# Optimizer & scheduler
# ----------------------------------------------------------------------------

def build_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """AdamW over only the parameters with ``requires_grad=True``."""
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(
        params,
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=weight_decay,
    )


def linear_warmup_decay_schedule(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup (0 → lr) followed by linear decay (lr → 0)."""
    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return float(step) / max(1, num_warmup_steps)
        return max(
            0.0,
            float(num_total_steps - step) / max(1, num_total_steps - num_warmup_steps),
        )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def cosine_warmup_schedule(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_total_steps: int,
    min_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup (0 → lr) followed by cosine decay (lr → lr*min_ratio).

    Cosine annealing gives a smoother late-training trajectory than linear
    decay and matches the protocol used by most ProtT5 LoRA fine-tunes
    (e.g. Bio-ELECTRA, ESM-LoRA).  ``min_ratio`` lets the LR settle at a
    floor instead of going all the way to zero (helps when early stopping
    fires late).
    """
    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return float(step) / max(1, num_warmup_steps)
        progress = float(step - num_warmup_steps) / max(
            1, num_total_steps - num_warmup_steps
        )
        progress = min(1.0, max(0.0, progress))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(min_ratio + (1.0 - min_ratio) * cosine_decay)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_scheduler(
    name: str,
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_total_steps: int,
    min_ratio: float = 0.0,
):
    name = name.lower()
    if name in {"linear", "linear_warmup_decay"}:
        return linear_warmup_decay_schedule(optimizer, num_warmup_steps, num_total_steps)
    if name in {"cosine", "cosine_warmup"}:
        return cosine_warmup_schedule(optimizer, num_warmup_steps, num_total_steps, min_ratio)
    raise ValueError(f"Unknown scheduler: {name!r}. Use 'linear' or 'cosine'.")


# ----------------------------------------------------------------------------
# Loss helper (matches paralora.model forward)
# ----------------------------------------------------------------------------

def compute_loss(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    loss_weights: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy on residues where ``labels != -100``.

    Always computed in fp32 to avoid the dtype mismatch that triggered
    "Attempting to unscale FP16 gradients" in the accelerate Trainer path.
    """
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    # Mask-aware attention: when cdr_masked_train is True the collator pads
    # non-CDR positions to ``attention_mask = 0`` via the masks in the dataset.
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        loss_weights=loss_weights,
    )
    loss = out.loss
    if loss is None:
        # Fallback: compute manually on full logits (rare; mostly when collator
        # accidentally strips all ``-100`` labels).
        active_logits = out.logits.float().view(-1, model.num_labels)
        active_labels = labels.view(-1).long()
        valid_mask = active_labels != -100
        loss = F.cross_entropy(
            active_logits[valid_mask],
            active_labels[valid_mask],
            weight=loss_weights.to(active_logits.device),
        )
    return loss.float()


# ----------------------------------------------------------------------------
# Metric helpers
# ----------------------------------------------------------------------------

@torch.no_grad()
def predict_residues(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run model over ``dataloader`` and collect (probs, labels) per residue.

    Only residues whose ``labels != -100`` are returned (i.e. those the
    DataCollatorForTokenClassification kept in the loss mask).  The
    dataloader is expected to expose ``attention_mask``; non-attended
    positions are filtered out by reading ``attention_mask`` and skipping
    those tokens.
    """
    model.eval()
    probs_chunks: List[np.ndarray] = []
    labels_chunks: List[np.ndarray] = []
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits.float()  # (B, T, 2)
        # Softmax probability for class 1 (paratope)
        p1 = torch.softmax(logits, dim=-1)[..., 1]
        # Keep tokens where attention_mask == 1 AND label != -100
        keep = (attention_mask == 1) & (labels != -100)
        probs_chunks.append(p1[keep].detach().cpu().numpy())
        labels_chunks.append(labels[keep].detach().cpu().numpy().astype(np.int64))
    return np.concatenate(probs_chunks), np.concatenate(labels_chunks)


def compute_metrics(probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Six metrics reported in the manuscript (Table VI)."""
    pred = (probs >= 0.5).astype(np.int64)
    return {
        "auc_roc": float(roc_auc_score(labels, probs)),
        "auc_pr": float(average_precision_score(labels, probs)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, pred)),
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "recall": float(recall_score(labels, pred, zero_division=0)),
    }


def sweep_threshold_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    thresholds: Optional[List[float]] = None,
) -> Tuple[Dict[str, float], float]:
    """Sweep a list of thresholds and return (metrics at best-F1 threshold, threshold).

    Threshold selection maximises F1; ties are broken by MCC. AUC-ROC and AUC-PR
    are threshold-free and therefore constant across the sweep.
    """
    if thresholds is None:
        thresholds = [round(0.02 * k, 2) for k in range(1, 26)]  # 0.02..0.50
    auc_roc = float(roc_auc_score(labels, probs))
    auc_pr = float(average_precision_score(labels, probs))
    best = {"f1": -1.0, "mcc": -1.0}
    best_t = 0.5
    for t in thresholds:
        pred = (probs >= t).astype(np.int64)
        f1 = float(f1_score(labels, pred, zero_division=0))
        mcc = float(matthews_corrcoef(labels, pred)) if pred.sum() > 0 and (1 - pred).sum() > 0 else 0.0
        prec = float(precision_score(labels, pred, zero_division=0))
        rec = float(recall_score(labels, pred, zero_division=0))
        if f1 > best["f1"] + 1e-6 or (
            abs(f1 - best["f1"]) < 1e-6 and mcc > best["mcc"]
        ):
            best = {"f1": f1, "mcc": mcc, "precision": prec, "recall": rec}
            best_t = t
    return (
        {
            "auc_roc": auc_roc,
            "auc_pr": auc_pr,
            "f1": best["f1"],
            "mcc": best["mcc"],
            "precision": best["precision"],
            "recall": best["recall"],
        },
        float(best_t),
    )


# ----------------------------------------------------------------------------
# Train / eval loops
# ----------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    loss_weights: torch.Tensor,
    device: torch.device,
    grad_clip: float,
    epoch: int,
    log_every: int,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    t0 = time.time()
    for step, batch in enumerate(dataloader, start=1):
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = compute_loss(model, batch, loss_weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=grad_clip,
        )
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        n_batches += 1
        if step % log_every == 0:
            cur_lr = optimizer.param_groups[0]["lr"]
            print(
                f"  epoch {epoch} step {step}/{len(dataloader)} "
                f"loss={loss.item():.4f} lr={cur_lr:.2e} "
                f"elapsed={time.time()-t0:.1f}s",
                flush=True,
            )
    return total_loss / max(1, n_batches)


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ParaLoRA custom training loop")
    parser.add_argument("--config", default="configs/paralora.json")
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--valid-data", required=True)
    parser.add_argument("--test-data", default=None,
                        help="Optional held-out test split (Parapred test set)")
    parser.add_argument("--train-format", choices=["csv", "paraperd", "pkl"], default=None)
    parser.add_argument("--valid-format", choices=["csv", "paraperd", "pkl"], default=None)
    parser.add_argument("--output-dir", default="../results/paralora_seed42")
    parser.add_argument("--checkpoint-out", default=None)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override config training.epochs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--scheduler", choices=["linear", "cosine"], default=None,
                        help="LR schedule: linear (warmup + linear decay) or cosine "
                             "(warmup + cosine annealing to min_lr_ratio).")
    parser.add_argument("--min-lr-ratio", type=float, default=0.0,
                        help="Floor of cosine schedule as a fraction of peak LR.")
    parser.add_argument("--early-stopping-patience", type=int, default=0,
                        help="Stop training if val AUC-ROC does not improve for N "
                             "epochs. 0 disables early stopping.")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-probs", action="store_true",
                        help="Save raw probs/labels for later analysis")
    return parser.parse_args()


def load_split_with_format(path: str, fmt: str | None) -> Split:
    if fmt == "pkl":
        return load_split(path, format="pkl")
    if fmt == "csv":
        return load_split(path, format="csv")
    if fmt == "paraperd":
        return load_split(path, format="paraperd")
    raise ValueError(f"Unknown split format: {fmt}")


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Force offline mode for HF hub (ProtT5 is local but peft/transformers
    # still probe the hub in some code paths).
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # ---------------- Load config ----------------
    with open(args.config) as handle:
        config = json.load(handle)

    # Overrides
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.lr is not None:
        config["training"]["lr"] = args.lr
    if args.weight_decay is not None:
        config["training"]["weight_decay"] = args.weight_decay
    if args.warmup_ratio is not None:
        config["training"]["warmup_ratio"] = args.warmup_ratio
    if args.scheduler is not None:
        config["training"]["scheduler"] = args.scheduler
    if "min_lr_ratio" not in config["training"]:
        config["training"]["min_lr_ratio"] = float(args.min_lr_ratio)
    if args.max_length is not None:
        config["data"]["max_length"] = args.max_length
    if args.batch_size is not None:
        config["training"]["batch"] = args.batch_size

    seed = args.seed
    config["training"]["seed"] = seed
    set_all_seeds(seed)
    print(f"Seed: {seed}", flush=True)
    print(f"Config:\n{json.dumps(config, indent=2)}", flush=True)

    # Force fp32 across the entire backbone. ``half_precision=True`` in the
    # config causes the released layer-norm to emit fp32 hidden states while
    # the surrounding ``nn.Linear`` modules stay in fp16; this combination
    # triggers a dtype mismatch inside ``SelfAttention`` and breaks training
    # on the current transformers/peft/accelerate stack.
    if config.get("half_precision", False):
        print(
            "[info] half_precision=True in config; overriding to fp32 to avoid "
            "layer_norm (fp32) vs Linear (fp16) dtype mismatch.",
            flush=True,
        )
        config["half_precision"] = False

    # ---------------- Build model ----------------
    model, tokenizer, lora_cfg, trainable = build_pt5_classifier(config=config)
    model = model.to(device)
    print(
        f"Trainable params: {trainable:,}; "
        f"lora: rank={lora_cfg.lora_rank}, alpha={lora_cfg.lora_alpha}, "
        f"layers={lora_cfg.lora_layers}",
        flush=True,
    )
    loss_weights = model.loss_weights.to(device)
    print(f"Loss weights [neg, pos] = {loss_weights.cpu().tolist()}", flush=True)

    # Cast the entire model (encoder, layer_norms, classifier head, LoRA) to fp32.
    model = model.float()

    # ---------------- Build datasets ----------------
    train_raw = load_split_with_format(args.train_data, args.train_format)
    valid_raw = load_split_with_format(args.valid_data, args.valid_format)
    print(
        f"Train: {len(train_raw.sequences)} seqs | "
        f"Valid: {len(valid_raw.sequences)} seqs",
        flush=True,
    )
    train_split = prepare_split(train_raw, cdr_mask=bool(config["data"].get("cdr_masked_train", True)))
    valid_split = prepare_split(valid_raw, cdr_mask=bool(config["data"].get("cdr_masked_eval", False)))

    train_set = create_dataset(
        tokenizer,
        train_split.sequences,
        train_split.labels,
        max_length=int(config["data"]["max_length"]),
    )
    valid_set = create_dataset(
        tokenizer,
        valid_split.sequences,
        valid_split.labels,
        max_length=int(config["data"]["max_length"]),
    )
    collator = DataCollatorForTokenClassification(tokenizer)
    batch_size = int(config["training"]["batch"])
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    # ---------------- Optimizer + scheduler ----------------
    epochs = int(config["training"]["epochs"])
    accum = max(1, int(args.grad_accum))
    steps_per_epoch = math.ceil(len(train_loader) / accum)
    total_steps = steps_per_epoch * epochs
    warmup_ratio = float(config["training"].get("warmup_ratio", 0.0))
    warmup_steps = int(total_steps * warmup_ratio)
    optimizer = build_optimizer(
        model,
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )
    scheduler_name = config["training"].get("scheduler", "cosine")
    scheduler = build_scheduler(
        name=scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_total_steps=total_steps,
        min_ratio=float(config["training"].get("min_lr_ratio", 0.0)),
    )
    print(
        f"Optimizer: AdamW(lr={config['training']['lr']}, "
        f"wd={config['training'].get('weight_decay', 0.0)}); "
        f"scheduler={scheduler_name}; "
        f"total_steps={total_steps} warmup={warmup_steps}; "
        f"epochs={epochs}; batch={batch_size}; accum={accum}; "
        f"early_stopping_patience={args.early_stopping_patience}",
        flush=True,
    )

    # ---------------- Train ----------------
    history: List[Dict[str, float]] = []
    best_val = {"auc_roc": -1.0}
    best_epoch = 0
    best_threshold = 0.5
    best_path = os.path.join(args.output_dir, "best_trainable_params.pt")
    epochs_no_improve = 0
    stopped_early = False
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_weights=loss_weights,
            device=device,
            grad_clip=args.grad_clip,
            epoch=epoch,
            log_every=args.log_every,
        )
        train_time = time.time() - t0

        # Evaluate on validation set
        t1 = time.time()
        probs, labels = predict_residues(model, valid_loader, device)
        eval_time = time.time() - t1
        val_metrics, val_threshold = sweep_threshold_metrics(probs, labels)

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_auc_roc": val_metrics["auc_roc"],
            "val_auc_pr": val_metrics["auc_pr"],
            "val_f1": val_metrics["f1"],
            "val_mcc": val_metrics["mcc"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_threshold": float(val_threshold),
            "train_time_s": float(train_time),
            "eval_time_s": float(eval_time),
            "lr": float(optimizer.param_groups[0]["lr"]),
        })
        print(
            f"[epoch {epoch}/{epochs}] "
            f"train_loss={train_loss:.4f} | "
            f"val AUC-ROC={val_metrics['auc_roc']:.4f} "
            f"AUC-PR={val_metrics['auc_pr']:.4f} "
            f"F1={val_metrics['f1']:.4f} "
            f"MCC={val_metrics['mcc']:.4f} "
            f"P={val_metrics['precision']:.4f} "
            f"R={val_metrics['recall']:.4f} "
            f"(thr={val_threshold:.2f}) "
            f"| train={train_time:.1f}s eval={eval_time:.1f}s",
            flush=True,
        )

        if val_metrics["auc_roc"] > best_val["auc_roc"]:
            best_val = val_metrics
            best_epoch = epoch
            best_threshold = val_threshold
            save_trainable_parameters(model, best_path)
            with open(os.path.join(args.output_dir, "best_val_probs.npz"), "wb") as handle:
                np.savez(handle, probs=probs, labels=labels)
            print(f"  -> new best val AUC-ROC={best_val['auc_roc']:.4f}; saved {best_path}", flush=True)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping: stop if val AUC-ROC has not improved for N epochs.
        if args.early_stopping_patience > 0 and epochs_no_improve >= args.early_stopping_patience:
            print(
                f"[early stop] no val AUC-ROC improvement for "
                f"{epochs_no_improve} epochs (patience={args.early_stopping_patience}); "
                f"stopping at epoch {epoch}.",
                flush=True,
            )
            stopped_early = True
            break

    # ---------------- Test set evaluation ----------------
    test_metrics = None
    test_threshold = 0.5
    if args.test_data is not None:
        test_raw = load_split_with_format(args.test_data, args.valid_format)
        # Use the best checkpoint
        best_state = torch.load(best_path, map_location="cpu")
        for name, param in model.named_parameters():
            if name in best_state:
                param.data.copy_(best_state[name].to(param.dtype))
        model = model.to(device)
        test_set = create_dataset(
            tokenizer,
            test_raw.sequences,
            test_raw.labels,
            max_length=int(config["data"]["max_length"]),
        )
        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
        )
        probs, labels = predict_residues(model, test_loader, device)

        # 论文口径：阈值只在 val 上调（best_threshold），应用到 test。
        test_metrics_at_val_thr = metrics_at_threshold(probs, labels, best_threshold)
        # test 自身 sweep 仅作泄漏参照，不用于论文报告。
        test_sweep_metrics, test_sweep_threshold = sweep_threshold_metrics(probs, labels)
        np.savez(
            os.path.join(args.output_dir, "test_probs.npz"),
            probs=probs,
            labels=labels,
        )
        print(
            f"[TEST @ val thr={best_threshold:.2f}] AUC-ROC={test_metrics_at_val_thr['auc_roc']:.4f} "
            f"AUC-PR={test_metrics_at_val_thr['auc_pr']:.4f} "
            f"F1={test_metrics_at_val_thr['f1']:.4f} "
            f"MCC={test_metrics_at_val_thr['mcc']:.4f} "
            f"P={test_metrics_at_val_thr['precision']:.4f} "
            f"R={test_metrics_at_val_thr['recall']:.4f}",
            flush=True,
        )
        print(
            f"[TEST sweep (leaky reference) thr={test_sweep_threshold:.2f}] "
            f"F1={test_sweep_metrics['f1']:.4f} MCC={test_sweep_metrics['mcc']:.4f}",
            flush=True,
        )

    # ---------------- Save summary ----------------
    summary = {
        "config": config,
        "args": vars(args),
        "best_val": best_val,
        "best_epoch": int(best_epoch),
        "best_threshold": float(best_threshold),
        "stopped_early": bool(stopped_early),
        "test_metrics": test_metrics,
        "test_threshold": float(test_threshold),
        "history": history,
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2, default=float)
    print(f"Saved summary → {summary_path}", flush=True)

    # Save final trainable params too (in case user wants both)
    final_path = os.path.join(args.output_dir, "final_trainable_params.pt")
    save_trainable_parameters(model, final_path)


if __name__ == "__main__":
    main()
