import argparse
import json
import math
import os
import re
import time
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForTokenClassification

from paralora.data import create_dataset, load_split, prepare_split
from paralora.model import build_pt5_classifier, save_trainable_parameters

from scripts._train_paralora_custom import (
    build_optimizer,
    build_scheduler,
    compute_loss,
    compute_metrics,
    predict_residues,
    set_all_seeds,
    sweep_threshold_metrics,
    train_one_epoch,
)
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def metrics_at_threshold(probs: np.ndarray, labels: np.ndarray, thr: float) -> Dict[str, float]:
    """Compute 6 metrics at a fixed threshold (paper-correct protocol)."""
    pred = (probs >= thr).astype(np.int64)
    return {
        "auc_roc": float(roc_auc_score(labels, probs)),
        "auc_pr": float(average_precision_score(labels, probs)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, pred)) if (pred.sum() > 0 and (1 - pred).sum() > 0) else 0.0,
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "recall": float(recall_score(labels, pred, zero_division=0)),
    }


def compute_balanced_weights(train_split) -> List[float]:
    """balanced class weights within CDR±2 region -> [w_neg, w_pos]."""
    pos = 0
    tot = 0
    for lab, msk in zip(train_split.labels, train_split.masks):
        for y, m in zip(lab, msk):
            if m == 1:
                tot += 1
                if y == 1:
                    pos += 1
    p = pos / max(1, tot)
    return [p, 1.0 - p]


def select_loss_weights(mode: str, train_split) -> List[float]:
    if mode == "paper":
        return [1.0, 0.1]
    if mode == "none":
        return [1.0, 1.0]
    if mode == "balanced":
        return compute_balanced_weights(train_split)
    raise ValueError(f"unknown class-weight mode: {mode}")


TRAINABLE_PRESETS = {
    "ln_lora": ".*layer_norm.*|.*lora_[ab].*",
    "lora_only": ".*lora_[ab].*",
}


def build_lora_config(layers: str, rank: int, init_scale: float, alpha: int,
                      trainable: str = "ln_lora") -> Dict:
    # `lora_layers` 在 lora.py 中以 `re.fullmatch(config.lora_layers, c_name)` 匹配子模块名,
    # c_name 取值 'q'/'k'/'v'/'o' 之一, 因此 multi-char 串必须用竖线分隔, 否则全部失配 → 0 LoRA 注入.
    if "|" not in layers and not any(ch in layers for ch in r"^$.+*?()[]{}\\"):
        layers = "|".join(layers)
    if trainable not in TRAINABLE_PRESETS:
        raise ValueError(
            f"unknown trainable preset: {trainable}; "
            f"choose from {sorted(TRAINABLE_PRESETS)}")
    return {
        "lora_rank": int(rank),
        "lora_init_scale": float(init_scale),
        "lora_alpha": int(alpha),
        "lora_modules": ".*SelfAttention|.*EncDecAttention",
        "lora_layers": layers,
        "trainable_param_names": TRAINABLE_PRESETS[trainable],
        "lora_scaling_rank": 0,
    }


def unfreeze_classifier(model):
    """分类头可训练（论文：task-specific linear classifier）。"""
    n = 0
    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True
            n += 1
    return n


def replace_head(model, kind: str, dropout: float):
    """可选项：把线性读头换成含 dropout 的线性 / MLP。"""
    hidden = model.config.hidden_size
    if kind == "linear":
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, model.num_labels),
        )
    elif kind == "mlp":
        model.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(256, model.num_labels),
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ParaLoRA fine-tuning (paper-faithful + free hyperparameters)")
    p.add_argument("--base-config", default="configs/paralora.json")
    p.add_argument("--train-data", required=True)
    p.add_argument("--valid-data", required=True)
    p.add_argument("--test-data", default=None)
    p.add_argument("--train-format", default="csv")
    p.add_argument("--valid-format", default="csv")
    p.add_argument("--test-format", default="csv")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    # LoRA
    p.add_argument("--lora-layers", default="q|k|v|o")
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-init-scale", type=float, default=0.02)
    p.add_argument("--lora-alpha", type=int, default=None, help="default 2*rank")
    p.add_argument("--trainable", default="ln_lora", choices=sorted(TRAINABLE_PRESETS),
                   help="可训练参数集：ln_lora=LayerNorm+LoRA(默认); "
                        "lora_only=仅 LoRA（P0-1 纯 LoRA 分离）")
    # head
    p.add_argument("--head", choices=["linear", "mlp"], default="linear")
    p.add_argument("--head-dropout", type=float, default=0.1)
    # loss
    p.add_argument("--class-weight", choices=["paper", "balanced", "none"], default="balanced")
    # eval / selection
    p.add_argument("--eval-scope", choices=["all", "cdr"], default="all",
                   help="eval on all residues (default) or CDR±2 only")
    p.add_argument("--selection-metric", choices=["auc_pr", "auc_roc", "mcc"], default="auc_pr")
    p.add_argument("--threshold-metric", choices=["f1", "mcc"], default="f1")
    # optimizer
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--scheduler", choices=["linear", "cosine"], default="cosine")
    p.add_argument("--min-lr-ratio", type=float, default=0.01)
    p.add_argument("--early-stopping-patience", type=int, default=8)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-probs", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    class_weight_mode = args.class_weight
    with open(args.base_config) as handle:
        base = json.load(handle)
    config = {
        "model_name_or_path": base["model_name_or_path"],
        "num_labels": 2,
        "half_precision": False,
        "lora": build_lora_config(
            layers=args.lora_layers,
            rank=args.lora_rank,
            init_scale=args.lora_init_scale,
            alpha=args.lora_alpha or 2 * args.lora_rank,
            trainable=args.trainable,
        ),

        "loss": {"pos_weight": 0.1, "neg_weight": 1.0}
        if class_weight_mode == "paper"
        else ({"pos_weight": 1.0, "neg_weight": 1.0}
              if class_weight_mode == "none"
              else {"pos_weight": None, "neg_weight": None, "mode": class_weight_mode}),
        "data": {"max_length": args.max_length, "cdr_masked_train": True, "cdr_masked_eval": args.eval_scope == "cdr"},
        "training": {
            "lr": args.lr, "weight_decay": args.weight_decay, "epochs": args.epochs,
            "warmup_ratio": args.warmup_ratio, "scheduler": args.scheduler,
            "min_lr_ratio": args.min_lr_ratio, "batch": args.batch_size, "seed": args.seed,
        },
    }
    set_all_seeds(args.seed)
    print(f"Config:\n{json.dumps(config, indent=2)}", flush=True)

    # ---- model ----
    model, tokenizer, lora_cfg, trainable_frozen = build_pt5_classifier(config=config)
    if args.head != "linear" or args.head_dropout > 0:
        replace_head(model, args.head, args.head_dropout)
    n_head = unfreeze_classifier(model)
    model = model.to(device).float()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,} (frozen-backbone base={trainable_frozen:,}, "
          f"classifier params now trainable={n_head})", flush=True)

    # ---- data ----
    train_raw = load_split(args.train_data, format=args.train_format)
    valid_raw = load_split(args.valid_data, format=args.valid_format)
    train_split = prepare_split(train_raw, cdr_mask=True)
    valid_split = prepare_split(valid_raw, cdr_mask=args.eval_scope == "cdr")
    train_set = create_dataset(tokenizer, train_split.sequences, train_split.labels, max_length=args.max_length)
    valid_set = create_dataset(tokenizer, valid_split.sequences, valid_split.labels, max_length=args.max_length)
    collator = DataCollatorForTokenClassification(tokenizer)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collator, num_workers=0)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, collate_fn=collator, num_workers=0)

    loss_weights = torch.tensor(select_loss_weights(args.class_weight, train_split), dtype=torch.float32, device=device)
    print(f"Loss weights [neg, pos] = {loss_weights.cpu().tolist()} "
          f"(mode={args.class_weight})", flush=True)

    # ---- optimizer / scheduler ----
    epochs = args.epochs
    steps_per_epoch = math.ceil(len(train_loader))
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(args.scheduler, optimizer, warmup_steps, total_steps, args.min_lr_ratio)
    print(f"Optimizer AdamW(lr={args.lr}, wd={args.weight_decay}); scheduler={args.scheduler}; "
          f"total_steps={total_steps} warmup={warmup_steps}; epochs={epochs}; patience={args.early_stopping_patience}",
          flush=True)

    # ---- train ----
    best_val = {args.selection_metric: -1.0}
    best_epoch = 0
    best_threshold = 0.5
    best_path = os.path.join(args.output_dir, "best_trainable_params.pt")
    no_improve = 0
    stopped_early = False
    history = []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, loss_weights, device, args.grad_clip, epoch, args.log_every)
        train_time = time.time() - t0
        t1 = time.time()
        probs, labels = predict_residues(model, valid_loader, device)
        eval_time = time.time() - t1
        val_metrics, val_threshold = sweep_threshold_metrics(probs, labels)
        sel = val_metrics[args.selection_metric]
        history.append({
            "epoch": epoch, "train_loss": float(train_loss),
            "val_auc_roc": val_metrics["auc_roc"], "val_auc_pr": val_metrics["auc_pr"],
            "val_f1": val_metrics["f1"], "val_mcc": val_metrics["mcc"],
            "val_precision": val_metrics["precision"], "val_recall": val_metrics["recall"],
            "val_threshold": float(val_threshold), "sel": float(sel),
            "train_time_s": train_time, "eval_time_s": eval_time,
            "lr": float(optimizer.param_groups[0]["lr"]),
        })
        print(f"[ep {epoch}/{epochs}] loss={train_loss:.4f} | val AUC-ROC={val_metrics['auc_roc']:.4f} "
              f"AUC-PR={val_metrics['auc_pr']:.4f} F1={val_metrics['f1']:.4f} MCC={val_metrics['mcc']:.4f} "
              f"P={val_metrics['precision']:.4f} R={val_metrics['recall']:.4f} (thr={val_threshold:.2f}) | "
              f"{args.selection_metric}={sel:.4f} | {train_time:.1f}s+{eval_time:.1f}s", flush=True)
        if sel > best_val[args.selection_metric]:
            best_val = val_metrics
            best_epoch = epoch
            best_threshold = val_threshold
            save_trainable_parameters(model, best_path)
            if args.save_probs:
                with open(os.path.join(args.output_dir, "best_val_probs.npz"), "wb") as h:
                    np.savez(h, probs=probs, labels=labels)
            print(f"  -> new best {args.selection_metric}={sel:.4f}; saved {best_path}", flush=True)
            no_improve = 0
        else:
            no_improve += 1
        if args.early_stopping_patience > 0 and no_improve >= args.early_stopping_patience:
            print(f"[early stop] no improvement {no_improve} epochs; stop at {epoch}.", flush=True)
            stopped_early = True
            break

    # ---- test (paper-correct: val threshold) ----
    test_metrics = None
    if args.test_data is not None:
        best_state = torch.load(best_path, map_location="cpu")
        for name, param in model.named_parameters():
            if name in best_state:
                param.data.copy_(best_state[name].to(param.dtype))
        model = model.to(device)
        test_raw = load_split(args.test_data, format=args.test_format)
        test_split = prepare_split(test_raw, cdr_mask=args.eval_scope == "cdr")
        test_set = create_dataset(tokenizer, test_split.sequences, test_split.labels, max_length=args.max_length)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collator, num_workers=0)
        probs, labels = predict_residues(model, test_loader, device)
        test_metrics = metrics_at_threshold(probs, labels, best_threshold)
        test_sweep, test_sweep_thr = sweep_threshold_metrics(probs, labels)
        if args.save_probs:
            with open(os.path.join(args.output_dir, "test_probs.npz"), "wb") as h:
                np.savez(h, probs=probs, labels=labels)
        print(f"[TEST @ val thr={best_threshold:.2f}] AUC-ROC={test_metrics['auc_roc']:.4f} "
              f"AUC-PR={test_metrics['auc_pr']:.4f} F1={test_metrics['f1']:.4f} MCC={test_metrics['mcc']:.4f} "
              f"P={test_metrics['precision']:.4f} R={test_metrics['recall']:.4f}", flush=True)
        print(f"[TEST sweep (leaky ref) thr={test_sweep_thr:.2f}] F1={test_sweep['f1']:.4f} MCC={test_sweep['mcc']:.4f}", flush=True)

    summary = {
        "scheme": "ParaLoRA-FT",
        "seed": args.seed,
        "lora": {"layers": args.lora_layers, "rank": args.lora_rank, "alpha": args.lora_alpha or 2 * args.lora_rank,
                 "init_scale": args.lora_init_scale},
        "head": args.head, "class_weight": args.class_weight, "eval_scope": args.eval_scope,
        "selection_metric": args.selection_metric, "threshold_metric": args.threshold_metric,
        "loss_weights": loss_weights.cpu().tolist(),
        "trainable_params": trainable,
        "best_epoch": best_epoch, "best_threshold": float(best_threshold), "stopped_early": stopped_early,
        "best_val": best_val,
        "test_metrics": test_metrics,
        "history": history,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as h:
        json.dump(summary, h, indent=2, default=float)
    print(f"Saved summary -> {os.path.join(args.output_dir, 'summary.json')}", flush=True)


if __name__ == "__main__":
    main()
