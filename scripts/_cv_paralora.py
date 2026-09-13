from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader
from transformers import DataCollatorForTokenClassification

from paralora.data import create_dataset
from paralora.model import build_pt5_classifier
from scripts._train_paralora_custom import (
    build_optimizer,
    build_scheduler,
    load_split_with_format,
    predict_residues,
    set_all_seeds,
    sweep_threshold_metrics,
    train_one_epoch,
)

KEYS = ["auc_roc", "auc_pr", "f1", "mcc", "precision", "recall"]


def metrics_at(probs: np.ndarray, labels: np.ndarray, thr: float) -> Dict[str, float]:
    pred = (probs >= thr).astype(np.int64)
    both = pred.sum() > 0 and (1 - pred).sum() > 0
    return {
        "auc_roc": float(roc_auc_score(labels, probs)),
        "auc_pr": float(average_precision_score(labels, probs)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, pred)) if both else 0.0,
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "recall": float(recall_score(labels, pred, zero_division=0)),
    }


def make_loader(tokenizer, sequences, labels, max_length: int, batch: int) -> DataLoader:
    dataset = create_dataset(tokenizer, sequences, labels, max_length=max_length)
    collator = DataCollatorForTokenClassification(tokenizer)
    return DataLoader(dataset, batch_size=batch, shuffle=False,
                      collate_fn=collator, num_workers=0)


def trainable_state(model) -> Dict[str, torch.Tensor]:
    return {n: p.detach().cpu().clone()
            for n, p in model.named_parameters() if p.requires_grad}


def load_state(model, state: Dict[str, torch.Tensor]) -> None:
    for name, param in model.named_parameters():
        if name in state:
            param.data.copy_(state[name].to(param.dtype))


def run_fold(
    fold: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    raw,
    config: dict,
    tokenizer,
    device: torch.device,
    args,
) -> Dict:
    inner_train = train_idx
    inner_val = val_idx

    set_all_seeds(args.seed)
    model, tok, lora_cfg, trainable = build_pt5_classifier(config=config)
    model = model.to(device).float()
    model.train()

    seqs_all = list(raw.sequences)
    labels_all = list(raw.labels)
    max_length = int(config["data"]["max_length"])
    batch = int(config["training"]["batch"])

    train_loader = make_loader(
        tok, [seqs_all[i] for i in inner_train],
        [labels_all[i] for i in inner_train], max_length, batch)
    val_loader = make_loader(
        tok, [seqs_all[i] for i in inner_val],
        [labels_all[i] for i in inner_val], max_length, batch)
    test_loader = make_loader(
        tok, [seqs_all[i] for i in test_idx],
        [labels_all[i] for i in test_idx], max_length, batch)

    train_cfg = config["training"]
    epochs = int(args.epochs or train_cfg["epochs"])
    lr = float(train_cfg["lr"])
    wd = float(train_cfg.get("weight_decay", 0.0))
    optimizer = build_optimizer(model, lr=lr, weight_decay=wd)
    steps_per_epoch = max(1, len(train_loader) // int(train_cfg.get("accum", 1)))
    warmup = max(1, int(float(train_cfg.get("warmup_ratio", 0.1)) * steps_per_epoch * epochs))
    scheduler = build_scheduler(
        train_cfg.get("scheduler", "cosine"), optimizer, warmup,
        steps_per_epoch * epochs, float(train_cfg.get("min_lr_ratio", 0.0)))
    loss_weights = model.loss_weights.to(device)

    best = {"auc_roc": -1.0}
    best_thr = 0.5
    best_state = None
    best_epoch = 0
    no_improve = 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(
            model=model, dataloader=train_loader, optimizer=optimizer,
            scheduler=scheduler, loss_weights=loss_weights, device=device,
            grad_clip=args.grad_clip, epoch=epoch, log_every=400)
        probs, labels = predict_residues(model, val_loader, device)
        val_metrics, thr = sweep_threshold_metrics(probs, labels)
        if val_metrics["auc_roc"] > best["auc_roc"]:
            best, best_thr, best_epoch = val_metrics, thr, epoch
            best_state = trainable_state(model)
            no_improve = 0
        else:
            no_improve += 1
        print(
            f"  [fold {fold}] ep{epoch} loss={loss:.4f} "
            f"val_auc={val_metrics['auc_roc']:.4f} thr={thr:.2f} "
            f"| best={best['auc_roc']:.4f}@{best_epoch} "
            f"({time.time() - t0:.0f}s)", flush=True)
        if args.patience > 0 and no_improve >= args.patience:
            print(f"  [fold {fold}] early stop at epoch {epoch}", flush=True)
            break

    if best_state is not None:
        load_state(model, best_state)
    probs, labels = predict_residues(model, test_loader, device)
    test_metrics = metrics_at(probs, labels, best_thr)
    test_metrics.update({
        "fold": int(fold),
        "threshold": float(best_thr),
        "best_epoch": int(best_epoch),
        "val_auc_roc": float(best["auc_roc"]),
        "fold_seconds": float(time.time() - t0),
        "n_test": int(len(test_idx)),
    })
    print(
        f"  [fold {fold}] TEST @val-thr={best_thr:.2f}: "
        f"AUC-ROC={test_metrics['auc_roc']:.4f} AUC-PR={test_metrics['auc_pr']:.4f} "
        f"F1={test_metrics['f1']:.4f} MCC={test_metrics['mcc']:.4f} "
        f"P={test_metrics['precision']:.4f} R={test_metrics['recall']:.4f}",
        flush=True)
    del model
    torch.cuda.empty_cache()
    return test_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paralora.json")
    parser.add_argument("--data", default="data/paralora/full.csv")
    parser.add_argument("--fold-seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0,
                        help="每折模型初始化的 seed（LoRA 重新初始化）")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--out", default="../results/paralora_10fold.json")
    args = parser.parse_args()

    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.config) as handle:
        config = json.load(handle)
    if config.get("half_precision", False):
        config["half_precision"] = False

    raw = load_split_with_format(args.data, "csv")
    n = len(raw.sequences)
    print(f"Parapred full split: {n} samples; device={device}", flush=True)

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.fold_seed)
    results: List[Dict] = []
    overall_t0 = time.time()
    for fold, (rest_idx, test_idx) in enumerate(kf.split(np.arange(n)), 1):
        inner_train, inner_val = train_test_split(
            rest_idx, test_size=1.0 / (args.folds - 1),
            random_state=args.fold_seed, shuffle=True)
        print(f"\n===== fold {fold}/{args.folds}: "
              f"train={len(inner_train)} val={len(inner_val)} test={len(test_idx)} =====",
              flush=True)
        res = run_fold(fold, np.array(inner_train), np.array(inner_val),
                       np.array(test_idx), raw, config, None, device, args)
        results.append(res)
        with open(args.out, "w") as handle:
            json.dump({"folds": results}, handle, indent=2, default=float)

    agg = {}
    for k in KEYS:
        vals = np.array([r[k] for r in results], dtype=float)
        agg[k] = {"mean": float(vals.mean()), "std": float(vals.std()),
                  "values": vals.tolist()}
    summary = {
        "protocol": f"{args.folds}-fold CV on Parapred ({n} complexes); "
                    f"inner val for early-stop + threshold; held-out fold evaluated "
                    f"at inner-val best threshold",
        "n_folds_done": len(results),
        "total_minutes": float((time.time() - overall_t0) / 60.0),
        "aggregate": agg,
        "thresholds": [r["threshold"] for r in results],
        "best_epochs": [r["best_epoch"] for r in results],
        "folds": results,
    }
    with open(args.out, "w") as handle:
        json.dump(summary, handle, indent=2, default=float)

    print(f"\n===== {len(results)}-fold mean±SD (held-out fold, val-tuned threshold) =====",
          flush=True)
    for k in KEYS:
        print(f"  {k:>10}: {agg[k]['mean']:.4f} ± {agg[k]['std']:.4f}", flush=True)
    print(f"  total: {summary['total_minutes']:.1f} min; saved → {args.out}", flush=True)


if __name__ == "__main__":
    main()
