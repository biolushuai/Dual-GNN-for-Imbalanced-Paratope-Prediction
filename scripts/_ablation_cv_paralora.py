from __future__ import annotations

import argparse
import json
import math
import os
import time
import traceback
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader
from transformers import DataCollatorForTokenClassification

from paralora.data import create_dataset, load_split, prepare_split
from paralora.model import build_pt5_classifier, load_trainable_parameters, save_trainable_parameters

from scripts._train_paralora_custom import (
    build_optimizer,
    build_scheduler,
    compute_loss,
    predict_residues,
    set_all_seeds,
    sweep_threshold_metrics,
    train_one_epoch,
)
from scripts._train_paralora_ft import (
    build_lora_config,
    metrics_at_threshold,
    select_loss_weights,
    unfreeze_classifier,
)


def static_loss_weights(mode: str) -> dict:
    if mode == "paper":
        return {"pos_weight": 0.1, "neg_weight": 1.0}
    if mode == "none":
        return {"pos_weight": 1.0, "neg_weight": 1.0}
    return {"pos_weight": None, "neg_weight": None, "mode": mode}


def build_config(base: dict, placement: str, rank: int, alpha: int,
                 class_weight: str, eval_scope: str, lr: float, wd: float,
                 epochs: int, scheduler: str, warmup_ratio: float,
                 min_lr_ratio: float, max_length: int, seed: int,
                 trainable: str = "ln_lora") -> dict:
    return {
        "model_name_or_path": base["model_name_or_path"],
        "num_labels": 2,
        "half_precision": False,
        "lora": build_lora_config(
            layers=placement, rank=rank, init_scale=0.02, alpha=alpha,
            trainable=trainable),
        "loss": static_loss_weights(class_weight),
        "data": {"max_length": max_length, "cdr_masked_train": True,
                 "cdr_masked_eval": eval_scope == "cdr"},
        "training": {"lr": lr, "weight_decay": wd, "epochs": epochs,
                     "warmup_ratio": warmup_ratio, "scheduler": scheduler,
                     "min_lr_ratio": min_lr_ratio, "batch": 1, "seed": seed},
    }


def make_loader(tokenizer, sequences, labels, max_length: int) -> DataLoader:
    ds = create_dataset(tokenizer, sequences, labels, max_length=max_length)
    collator = DataCollatorForTokenClassification(tokenizer)
    return DataLoader(ds, batch_size=1, shuffle=False,
                      collate_fn=collator, num_workers=0)


def run_fold(fold: int, train_idx, val_idx, test_idx, raw, config: dict,
             tokenizer, device: torch.device, args,
             fold_params_path: str) -> Optional[dict]:
    set_all_seeds(args.seed)
    model, tok, lora_cfg, trainable = build_pt5_classifier(config=config)
    unfreeze_classifier(model)
    model = model.to(device).float()
    model.train()

    n_train = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    n_lora_t = sum(1 for nm, _ in model.named_parameters()
                   if ".lora_a" in nm or ".lora_b" in nm)
    n_ln_t = sum(1 for nm, p in model.named_parameters()
                 if "layer_norm" in nm and p.requires_grad)
    preset = config.get("lora", {}).get("trainable_param_names", "")
    print(f"  [fold {fold+1}] preset='{preset}' trainable={n_train:,} "
          f"| lora_tensors={n_lora_t} | layer_norm_trainable={n_ln_t}", flush=True)

    seqs = list(raw.sequences)
    labs = list(raw.labels)
    max_length = int(config["data"]["max_length"])

    train_split = prepare_split(
        type(raw)([seqs[i] for i in train_idx], [labs[i] for i in train_idx],
                  [raw.masks[i] for i in train_idx]), cdr_mask=True)
    val_split = prepare_split(
        type(raw)([seqs[i] for i in val_idx], [labs[i] for i in val_idx],
                  [raw.masks[i] for i in val_idx]), cdr_mask=False)
    test_split = prepare_split(
        type(raw)([seqs[i] for i in test_idx], [labs[i] for i in test_idx],
                  [raw.masks[i] for i in test_idx]), cdr_mask=False)

    train_loader = make_loader(tok, train_split.sequences, train_split.labels, max_length)
    val_loader = make_loader(tok, val_split.sequences, val_split.labels, max_length)
    test_loader = make_loader(tok, test_split.sequences, test_split.labels, max_length)

    loss_weights = torch.tensor(
        select_loss_weights(args.class_weight, train_split),
        dtype=torch.float32, device=device)

    epochs = args.epochs
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(args.scheduler, optimizer, warmup_steps,
                                total_steps, args.min_lr_ratio)

    best_val = {"auc_pr": -1.0}
    best_thr = 0.5
    best_epoch = 0
    best_state: Optional[dict] = None
    no_improve = 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, scheduler,
                               loss_weights, device, args.grad_clip, epoch, 100)
        vprobs, vlabels = predict_residues(model, val_loader, device)
        val_metrics, val_thr = sweep_threshold_metrics(vprobs, vlabels)
        if val_metrics["auc_pr"] > best_val["auc_pr"]:
            best_val = val_metrics
            best_thr = val_thr
            best_epoch = epoch
            best_state = {n: p.detach().cpu().clone()
                          for n, p in model.named_parameters() if p.requires_grad}
            no_improve = 0
        else:
            no_improve += 1
        if args.patience > 0 and no_improve >= args.patience:
            break

    os.makedirs(os.path.dirname(fold_params_path) or ".", exist_ok=True)
    if best_state is not None:
        torch.save(best_state, fold_params_path)
    else:
        save_trainable_parameters(model, fold_params_path)

    if best_state is not None:
        for name, param in model.named_parameters():
            if name in best_state:
                param.data.copy_(best_state[name].to(param.dtype))
    tprobs, tlabels = predict_residues(model, test_loader, device)
    test_metrics = metrics_at_threshold(tprobs, tlabels, best_thr)
    val_auc_pr = float(best_val["auc_pr"])

    result = {
        "fold": int(fold),
        "threshold": float(best_thr),
        "best_epoch": int(best_epoch),
        "val_auc_pr": val_auc_pr,
        "val_auc_roc": float(best_val["auc_roc"]),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "fold_seconds": float(time.time() - t0),
        **test_metrics,
    }
    del model
    torch.cuda.empty_cache()
    return result


def run_config(name: str, config: dict, raw, device, args, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "train_config.json"), "w") as h:
        json.dump(config, h, indent=2)

    n = len(raw.sequences)
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.fold_seed)
    fold_results: List[dict] = []

    for f in range(args.folds):
        fp = os.path.join(out_dir, f"fold{f:02d}_best_params.pt")
        fs = os.path.join(out_dir, f"fold{f:02d}_summary.json")
        if os.path.exists(fp) and os.path.exists(fs):
            fold_results.append(json.load(open(fs)))
            print(f"  [{name}] fold {f+1}: skipped (exists)", flush=True)
            continue

        splits = list(kf.split(np.arange(n)))
        rest_idx, test_idx = splits[f]
        inner_test_size = 0.5 if args.folds == 2 else 1.0 / (args.folds - 1)
        inner_train, inner_val = train_test_split(
            rest_idx, test_size=inner_test_size,
            random_state=args.fold_seed, shuffle=True)
        print(f"\n===== [{name}] fold {f+1}/{args.folds}: "
              f"train={len(inner_train)} val={len(inner_val)} test={len(test_idx)} =====",
              flush=True)
        try:
            res = run_fold(f, inner_train, inner_val, test_idx, raw, config,
                           None, device, args, fp)
        except Exception as e:
            print(f"  [{name}] fold {f+1} FAILED: {e}", flush=True)
            traceback.print_exc()
            continue
        with open(fs, "w") as h:
            json.dump(res, h, indent=2, default=float)
        fold_results.append(res)
        print(f"  [{name}] fold {f+1} TEST@val-thr={res['threshold']:.2f}: "
              f"AUC-ROC={res['auc_roc']:.4f} AUC-PR={res['auc_pr']:.4f} "
              f"F1={res['f1']:.4f} MCC={res['mcc']:.4f} "
              f"P={res['precision']:.4f} R={res['recall']:.4f} "
              f"({res['fold_seconds']:.0f}s)", flush=True)

    if len(fold_results) < args.folds:
        print(f"  [{name}] only {len(fold_results)}/{args.folds} folds done; "
              f"summary not finalized.", flush=True)
        return {"name": name, "config": config, "folds": fold_results,
                "complete": False}

    KEYS = ["auc_roc", "auc_pr", "f1", "mcc", "precision", "recall"]
    agg = {}
    for k in KEYS:
        vals = np.array([r[k] for r in fold_results], dtype=float)
        agg[k] = {"mean": float(vals.mean()), "std": float(vals.std()),
                  "values": vals.tolist()}

    best_fold = int(np.argmax([r["val_auc_pr"] for r in fold_results]))

    import shutil
    src = os.path.join(out_dir, f"fold{best_fold:02d}_best_params.pt")
    if os.path.exists(src):
        shutil.copy(src, os.path.join(out_dir, "best_for_paradg.pt"))
        shutil.copy(src, os.path.join(out_dir, "best_trainable_params.pt"))

    summary = {
        "name": name,
        "config": config,
        "n_folds_done": len(fold_results),
        "aggregate": agg,
        "recommended_fold": best_fold,
        "recommended_ckpt": "best_for_paradg.pt",
        "folds": fold_results,
    }
    with open(os.path.join(out_dir, "ablation_summary.json"), "w") as h:
        json.dump(summary, h, indent=2, default=float)
    print(f"\n===== [{name}] 10-fold mean±SD =====", flush=True)
    for k in KEYS:
        print(f"  {k:>10}: {agg[k]['mean']:.4f} ± {agg[k]['std']:.4f}", flush=True)
    return {"name": name, "config": config, "aggregate": agg,
            "recommended_fold": best_fold, "complete": True}


def write_master(out_root: str, entries: List[dict]) -> None:
    master = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
              "configs": []}
    for e in entries:
        if not e.get("complete"):
            continue
        master["configs"].append({
            "name": e["name"],
            "aggregate": e["aggregate"],
            "recommended_fold": e["recommended_fold"],
            "ckpt": os.path.join("results/ablation", e["name"], "best_for_paradg.pt"),
        })
    with open(os.path.join(out_root, "ablation_master.json"), "w") as h:
        json.dump(master, h, indent=2, default=float)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/paralora/full.csv")
    p.add_argument("--base-config", default="configs/paralora.json")
    p.add_argument("--out-root", default="../results/ablation")
    p.add_argument("--fold-seed", type=int, default=42)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--placements", nargs="+", default=["qv", "qkv", "qkvo"])
    p.add_argument("--ranks", nargs="+", type=int, default=[4, 8, 16])
    p.add_argument("--pairs", nargs="+", default=None,
                   help="placement:rank 配对列表 (例: 'q:8' 'qkvo:2'). "
                        "若提供则忽略 --placements 与 --ranks")
    p.add_argument("--alpha", type=int, default=8)
    p.add_argument("--trainable", default="ln_lora",
                   choices=["ln_lora", "lora_only"],
                   help="可训练参数集：ln_lora=LayerNorm+LoRA(默认); "
                        "lora_only=仅 LoRA，冻结 LayerNorm（P0-1）")
    p.add_argument("--tag", default="",
                   help="目录名后缀，用于区分同一 placement/rank 下的不同变体 "
                        "(例: --tag loraonly -> lora_v_r8_loraonly)")
    p.add_argument("--class-weight", default="paper",
                   choices=["paper", "balanced", "none"])
    p.add_argument("--eval-scope", default="all", choices=["all", "cdr"])
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--scheduler", default="cosine")
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--min-lr-ratio", type=float, default=0.01)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--grad-clip", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.base_config) as h:
        base = json.load(h)
    raw = load_split(args.data, "csv")
    if args.pairs:
        pairs: List[Tuple[str, int]] = []
        for spec in args.pairs:
            p, r = spec.split(":")
            pairs.append((p, int(r)))
    else:
        pairs = [(pl, rk) for pl in args.placements for rk in args.ranks]
    print(f"Parapred full: {len(raw.sequences)} complexes; device={device}; "
          f"pairs={pairs}", flush=True)

    os.makedirs(args.out_root, exist_ok=True)
    entries: List[dict] = []
    overall_t0 = time.time()
    for placement, rank in pairs:
            name = f"lora_{placement}_r{rank}"
            if args.tag:
                name = f"{name}_{args.tag}"
            out_dir = os.path.join(args.out_root, name)
            if os.path.exists(os.path.join(out_dir, "ablation_summary.json")):
                # 已完成的配置：载入汇总，跳过
                try:
                    s = json.load(open(os.path.join(out_dir, "ablation_summary.json")))
                    print(f"[{name}] already complete; skipped.", flush=True)
                    entries.append({"name": name, "config": s.get("config"),
                                   "aggregate": s["aggregate"],
                                   "recommended_fold": s["recommended_fold"],
                                   "complete": True})
                    continue
                except Exception:
                    pass
            config = build_config(
                base, placement, rank, args.alpha, args.class_weight,
                args.eval_scope, args.lr, args.weight_decay, args.epochs,
                args.scheduler, args.warmup_ratio, args.min_lr_ratio,
                args.max_length, args.seed, args.trainable)
            print(f"\n########## CONFIG {name} ##########", flush=True)
            res = run_config(name, config, raw, device, args, out_dir)
            entries.append(res)
            write_master(args.out_root, entries)
            print(f"[{name}] elapsed {time.time()-overall_t0:.0f}s", flush=True)

    write_master(args.out_root, entries)
    print(f"\nALL DONE. total {time.time()-overall_t0:.0f}s. "
          f"master -> {os.path.join(args.out_root, 'ablation_master.json')}", flush=True)


if __name__ == "__main__":
    main()
