#!/usr/bin/env python
"""用 ParaLoRA 微调后的模型重算 PECAN 抗体残基嵌入，作为 ParaDG 节点特征。

**重要更正（2026-09-09 实证）**：``pecan-paratope-*-all-paragraph.pkl`` 里的
``ab_feature`` **不是**原始 ProtT5 嵌入，而是**论文 ParaLoRA 微调后模型**的
编码器隐状态。依据：
  1. ``Code/README.md``：``load_PT5_LoRA_Finetuning_pecan.ipynb`` "re-loads a
     **fine-tuned checkpoint**, embeds a pickle file ... under ``ab_feature``"
  2. 该笔记本 cell 27 ``load_model("../finetuned_models/paragraph_nopecan_1_1_e1.pth")``
     → cell 32 用该模型编码 ``antibody_sequence`` 并 ``data['ab_feature'] = features[i]``
  3. 实测（``results/_verify_feature_identity.py``，val 前 12 条）：
     cos(ab_feature, 本机原始 ProtT5-half-uniref50) = **0.7525**，
     残基范数 8.48（pkl） vs 6.27（原始 ProtT5） → 显著不同，故为微调版。

因此本脚本的语义是「**用我们复现的 ParaLoRA 权重替换论文自带的 ParaLoRA 特征**」，
而不是「把原始特征升级为微调特征」。两者不可混淆。

本脚本：
1. 按 seed 重建 ParaLoRA 微调模型（set_all_seeds → build → 加载 ckpt；
   分类头为冻结随机头，可精确重建，方法同 _eval_val_threshold.py）
2. 对每个 pkl 记录的 ``antibody_sequence`` 跑编码器，取最后一层隐状态
   (L, 1024)，替换 ab_feature
3. 其余字段（labels / adjacency / surface index）原样保留，写回新目录
   （文件名保持不变，供 paradg/data.py 直接读取）

用法（从 ParaLoRA/ 目录执行）：
    python -m scripts._gen_paralora_embeddings \
        --ckpt-dir ../results/paralora_30ep_s2 --seed 2 \
        --src-dir /mnt/d/ProjectsData/ParaLoRADG \
        --out-dir /mnt/d/ProjectsData/ParaLoRADG_paralora
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from typing import Dict, List

import numpy as np
import torch

from paralora.model import build_pt5_classifier, load_trainable_parameters
from scripts._train_paralora_custom import set_all_seeds

SPLITS = {
    "train": "pecan-paratope-train-all-paragraph.pkl",
    "val": "pecan-paratope-val-all-paragraph.pkl",
    "test": "pecan-paratope-test-all-paragraph.pkl",
}


def embed_sequence(model, tokenizer, seq: str, device: torch.device) -> np.ndarray:
    """返回 (L, 1024) 的逐残基嵌入（去掉结尾 </s>）。"""
    encoded = tokenizer(
        " ".join(list(seq)),
        return_tensors="pt",
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    with torch.no_grad():
        hidden = model.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
    hidden = hidden[0].detach().cpu().float().numpy()
    n_tokens = int(attention_mask.sum().item())
    residues = hidden[: n_tokens - 1]  # 去掉 </s>
    if residues.shape[0] != len(seq):
        raise ValueError(
            f"token/residue mismatch: {residues.shape[0]} vs {len(seq)}"
        )
    return residues


def process_split(
    model, tokenizer, path: str, device: torch.device, log_every: int = 25
) -> List[Dict]:
    with open(path, "rb") as handle:
        records = pickle.load(handle)
    print(f"  {os.path.basename(path)}: {len(records)} records", flush=True)
    out: List[Dict] = []
    t0 = time.time()
    for i, rec in enumerate(records, 1):
        seq = rec["antibody_sequence"]
        seq = "".join(c for c in seq if c.strip())
        try:
            feat = embed_sequence(model, tokenizer, seq, device)
        except ValueError as exc:
            print(f"    [skip {i}] {exc}", flush=True)
            feat = np.asarray(rec["ab_feature"]).astype(np.float32)
        new_rec = dict(rec)
        new_rec["ab_feature"] = feat.astype(np.float16)
        # surface_ab_feature 在 paraDG 代码中未被使用，但保持一致性
        idx = np.asarray(rec.get("antibody_surface_index", []), dtype=int)
        if idx.size:
            new_rec["surface_ab_feature"] = feat[idx].astype(np.float16)
        out.append(new_rec)
        if i % log_every == 0 or i == len(records):
            el = time.time() - t0
            print(
                f"    {i}/{len(records)} ({el:.0f}s, {el / i:.2f}s/rec, "
                f"L={len(seq)})",
                flush=True,
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True,
                        help="含 best_trainable_params.pt 的 ParaLoRA 结果目录")
    parser.add_argument("--seed", type=int, required=True,
                        help="该 ckpt 对应的训练 seed（用于重建冻结的分类头）")
    parser.add_argument("--config", default="configs/paralora.json")
    parser.add_argument("--src-dir", default="/mnt/d/ProjectsData/ParaLoRADG")
    parser.add_argument("--out-dir", default="/mnt/d/ProjectsData/ParaLoRADG_paralora")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS.keys()))
    args = parser.parse_args()

    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.config) as handle:
        config = json.load(handle)
    if config.get("half_precision", False):
        config["half_precision"] = False

    print(f"Device: {device}", flush=True)
    print(f"Rebuilding ParaLoRA model from {args.ckpt_dir} (seed={args.seed})", flush=True)
    set_all_seeds(args.seed)
    model, tokenizer, lora_cfg, trainable = build_pt5_classifier(config=config)
    model = model.to(device).float()
    n = load_trainable_parameters(
        model, os.path.join(args.ckpt_dir, "best_trainable_params.pt")
    )
    model.eval()
    print(f"Loaded {n:,} params; model in eval mode", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    for split in args.splits:
        src = os.path.join(args.src_dir, SPLITS[split])
        dst = os.path.join(args.out_dir, SPLITS[split])
        print(f"[{split}] {src} → {dst}", flush=True)
        out = process_split(model, tokenizer, src, device)
        with open(dst, "wb") as handle:
            pickle.dump(out, handle)
        # 自检：形状与标签长度一致
        bad = [i for i, r in enumerate(out)
               if r["ab_feature"].shape[0] != len(r["antibody_labels"])]
        print(f"  saved {len(out)} records; shape-mismatch={len(bad)}", flush=True)
        print(f"  → {dst}", flush=True)


if __name__ == "__main__":
    main()
