#!/usr/bin/env python
"""ParaLoRA 端到端 sanity check：加载数据 + 1 batch 训练"""
import json
import os
import time

import torch
from transformers import T5Tokenizer

from paralora.data import load_split
from paralora.model import build_pt5_classifier


def main():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")

    config = json.load(open("configs/paralora.json"))
    config["model_name_or_path"] = "/mnt/d/UbuntuData/ProgramData/ProtTrans-models/prot_t5_xl_half_uniref50-enc"
    config["loss"]["pos_weight"] = 0.1
    config["loss"]["neg_weight"] = 1.0
    config["training"]["deepspeed"] = False
    config["training"]["fp16"] = False
    config["half_precision"] = False
    config["training"]["epochs"] = 1
    config["training"]["batch_size"] = 4

    print("\n=== Loading Parapred CSV (552 rows) ===")
    csv_path = "/mnt/d/研究资料/论文投稿/2026年/03-ParaLoRAandDG/Code/v1/做数据集和第一个实验/data/processed_dataset_paraperd.csv"
    split = load_split(csv_path, format="paraperd")
    print(f"Loaded {len(split.sequences)} samples")
    n_pos = sum(sum(l) for l in split.labels)
    n_tot = sum(len(l) for l in split.labels)
    print(f"Total residues: {n_tot}; paratope residues: {n_pos} ({100*n_pos/n_tot:.2f}%)")
    # Verify mask aligns with label positions
    pos_in_mask = sum(sum(l) for l, m in zip(split.labels, split.masks) for l in [l])
    pos_unmasked = 0
    for l, m in zip(split.labels, split.masks):
        for li, mi in zip(l, m):
            if li == 1 and mi == 0:
                pos_unmasked += 1
    print(f"Paratope residues outside CDR+/-2 mask: {pos_unmasked} ({100*pos_unmasked/n_pos:.2f}%)")

    print("\n=== Building ParaLoRA model ===")
    t0 = time.time()
    model, tokenizer, lora_cfg, trainable = build_pt5_classifier(config=config)
    model = model.to(device)
    print(f"Build time: {time.time()-t0:.1f}s; Trainable params: {trainable:,}")
    print(f"lora: rank={lora_cfg.lora_rank}, alpha={lora_cfg.lora_alpha}, layers={lora_cfg.lora_layers}")
    print(f"loss_weights: {model.loss_weights.cpu().tolist()}")

    print("\n=== 4-sample forward + backward ===")
    seqs = split.sequences[:4]
    labels = split.labels[:4]
    masks = split.masks[:4]
    pad_id = tokenizer.pad_token_id
    token_lens = [len(tokenizer(" ".join(s), add_special_tokens=True)["input_ids"]) for s in seqs]
    label_lens = [len(l) for l in labels]
    # Tokenizer adds 2 special tokens for ProtT5, so token_len >= residue_len
    max_len = max(max(token_lens), max(label_lens))
    X = torch.full((4, max_len), pad_id, dtype=torch.long)
    L = torch.zeros((4, max_len), dtype=torch.long)
    M = torch.zeros((4, max_len), dtype=torch.long)
    for i, (s, l, m) in enumerate(zip(seqs, labels, masks)):
        toks = tokenizer(" ".join(s), add_special_tokens=True)["input_ids"]
        n = len(toks)
        # Labels may be shorter than tokens (tokenizer adds special tokens)
        # In a real trainer, labels and masks are padded to max batch length; here we just write to token positions
        ln = min(n, len(l))
        mn = min(n, len(m))
        X[i, :n] = torch.tensor(toks, dtype=torch.long)
        L[i, :ln] = torch.tensor(l[:ln], dtype=torch.long)
        M[i, :mn] = torch.tensor(m[:mn], dtype=torch.long)
    X, L, M = X.to(device), L.to(device), M.to(device)
    print(f"Batch X={X.shape} L={L.shape} M={M.shape}")

    # Model uses attention_mask as active mask and labels where masked positions -> -100
    attn_mask = M.clone()  # CDR+/-2 mask acts as attention mask
    L_for_loss = L.clone()
    L_for_loss[M == 0] = -100  # ignore non-CDR positions in loss

    t0 = time.time()
    out = model(input_ids=X, attention_mask=attn_mask, labels=L_for_loss)
    print(f"Forward: logits.shape={out.logits.shape} loss={out.loss.item():.4f} time={time.time()-t0:.3f}s")

    out.loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"Backward OK; grad_norm={grad_norm:.2f}")


if __name__ == "__main__":
    main()