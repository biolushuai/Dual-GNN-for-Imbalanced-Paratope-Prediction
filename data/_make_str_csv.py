#!/usr/bin/env python
"""Prepare ParaLoRA CSV splits in 'sequence,label,mask' format compatible with ParaLoRA CLI _from_csv.

The released sequence-splits helper writes label as a JSON list string, but ParaLoRA's _from_csv
expects a flat per-residue digit string like "011010...". This script rebuilds the splits in the
latter format from the raw Parapred CSV.
"""
import os
import pandas as pd

SRC = "/mnt/d/.../processed_dataset_paraperd.csv"
OUT_DIR = "/mnt/d/ProjectData/ParaLoRAandDG/ParaLoRA/data/paralora_str"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(SRC)
print(f"Loaded {len(df)} rows; columns: {df.columns.tolist()}")

# Build mask from cdrs
import json
def build_mask(label_str, cdrs_str):
    label = [int(c) for c in str(label_str)]
    cdrs = json.loads(cdrs_str)
    mask = [0] * len(label)
    for idx in cdrs:
        mask[int(idx)] = 1
    return "".join(str(c) for c in label), "".join(str(m) for m in mask)

# Use the existing 80/10/10 split IDs (train.csv / val.csv / test.csv have pdb columns)
def get_ids(name):
    sub = pd.read_csv(f"/mnt/d/WorkBuddy/ParaLoRAandDG/ParaLoRA/data/paralora/{name}.csv")
    return set(sub["sequence"].tolist())

# These were generated from sequence-splits helper; we just reuse the row ordering
for split in ["train", "val", "test"]:
    sub = pd.read_csv(f"/mnt/d/WorkBuddy/ParaLoRAandDG/ParaLoRA/data/paralora/{split}.csv")
    out_rows = []
    for _, row in sub.iterrows():
        # find matching original row by sequence
        orig = df[df["sequence"] == row["sequence"]].iloc[0]
        label_str, mask_str = build_mask(orig["paratope"], orig["cdrs"])
        out_rows.append({"sequence": row["sequence"], "label": label_str, "mask": mask_str})
    out_df = pd.DataFrame(out_rows)
    out_path = os.path.join(OUT_DIR, f"{split}.csv")
    out_df.to_csv(out_path, index=False)
    print(f"  {split}: {len(out_df)} rows -> {out_path}")
print("Done. Splits in", OUT_DIR)
