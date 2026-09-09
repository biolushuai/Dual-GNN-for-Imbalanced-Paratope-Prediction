"""Prepare the PECAN splits used by the ParaLoRA sequence branch.

The released ParaLoRA experiments used two upstream data layouts:

* ``data/pecan-paratope-train.pkl`` -- one pickle per split, every entry
  is ``{"ab_sequence": str, "ab_label": list[int]}``. Used for the
  initial cross-validation runs (90 % paratope accuracy was reported on
  this layout).

* ``data/processed_dataset_paraperd.csv`` -- a single CSV with columns
  ``pdb, chain_type, sequence, paratope, cdrs``. Used for the released
  model checkpoint.

This script converts both layouts into the canonical format consumed by
``paralora.data.load_split``:

    columns = sequence | label | mask

The CDR positions come from the ``cdrs`` column when available. When
the CSV is missing, the mask is set to all-ones (everything contributes
to the loss during evaluation).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def from_pkl_split(pkl_path: str) -> pd.DataFrame:
    with open(pkl_path, "rb") as handle:
        raw = pickle.load(handle)
    rows: List[Dict] = []
    for entry in raw:
        rows.append(
            {
                "sequence": entry["ab_sequence"],
                "label": [int(v) for v in entry["ab_label"]],
                "mask": [1] * len(entry["ab_sequence"]),
            }
        )
    return pd.DataFrame(rows)


def from_paraperd_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={"paratope": str})
    rows: List[Dict] = []
    for seq, label_str, cdr_str in zip(df["sequence"].astype(str).tolist(),
                                        df["paratope"].astype(str).tolist(),
                                        df["cdrs"].astype(str).tolist()):
        label = [int(c) for c in label_str]
        mask = [0] * len(label)
        if cdr_str and cdr_str[0] in "[{":
            for idx in json.loads(cdr_str):
                mask[int(idx)] = 1
        rows.append({"sequence": seq, "label": label, "mask": mask})
    return pd.DataFrame(rows)


def write_split(rows: pd.DataFrame, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    rows.to_csv(output_path, index=False)
    print(f"Wrote {len(rows)} rows to {output_path}")


def split_train_test(
    rows: pd.DataFrame,
    test_frac: float = 0.3,
    val_frac: float = 0.5,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    from sklearn.model_selection import train_test_split

    train_set, temp_set = train_test_split(rows, test_size=test_frac, random_state=seed)
    val_set, test_set = train_test_split(temp_set, test_size=val_frac, random_state=seed)

    # The released training script forces the test mask to all-ones so every
    # residue contributes to the metric.
    test_set = test_set.copy()
    test_set["mask"] = test_set["mask"].apply(lambda m: [1] * len(m))
    return {"train": train_set, "val": val_set, "test": test_set}


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare PECAN splits for the ParaLoRA sequence branch.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_from_pkl = sub.add_parser("from-pkl", help="Convert a *.pkl split into a sequence/label/mask CSV.")
    p_from_pkl.add_argument("--input", required=True)
    p_from_pkl.add_argument("--output", required=True)

    p_from_csv = sub.add_parser("from-paraperd", help="Convert a paraperd-style CSV into sequence/label/mask CSV.")
    p_from_csv.add_argument("--input", required=True)
    p_from_csv.add_argument("--output", required=True)

    p_split = sub.add_parser("train-val-test", help="70/15/15 split from a sequence/label/mask CSV.")
    p_split.add_argument("--input", required=True)
    p_split.add_argument("--out-dir", required=True)
    p_split.add_argument("--seed", type=int, default=42)

    args = parser.parse_args(argv)

    if args.cmd == "from-pkl":
        rows = from_pkl_split(args.input)
        write_split(rows, args.output)
    elif args.cmd == "from-paraperd":
        rows = from_paraperd_csv(args.input)
        write_split(rows, args.output)
    elif args.cmd == "train-val-test":
        rows = pd.read_csv(args.input, dtype={"label": str, "mask": str})
        for name, sub_df in split_train_test(rows, seed=args.seed).items():
            write_split(sub_df, os.path.join(args.out_dir, f"{name}.csv"))


if __name__ == "__main__":
    main()