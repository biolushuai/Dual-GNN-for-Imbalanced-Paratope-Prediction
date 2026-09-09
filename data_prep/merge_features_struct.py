"""Merge the qkvo r8 ParaLoRA feature pickles with the DSSP-derived structure
pickles into a single dataset that can drive every ParaDG ablation.

The output keeps *everything* the feature pickle already carried (``ab_feature``
= qkvo r8 embedding, paratope labels, the pre-computed 4.5 A heavy-atom
``antibody_adjacency_labels`` and the original surface labels) and adds two
fields recovered from the PDB files:

``antibody_coords``
    Calpha coordinates, so the global view can be rebuilt as a Calpha-Calpha
    contact map at any cutoff (paper Eq. 8) -> contact-cutoff ablation.
``antibody_rasa``
    Continuous DSSP relative solvent accessibility -> surface-cutoff ablation
    (re-deriving the mask at load time) and the ``surface_mode='feature'``
    variant that feeds rASA in as an extra node channel.

Because the rASA cutoff and the graph source are resolved at *load* time by
``paradg.data`` (``--rasa-threshold`` / ``--graph-source``), one merged copy is
enough - no need to materialise one dataset per ablation value.

Records whose structure could not be rebuilt keep their original fields and are
reported; they simply cannot take part in the structure-dependent ablations.
"""

from __future__ import annotations

import argparse
import os
import pickle
from typing import Dict, List

import numpy as np

SPLITS = ("train", "val", "test")
TEMPLATE = "pecan-paratope-{split}-all-paragraph.pkl"


def _load(path: str):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-dir", required=True,
                    help="qkvo r8 feature pickles")
    ap.add_argument("--struct-dir", required=True,
                    help="structure pickles from build_structure_from_pkls.py")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--require-structure", action="store_true",
                    help="Drop records without a structure record instead of keeping them")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    report: Dict[str, dict] = {}

    for split in SPLITS:
        feat = _load(os.path.join(args.feature_dir, TEMPLATE.format(split=split)))
        struct = _load(os.path.join(args.struct_dir, TEMPLATE.format(split=split)))
        by_name = {r["protein_name"]: r for r in struct}

        out: List[dict] = []
        n_struct = 0
        n_missing = 0
        n_lenbad = 0
        missing_names: List[str] = []

        for frec in feat:
            name = frec["protein_name"]
            new = dict(frec)
            srec = by_name.get(name)
            length = len(np.asarray(frec["antibody_labels"]))

            if srec is None:
                n_missing += 1
                missing_names.append(name)
                if args.require_structure:
                    continue
                out.append(new)
                continue

            coords = np.asarray(srec["antibody_coords"], dtype=np.float64)
            rasa = np.nan_to_num(np.asarray(srec["antibody_rasa"], dtype=np.float32), nan=0.0)

            if coords.shape[0] != length or rasa.shape[0] != length:
                n_lenbad += 1
                missing_names.append(f"{name}(len {coords.shape[0]}/{rasa.shape[0]} vs {length})")
                if args.require_structure:
                    continue
                out.append(new)
                continue

            new["antibody_coords"] = coords
            new["antibody_rasa"] = rasa
            n_struct += 1
            out.append(new)

        out_path = os.path.join(args.out_dir, TEMPLATE.format(split=split))
        with open(out_path, "wb") as handle:
            pickle.dump(out, handle)

        report[split] = {
            "written": len(out),
            "with_structure": n_struct,
            "missing_structure": n_missing,
            "length_mismatch": n_lenbad,
        }
        print(f"[{split}] wrote {len(out)} records "
              f"(structure {n_struct}, missing {n_missing}, len-mismatch {n_lenbad}) "
              f"-> {out_path}", flush=True)
        if missing_names:
            print(f"         affected: {missing_names[:10]}"
                  f"{' ...' if len(missing_names) > 10 else ''}", flush=True)

    total = sum(v["written"] for v in report.values())
    struct_total = sum(v["with_structure"] for v in report.values())
    print(f"\nDone. {struct_total}/{total} records carry Calpha coords + continuous rASA.")
    print(f"Data dir: {args.out_dir}")


if __name__ == "__main__":
    main()
