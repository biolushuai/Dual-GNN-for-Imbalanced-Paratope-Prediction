"""Build the PECAN structure-dataset (Calpha coords + continuous rASA) for the
exact set of complexes present in the ParaLoRA feature pickles.

Unlike ``build_structure_dataset.py`` (which zips the three PDB directories by
*sorted filename* and therefore mis-pairs antibody/antigen files for complexes
whose filenames do not sort-align, e.g. complex ``1A3R_LH_P`` whose antibody
file is ``1A3R_LH.pdb`` and antigen file is ``1A3R_P.pdb``), this driver walks
the feature pickles' ``protein_name`` manifest and locates each complex's
antibody / antigen PDB by ``<pdbid>_*.pdb`` glob + chain-subset matching.

Outputs one pickle per split (same filenames as the feature pickles) so the
merge step and ``paradg.data.load_protein_data`` can consume them directly.
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
from typing import List, Optional

from Bio.PDB import PDBParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_prep.build_structure_dataset import calculate_binding_and_surface

_PARSER = PDBParser(QUIET=True)


def _chains(pdb: str) -> set:
    struct = _PARSER.get_structure("x", pdb)
    return {c.id for c in struct[0].get_chains()}


def _pick(cands: List[str], complex_chains: set) -> Optional[str]:
    """Among candidate antibody/antigen files, keep those whose chain set is a
    subset of the complex's chains, then prefer the one with the most chains
    (the combined antibody/antigen file)."""
    subs = []
    for c in cands:
        try:
            ch = _chains(c)
        except Exception:
            continue
        if ch and ch <= complex_chains:
            subs.append((len(ch), c))
    if not subs:
        return None
    subs.sort(reverse=True)
    return subs[0][1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-dir", required=True,
                    help="qkvo r8 feature pickles (manifest source of protein_name)")
    ap.add_argument("--complex-dir", required=True)
    ap.add_argument("--ab-dir", required=True)
    ap.add_argument("--ag-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dssp-executable", default=None)
    ap.add_argument("--rasa-threshold", type=float, default=0.25)
    ap.add_argument("--distance-threshold", type=float, default=4.5)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                    help="Which splits to build (useful for resuming)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for split in args.splits:
        src = os.path.join(args.feature_dir, f"pecan-paratope-{split}-all-paragraph.pkl")
        with open(src, "rb") as handle:
            recs = pickle.load(handle)
        names = [r["protein_name"] for r in recs]
        out: List[dict] = []
        skipped = 0
        for name in names:
            pdbid = name.split("_")[0]
            complex_file = os.path.join(args.complex_dir, f"{name}.pdb")
            ab_cands = glob.glob(os.path.join(args.ab_dir, f"{pdbid}_*.pdb"))
            ag_cands = glob.glob(os.path.join(args.ag_dir, f"{pdbid}_*.pdb"))
            if not os.path.exists(complex_file) or not ab_cands or not ag_cands:
                print(f"[skip {name}] missing files c={os.path.exists(complex_file)} "
                      f"ab={len(ab_cands)} ag={len(ag_cands)}", flush=True)
                skipped += 1
                continue
            cchains = _chains(complex_file)
            ab_file = _pick(ab_cands, cchains)
            ag_file = _pick(ag_cands, cchains)
            if ab_file is None or ag_file is None:
                print(f"[skip {name}] could not pick ab/ag (cchains={cchains})", flush=True)
                skipped += 1
                continue
            try:
                rec = calculate_binding_and_surface(
                    abag_file=complex_file,
                    ab_file=ab_file,
                    ag_file=ag_file,
                    distance_threshold=args.distance_threshold,
                    neighbor_distance_threshold=4.5,
                    rasa_threshold=args.rasa_threshold,
                    dssp_executable=args.dssp_executable,
                )
            except Exception as exc:  # one malformed PDB must not kill the split
                print(f"[skip {name}] {type(exc).__name__}: {exc} "
                      f"(ab={os.path.basename(ab_file)} ag={os.path.basename(ag_file)})",
                      flush=True)
                skipped += 1
                continue
            if rec is None:
                print(f"[skip {name}] calculate returned None (length mismatch?)", flush=True)
                skipped += 1
                continue
            out.append(rec)
        with open(os.path.join(args.out_dir, f"pecan-paratope-{split}-all-paragraph.pkl"), "wb") as handle:
            pickle.dump(out, handle)
        print(f"[{split}] saved {len(out)} records ({skipped} skipped) -> "
              f"{args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
