from __future__ import annotations

import argparse
import os
import re
import shutil
from typing import Dict, List


def process_chain(chain_str: str) -> str:
    """Remove ';' separators used by PDB chain annotations."""
    return chain_str.replace(";", "")


def parse_split(txt_path: str) -> List[Dict[str, str]]:
    """Parse the official PECAN split text file into a list of records."""
    records: List[Dict[str, str]] = []
    with open(txt_path, "r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 4:
                print(f"[warn] line {line_no}: expected 4 fields, got {len(parts)}; skipping")
                continue
            pdb_id, c1, c2, raw_chain = parts
            chain_clean = process_chain(raw_chain)
            records.append(
                {
                    "pdb_id": pdb_id,
                    "c1": c1,
                    "c2": c2,
                    "chain_clean": chain_clean,
                }
            )
    return records


def fetch_pdb_files(
    txt_path: str,
    src_complex: str,
    src_ab: str,
    src_ag: str,
    out_complex: str,
    out_ab: str,
    out_ag: str,
    missing_log: str,
) -> Dict[str, int]:
    """Copy the three PDB files for each entry of a PECAN split.

    Returns a counter with the number of files found and missing for each
    of the three layouts.
    """
    os.makedirs(out_complex, exist_ok=True)
    os.makedirs(out_ab, exist_ok=True)
    os.makedirs(out_ag, exist_ok=True)

    counts = {"complex": 0, "ab": 0, "ag": 0, "missing": 0}
    missing: List[str] = []

    records = parse_split(txt_path)
    for record in records:
        ch_combine = f"{record['c1']}{record['c2']}"
        name_complex = f"{record['pdb_id']}_{ch_combine}_{record['chain_clean']}.pdb"
        name_ab = f"{record['pdb_id']}_{ch_combine}.pdb"
        name_ag = f"{record['pdb_id']}_{record['chain_clean']}.pdb"

        pairs = [
            (os.path.join(src_complex, name_complex), os.path.join(out_complex, name_complex), "complex"),
            (os.path.join(src_ab, name_ab), os.path.join(out_ab, name_ab), "ab"),
            (os.path.join(src_ag, name_ag), os.path.join(out_ag, name_ag), "ag"),
        ]
        for src, dst, tag in pairs:
            if os.path.exists(src):
                shutil.copy2(src, dst)
                counts[tag] += 1
            else:
                counts["missing"] += 1
                missing.append(f"[missing {tag}] {src}")

    with open(missing_log, "w", encoding="utf-8") as handle:
        handle.write("\n".join(missing) + ("\n" if missing else ""))
    print(f"Fetched complexes={counts['complex']} abs={counts['ab']} ags={counts['ag']} "
          f"missing={counts['missing']} (log: {missing_log})")
    return counts


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fetch PECAN PDB files for a split.")
    parser.add_argument("--split-list", required=True, help="Path to the PECAN split text file (e.g. pecan-paratope-test.txt).")
    parser.add_argument("--src-complex", required=True, help="Folder holding {id}_{c1c2}_{chain}.pdb")
    parser.add_argument("--src-ab", required=True, help="Folder holding {id}_{c1c2}.pdb")
    parser.add_argument("--src-ag", required=True, help="Folder holding {id}_{chain}.pdb")
    parser.add_argument("--out-complex", required=True)
    parser.add_argument("--out-ab", required=True)
    parser.add_argument("--out-ag", required=True)
    parser.add_argument("--missing-log", default="missing_pdb.log")
    args = parser.parse_args(argv)

    fetch_pdb_files(
        txt_path=args.split_list,
        src_complex=args.src_complex,
        src_ab=args.src_ab,
        src_ag=args.src_ag,
        out_complex=args.out_complex,
        out_ab=args.out_ab,
        out_ag=args.out_ag,
        missing_log=args.missing_log,
    )


if __name__ == "__main__":
    main()
