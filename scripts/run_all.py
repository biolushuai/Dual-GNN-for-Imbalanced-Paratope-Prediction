"""Convenience wrappers for the data prep and analysis scripts.

Running ``python -m scripts.run_all`` chains the data preparation and
connectivity check in the right order so that reviewers can reproduce
the figures of the manuscript with a single command.

The actual scripts remain first-class modules; this file just provides a
minimal convenience entry point.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(module: str, args: List[str]) -> int:
    cmd = [sys.executable, "-m", module, *args]
    print(f"\n>>> {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=REPO_ROOT)


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="End-to-end reproduction script.")
    sub_p = parser.add_subparsers(dest="cmd", required=True)

    p_fetch = sub_p.add_parser("fetch-pdb", help="Run fetch_pdb_files on every split.")
    p_fetch.add_argument("--split-list-dir", required=True,
                         help="Folder containing pecan-paratope-{train,val,test}.txt")
    p_fetch.add_argument("--src-complex", required=True)
    p_fetch.add_argument("--src-ab", required=True)
    p_fetch.add_argument("--src-ag", required=True)
    p_fetch.add_argument("--out-dir", required=True)
    p_fetch.add_argument("--rasa-threshold", type=float, default=0.25)

    p_struct = sub_p.add_parser("structure-dataset", help="Run build_structure_dataset on every split.")
    p_struct.add_argument("--data-dir", required=True,
                          help="Folder with pecan-paratope-{split}/{complex,ab,ag} subfolders.")
    p_struct.add_argument("--out-dir", required=True)
    p_struct.add_argument("--rasa-threshold", type=float, default=0.25)

    p_embed = sub_p.add_parser("embed", help="Run embed_with_prot_t5 on every split.")
    p_embed.add_argument("--input-dir", required=True)
    p_embed.add_argument("--output-dir", required=True)

    p_seq = sub_p.add_parser("sequence-splits",
 help="Prepare ParaLoRA sequence splits from a paraperd CSV.")
    p_seq.add_argument("--paraperd-csv", required=True)
    p_seq.add_argument("--output-dir", required=True)

    p_connect = sub_p.add_parser("connectivity",
 help="Run the graph connectivity analysis.")
    p_connect.add_argument("--config", default="configs/paradg.json")
    p_connect.add_argument("--output", required=True)

    args = parser.parse_args(argv)

    if args.cmd == "fetch-pdb":
        for split in ("train", "val", "test"):
            _run(
                "data_prep.fetch_pdb_files",
                [
                    "--split-list", os.path.join(args.split_list_dir, f"pecan-paratope-{split}.txt"),
                    "--src-complex", args.src_complex,
                    "--src-ab", args.src_ab,
                    "--src-ag", args.src_ag,
                    "--out-complex", os.path.join(args.out_dir, split),
                    "--out-ab", os.path.join(args.out_dir, f"{split}-ab"),
                    "--out-ag", os.path.join(args.out_dir, f"{split}-ag"),
                    "--missing-log", os.path.join(args.out_dir, f"missing-{split}.log"),
                ],
            )

    elif args.cmd == "structure-dataset":
        for split in ("train", "val", "test"):
            _run(
                "data_prep.build_structure_dataset",
                [
                    "--complex-dir", os.path.join(args.data_dir, split),
                    "--ab-dir", os.path.join(args.data_dir, f"{split}-ab"),
                    "--ag-dir", os.path.join(args.data_dir, f"{split}-ag"),
                    "--output", os.path.join(args.out_dir, f"pecan-paratope-{split}-all.pkl"),
                    "--rasa-threshold", str(args.rasa_threshold),
                ],
            )

    elif args.cmd == "embed":
        for split in ("train", "val", "test"):
            _run(
                "data_prep.embed_with_prot_t5",
                [
                    "--input", os.path.join(args.input_dir, f"pecan-paratope-{split}-all.pkl"),
                    "--output", os.path.join(args.output_dir, f"pecan-paratope-{split}-all-embedded.pkl"),
                ],
            )

    elif args.cmd == "sequence-splits":
        _run(
            "data_prep.prepare_sequence_splits",
            [
                "from-paraperd",
                "--input", args.paraperd_csv,
                "--output", os.path.join(args.output_dir, "all.csv"),
            ],
        )
        _run(
            "data_prep.prepare_sequence_splits",
            [
                "train-val-test",
                "--input", os.path.join(args.output_dir, "all.csv"),
                "--out-dir", args.output_dir,
            ],
        )

    elif args.cmd == "connectivity":
        _run(
            "analysis.graph_connectivity",
            ["--config", args.config, "--output", args.output],
        )


if __name__ == "__main__":
    main()