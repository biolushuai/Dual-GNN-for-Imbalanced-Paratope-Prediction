from __future__ import annotations

import argparse
import os
import pickle
import sys
import tempfile
import warnings
from typing import List, Optional

import numpy as np
from Bio.PDB import NeighborSearch, PDBParser
from Bio.PDB.DSSP import DSSP
from Bio.SeqUtils import seq1

warnings.filterwarnings("ignore")


#: Record types kept when sanitising a PDB file for DSSP. Everything DSSP needs
#: to compute solvent accessibility lives in the coordinate section; the header
#: and annotation sections only ever caused parse failures here.
_DSSP_KEEP_RECORDS = ("HEADER", "CRYST1", "ATOM  ", "HETATM", "TER", "MODEL ",
                      "ENDMDL", "END")


def _preprocess_pdb_for_dssp(path: str) -> str:
    """Return a temp copy of ``path`` that DSSP 4.x can parse.

    The PDB files in this dataset come in two flavours - hand-stripped in-house
    files and a handful of near-verbatim RCSB downloads - and DSSP 4.x aborts on
    several quirks found in both:

      * CRLF line endings -> "Not a valid integer in PDB record"
      * un-numbered ``REMARK`` lines (``REMARK Extract antibody...``) ->
        "Trying to parse 'Ex'"
      * **dangling ``TER`` records** for chains that were stripped from the
        file. An antigen-only ``*_C.pdb`` still carries the antibody's
        ``TER 844 LYS A 107`` lines while chains A/B have no atoms ->
        "the field TER is not at the correct location"
      * ``ANISOU`` records -> "the field ANISOU is not at the correct location"
      * ``HET`` / ``SHEET`` / ``LINK`` / ``SITE`` annotations that reference
        atoms of the stripped chain -> "Links for ... are incomplete"

    Rather than patching each symptom we keep only the coordinate section
    (:data:`_DSSP_KEEP_RECORDS`) and drop ``TER`` records whose chain carries no
    atoms. This is safe for the quantity we consume: relative solvent
    accessibility is a purely geometric function of the atom coordinates, so
    discarding metadata records cannot change it. No file in the antibody /
    antigen directories contains ``HETATM`` records, so no atoms are lost.

    Chain ids and residue numbering are left untouched so the DSSP output still
    maps back onto the parsed BioPython model.
    """
    with open(path, "r", errors="ignore") as handle:
        raw = handle.read()
    lines = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    # Chain ids that actually carry atoms in this file.
    atom_chains = {
        line[21] for line in lines
        if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 22
    }

    kept: List[str] = []
    for line in lines:
        if not line.startswith(_DSSP_KEEP_RECORDS):
            continue
        if line.startswith("TER") and len(line) >= 22 and line[21] not in atom_chains:
            continue
        kept.append(line)

    tmp = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w")
    tmp.write("\n".join(kept) + "\n")
    tmp.close()
    return tmp.name


def _filter_heavy_atoms(model):
    return [atom for atom in model.get_atoms() if not atom.name.startswith("H")]


def _collect_chains(model):
    return list(model.get_chains())


def _split_complex_chains(complex_model, ab_chains):
    antibody_complex: List = []
    antigen_complex: List = []
    # Match by chain *id*, not object identity: ``ab_model`` and ``abag_model``
    # are parsed from separate files, so their ``Chain`` objects are distinct
    # even when they represent the same physical chain.
    ab_set = set(ab_chains)
    for chain in _collect_chains(complex_model):
        if chain.id in ab_set:
            antibody_complex.append(chain)
        else:
            antigen_complex.append(chain)
    return antibody_complex, antigen_complex


def calculate_binding_and_surface(
    abag_file: str,
    ab_file: str,
    ag_file: str,
    distance_threshold: float = 4.5,
    neighbor_distance_threshold: float = 4.5,
    rasa_threshold: float = 0.25,
    dssp_executable: Optional[str] = None,
) -> Optional[dict]:
    parser = PDBParser(QUIET=True)
    abag_structure = parser.get_structure(os.path.basename(abag_file), abag_file)
    ab_structure = parser.get_structure(os.path.basename(ab_file), ab_file)
    ag_structure = parser.get_structure(os.path.basename(ag_file), ag_file)
    abag_model = abag_structure[0]
    ab_model = ab_structure[0]
    ag_model = ag_structure[0]

    antibody_chains = list(_collect_chains(ab_model))
    antigen_chains = list(_collect_chains(ag_model))
    antibody_chain_ids = [c.id for c in antibody_chains]
    antibody_complex_chains, antigen_complex_chains = _split_complex_chains(abag_model, antibody_chain_ids)

    antibody_residues = [res for chain in antibody_chains for res in chain]
    antigen_residues = [res for chain in antigen_chains for res in chain]
    antibody_complex_residues = [res for chain in antibody_complex_chains for res in chain]
    antigen_complex_residues = [res for chain in antigen_complex_chains for res in chain]

    if len(antibody_complex_residues) != len(antibody_residues):
        print(f"[length mismatch] antibody complex vs isolated ({abag_file})")
        return None
    if len(antigen_complex_residues) != len(antigen_residues):
        print(f"[length mismatch] antigen complex vs isolated ({abag_file})")
        return None

    abag_atoms = _filter_heavy_atoms(abag_model)
    ab_atoms = _filter_heavy_atoms(ab_model)
    ag_atoms = _filter_heavy_atoms(ag_model)
    abag_ns = NeighborSearch(abag_atoms)
    ab_ns = NeighborSearch(ab_atoms)
    ag_ns = NeighborSearch(ag_atoms)

    ab_labels = [0] * len(antibody_residues)
    ag_labels = [0] * len(antigen_residues)
    ab_surface_labels = [0] * len(antibody_residues)
    ag_surface_labels = [0] * len(antigen_residues)
    ab_surface_index: List[int] = []
    ag_surface_index: List[int] = []

    ab_dssp_kwargs = {"dssp": dssp_executable} if dssp_executable else {}
    ab_dssp = DSSP(ab_model, _preprocess_pdb_for_dssp(ab_file), **ab_dssp_kwargs)
    ag_dssp = DSSP(ag_model, _preprocess_pdb_for_dssp(ag_file), **ab_dssp_kwargs)

    ab_index = {res: i for i, res in enumerate(antibody_residues)}
    ag_index = {res: i for i, res in enumerate(antigen_residues)}
    ab_complex_index = {res: i for i, res in enumerate(antibody_complex_residues)}
    ag_complex_index = {res: i for i, res in enumerate(antigen_complex_residues)}

    ab_rasa = [np.nan] * len(antibody_residues)
    ag_rasa = [np.nan] * len(antigen_residues)
    for key in ab_dssp.keys():
        chain_id, res_id = key
        res = ab_model[chain_id][res_id]
        i = ab_index[res]
        rasa = ab_dssp[key][3]
        ab_rasa[i] = float(rasa)
        if rasa is not None and rasa >= rasa_threshold:
            ab_surface_labels[i] = 1
            ab_surface_index.append(i)
    for key in ag_dssp.keys():
        chain_id, res_id = key
        res = ag_model[chain_id][res_id]
        i = ag_index[res]
        rasa = ag_dssp[key][3]
        ag_rasa[i] = float(rasa)
        if rasa is not None and rasa >= rasa_threshold:
            ag_surface_labels[i] = 1
            ag_surface_index.append(i)

    ab_surface_residues = [antibody_residues[i] for i in ab_surface_index]
    ag_surface_residues = [antigen_residues[i] for i in ag_surface_index]
    ab_surface_index_map = {res: i for i, res in enumerate(ab_surface_residues)}
    ag_surface_index_map = {res: i for i, res in enumerate(ag_surface_residues)}

    def adjacency(residues, ns, distance):
        # Local index into *this* residue list (the caller may pass the
        # surface-residue subset, whose length differs from the full antibody /
        # antigen residue lists that ``ab_index`` / ``ag_index`` index).
        local_index = {res: i for i, res in enumerate(residues)}
        adj = []
        for i, res in enumerate(residues):
            row = [0] * len(residues)
            row[i] = 1
            for atom in res:
                for other in ns.search(atom.coord, distance):
                    other_res = other.get_parent()
                    if other_res in local_index:
                        row[local_index[other_res]] = 1
            adj.append(row)
        return np.asarray(adj, dtype=np.int8)

    ab_adjacency = adjacency(antibody_residues, ab_ns, neighbor_distance_threshold)
    ag_adjacency = adjacency(antigen_residues, ag_ns, neighbor_distance_threshold)

    ab_surface_adjacency = adjacency(ab_surface_residues, ab_ns, neighbor_distance_threshold)
    ag_surface_adjacency = adjacency(ag_surface_residues, ag_ns, neighbor_distance_threshold)

    ab_surface_labels_on_surface = [0] * len(ab_surface_residues)
    ag_surface_labels_on_surface = [0] * len(ag_surface_residues)

    # Mark interface residues.
    for i, ab_res in enumerate(antibody_complex_residues):
        for atom in ab_res:
            for other in abag_ns.search(atom.coord, distance_threshold):
                other_res = other.get_parent()
                if other_res in ag_complex_index:
                    ab_labels[i] = 1
                    ag_labels[ag_complex_index[other_res]] = 1
    # Restrict binding labels to surface residues.
    for i, ab_res in enumerate(antibody_complex_residues):
        if ab_labels[i] == 1:
            isolated_res = list(ab_index.keys())[i]
            if isolated_res in ab_surface_residues:
                ab_surface_labels_on_surface[ab_surface_index_map[isolated_res]] = 1
    for i, ag_res in enumerate(antigen_complex_residues):
        if ag_labels[i] == 1:
            isolated_res = list(ag_index.keys())[i]
            if isolated_res in ag_surface_residues:
                ag_surface_labels_on_surface[ag_surface_index_map[isolated_res]] = 1

    antibody_sequence = "".join(seq1(res.get_resname()) for res in antibody_residues)
    antigen_sequence = "".join(seq1(res.get_resname()) for res in antigen_residues)

    def _ca_or_mean_coord(res):
        """Calpha coordinate, falling back to the mean of all heavy atoms."""
        for atom in res:
            if atom.name == "CA":
                return atom.coord
        return np.mean([atom.coord for atom in res], axis=0)

    antibody_coords = np.asarray(
        [_ca_or_mean_coord(res) for res in antibody_residues],
        dtype=np.float64,
    )

    return {
        "protein_name": os.path.basename(abag_file).replace(".pdb", ""),
        "antibody_sequence": antibody_sequence,
        "antigen_sequence": antigen_sequence,
        "antibody_labels": np.asarray(ab_labels, dtype=np.int8),
        "antigen_labels": np.asarray(ag_labels, dtype=np.int8),
        "antibody_surface_labels": np.asarray(ab_surface_labels, dtype=np.int8),
        "antigen_surface_labels": np.asarray(ag_surface_labels, dtype=np.int8),
        "antibody_surface_index": np.asarray(ab_surface_index, dtype=np.int64),
        "antigen_surface_index": np.asarray(ag_surface_index, dtype=np.int64),
        "antibody_labels_onsurface": np.asarray(ab_surface_labels_on_surface, dtype=np.int8),
        "antigen_labels_onsurface": np.asarray(ag_surface_labels_on_surface, dtype=np.int8),
        "antibody_adjacency_labels": ab_adjacency,
        "antigen_adjacency_labels": ag_adjacency,
        "antibody_adjacency_labels_onsurface": ab_surface_adjacency,
        "antigen_adjacency_labels_onsurface": ag_surface_adjacency,
        "antibody_rasa": np.asarray(ab_rasa, dtype=np.float32),
        "antigen_rasa": np.asarray(ag_rasa, dtype=np.float32),
        "antibody_coords": antibody_coords,
    }


def process_split(
    complex_dir: str,
    ab_dir: str,
    ag_dir: str,
    output_path: str,
    distance_threshold: float = 4.5,
    neighbor_distance_threshold: float = 4.5,
    rasa_threshold: float = 0.25,
    dssp_executable: Optional[str] = None,
) -> int:
    complex_files = sorted(f for f in os.listdir(complex_dir) if f.endswith(".pdb"))
    ab_files = sorted(f for f in os.listdir(ab_dir) if f.endswith(".pdb"))
    ag_files = sorted(f for f in os.listdir(ag_dir) if f.endswith(".pdb"))

    if not (len(complex_files) == len(ab_files) == len(ag_files)):
        raise ValueError(
            f"Folder sizes disagree: complex={len(complex_files)}, "
            f"ab={len(ab_files)}, ag={len(ag_files)}"
        )

    results = []
    for complex_file, ab_file, ag_file in zip(complex_files, ab_files, ag_files):
        record = calculate_binding_and_surface(
            os.path.join(complex_dir, complex_file),
            os.path.join(ab_dir, ab_file),
            os.path.join(ag_dir, ag_file),
            distance_threshold=distance_threshold,
            neighbor_distance_threshold=neighbor_distance_threshold,
            rasa_threshold=rasa_threshold,
            dssp_executable=dssp_executable,
        )
        if record is not None:
            results.append(record)
    if not results:
        raise RuntimeError("No record produced; check the input folders.")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as handle:
        pickle.dump(results, handle)
    print(f"Saved {len(results)} records to {output_path}")
    return len(results)


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the PECAN structure-dataset pickle.")
    parser.add_argument("--complex-dir", required=True)
    parser.add_argument("--ab-dir", required=True)
    parser.add_argument("--ag-dir", required=True)
    parser.add_argument("--output", required=True, help="Output pickle path.")
    parser.add_argument("--distance-threshold", type=float, default=4.5,
                        help="Heavy-atom contact cutoff (Angstrom) used to flag interface residues.")
    parser.add_argument("--neighbor-threshold", type=float, default=4.5,
                        help="Neighbor-search cutoff used when building the residue adjacency matrix.")
    parser.add_argument("--rasa-threshold", type=float, default=0.25,
                        help="rASA cutoff that defines a surface residue (default 0.25 = 25 %; ablation uses 0.15, 0.20, 0.30).")
    parser.add_argument("--dssp-executable", default=None,
                        help="Optional DSSP binary path. Defaults to the BioPython auto-discovery.")
    args = parser.parse_args(argv)

    process_split(
        complex_dir=args.complex_dir,
        ab_dir=args.ab_dir,
        ag_dir=args.ag_dir,
        output_path=args.output,
        distance_threshold=args.distance_threshold,
        neighbor_distance_threshold=args.neighbor_threshold,
        rasa_threshold=args.rasa_threshold,
        dssp_executable=args.dssp_executable,
    )


if __name__ == "__main__":
    sys.exit(main())
