"""Graph construction for the structure-based branch (ParaDG).

Each antibody is represented as a dual-view graph over the *same* residue node set:

* global view  : edges between Calpha pairs within ``distance_threshold`` (A' in Eq. 8)
* surface view : the same adjacency masked to surface residues
  (A_surf = M_surf * A' * M_surf^T in Eq. 14)

Two surface-handling modes are supported because they are compared in the
revision experiments:

``mask`` (default)
    Hard rASA threshold: only surface residues keep edges in the surface view.
``feature``
    No hard masking. The continuous rASA value is appended as an extra node
    feature and the surface view uses the full graph, letting the network learn
    the exposure/paratope correlation itself.
"""

from __future__ import annotations

import os
import pickle
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch_geometric.data import Data


DEFAULT_SPLIT_FILES = {
    "train": "pecan-paratope-train-all-paragraph.pkl",
    "val": "pecan-paratope-val-all-paragraph.pkl",
    "test": "pecan-paratope-test-all-paragraph.pkl",
}


def adjacency_to_edge_index(adj: np.ndarray) -> torch.Tensor:
    """Convert a dense adjacency matrix (0/1) to a ``[2, num_edges]`` index tensor."""
    rows, cols = np.nonzero(adj)
    if rows.size == 0:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.from_numpy(np.vstack([rows, cols]).astype(np.int64))


def coords_to_adjacency(coords: np.ndarray, distance_threshold: float) -> np.ndarray:
    """Build a binary Calpha-Calpha contact map from coordinates."""
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    return (dist <= distance_threshold).astype(np.float32)


def load_protein_data(
    data_dir: str,
    split_files: Optional[Dict[str, str]] = None,
    splits: Sequence[str] = ("train", "val", "test"),
    rasa_threshold: Optional[float] = None,
) -> Dict[str, List[dict]]:
    """Load and validate the pre-processed antibody pickles.

    Every record is expected to provide ``ab_feature`` (ProtT5/ParaLoRA residue
    embeddings) and ``antibody_labels`` (binary paratope labels). Structural
    fields are optional and only required by the structure-based branch.

    When ``rasa_threshold`` is given, the surface mask is re-derived from the
    continuous ``antibody_rasa`` field instead of using the stored surface
    labels. This keeps the rASA-cutoff ablation a pure load-time switch.
    """
    split_files = split_files or DEFAULT_SPLIT_FILES
    loaded: Dict[str, List[dict]] = {}

    for split in splits:
        path = os.path.join(data_dir, split_files[split])
        with open(path, "rb") as handle:
            raw = pickle.load(handle)

        records = []
        for item in raw:
            parsed = _parse_record(item, rasa_threshold=rasa_threshold)
            if parsed is not None:
                records.append(parsed)
        if not records:
            raise ValueError(f"No usable records found in {path}")
        surf_rate = float(np.mean([r["surface"].mean() for r in records]))
        print(
            f"[{split}] loaded {len(records)}/{len(raw)} complexes from {path} "
            f"(surface rate {surf_rate:.3f})"
        )
        loaded[split] = records

    return loaded


def _parse_record(item: dict, rasa_threshold: Optional[float] = None) -> Optional[dict]:
    features = np.asarray(item.get("ab_feature", []))
    labels = np.asarray(item.get("antibody_labels", []))
    if features.ndim != 2 or labels.ndim != 1 or len(features) == 0:
        return None
    if len(features) != len(labels):
        return None

    record = {
        "name": item.get("protein_name", ""),
        "features": features.astype(np.float32),
        "labels": labels.astype(np.float32),
    }

    rasa = item.get("antibody_rasa", None)
    if rasa is not None:
        # A handful of residues have no DSSP entry; treat them as fully buried
        # rather than letting NaN propagate into the graph / node features.
        rasa = np.nan_to_num(np.asarray(rasa, dtype=np.float32), nan=0.0)
    record["rasa"] = rasa

    if rasa_threshold is not None:
        if rasa is None:
            raise ValueError("rasa_threshold requires the 'antibody_rasa' field")
        surface_labels = (rasa >= rasa_threshold).astype(np.float32)
    else:
        surface_labels = item.get("antibody_surface_labels", None)
        if surface_labels is None:
            surface_index = np.asarray(item.get("antibody_surface_index", []), dtype=int)
            surface_labels = np.zeros(len(labels), dtype=np.float32)
            if surface_index.size:
                surface_labels[surface_index] = 1.0
    record["surface"] = np.asarray(surface_labels, dtype=np.float32)

    coords = item.get("antibody_coords", None)
    record["coords"] = None if coords is None else np.asarray(coords, dtype=np.float64)

    global_adj = item.get("antibody_adjacency_labels", None)
    record["global_adj"] = None if global_adj is None else np.asarray(global_adj)

    return record


def build_graph(
    record: dict,
    surface_mode: str = "mask",
    distance_threshold: float = 8.0,
    repair_isolated: bool = True,
    repair_radius: float = 8.0,
    graph_source: str = "auto",
) -> Data:
    """Build one dual-view PyG graph from a parsed antibody record.

    ``graph_source`` selects the global view:

    ``coords``
        Recompute the Calpha-Calpha contact map at ``distance_threshold``
        (paper Eq. 8). Required for the contact-cutoff ablation.
    ``stored``
        Use the pre-computed ``antibody_adjacency_labels`` shipped with the
        dataset (a 4.5 A heavy-atom neighbourhood), ignoring the coordinates.
    ``auto``
        Prefer coordinates and fall back to the stored adjacency.
    """
    if surface_mode not in ("mask", "feature"):
        raise ValueError(f"Unknown surface_mode: {surface_mode}")
    if graph_source not in ("auto", "coords", "stored"):
        raise ValueError(f"Unknown graph_source: {graph_source}")

    num_nodes = len(record["labels"])

    coords = record.get("coords")
    has_coords = coords is not None and len(coords) == num_nodes
    stored = record.get("global_adj")

    if graph_source == "coords":
        if not has_coords:
            raise ValueError("graph_source='coords' requires the 'antibody_coords' field")
        global_adj = coords_to_adjacency(coords, distance_threshold)
    elif graph_source == "stored":
        if stored is None:
            raise ValueError("graph_source='stored' requires 'antibody_adjacency_labels'")
        global_adj = (np.asarray(stored) > 0).astype(np.float32)
    elif has_coords:
        global_adj = coords_to_adjacency(coords, distance_threshold)
    elif stored is not None:
        global_adj = (np.asarray(stored) > 0).astype(np.float32)
    else:
        raise ValueError("Record provides neither Calpha coordinates nor a stored adjacency")

    global_adj = _with_self_loops(global_adj)

    if surface_mode == "mask":
        surface = record["surface"].astype(bool)
        mask = np.outer(surface, surface)
        surface_adj = global_adj * mask
        x = record["features"]
        if repair_isolated and record.get("coords") is not None:
            surface_adj = _repair_isolated_nodes(
                surface_adj, surface, record["coords"], repair_radius
            )
    else:
        rasa = record.get("rasa")
        if rasa is None:
            raise ValueError("surface_mode='feature' requires the 'antibody_rasa' field")
        x = np.concatenate([record["features"], rasa[:, None]], axis=1)
        surface_adj = global_adj.copy()
        surface = np.ones(num_nodes, dtype=bool)

    return Data(
        x=torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
        y=torch.from_numpy(record["labels"].astype(np.float32)),
        edge_index=adjacency_to_edge_index(global_adj),
        surface_edge_index=adjacency_to_edge_index(surface_adj),
        surface_mask=torch.from_numpy(surface),
        name=record.get("name", ""),
    )


def _with_self_loops(adj: np.ndarray) -> np.ndarray:
    adj = adj.copy()
    np.fill_diagonal(adj, 1.0)
    return adj


def _repair_isolated_nodes(
    surface_adj: np.ndarray,
    surface: np.ndarray,
    coords: np.ndarray,
    repair_radius: float,
) -> np.ndarray:
    """Re-attach isolated surface residues to their closest surface neighbour.

    A hard rASA cutoff can leave single residues without any surface-view edge,
    which blocks message passing in the GAT branch. Following the mitigation
    described in the response to reviewers, an isolated surface residue is
    connected to its nearest surface neighbour when that neighbour lies within
    ``repair_radius`` Angstrom.
    """
    adj = surface_adj.copy()
    surface_idx = np.flatnonzero(surface)
    if surface_idx.size < 2:
        return adj

    sub = coords[surface_idx]
    diff = sub[:, None, :] - sub[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(dist, np.inf)

    for local, node in enumerate(surface_idx):
        if adj[node].sum() > 1:  # self-loop already counts as one edge
            continue
        nearest = int(np.argmin(dist[local]))
        if dist[local, nearest] <= repair_radius:
            neighbour = int(surface_idx[nearest])
            adj[node, neighbour] = 1.0
            adj[neighbour, node] = 1.0
    return adj


class AntibodyGraphDataset:
    """Container that turns parsed records into PyG graphs and exposes loaders."""

    def __init__(
        self,
        records: Sequence[dict],
        surface_mode: str = "mask",
        distance_threshold: float = 8.0,
        repair_isolated: bool = True,
        repair_radius: float = 8.0,
        graph_source: str = "auto",
    ):
        self.graphs: List[Data] = []
        skipped = 0
        for record in records:
            try:
                self.graphs.append(
                    build_graph(
                        record,
                        surface_mode=surface_mode,
                        distance_threshold=distance_threshold,
                        repair_isolated=repair_isolated,
                        repair_radius=repair_radius,
                        graph_source=graph_source,
                    )
                )
            except Exception as exc:  # keep the pipeline alive on malformed entries
                skipped += 1
                print(f"Skipping {record.get('name', '<unnamed>')}: {exc}")
        print(f"Built {len(self.graphs)} graphs ({skipped} skipped)")

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Data:
        return self.graphs[idx]

    def summary(self) -> Dict[str, float]:
        num_nodes = sum(g.num_nodes for g in self.graphs)
        positives = sum(int(g.y.sum()) for g in self.graphs)
        return {
            "num_graphs": len(self.graphs),
            "num_residues": num_nodes,
            "num_paratope": positives,
            "positive_ratio": positives / max(num_nodes, 1),
        }
