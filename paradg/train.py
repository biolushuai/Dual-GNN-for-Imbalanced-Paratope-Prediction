#!/usr/bin/env python
"""Train and evaluate ParaDG on the PECAN paratope benchmark.

Examples
--------
Single run with the released configuration::

    python -m paradg.train --data-dir data/pecan --save-dir results/paradg

Reproduce the multi-seed statistics (Table IX of the revision)::

    python -m paradg.train --data-dir data/pecan --save-dir results/paradg \
        --seeds 0 1 2 3 4

Ablations requested by the reviewers::

    # rASA cutoff (15 / 20 / 25 / 30 %)
    python -m paradg.train --rasa-threshold 0.20

    # continuous rASA as a node feature instead of hard masking
    python -m paradg.train --surface-mode feature

    # graph distance threshold (6 / 8 / 10 / 12 A)
    python -m paradg.train --distance-threshold 10.0

    # oversampling ratio
    python -m paradg.train --sample-ratio 2.5
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader as GeometricDataLoader

from .data import AntibodyGraphDataset, load_protein_data
from .evaluate import aggregate_seeds, compute_metrics, predict, select_threshold
from .models import ParaDG, WeightedBCELoss
from .oversampling import StructurePreservingGraphOversampler

METRIC_NAMES = ["auc_roc", "auc_pr", "f1", "mcc", "precision", "recall", "specificity"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Trainer:
    """Epoch loop with early stopping, LR scheduling and metric logging."""

    def __init__(self, model, device, config, save_dir):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.criterion = WeightedBCELoss(
            pos_weight=config["loss"]["pos_weight"],
            neg_weight=config["loss"]["neg_weight"],
        )
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=config["training"]["lr_scheduler_factor"],
            patience=config["training"]["lr_scheduler_patience"],
            min_lr=config["training"]["min_lr"],
        )

        self.best_model_path = os.path.join(save_dir, "best_model.pth")
        # ``selection_metric`` 决定该初值是最小化还是最大化目标
        selection = config["training"].get("selection_metric", "val_loss")
        self.best_val_loss = (
            -float("inf") if selection in ("auc_pr", "auc_roc") else float("inf")
        )
        self.patience_counter = 0
        self.loss_window = 5
        self.val_loss_history: List[float] = []
        self.train_loss_history: List[float] = []
        self.history: List[dict] = []

    def moving_average(self, values: List[float]) -> float:
        window = values[-self.loss_window:]
        return sum(window) / len(window)

    def train_epoch(self, loader) -> float:
        self.model.train()
        total = 0.0
        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(batch.x, batch.edge_index, batch.surface_edge_index, batch.batch)
            syn_weight = self.config["loss"].get("syn_loss_weight", 1.0)
            node_weight = None
            if syn_weight != 1.0 and hasattr(batch, "syn_mask") and batch.syn_mask is not None:
                node_weight = torch.where(
                    batch.syn_mask.bool(),
                    torch.full_like(logits, float(syn_weight)),
                    torch.ones_like(logits),
                )
            loss = self.criterion(logits, batch.y, node_weight=node_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config["training"]["grad_clip"]
            )
            self.optimizer.step()
            total += float(loss.item())
        avg = total / max(len(loader), 1)
        self.train_loss_history.append(avg)
        return avg

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        total = 0.0
        scores, labels = [], []
        for batch in loader:
            batch = batch.to(self.device)
            logits = self.model(batch.x, batch.edge_index, batch.surface_edge_index, batch.batch)
            total += float(self.criterion(logits, batch.y).item())
            scores.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(batch.y.cpu().numpy())
        avg_loss = total / max(len(loader), 1)
        self.val_loss_history.append(avg_loss)
        return avg_loss, np.concatenate(scores), np.concatenate(labels)

    def fit(self, train_loader, val_loader, test_loader=None) -> dict:
        best_state = None
        for epoch in range(1, self.config["training"]["num_epochs"] + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_scores, val_labels = self.validate(val_loader)
            smoothed = self.moving_average(self.val_loss_history)
            self.scheduler.step(smoothed)

            val_metrics = compute_metrics(val_labels, val_scores, threshold=0.5)
            # 排序质量指标（与阈值无关），用于可选的模型选择准则
            try:
                val_metrics["auc_roc"] = float(
                    roc_auc_score(val_labels, val_scores)
                ) if len(np.unique(val_labels)) > 1 else 0.0
            except ValueError:
                val_metrics["auc_roc"] = 0.0
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_smoothed_loss": smoothed,
                "lr": self.optimizer.param_groups[0]["lr"],
                "val": val_metrics,
            }
            self.history.append(record)
            print(
                f"[epoch {epoch:03d}] train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_auc_pr={val_metrics['auc_pr']:.4f} "
                f"val_mcc={val_metrics['mcc']:.4f}"
            )

            # 模型选择准则：默认沿用 smoothed val loss（历史行为）；
            # 但对极度不平衡的 paratope 预测，val loss 会在早期过拟合后迅速爆炸，
            # 用排序质量指标（auc_pr / auc_roc）选 epoch 通常更稳。
            selection = self.config["training"].get("selection_metric", "val_loss")
            if selection in ("auc_pr", "auc_roc"):
                current = float(val_metrics[selection])
                improved = current > self.best_val_loss + 1e-6
            else:
                current = smoothed
                improved = smoothed < self.best_val_loss - 1e-6

            if improved:
                self.best_val_loss = current
                self.patience_counter = 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config["training"]["early_stopping_patience"]:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            torch.save(best_state, self.best_model_path)

        with open(os.path.join(self.save_dir, "history.json"), "w") as handle:
            json.dump(self.history, handle, indent=2)

        return self._finalise(val_loader, test_loader)

    def _finalise(self, val_loader, test_loader) -> dict:
        val_loss, val_scores, val_labels = self.validate(val_loader)
        threshold = select_threshold(val_labels, val_scores)
        val_metrics = compute_metrics(val_labels, val_scores, threshold)

        results = {
            "val": val_metrics,
            "threshold": threshold,
            "best_val_loss": self.best_val_loss,
        }

        if test_loader is not None:
            test_scores, test_labels = predict(self.model, test_loader, self.device)
            results["test"] = compute_metrics(test_labels, test_scores, threshold)
            np.savez(
                os.path.join(self.save_dir, "test_predictions.npz"),
                y_true=test_labels,
                y_score=test_scores,
            )

        with open(os.path.join(self.save_dir, "metrics.json"), "w") as handle:
            json.dump(results, handle, indent=2)
        return results


def build_datasets(args, config):
    records = load_protein_data(args.data_dir, rasa_threshold=config.get("rasa_threshold"))

    def make(split):
        return AntibodyGraphDataset(
            records[split],
            surface_mode=config["surface_mode"],
            distance_threshold=config["distance_threshold"],
            repair_isolated=config["repair_isolated"],
            repair_radius=config["repair_radius"],
            graph_source=config.get("graph_source", "auto"),
        )

    train = make("train")
    val = make("val")
    test = make("test")

    if config["oversampling"]["enabled"]:
        sampler = StructurePreservingGraphOversampler(
            k_neighbors=config["oversampling"]["k_neighbors"],
            target_pos_ratio=config["oversampling"]["target_pos_ratio"],
            sample_ratio=config["oversampling"]["sample_ratio"],
            metric=config["oversampling"]["metric"],
            synthetic_edges=config["oversampling"].get("synthetic_edges", "bidirectional"),
            lambda_max=config["oversampling"].get("lambda_max", 1.0),
        )
        train.graphs = sampler.oversample(train.graphs)

    print("train:", train.summary())
    print("val:  ", val.summary())
    print("test: ", test.summary())
    return train, val, test


def run_seed(seed: int, args, config, train, val, test, device) -> dict:
    set_seed(seed)
    save_dir = os.path.join(args.save_dir, f"seed_{seed}")
    # ``num_node_features`` already accounts for the extra rASA channel of
    # ``surface_mode='feature'`` (adjusted once in ``main``).
    model = ParaDG(**config["model"])

    loader_kwargs = {
        "batch_size": config["training"]["batch_size"],
        "num_workers": config["training"]["num_workers"],
    }
    train_loader = GeometricDataLoader(train, shuffle=True, **loader_kwargs)
    val_loader = GeometricDataLoader(val, shuffle=False, **loader_kwargs)
    test_loader = GeometricDataLoader(test, shuffle=False, **loader_kwargs)

    trainer = Trainer(model, device, config, save_dir)
    results = trainer.fit(train_loader, val_loader, test_loader)
    print(f"[seed {seed}] test:", json.dumps(results.get("test", {}), indent=2))
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train ParaDG for antibody paratope prediction")
    parser.add_argument("--data-dir", required=True, help="Directory with the PECAN pickles")
    parser.add_argument("--save-dir", default="results/paradg")
    parser.add_argument("--config", default="configs/paradg.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument(
        "--surface-mode",
        choices=["mask", "feature"],
        help="mask: hard rASA cutoff (default); feature: continuous rASA as node feature",
    )
    parser.add_argument("--distance-threshold", type=float, help="Calpha distance cutoff in Angstrom")
    parser.add_argument(
        "--repair-isolated",
        dest="repair_isolated",
        action="store_true",
        default=None,
        help="Re-attach isolated surface residues to their nearest surface neighbour",
    )
    parser.add_argument(
        "--no-repair-isolated",
        dest="repair_isolated",
        action="store_false",
        help="Disable the isolated-node repair (ablation)",
    )
    parser.add_argument("--repair-radius", type=float, help="Radius used to re-attach isolated surface residues")
    parser.add_argument("--sample-ratio", type=float, help="Oversampling ratio (overrides target_pos_ratio)")
    parser.add_argument("--target-pos-ratio", type=float,
                        help="Desired positive ratio used by Algorithm 1 (overridden by --sample-ratio). "
                             "Lower values inject fewer synthetic nodes and keep the training prior closer "
                             "to the natural test prior, reducing the train/inference prior shift.")
    parser.add_argument("--lambda-max", type=float,
                        help="Upper bound for the interpolation coefficient lambda of Algorithm 1 "
                             "(default 1.0 = manuscript). Values < 1 keep synthetic paratope nodes "
                             "close to their real parent instead of placing them at the midpoint "
                             "between parent and partner, which is a low-density point in a sparse "
                             "paratope region and therefore injects label noise.")
    parser.add_argument("--dropout", type=float,
                        help="Dropout inside the dual-view blocks (config default 0.5). "
                             "The manuscript does not state a value; with ~430-residue graphs and "
                             "only 195 training complexes, 0.5 makes the validation curve oscillate "
                             "wildly (AUC-PR swinging 0.1-0.6 across epochs) and triggers early "
                             "stopping after ~16 epochs.")
    parser.add_argument("--hidden-channels", type=int,
                        help="Width of the dual-view blocks (config default 256).")
    parser.add_argument("--patience", type=int,
                        help="Early-stopping patience in epochs (config default 15).")
    parser.add_argument("--num-layers", type=int, help="Number of dual-view blocks (Table VIII ablation)")
    parser.add_argument("--no-surface-view", action="store_true",
                        help="Drop the surface-view GAT branch (Table VII 'w/o Surface-view' arm)")
    parser.add_argument("--k-neighbors", type=int, help="Oversampling kNN neighbours (Table IX ablation)")
    parser.add_argument("--metric", choices=["euclidean", "cosine"],
                        help="Oversampling kNN distance metric (S2 ablation)")
    parser.add_argument("--synthetic-edges",
                        choices=["bidirectional", "incoming", "isolated"],
                        help="How synthetic paratope nodes are wired in. "
                             "'bidirectional' = Algorithm 1 as printed; 'incoming' = "
                             "synthetic nodes only receive from the parent's "
                             "neighbourhood, so real residues see the same graph as at "
                             "inference time; 'isolated' = self-loop only")
    parser.add_argument("--rasa-threshold", type=float,
                        help="Re-derive the surface mask from the continuous rASA at this "
                             "cutoff (S1 ablation); default keeps the stored surface labels")
    parser.add_argument("--use-stored-surface", action="store_true",
                        help="Ignore any rasa_threshold from the config and keep the surface "
                             "labels shipped in the pickles. Required for datasets that carry "
                             "no continuous 'antibody_rasa' field.")
    parser.add_argument("--graph-source", choices=["auto", "coords", "stored"],
                        help="Global-view source: 'coords' recomputes the Calpha contact map "
                             "at --distance-threshold (S3 ablation), 'stored' uses the "
                             "pre-computed 4.5 A heavy-atom adjacency shipped with the dataset")
    parser.add_argument("--pos-weight", type=float, help="Positive class weight of the weighted CE loss")
    parser.add_argument("--neg-weight", type=float, help="Negative class weight of the weighted CE loss")
    parser.add_argument("--syn-loss-weight", type=float,
                        help="Loss weight applied to over-sampled synthetic nodes "
                             "(1.0 = manuscript, <1.0 down-weights the interpolants)")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    with open(args.config) as handle:
        config = json.load(handle)

    if args.surface_mode:
        config["surface_mode"] = args.surface_mode
    if args.distance_threshold is not None:
        config["distance_threshold"] = args.distance_threshold
    if args.repair_radius is not None:
        config["repair_radius"] = args.repair_radius
    if args.repair_isolated is not None:
        config["repair_isolated"] = args.repair_isolated
    if args.num_layers is not None:
        config["model"]["num_layers"] = args.num_layers
    if args.no_surface_view:
        config["model"]["use_surface_view"] = False
    if args.k_neighbors is not None:
        config["oversampling"]["k_neighbors"] = args.k_neighbors
    if args.metric is not None:
        config["oversampling"]["metric"] = args.metric
    if args.synthetic_edges is not None:
        config["oversampling"]["synthetic_edges"] = args.synthetic_edges
    if args.rasa_threshold is not None:
        config["rasa_threshold"] = args.rasa_threshold
    if args.use_stored_surface:
        config["rasa_threshold"] = None
    if args.graph_source is not None:
        config["graph_source"] = args.graph_source
    if args.sample_ratio is not None:
        config["oversampling"]["sample_ratio"] = args.sample_ratio
    if args.target_pos_ratio is not None:
        config["oversampling"]["target_pos_ratio"] = args.target_pos_ratio
        # a single target ratio must not coexist with an explicit sample-ratio override
        config["oversampling"]["sample_ratio"] = None
    if args.lambda_max is not None:
        config["oversampling"]["lambda_max"] = args.lambda_max
    if args.pos_weight is not None:
        config["loss"]["pos_weight"] = args.pos_weight
    if args.dropout is not None:
        config["model"]["dropout"] = args.dropout
    if args.hidden_channels is not None:
        config["model"]["hidden_channels"] = args.hidden_channels
    if args.patience is not None:
        config["training"]["early_stopping_patience"] = args.patience
    if args.neg_weight is not None:
        config["loss"]["neg_weight"] = args.neg_weight
    if args.syn_loss_weight is not None:
        config["loss"]["syn_loss_weight"] = args.syn_loss_weight
    if args.epochs:
        config["training"]["num_epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate

    if config["surface_mode"] == "feature":
        config["model"]["num_node_features"] += 1

    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    train, val, test = build_datasets(args, config)

    per_seed = []
    for seed in args.seeds:
        results = run_seed(seed, args, config, train, val, test, device)
        per_seed.append(results.get("test", results["val"]))

    if len(args.seeds) > 1:
        summary = aggregate_seeds(per_seed, METRIC_NAMES)
        print("\nMulti-seed summary (mean +/- 95% CI):")
        for name, stat in summary.items():
            print(f"  {name:12s} {stat['mean']:.4f} +/- {stat['ci95']:.4f}  (n={stat['n']})")
        with open(os.path.join(args.save_dir, "multi_seed_summary.json"), "w") as handle:
            json.dump({"summary": summary, "per_seed": per_seed}, handle, indent=2)


if __name__ == "__main__":
    main()
