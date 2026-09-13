import json
import os
import time

import torch
from torch_geometric.loader import DataLoader

from paradg.data import AntibodyGraphDataset, load_protein_data
from paradg.models import ParaDG, WeightedBCELoss


def main():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")

    config = json.load(open("configs/paradg.json"))
    config["loss"]["pos_weight"] = 1.0
    config["loss"]["neg_weight"] = 1.0

    print("\n=== Loading ParaDG data ===")
    t0 = time.time()
    # Use absolute WSL path directly to avoid DrvFs cache flakiness
    data_dir = "/mnt/d/ProjectsData/ParaLoRADG"
    records = load_protein_data(data_dir, splits=("train", "val", "test"))
    print(f"Load time: {time.time() - t0:.1f}s")

    print("\n=== Building PyG graphs ===")
    t0 = time.time()
    train_ds = AntibodyGraphDataset(records["train"], surface_mode="mask", distance_threshold=8.0)
    val_ds = AntibodyGraphDataset(records["val"], surface_mode="mask", distance_threshold=8.0)
    test_ds = AntibodyGraphDataset(records["test"], surface_mode="mask", distance_threshold=8.0)
    print(f"Graph build time: {time.time() - t0:.1f}s")

    print("\n=== Dataset summary ===")
    print("train:", train_ds.summary())
    print("val:  ", val_ds.summary())
    print("test: ", test_ds.summary())

    print("\n=== Building model ===")
    model = ParaDG(**config["model"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {n_params:,}; Trainable: {n_trainable:,}")

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

    print("\n=== 1-batch sanity check ===")
    batch = next(iter(train_loader)).to(device)
    print(f"Batch: x={batch.x.shape} edge_index={batch.edge_index.shape} surf_edge_index={batch.surface_edge_index.shape} y={batch.y.shape} batch={batch.batch.shape}")
    t0 = time.time()
    logits = model(batch.x, batch.edge_index, batch.surface_edge_index, batch.batch)
    print(f"Forward: logits.shape={logits.shape} time={time.time()-t0:.3f}s")
    loss_fn = WeightedBCELoss(pos_weight=1.0, neg_weight=1.0)
    loss = loss_fn(logits, batch.y)
    print(f"Loss: {loss.item():.4f}")
    loss.backward()
    print("Backward OK")

    print("\n=== 1 full epoch train ===")
    optim = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    model.train()
    t0 = time.time()
    total = 0.0
    n_batch = 0
    for batch in train_loader:
        batch = batch.to(device)
        optim.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.surface_edge_index, batch.batch)
        loss = loss_fn(logits, batch.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optim.step()
        total += float(loss.item())
        n_batch += 1
    print(f"1 epoch: avg_loss={total/n_batch:.4f} time={time.time()-t0:.1f}s ({n_batch} batches)")


if __name__ == "__main__":
    main()
