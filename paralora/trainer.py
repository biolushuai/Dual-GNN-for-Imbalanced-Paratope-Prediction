from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .data import Split, create_dataset, prepare_split
from .model import (
    build_pt5_classifier,
    load_trainable_parameters,
    save_trainable_parameters,
)


DEEPSPEED_CONFIG = {
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1,
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto",
        },
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto",
        },
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "cpu", "pin_memory": True},
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": False,
}


def set_seeds(seed: int) -> None:
    """Seed every random source touched by the trainer."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    set_seed(seed)


def _to_dataset(split: Split, tokenizer, max_length: int) -> Dataset:
    return create_dataset(tokenizer, split.sequences, split.labels, max_length=max_length)


def _build_training_args(
    output_dir: str,
    train_cfg: Dict,
    use_deepspeed: bool,
    deepspeed_cfg_path: Optional[str] = None,
) -> TrainingArguments:
    if deepspeed_cfg_path is not None:
        ds_arg: Optional[object] = deepspeed_cfg_path
    elif use_deepspeed:
        ds_arg = DEEPSPEED_CONFIG
    else:
        ds_arg = None

    return TrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        eval_steps=train_cfg.get("eval_steps", 50),
        logging_strategy="epoch",
        save_strategy="no",
        learning_rate=train_cfg["lr"],
        per_device_train_batch_size=train_cfg["batch"],
        per_device_eval_batch_size=train_cfg["batch"],
        gradient_accumulation_steps=train_cfg["accum"],
        num_train_epochs=train_cfg["epochs"],
        seed=train_cfg["seed"],
        fp16=bool(train_cfg.get("fp16", False)),
        bf16=bool(train_cfg.get("bf16", False)),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.0),
        weight_decay=train_cfg.get("weight_decay", 0.0),
        deepspeed=ds_arg,
        report_to=[],
        disable_tqdm=False,
        dataloader_drop_last=False,
        # Disable AMP mixed precision: peft+accelerate version combo on RTX 4090 + torch 2.3
        # raises "Attempting to unscale FP16 gradients" when LoRA params are cast.
        # Trade-off: ~2x slower training in exchange for stable gradients.
        fp16_full_eval=False,
    )


def train_per_residue(
    train_split: Split,
    valid_split: Split,
    config: Dict,
    output_dir: str = "./runs/paralora",
    deepspeed: Optional[bool] = None,
    deepspeed_config_path: Optional[str] = None,
    return_trainer: bool = False,
) -> Tuple:
    """Train the LoRA-augmented T5 classifier on paratope labels.

    Args:
        train_split: Training set (sequence / label / mask).
        valid_split: Validation set.
        config: Full ParaLoRA configuration.
        output_dir: Where the trainer writes intermediate files.
        deepspeed: Override the JSON config flag. ``None`` falls back to
            ``config["training"]["deepspeed"]``.
        deepspeed_config_path: Optional path to an external DeepSpeed JSON.

    Returns:
        ``(tokenizer, model, log_history, trainer)`` -- ``trainer`` is
        always returned so callers can run predictions without
        re-loading the model.
    """
    if deepspeed is None:
        deepspeed = bool(config["training"].get("deepspeed", False))

    set_seeds(config["training"]["seed"])

    model, tokenizer, _, trainable = build_pt5_classifier(config=config)
    print(f"Trainable parameters: {trainable:,}")

    train_set = _to_dataset(
        prepare_split(train_split, cdr_mask=bool(config["data"].get("cdr_masked_train", True))),
        tokenizer,
        max_length=int(config["data"]["max_length"]),
    )
    valid_set = _to_dataset(
        prepare_split(valid_split, cdr_mask=bool(config["data"].get("cdr_masked_eval", False))),
        tokenizer,
        max_length=int(config["data"]["max_length"]),
    )

    training_args = _build_training_args(
        output_dir=output_dir,
        train_cfg=config["training"],
        use_deepspeed=deepspeed,
        deepspeed_cfg_path=deepspeed_config_path,
    )
    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=valid_set,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()

    if return_trainer:
        return tokenizer, model, trainer.state.log_history, trainer
    return tokenizer, model, trainer.state.log_history, trainer


# --------------------------------------------------------------------------------------
# CLI wrapper
# --------------------------------------------------------------------------------------


def _load_split_from_args(path: str, fmt: Optional[str]) -> Split:
    from .data import load_split
    return load_split(path, format=fmt)


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Train the ParaLoRA sequence branch")
    parser.add_argument("--config", default="configs/paralora.json")
    parser.add_argument("--train-data", required=True, help="Path to training split")
    parser.add_argument("--valid-data", required=True, help="Path to validation split")
    parser.add_argument("--train-format", choices=["csv", "paraperd", "pkl"], default=None)
    parser.add_argument("--valid-format", choices=["csv", "paraperd", "pkl"], default=None)
    parser.add_argument("--output-dir", default="./runs/paralora")
    parser.add_argument("--checkpoint-out", default="./runs/paralora/trainable_params.pt")
    parser.add_argument("--no-deepspeed", action="store_true")
    parser.add_argument("--deepspeed-config", default=None)
    args = parser.parse_args(argv)

    with open(args.config) as handle:
        config = json.load(handle)

    train_split = _load_split_from_args(args.train_data, args.train_format)
    valid_split = _load_split_from_args(args.valid_data, args.valid_format)

    tokenizer, model, history, trainer = train_per_residue(
        train_split=train_split,
        valid_split=valid_split,
        config=config,
        output_dir=args.output_dir,
        deepspeed=(not args.no_deepspeed) and bool(config["training"].get("deepspeed", False)),
        deepspeed_config_path=args.deepspeed_config,
    )
    with open(os.path.join(args.output_dir, "history.json"), "w") as handle:
        json.dump(history, handle, indent=2, default=float)

    saved = save_trainable_parameters(model, args.checkpoint_out)
    print(f"Saved {saved:,} trainable parameters to {args.checkpoint_out}")


if __name__ == "__main__":
    main()
