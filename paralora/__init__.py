"""ParaLoRA: sequence-based paratope prediction.

This package wraps a Hugging Face T5 encoder with parameter-efficient
adapter training (LoRA) for residue-level binary classification.

Public modules:
    ``lora``         -- LoRA adapter implementation (LoRAConfig, LoRALinear, modify_with_lora)
    ``model``        -- T5EncoderForTokenClassification and ``build_pt5_classifier``
    ``data``         -- sequence/labelled dataset construction
    ``trainer``      -- ``train_per_residue`` (HuggingFace Trainer + DeepSpeed wrapper)
    ``evaluate``     -- inference helpers and metric computation
"""

from .lora import LoRAConfig, LoRALinear, modify_with_lora
from .model import ClassConfig, T5EncoderForTokenClassification, build_pt5_classifier
from .data import create_dataset, prepare_split
from .trainer import train_per_residue, set_seeds
from .evaluate import evaluate_paratope, extract_embeddings

__all__ = [
    "LoRAConfig",
    "LoRALinear",
    "modify_with_lora",
    "ClassConfig",
    "T5EncoderForTokenClassification",
    "build_pt5_classifier",
    "create_dataset",
    "prepare_split",
    "train_per_residue",
    "set_seeds",
    "evaluate_paratope",
    "extract_embeddings",
]