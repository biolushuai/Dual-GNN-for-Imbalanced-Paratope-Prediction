from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import T5Config, T5EncoderModel, T5Tokenizer
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.t5.modeling_t5 import T5PreTrainedModel, T5Stack

from .lora import LoRAConfig, freeze_backbone_and_unfreeze_lora, modify_with_lora


@dataclass
class ClassConfig:
    """Hyperparameters for the token classification head."""

    dropout_rate: float = 0.2
    num_labels: int = 2

    def to_dict(self) -> Dict[str, object]:
        return {"dropout_rate": self.dropout_rate, "num_labels": self.num_labels}


class T5EncoderForTokenClassification(T5PreTrainedModel):
    """T5 encoder wrapped with a single linear token classifier.

    Mirrors the structure of ``transformers.T5ForTokenClassification`` but
    drops the decoder side. Loss is masked with ``-100`` on padded tokens
    so it is naturally compatible with
    ``transformers.DataCollatorForTokenClassification``.
    """

    def __init__(self, config: T5Config, class_config: ClassConfig):
        super().__init__(config)
        self.num_labels = class_config.num_labels
        self.config = config

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        self.dropout = nn.Dropout(class_config.dropout_rate)
        self.classifier = nn.Linear(config.hidden_size, class_config.num_labels)

        # Initialize weights and apply final processing.
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        loss_weights: Optional[torch.Tensor] = None,
    ) -> TokenClassifierOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = self.dropout(outputs[0])
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            active_mask = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_mask,
                labels.view(-1),
                torch.tensor(-100, dtype=labels.dtype, device=labels.device),
            )
            valid_logits = active_logits[active_labels != -100]
            valid_labels = active_labels[active_labels != -100].long()

            if loss_weights is None:
                loss_weights = torch.tensor([1.0, 1.0], device=valid_logits.device)
            else:
                loss_weights = loss_weights.to(valid_logits.device)

            loss_fct = CrossEntropyLoss(weight=loss_weights)
            # Cast logits to match label dtype to avoid fp16/fp32 mismatch
            loss = loss_fct(valid_logits.float(), valid_labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def load_prot_t5_encoder(
    model_name_or_path: str,
    half_precision: bool,
    cache_dir: Optional[str] = None,
) -> Tuple[T5EncoderModel, T5Tokenizer]:
    """Load a ProtT5 encoder and tokenizer. Half precision is GPU-only."""
    if half_precision and not torch.cuda.is_available():
        raise RuntimeError("Half precision requires a CUDA-capable GPU.")

    tokenizer = T5Tokenizer.from_pretrained(
        model_name_or_path,
        do_lower_case=False,
        cache_dir=cache_dir,
    )
    dtype = torch.float16 if half_precision else None
    model = T5EncoderModel.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        cache_dir=cache_dir,
    )
    if half_precision:
        model = model.to(torch.device("cuda"))
    return model, tokenizer


def build_pt5_classifier(
    config_path: Optional[str] = None,
    config: Optional[Dict] = None,
    loss_weights: Optional[torch.Tensor] = None,
    cache_dir: Optional[str] = None,
) -> Tuple[T5EncoderForTokenClassification, T5Tokenizer, LoRAConfig, int]:
    """Build a T5EncoderForTokenClassification + LoRA + freeze pipeline.

    Args:
        config_path: Path to a JSON config file with sections ``model_*``,
            ``lora`` and ``loss`` (see ``configs/paralora.json``).
        config: In-memory config dictionary. Mutually exclusive with
            ``config_path``.
        loss_weights: Optional ``[neg, pos]`` weight tensor passed to
            ``CrossEntropyLoss``. If ``None``, weights are derived from
            ``config["loss"]["pos_weight"]``.
        cache_dir: Optional Hugging Face cache directory for the backbone
            checkpoint.

    Returns:
        ``(model, tokenizer, lora_config, trainable_params)``
    """
    if (config_path is None) == (config is None):
        raise ValueError("Provide exactly one of `config_path` or `config`.")
    if config is None:
        with open(config_path) as handle:
            config = json.load(handle)

    model_name_or_path = config["model_name_or_path"]
    half_precision = bool(config.get("half_precision", False))
    encoder, tokenizer = load_prot_t5_encoder(
        model_name_or_path,
        half_precision=half_precision,
        cache_dir=cache_dir,
    )

    classifier_model = T5EncoderForTokenClassification(
        encoder.config,
        ClassConfig(num_labels=config["num_labels"]),
    )
    # Re-use the encoder weights without copying.
    classifier_model.shared = encoder.shared
    classifier_model.encoder = encoder.encoder
    del encoder

    lora_config = LoRAConfig.from_dict(config["lora"])
    modify_with_lora(classifier_model, lora_config)

    trainable = freeze_backbone_and_unfreeze_lora(classifier_model, lora_config)

    if loss_weights is None:
        pos = float(config["loss"]["pos_weight"])
        neg = float(config["loss"]["neg_weight"])
        loss_weights = torch.tensor([neg, pos], dtype=torch.float32)
    classifier_model.loss_weights = loss_weights

    if half_precision:
        classifier_model = classifier_model.half()

    return classifier_model, tokenizer, lora_config, trainable


def save_trainable_parameters(model: nn.Module, path: str) -> int:
    """Save only the parameters with ``requires_grad=True``.

    This keeps the released checkpoint small (~3 MB instead of ~3 GB).
    Returns the number of parameters saved.
    """
    state = {name: param.detach().cpu() for name, param in model.named_parameters() if param.requires_grad}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(state, path)
    return sum(t.numel() for t in state.values())


def load_trainable_parameters(model: nn.Module, path: str, strict: bool = False) -> int:
    """Load trainable-only weights saved by :func:`save_trainable_parameters`."""
    state = torch.load(path, map_location="cpu")
    target = model
    if isinstance(target, T5EncoderForTokenClassification) is False:
        # Accept both bare modules and `torch.nn.parallel`-wrapped ones.
        target = target.module if hasattr(target, "module") else target
    for name, param in target.named_parameters():
        if name in state:
            param.data.copy_(state[name].to(param.dtype))
        elif strict:
            raise KeyError(f"Missing parameter {name} in checkpoint.")
    return sum(t.numel() for t in state.values())
