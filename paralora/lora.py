"""LoRA adapter module adapted from the ProtT5 LoRA tutorial.

The implementation is identical to the released notebook, but the API is
re‑organised so the rest of the package can drive it from configuration
files and CLI arguments.

LoRA hyperparameters follow Section III-B-2 of the manuscript:

* rank  ``r = 4``
* scaling ``alpha = 8`` (effective scaling ``alpha / r = 2``)
* initialisation ``A ~ N(0, 0.01)``, ``B = 0``
* target projections ``q, v`` of every self-attention layer

The ``modify_with_lora`` helper walks ``transformer.named_modules()``
and replaces the selected ``nn.Linear`` children with :class:`LoRALinear`.
The base weight is shared with the original layer (no copy) so the
state-dict of a checkpoint matches the released ParaLoRA weights
byte-for-byte when only the LoRA parameters are stored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LoRAConfig:
    """Configuration for LoRA injection.

    Attributes:
        lora_rank: Bottleneck rank ``r``. Set to 0 to disable LoRA (will keep
            only the base linear layer).
        lora_alpha: LoRA scaling factor. Effective scaling during forward is
            ``alpha / r``.
        lora_init_scale: Standard deviation of the Gaussian used to sample
            ``lora_a``. Negative values trigger a "signed" initialisation
            where both ``lora_a`` and ``lora_b`` are sampled from the same
            scaled Gaussian (used by the ia3/lora scaling modes).
        lora_modules: Regular expression matching the attention module
            *names* that should receive LoRA. The released model uses
            ``".*SelfAttention|.*EncDecAttention"``.
        lora_layers: Regular expression matching the child module *names*
            inside each attention block. The released model uses ``"q|v"``;
            ``"q|k|v|o"`` reproduces the ablation of the same name.
        trainable_param_names: Regular expression matching parameter names
            that should remain trainable. Default keeps only the LoRA
            parameters and the layer-norm scales/biases.
        lora_scaling_rank: Rank of the optional LoRA-scaling (multi-LoRA)
            branch. ``0`` disables it.
    """

    lora_rank: int = 4
    lora_alpha: int = 8
    lora_init_scale: float = 0.01
    lora_modules: str = ".*SelfAttention|.*EncDecAttention"
    lora_layers: str = "q|v"
    trainable_param_names: str = ".*layer_norm.*|.*lora_[ab].*"
    lora_scaling_rank: int = 0

    def to_dict(self) -> Dict[str, object]:
        return {
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_init_scale": self.lora_init_scale,
            "lora_modules": self.lora_modules,
            "lora_layers": self.lora_layers,
            "trainable_param_names": self.trainable_param_names,
            "lora_scaling_rank": self.lora_scaling_rank,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "LoRAConfig":
        # Drop keys starting with an underscore so callers can attach free-form
        # documentation to the JSON (e.g. ``_note``) without breaking ``cls(**payload)``.
        valid_keys = set(cls.to_dict(cls()).keys())
        clean = {k: v for k, v in payload.items() if k in valid_keys}
        return cls(**clean)


class LoRALinear(nn.Module):
    """Linear layer augmented with a LoRA adapter.

    The original ``nn.Linear`` weight is *not* copied; we keep a reference
    so that loading a state-dict keyed by the original parameter name
    (``layer.q.weight``) still works. Only ``lora_a`` / ``lora_b`` and the
    optional ``multi_lora_*`` are stored separately.
    """

    def __init__(
        self,
        linear_layer: nn.Linear,
        rank: int,
        scaling_rank: int,
        init_scale: float,
        alpha: int = 8,
    ):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rank = rank
        self.scaling_rank = scaling_rank
        self.alpha = alpha
        self.scaling = alpha / rank if rank > 0 else 0.0
        # Share the base weight with the original linear layer.
        self.weight = linear_layer.weight
        self.bias = linear_layer.bias

        if self.rank > 0:
            self.lora_a = nn.Parameter(torch.randn(rank, linear_layer.in_features) * init_scale)
            if init_scale < 0:
                self.lora_b = nn.Parameter(
                    torch.randn(linear_layer.out_features, rank) * init_scale
                )
            else:
                self.lora_b = nn.Parameter(torch.zeros(linear_layer.out_features, rank))

        if self.scaling_rank:
            self.multi_lora_a = nn.Parameter(
                torch.ones(self.scaling_rank, linear_layer.in_features)
                + torch.randn(self.scaling_rank, linear_layer.in_features) * init_scale
            )
            if init_scale < 0:
                self.multi_lora_b = nn.Parameter(
                    torch.ones(linear_layer.out_features, self.scaling_rank)
                    + torch.randn(linear_layer.out_features, self.scaling_rank) * init_scale
                )
            else:
                self.multi_lora_b = nn.Parameter(
                    torch.ones(linear_layer.out_features, self.scaling_rank)
                )

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002 -- matches nn.Linear
        if self.scaling_rank == 1 and self.rank == 0:
            # ia3 / LoRA-scaling parsimonious implementation.
            if self.multi_lora_a.requires_grad:
                hidden = F.linear(input * self.multi_lora_a.flatten(), self.weight, self.bias)
            else:
                hidden = F.linear(input, self.weight, self.bias)
            if self.multi_lora_b.requires_grad:
                hidden = hidden * self.multi_lora_b.flatten()
            return hidden

        weight = self.weight
        if self.scaling_rank:
            weight = weight * torch.matmul(self.multi_lora_b, self.multi_lora_a) / self.scaling_rank
        if self.rank:
            weight = weight + self.scaling * torch.matmul(self.lora_b, self.lora_a)
        weight = weight.to(input.dtype)
        return F.linear(input, weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, rank={self.rank}, scaling_rank={self.scaling_rank}"
        )


def modify_with_lora(transformer: nn.Module, config: LoRAConfig) -> nn.Module:
    """Replace selected linear layers inside ``transformer`` with LoRA versions.

    The function only walks the module tree once and uses ``setattr`` so
    that subsequent code keeps the original attribute names (e.g.
    ``encoder.block[0].layer[0].SelfAttention.q``).
    """
    for m_name, module in dict(transformer.named_modules()).items():
        if not re.fullmatch(config.lora_modules, m_name):
            continue
        for c_name, layer in dict(module.named_children()).items():
            if not re.fullmatch(config.lora_layers, c_name):
                continue
            if not isinstance(layer, nn.Linear):
                raise TypeError(
                    f"LoRA can only be applied to nn.Linear; got {type(layer)} at "
                    f"{m_name}.{c_name}."
                )
            setattr(
                module,
                c_name,
                LoRALinear(
                    layer,
                    rank=config.lora_rank,
                    scaling_rank=config.lora_scaling_rank,
                    init_scale=config.lora_init_scale,
                    alpha=config.lora_alpha,
                ),
            )
    return transformer


def freeze_backbone_and_unfreeze_lora(model: nn.Module, config: LoRAConfig) -> int:
    """Freeze the entire encoder and unfreeze LoRA + layer-norm parameters.

    Returns the number of trainable parameters after freezing.
    """
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if re.fullmatch(config.trainable_param_names, name):
            param.requires_grad = True
    return sum(int(p.numel()) for p in model.parameters() if p.requires_grad)