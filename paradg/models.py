from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "elu":
        return nn.ELU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


class FeatureProjection(nn.Module):
    """BN -> MLP projection that maps raw residue embeddings into the latent space.

    Implements Eq. (10)-(11):  H_hat = BN(H0);  X0 = W2 * act(W1 * H_hat + b1) + b2
    """

    def __init__(self, in_dim: int, hidden: int, dropout: float, activation: nn.Module):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden * 4),
            nn.BatchNorm1d(hidden * 4),
            activation,
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, hidden * 2),
            nn.BatchNorm1d(hidden * 2),
            activation,
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.bn(x))


class ParaDG(nn.Module):
    """Dual-view graph network for residue-level paratope prediction.

    Args:
        num_node_features: Dimension of the input residue embeddings (1024 for ProtT5).
        hidden_channels: Width of every hidden layer.
        num_layers: Number of dual-view blocks (3 in the manuscript).
        dropout: Dropout probability.
        gat_heads: Number of attention heads in the surface-view GAT branch.
        activation: Non-linearity, ``"relu"`` (manuscript) or ``"elu"`` (released model).
        classifier: ``"mlp"`` (released model) or ``"linear"`` (Eq. 18 in the manuscript).
        use_surface_view: When False the surface-view GAT branch and the
            cross-view fusion are removed, leaving a single-view GCN stack.
            This is the ``w/o Surface-view`` arm of Table VII.
    """

    def __init__(
        self,
        num_node_features: int = 1024,
        hidden_channels: int = 256,
        num_layers: int = 3,
        dropout: float = 0.5,
        gat_heads: int = 4,
        activation: str = "elu",
        classifier: str = "mlp",
        use_surface_view: bool = True,
    ):
        super().__init__()
        if hidden_channels % gat_heads != 0:
            if use_surface_view:
                raise ValueError("hidden_channels must be divisible by gat_heads")

        self.num_layers = num_layers
        self.dropout = dropout
        self.act = _activation(activation)
        self.use_surface_view = bool(use_surface_view)

        self.feature_proj = FeatureProjection(
            num_node_features, hidden_channels, dropout, self.act
        )

        # Global-view GCN branch (Eq. 12) and surface-view GAT branch (Eq. 15).
        self.global_convs = nn.ModuleList()
        self.global_norms = nn.ModuleList()
        self.surface_convs = nn.ModuleList()
        self.surface_norms = nn.ModuleList()

        for layer in range(num_layers):
            if self.use_surface_view:
                in_dim = hidden_channels if layer == 0 else hidden_channels * 2
            else:
                # single view: each block consumes the previous block output only
                in_dim = hidden_channels
            self.global_convs.append(GCNConv(in_dim, hidden_channels))
            self.global_norms.append(nn.BatchNorm1d(hidden_channels))
            if self.use_surface_view:
                self.surface_convs.append(
                    GATConv(in_dim, hidden_channels // gat_heads, heads=gat_heads, concat=True)
                )
                self.surface_norms.append(nn.BatchNorm1d(hidden_channels))

        # Cross-level fusion (Eq. 16) and global context pooling.
        self.fusion = nn.ModuleList()
        self.context = nn.ModuleList()
        for _ in range(num_layers):
            if self.use_surface_view:
                fusion_in = hidden_channels * 2
            else:
                fusion_in = hidden_channels
            self.fusion.append(
                nn.Sequential(
                    nn.Linear(fusion_in, hidden_channels),
                    nn.BatchNorm1d(hidden_channels),
                    self.act,
                    nn.Dropout(dropout),
                )
            )
            self.context.append(
                nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.BatchNorm1d(hidden_channels),
                    self.act,
                    nn.Dropout(dropout),
                )
            )

        # Final classifier over the concatenation of all layer outputs (Eq. 17-18).
        final_dim = hidden_channels * num_layers
        if classifier == "linear":
            self.node_pred = nn.Linear(final_dim, 1)
        elif classifier == "mlp":
            dims = [hidden_channels * 2, hidden_channels, hidden_channels // 2]
            layers = []
            current = final_dim
            for dim in dims:
                layers += [
                    nn.Linear(current, dim),
                    nn.BatchNorm1d(dim),
                    _activation(activation),
                    nn.Dropout(dropout),
                ]
                current = dim
            layers.append(nn.Linear(current, 1))
            self.node_pred = nn.Sequential(*layers)
        else:
            raise ValueError(f"Unsupported classifier head: {classifier}")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        surface_edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Run the dual-view network.

        Args:
            x: Node features ``[num_nodes, num_node_features]``.
            edge_index: Global-view graph connectivity ``[2, num_edges]``.
            surface_edge_index: Surface-view (rASA-masked) connectivity ``[2, num_edges]``.
            batch: Graph assignment vector ``[num_nodes]``.

        Returns:
            Per-residue logits of shape ``[num_nodes]``.
        """
        x = self.feature_proj(x)

        global_x = x
        surface_x = x
        layer_outputs = []
        previous_fused: Optional[torch.Tensor] = None

        for layer in range(self.num_layers):
            # Eq. (13): each branch consumes the concatenation of both branch outputs.
            if self.use_surface_view:
                global_in = torch.cat([global_x, surface_x], dim=-1) if layer > 0 else global_x
            else:
                global_in = global_x
            global_x = self.global_convs[layer](global_in, edge_index)
            global_x = self.act(self.global_norms[layer](global_x))
            global_x = F.dropout(global_x, p=self.dropout, training=self.training)

            if self.use_surface_view:
                surface_in = torch.cat([surface_x, global_x], dim=-1) if layer > 0 else surface_x
                surface_x = self.surface_convs[layer](surface_in, surface_edge_index)
                surface_x = self.act(self.surface_norms[layer](surface_x))
                surface_x = F.dropout(surface_x, p=self.dropout, training=self.training)

            # Eq. (16): fuse the two views, then add a graph-level context vector.
            if self.use_surface_view:
                fused = self.fusion[layer](torch.cat([global_x, surface_x], dim=-1))
            else:
                fused = self.fusion[layer](global_x)
            context = self.context[layer](global_mean_pool(fused, batch))[batch]
            fused = fused + context

            # Residual path: each branch is residual-connected to the previous
            # fused representation, which stabilises deep dual-view stacking.
            if previous_fused is not None:
                global_x = global_x + previous_fused
                if self.use_surface_view:
                    surface_x = surface_x + previous_fused
            previous_fused = fused

            layer_outputs.append(fused)

        x = torch.cat(layer_outputs, dim=-1)
        return self.node_pred(x).squeeze(-1)


class WeightedBCELoss(nn.Module):
    """Weighted binary cross-entropy with logits (Eq. 5 of the manuscript).

    L = -1/|Omega| sum_i [ w_p * y_i * log p_i + w_n * (1 - y_i) * log (1 - p_i) ]

    Args:
        pos_weight: Weight applied to paratope (positive) residues.
        neg_weight: Weight applied to non-paratope (negative) residues.
    """

    def __init__(self, pos_weight: float = 5.0, neg_weight: float = 1.0):
        super().__init__()
        self.pos_weight = float(pos_weight)
        self.neg_weight = float(neg_weight)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        node_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        weight = torch.full_like(targets, self.neg_weight)
        weight[targets == 1] = self.pos_weight
        if node_weight is not None:
            weight = weight * node_weight.to(weight.dtype)
        return F.binary_cross_entropy_with_logits(logits, targets, weight=weight, reduction="mean")

    def extra_repr(self) -> str:
        return f"pos_weight={self.pos_weight}, neg_weight={self.neg_weight}"
