"""
NationalSoilNet: ResNet-style architecture for multi-target soil property prediction.

Predicts 7 soil properties (pH, CEC, ESP, SOC, Ca, Mg, Na) from 64-band
AlphaEarth satellite embeddings using residual blocks with multi-head outputs.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn

DEFAULT_TARGETS = ["ph", "cec", "esp", "soc", "ca", "mg", "na"]


class ResidualBlock(nn.Module):
    """Residual block with skip connection for tabular data."""

    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.block(x)
        out = out + residual
        out = self.activation(out)
        out = self.dropout(out)
        return out


class NationalSoilNet(nn.Module):
    """
    ResNet-style model for multi-target soil property prediction.

    Architecture:
        Input(64) -> Linear(128) -> BN -> SiLU -> [ResBlocks x2] -> 7 output heads

    Each output head: Linear(hidden->hidden//2) -> SiLU -> Linear(hidden//2->1)
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
        num_res_blocks: int = 2,
        dropout: float = 0.2,
        target_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.target_names = target_names or DEFAULT_TARGETS
        self.num_targets = len(self.target_names)

        # Input projection
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
        )

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(num_res_blocks)]
        )

        # Multi-head output layers (one per target property)
        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            for name in self.target_names
        })

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass returning dict of {target_name: predictions}."""
        x = self.input_layer(x)
        x = self.res_blocks(x)
        return {name: head(x).squeeze(-1) for name, head in self.heads.items()}

    def forward_stacked(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning stacked tensor of shape [batch, num_targets]."""
        outputs = self.forward(x)
        return torch.stack([outputs[name] for name in self.target_names], dim=1)
