"""PyTorch Dataset for soil property prediction."""

from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class SoilDataset(Dataset):
    """PyTorch Dataset wrapping feature and target arrays for soil prediction."""

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
    ):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.sample_weights: Optional[torch.Tensor] = None
        if sample_weights is not None:
            if len(sample_weights) != len(features):
                raise ValueError(
                    f"sample_weights length {len(sample_weights)} != n_samples {len(features)}"
                )
            self.sample_weights = torch.FloatTensor(sample_weights)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(
        self,
        idx: int,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if self.sample_weights is None:
            return self.features[idx], self.targets[idx]
        return self.features[idx], self.targets[idx], self.sample_weights[idx]
