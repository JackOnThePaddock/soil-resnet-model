"""PyTorch Dataset for soil property prediction."""

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class SoilDataset(Dataset):
    """PyTorch Dataset wrapping feature and target arrays for soil prediction."""

    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]
