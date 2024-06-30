import logging
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .custom_generator import Generator


log = logging.getLogger(__name__)


class GenDataset(Dataset):
    def __init__(
        self,
        dataset_len: int = -1,
        transform: Optional[Callable] = None,
        generator: Optional[Generator] = None,
    ):
        self.transform = transform
        self.generator = generator

        self.dataset_len = len(generator) if dataset_len == -1 else dataset_len

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        img, label = self.generator.get(index)

        if self.transform is not None:
            img = self.transform(img)

        return img, label


def imgcv_from_tensor(img_t: torch.Tensor) -> np.ndarray:
    img_np = img_t.numpy()
    img_np = np.transpose(img_np, (1, 2, 0))
    img_np = (img_np * 255).astype(np.uint8)
    return img_np
