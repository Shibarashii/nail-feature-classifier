# dataloader.py
import os
from torch.utils.data import DataLoader


def get_dataloader(dataset, batch_size=32, shuffle=False, num_workers=None):
    """
    Returns a PyTorch dataloader.
    """
    if num_workers is None:
        num_workers = os.cpu_count()

    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers)
