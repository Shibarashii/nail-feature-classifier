# dataset.py
from torchvision.datasets import ImageFolder
from pathlib import Path


def get_dataset(data_dir: str, split: str, transform=None):
    """
    Returns a PyTorch dataset for a given split ('train', 'valid', 'test').

    Args:
        data_dir (str or Path): Base data directory.
        split (str): One of 'train', 'valid', or 'test'.
        transform: torchvision transforms to apply.

    Returns:
        ImageFolder dataset.
    """
    data_path = Path(data_dir) / split
    dataset = ImageFolder(root=data_path, transform=transform)
    return dataset
