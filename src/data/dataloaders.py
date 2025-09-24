# src/data/dataloaders.py
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from pathlib import Path

NUM_WORKERS = os.cpu_count()
ROOT_DIR = Path(__file__).resolve().parent.parent.parent  # src/data -> root
train_dir = ROOT_DIR / "data" / "Nail Classification" / "train"
val_dir = ROOT_DIR / "data" / "Nail Classification" / "valid"
test_dir = ROOT_DIR / "data" / "Nail Classification" / "test"


def create_dataloaders(
    train_transform: transforms.Compose,
    test_transform: transforms.Compose,
    batch_size: int,
    num_workers: int = NUM_WORKERS,
    test_run: bool = False
):
    """Creates training and testing DataLoaders.

    If test_run=True, only small subsets of the datasets are used
    to make training very fast.
    """

    # Turn image data into datasets
    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    val_data = datasets.ImageFolder(val_dir, transform=test_transform)
    test_data = datasets.ImageFolder(test_dir, transform=test_transform)

    # Use small subset for fast test run
    if test_run:
        train_data = Subset(train_data, range(min(100, len(train_data))))
        val_data = Subset(val_data, range(min(50, len(val_data))))
        test_data = Subset(test_data, range(min(50, len(test_data))))

    # Get class names
    class_names = train_data.dataset.classes if isinstance(
        train_data, Subset) else train_data.classes
    class_to_idx = train_data.dataset.class_to_idx if isinstance(
        train_data, Subset) else train_data.class_to_idx
    num_classes = len(class_names)

    # Turn images into data loaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, val_dataloader, test_dataloader, class_names, class_to_idx, num_classes
