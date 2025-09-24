import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): Random seed value (default: 42)
    """
    # Python & NumPy
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set hash seed for full reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"[INFO] Seed set to {seed}")
