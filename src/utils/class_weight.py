import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import torch


def get_class_weight(device: torch.device):
    train_counts = [753, 456, 612, 783, 643, 537, 336, 690, 657, 894]
    class_weights = compute_class_weight(class_weight='balanced',
                                         classes=np.arange(len(train_counts)),
                                         y=np.repeat(np.arange(len(train_counts)), train_counts))

    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    return weights_tensor
