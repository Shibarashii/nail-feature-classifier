"""
Utility functions for gradual unfreezing of model layers.
"""
import torch.nn as nn
from typing import List


def unfreeze_layer_group(layer_group):
    """
    Unfreeze all parameters in a layer group.

    Parameters
    ----------
    layer_group : nn.Module or list of nn.Module
        The layer(s) to unfreeze.
    """
    if isinstance(layer_group, (list, tuple)):
        for layer in layer_group:
            for param in layer.parameters():
                param.requires_grad = True
    else:
        for param in layer_group.parameters():
            param.requires_grad = True


def get_unfreeze_schedule(num_epochs: int, num_groups: int) -> List[int]:
    """
    Create a schedule for when to unfreeze each layer group.

    Parameters
    ----------
    num_epochs : int
        Total number of training epochs.
    num_groups : int
        Number of layer groups (excluding the classifier, which is always unfrozen).

    Returns
    -------
    List[int]
        Epochs at which to unfreeze each group. Group 0 (classifier) is always unfrozen.

    Example
    -------
    >>> get_unfreeze_schedule(30, 8)
    [0, 4, 8, 12, 16, 20, 24, 28]  # Unfreeze a group every ~4 epochs
    """
    if num_groups <= 1:
        return [0]

    # Distribute unfreezing events across training
    # Start at epoch 0 (classifier already unfrozen)
    # Unfreeze remaining groups at regular intervals
    epochs_per_group = num_epochs / num_groups
    schedule = [int(i * epochs_per_group) for i in range(num_groups)]

    return schedule


def should_unfreeze_next_group(epoch: int, unfreeze_schedule: List[int], current_group: int) -> bool:
    """
    Determine if the next layer group should be unfrozen at the current epoch.

    Parameters
    ----------
    epoch : int
        Current epoch (1-indexed).
    unfreeze_schedule : List[int]
        Schedule of epochs for unfreezing.
    current_group : int
        Index of the current unfrozen group.

    Returns
    -------
    bool
        True if the next group should be unfrozen.
    """
    if current_group + 1 >= len(unfreeze_schedule):
        return False  # All groups already unfrozen

    return epoch >= unfreeze_schedule[current_group + 1]


def print_trainable_parameters(model: nn.Module):
    """
    Print the number and percentage of trainable parameters.

    Parameters
    ----------
    model : nn.Module
        The model to analyze.
    """
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_percentage = 100 * trainable_params / total_params

    print(
        f"Trainable params: {trainable_params:,} / {total_params:,} ({trainable_percentage:.2f}%)")
