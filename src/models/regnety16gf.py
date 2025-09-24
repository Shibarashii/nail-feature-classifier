# src/models/regnety16gf.py
import torch
from torchvision import models
from torchvision.models import RegNet_Y_16GF_Weights


def get_from_scratch_model(num_classes: int):
    """Return a RegNetY-16GF model initialized from scratch (no pretrained weights)."""
    model = models.regnet_y_16gf(weights=None)  # No pretrained weights
    model.fc = torch.nn.Linear(
        in_features=model.fc.in_features,
        out_features=num_classes
    )
    return model


def get_baseline_model(num_classes: int):
    """
    Returns a baseline RegNetY-16GF where only the classifier head is trainable.
    Backbone is frozen.
    """
    model = models.regnet_y_16gf(weights=RegNet_Y_16GF_Weights.DEFAULT)
    model.fc = torch.nn.Linear(
        in_features=model.fc.in_features,
        out_features=num_classes
    )
    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze classifier
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def get_full_finetune_model(num_classes: int):
    """Return a RegNetY-16GF model for full fine-tuning (all layers trainable)."""
    model = models.regnet_y_16gf(weights=RegNet_Y_16GF_Weights.DEFAULT)
    model.fc = torch.nn.Linear(
        in_features=model.fc.in_features,
        out_features=num_classes
    )
    return model


def get_gradual_unfreeze_model(num_classes: int):
    """
    Return a RegNetY-16GF model for gradual unfreezing.
    Initially freeze backbone; unfreezing is handled in the training loop.
    """
    model = models.regnet_y_16gf(weights=RegNet_Y_16GF_Weights.DEFAULT)
    model.fc = torch.nn.Linear(
        in_features=model.fc.in_features,
        out_features=num_classes
    )
    # Initially freeze backbone
    for param in model.parameters():
        param.requires_grad = False
    # Keep classifier trainable
    for param in model.fc.parameters():
        param.requires_grad = True

    return model
