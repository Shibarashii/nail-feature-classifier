# src/models/efficientnetv2s.py
import torch
from torchvision import models


def get_from_scratch_model(num_classes: int):
    """Return a model initialized from scratch (no pretrained weights)."""
    model = models.efficientnet_v2_s(pretrained=False)  # No pretrained weights
    model.classifier[1] = torch.nn.Linear(
        in_features=model.classifier[1].in_features,
        out_features=num_classes
    )
    return model


def get_baseline_model(num_classes: int):
    """
    Returns a baseline model where only the classifier head is trainable.
    Backbone is frozen.
    """
    model = models.efficientnet_v2_s(pretrained=True)
    model.classifier[1] = torch.nn.Linear(
        in_features=model.classifier[1].in_features,
        out_features=num_classes
    )
    # Freeze backbone
    for param in model.features.parameters():
        param.requires_grad = False

    return model


def get_full_finetune_model(num_classes: int):
    """Return a model for full fine-tuning (all layers trainable)."""
    model = models.efficientnet_v2_s(pretrained=True)
    model.classifier[1] = torch.nn.Linear(
        in_features=model.classifier[1].in_features,
        out_features=num_classes
    )
    # All layers trainable by default
    return model


def get_gradual_unfreeze_model(num_classes: int):
    """
    Return a model for gradual unfreezing.
    Freezing/unfreezing per epoch is handled in engine.py.
    """
    model = models.efficientnet_v2_s(pretrained=True)
    model.classifier[1] = torch.nn.Linear(
        in_features=model.classifier[1].in_features,
        out_features=num_classes
    )
    # Initially freeze backbone; unfreezing is handled in the training loop
    for param in model.features.parameters():
        param.requires_grad = False
    return model
