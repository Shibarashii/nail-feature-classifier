# src/models/convnexttiny.py
import torch
from torchvision import models
from torchvision.models import ConvNeXt_Tiny_Weights


def get_from_scratch_model(num_classes: int):
    """Return a ConvNeXt-Tiny model initialized from scratch (no pretrained weights)."""
    model = models.convnext_tiny(weights=None)  # No pretrained weights
    model.classifier[2] = torch.nn.Linear(
        in_features=model.classifier[2].in_features,
        out_features=num_classes
    )
    return model


def get_baseline_model(num_classes: int):
    """
    Returns a baseline ConvNeXt-Tiny where only the classifier head is trainable.
    Backbone is frozen.
    """
    model = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
    model.classifier[2] = torch.nn.Linear(
        in_features=model.classifier[2].in_features,
        out_features=num_classes
    )
    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False
    # Keep classifier trainable
    for param in model.classifier[2].parameters():
        param.requires_grad = True

    return model


def get_full_finetune_model(num_classes: int):
    """Return a ConvNeXt-Tiny model for full fine-tuning (all layers trainable)."""
    model = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
    model.classifier[2] = torch.nn.Linear(
        in_features=model.classifier[2].in_features,
        out_features=num_classes
    )
    return model


def get_gradual_unfreeze_model(num_classes: int):
    """
    Return a ConvNeXt-Tiny model for gradual unfreezing.
    Initially freeze backbone; unfreezing is handled in the training loop.
    """
    model = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
    model.classifier[2] = torch.nn.Linear(
        in_features=model.classifier[2].in_features,
        out_features=num_classes
    )
    # Initially freeze backbone
    for param in model.parameters():
        param.requires_grad = False
    # Keep classifier trainable
    for param in model.classifier[2].parameters():
        param.requires_grad = True

    return model
