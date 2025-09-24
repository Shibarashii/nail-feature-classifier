# src/models/swinv2t.py
import torch
from torchvision import models
from torchvision.models import Swin_V2_T_Weights


def get_from_scratch_model(num_classes: int):
    """Return a SwinV2-T model initialized from scratch (no pretrained weights)."""
    model = models.swin_v2_t(weights=None)
    model.head = torch.nn.Linear(
        in_features=model.head.in_features,
        out_features=num_classes
    )
    return model


def get_baseline_model(num_classes: int):
    """
    Returns a baseline SwinV2-T where only the classifier head is trainable.
    Backbone is frozen.
    """
    model = models.swin_v2_t(weights=Swin_V2_T_Weights.DEFAULT)
    model.head = torch.nn.Linear(
        in_features=model.head.in_features,
        out_features=num_classes
    )
    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False
    # Keep classifier trainable
    for param in model.head.parameters():
        param.requires_grad = True

    return model


def get_full_finetune_model(num_classes: int):
    """Return a SwinV2-T model for full fine-tuning (all layers trainable)."""
    model = models.swin_v2_t(weights=Swin_V2_T_Weights.DEFAULT)
    model.head = torch.nn.Linear(
        in_features=model.head.in_features,
        out_features=num_classes
    )
    return model


def get_gradual_unfreeze_model(num_classes: int):
    """
    Return a SwinV2-T model for gradual unfreezing.
    Initially freeze backbone; unfreezing is handled in the training loop.
    """
    model = models.swin_v2_t(weights=Swin_V2_T_Weights.DEFAULT)
    model.head = torch.nn.Linear(
        in_features=model.head.in_features,
        out_features=num_classes
    )
    # Initially freeze backbone
    for param in model.parameters():
        param.requires_grad = False
    # Keep classifier trainable
    for param in model.head.parameters():
        param.requires_grad = True

    return model
