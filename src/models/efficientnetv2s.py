# src/models/efficientnetv2s.py
import torch
from torchvision import models
from torchvision.models import EfficientNet_V2_S_Weights


def get_from_scratch_model(num_classes: int):
    """Return a model initialized from scratch (no pretrained weights)."""
    model = models.efficientnet_v2_s(weights=None)  # No pretrained weights
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
    model = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
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
    model = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    model.classifier[1] = torch.nn.Linear(
        in_features=model.classifier[1].in_features,
        out_features=num_classes
    )
    # All layers trainable by default
    return model


def get_gradual_unfreeze_model(num_classes: int):
    """
    Returns a model configured for gradual unfreezing.
    Initially, only the classifier head is trainable.
    """
    model = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    model.classifier[1] = torch.nn.Linear(
        in_features=model.classifier[1].in_features,
        out_features=num_classes
    )

    # Freeze all backbone layers initially
    for param in model.features.parameters():
        param.requires_grad = False

    return model


def get_layer_groups(model):
    """
    Split EfficientNetV2-S into layer groups for gradual unfreezing.
    Returns groups from head to early layers.
    """
    groups = [
        model.classifier,  # Group 0: Classifier head
        model.features[7],  # Group 1: Last stage
        model.features[6],  # Group 2
        model.features[5],  # Group 3
        model.features[4],  # Group 4
        model.features[3],  # Group 5
        model.features[2],  # Group 6
        model.features[1],  # Group 7
        model.features[0],  # Group 8: Stem
    ]
    return groups
