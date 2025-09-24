# src/models/mobilenetv3s.py
import torch
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights


# The purpose of this file is to have test runs, to ensure the model is working as expected.
# These models are not used in the actual training/evaluation pipeline.
def get_from_scratch_model(num_classes: int):
    """Return a model initialized from scratch (no pretrained weights)."""
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = torch.nn.Linear(
        in_features=model.classifier[3].in_features,
        out_features=num_classes
    )
    return model


def get_baseline_model(num_classes: int):
    """
    Returns a baseline model where only the classifier head is trainable.
    Backbone is frozen.
    """
    model = models.mobilenet_v3_small(
        weights=MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[3] = torch.nn.Linear(
        in_features=model.classifier[3].in_features,
        out_features=num_classes
    )
    # Freeze backbone
    for param in model.features.parameters():
        param.requires_grad = False

    return model


def get_full_finetune_model(num_classes: int):
    """Return a model for full fine-tuning (all layers trainable)."""
    model = models.mobilenet_v3_small(
        weights=MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[3] = torch.nn.Linear(
        in_features=model.classifier[3].in_features,
        out_features=num_classes
    )
    return model


def get_gradual_unfreeze_model(num_classes: int):
    """
    Return a model for gradual unfreezing.
    Freezing/unfreezing per epoch is handled in engine.py.
    """
    model = models.mobilenet_v3_small(
        weights=MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[3] = torch.nn.Linear(
        in_features=model.classifier[3].in_features,
        out_features=num_classes
    )
    # Initially freeze backbone
    for param in model.features.parameters():
        param.requires_grad = False
    return model
