# src/models/vgg16.py
import torch
from torchvision import models
from torchvision.models import VGG16_Weights


def get_from_scratch_model(num_classes: int):
    """Return a VGG-16 model initialized from scratch (no pretrained weights)."""
    model = models.vgg16(weights=None)  # No pretrained weights
    # Replace the last classifier layer
    model.classifier[6] = torch.nn.Linear(
        in_features=model.classifier[6].in_features,
        out_features=num_classes
    )
    return model


def get_baseline_model(num_classes: int):
    """
    Returns a baseline VGG-16 where only the classifier head is trainable.
    Backbone is frozen.
    """
    model = models.vgg16(weights=VGG16_Weights.DEFAULT)
    model.classifier[6] = torch.nn.Linear(
        in_features=model.classifier[6].in_features,
        out_features=num_classes
    )
    # Freeze backbone
    for param in model.features.parameters():
        param.requires_grad = False
    # Keep classifier trainable
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model


def get_full_finetune_model(num_classes: int):
    """Return a VGG-16 model for full fine-tuning (all layers trainable)."""
    model = models.vgg16(weights=VGG16_Weights.DEFAULT)
    model.classifier[6] = torch.nn.Linear(
        in_features=model.classifier[6].in_features,
        out_features=num_classes
    )
    return model


def get_gradual_unfreeze_model(num_classes: int):
    """
    Return a VGG-16 model for gradual unfreezing.
    Initially freeze backbone; unfreezing is handled in the training loop.
    """
    model = models.vgg16(weights=VGG16_Weights.DEFAULT)
    model.classifier[6] = torch.nn.Linear(
        in_features=model.classifier[6].in_features,
        out_features=num_classes
    )
    # Initially freeze backbone
    for param in model.features.parameters():
        param.requires_grad = False
    # Keep classifier trainable
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model
