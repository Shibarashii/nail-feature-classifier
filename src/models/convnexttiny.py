import torch
from torchvision import models
from torchvision.models import ConvNeXt_Tiny_Weights


def get_from_scratch_model(num_classes: int):
    """Return a model initialized from scratch (no pretrained weights)."""
    model = models.convnext_tiny(weights=None)
    model.classifier[2] = torch.nn.Linear(
        in_features=model.classifier[2].in_features,
        out_features=num_classes
    )
    return model


def get_baseline_model(num_classes: int):
    """
    Returns a baseline model where only the classifier head is trainable.
    Backbone is frozen.
    """
    model = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
    model.classifier[2] = torch.nn.Linear(
        in_features=model.classifier[2].in_features,
        out_features=num_classes
    )
    # Freeze backbone
    for param in model.features.parameters():
        param.requires_grad = False
    return model


def get_full_finetune_model(num_classes: int):
    """Return a model for full fine-tuning (all layers trainable)."""
    model = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
    model.classifier[2] = torch.nn.Linear(
        in_features=model.classifier[2].in_features,
        out_features=num_classes
    )
    return model


def get_gradual_unfreeze_model(num_classes: int):
    """
    Returns a model configured for gradual unfreezing.
    Initially, only the classifier head is trainable.
    """
    model = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
    model.classifier[2] = torch.nn.Linear(
        in_features=model.classifier[2].in_features,
        out_features=num_classes
    )
    # Freeze all backbone layers initially
    for param in model.features.parameters():
        param.requires_grad = False
    return model


def get_layer_groups(model):
    """
    Split ConvNeXt-Tiny into layer groups for gradual unfreezing.
    Returns groups from head to early layers.

    ConvNeXt architecture has 4 stages, we'll unfreeze stage by stage.
    """
    groups = [
        model.classifier,  # Group 0: Classifier head
        model.features[7],  # Group 1: Stage 4 (last stage)
        model.features[6],  # Group 2: Downsampling layer 3
        model.features[5],  # Group 3: Stage 3
        model.features[4],  # Group 4: Downsampling layer 2
        model.features[3],  # Group 5: Stage 2
        model.features[2],  # Group 6: Downsampling layer 1
        model.features[1],  # Group 7: Stage 1
        model.features[0],  # Group 8: Stem
    ]
    return groups
