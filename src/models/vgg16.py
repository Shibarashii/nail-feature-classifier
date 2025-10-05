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


def get_layer_groups(model):
    """
    Split VGG16 into layer groups for gradual unfreezing.
    Returns groups from head to early layers.

    VGG16 Architecture:
    - features: 5 conv blocks (conv1 -> conv5)
    - classifier: 3 FC layers

    Layer groups (8 total):
    - Group 0: Final FC layer (classifier[6])
    - Group 1: FC + ReLU + Dropout (classifier[3:6])
    - Group 2: First FC + ReLU + Dropout (classifier[0:3])
    - Group 3: Conv block 5 (3 conv + pool)
    - Group 4: Conv block 4 (3 conv + pool)
    - Group 5: Conv block 3 (3 conv + pool)
    - Group 6: Conv block 2 (2 conv + pool)
    - Group 7: Conv block 1 (2 conv + pool)
    """
    groups = [
        model.classifier[6],      # Group 0: Final classifier layer
        # Group 1: Second FC layer (Linear + ReLU + Dropout)
        model.classifier[3:6],
        # Group 2: First FC layer (Linear + ReLU + Dropout)
        model.classifier[0:3],
        # Group 3: Conv5 block (conv5_1, conv5_2, conv5_3, pool5)
        model.features[24:],
        # Group 4: Conv4 block (conv4_1, conv4_2, conv4_3, pool4)
        model.features[17:24],
        # Group 5: Conv3 block (conv3_1, conv3_2, conv3_3, pool3)
        model.features[10:17],
        # Group 6: Conv2 block (conv2_1, conv2_2, pool2)
        model.features[5:10],
        # Group 7: Conv1 block (conv1_1, conv1_2, pool1)
        model.features[0:5],
    ]
    return groups
