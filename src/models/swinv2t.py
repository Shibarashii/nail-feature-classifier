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


def get_layer_groups(model):
    """
    Split Swin V2 Tiny into layer groups for gradual unfreezing.
    Returns groups from head to early layers.

    Swin V2 Tiny architecture:
    - features[0]: PatchEmbedding (stem)
    - features[1]: Stage 1 - 2 Swin Transformer blocks
    - features[2]: PatchMerging (downsampling)
    - features[3]: Stage 2 - 2 Swin Transformer blocks
    - features[4]: PatchMerging (downsampling)
    - features[5]: Stage 3 - 6 Swin Transformer blocks
    - features[6]: PatchMerging (downsampling)
    - features[7]: Stage 4 - 2 Swin Transformer blocks (deepest)
    - norm: Layer normalization
    - head: Classification head
    """
    groups = [
        [model.head, model.norm],  # Group 0: Head + final normalization
        model.features[7],         # Group 1: Stage 4 (deepest, 2 blocks)
        model.features[6],         # Group 2: Patch merging 3
        model.features[5],         # Group 3: Stage 3 (6 blocks)
        model.features[4],         # Group 4: Patch merging 2
        model.features[3],         # Group 5: Stage 2 (2 blocks)
        model.features[2],         # Group 6: Patch merging 1
        model.features[1],         # Group 7: Stage 1 (2 blocks)
        model.features[0],         # Group 8: Patch embedding (stem)
    ]
    return groups
