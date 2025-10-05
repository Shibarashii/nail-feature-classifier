# src/models/resnet50.py
import torch
from torchvision import models
from torchvision.models import ResNet50_Weights


def get_from_scratch_model(num_classes: int):
    """Return a ResNet-50 model initialized from scratch (no pretrained weights)."""
    model = models.resnet50(weights=None)  # No pretrained weights
    model.fc = torch.nn.Linear(
        in_features=model.fc.in_features,
        out_features=num_classes
    )
    return model


def get_baseline_model(num_classes: int):
    """
    Returns a baseline ResNet-50 where only the classifier head is trainable.
    Backbone is frozen.
    """
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Linear(
        in_features=model.fc.in_features,
        out_features=num_classes
    )
    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False
    # Keep classifier trainable
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def get_full_finetune_model(num_classes: int):
    """Return a ResNet-50 model for full fine-tuning (all layers trainable)."""
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Linear(
        in_features=model.fc.in_features,
        out_features=num_classes
    )
    return model


def get_gradual_unfreeze_model(num_classes: int):
    """
    Return a ResNet-50 model for gradual unfreezing.
    Initially freeze backbone; unfreezing is handled in the training loop.
    """
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
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


def get_layer_groups(model):
    """
    Split ResNet50 into layer groups for gradual unfreezing.
    Returns groups from head to early layers.

    ResNet50 architecture:
    - conv1: Initial 7x7 convolution
    - bn1: Batch normalization
    - relu: ReLU activation
    - maxpool: Max pooling
    - layer1: 3 bottleneck blocks (output: 256 channels)
    - layer2: 4 bottleneck blocks (output: 512 channels)
    - layer3: 6 bottleneck blocks (output: 1024 channels)
    - layer4: 3 bottleneck blocks (output: 2048 channels)
    - avgpool: Adaptive average pooling
    - fc: Fully connected layer (classifier)

    We unfreeze from deepest (layer4) to shallowest (conv1/stem).
    """
    groups = [
        model.fc,      # Group 0: Classifier head
        # Group 1: Layer 4 (deepest residual blocks, 3 bottlenecks)
        model.layer4,
        model.layer3,  # Group 2: Layer 3 (6 bottleneck blocks)
        model.layer2,  # Group 3: Layer 2 (4 bottleneck blocks)
        model.layer1,  # Group 4: Layer 1 (3 bottleneck blocks)
        [model.conv1, model.bn1, model.relu, model.maxpool],  # Group 5: Stem
    ]
    return groups
