# src/utils/gradual_unfreeze.py
"""
ULMFiT-style gradual unfreezing utilities.
Progressively unfreezes layer groups from head to backbone.
"""
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def get_layer_groups_efficientnetv2s(model):
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


def get_layer_groups_resnet50(model):
    """
    Split ResNet50 into layer groups for gradual unfreezing.
    Returns groups from head to early layers.
    """
    groups = [
        model.fc,           # Group 0: Classifier head
        model.layer4,       # Group 1: Last residual block
        model.layer3,       # Group 2
        model.layer2,       # Group 3
        model.layer1,       # Group 4
        [model.conv1, model.bn1],  # Group 5: Stem
    ]
    return groups


def get_layer_groups_vgg16(model):
    """
    Split VGG16 into layer groups for gradual unfreezing.
    Returns groups from head to early layers.
    """
    groups = [
        model.classifier,   # Group 0: Classifier head
        model.features[24:],  # Group 1: Conv5 block
        model.features[17:24],  # Group 2: Conv4 block
        model.features[10:17],  # Group 3: Conv3 block
        model.features[5:10],   # Group 4: Conv2 block
        model.features[0:5],    # Group 5: Conv1 block
    ]
    return groups


def get_layer_groups_convnext_tiny(model):
    """
    Split ConvNeXt-Tiny into layer groups for gradual unfreezing.
    Returns groups from head to early layers.
    """
    groups = [
        model.classifier,   # Group 0: Classifier head
        model.features[7],  # Group 1: Last stage + norm
        model.features[6],  # Group 2: Stage 3
        model.features[4],  # Group 3: Stage 2
        model.features[2],  # Group 4: Stage 1
        model.features[0],  # Group 5: Stem
    ]
    return groups


def get_layer_groups_swin_v2_t(model):
    """
    Split Swin-V2-T into layer groups for gradual unfreezing.
    Returns groups from head to early layers.
    """
    groups = [
        model.head,         # Group 0: Classifier head
        model.features[7],  # Group 1: Norm + permute
        model.features[6],  # Group 2: Last stage
        model.features[5],  # Group 3: Stage 3
        model.features[4],  # Group 4: Stage 2
        model.features[2:4],  # Group 5: Stage 1
        model.features[0:2],  # Group 6: Patch embedding
    ]
    return groups


def get_layer_groups(model, model_name):
    """
    Get layer groups for a given model architecture.

    Parameters
    ----------
    model : nn.Module
        The model to get layer groups from.
    model_name : str
        Name of the model architecture.

    Returns
    -------
    list
        List of layer groups from head to early layers.
    """
    layer_group_funcs = {
        'efficientnetv2s': get_layer_groups_efficientnetv2s,
        'resnet50': get_layer_groups_resnet50,
        'vgg16': get_layer_groups_vgg16,
        'convnext_tiny': get_layer_groups_convnext_tiny,
        'swin_v2_t': get_layer_groups_swin_v2_t,
    }

    if model_name not in layer_group_funcs:
        raise ValueError(f"Unsupported model: {model_name}. "
                         f"Supported models: {list(layer_group_funcs.keys())}")

    return layer_group_funcs[model_name](model)


def discriminative_lrs(base_lr, n_groups, div_factor=2.6):
    """
    Assign progressively smaller LRs to earlier layers.
    """
    return [base_lr / (div_factor ** i) for i in range(n_groups)][::-1]


def slanted_triangular_scheduler(optimizer, num_training_steps, cut_frac=0.1, ratio=32):
    """
    Slanted triangular schedule from ULMFiT.
    Increases LR linearly then decays.
    """
    cut = int(num_training_steps * cut_frac)

    def lr_lambda(step):
        if step < cut:
            p = step / cut
        else:
            p = 1 - (step - cut) / (cut * (ratio - 1))
        return max(0, p)
    return LambdaLR(optimizer, lr_lambda)


def freeze_all_groups(layer_groups):
    """Freeze all layer groups."""
    for group in layer_groups:
        if isinstance(group, list):
            for layer in group:
                for param in layer.parameters():
                    param.requires_grad = False
        else:
            for param in group.parameters():
                param.requires_grad = False


def unfreeze_group(layer_group):
    """Unfreeze a specific layer group."""
    if isinstance(layer_group, list):
        for layer in layer_group:
            for param in layer.parameters():
                param.requires_grad = True
    else:
        for param in layer_group.parameters():
            param.requires_grad = True


def unfreeze_up_to_group(layer_groups, group_idx):
    """
    Unfreeze all groups from 0 (head) up to and including group_idx.

    Parameters
    ----------
    layer_groups : list
        List of layer groups.
    group_idx : int
        Index of the last group to unfreeze (inclusive).
    """
    for i in range(group_idx + 1):
        unfreeze_group(layer_groups[i])


def get_unfreeze_schedule(num_groups, num_epochs, epochs_per_unfreeze=None):
    """
    Create a schedule for when to unfreeze each layer group.

    Parameters
    ----------
    num_groups : int
        Total number of layer groups.
    num_epochs : int
        Total number of training epochs.
    epochs_per_unfreeze : int, optional
        Number of epochs to train before unfreezing the next group.
        If None, distributes unfreezing evenly across epochs.

    Returns
    -------
    dict
        Mapping of epoch number to group index to unfreeze up to.
        Example: {0: 0, 3: 1, 6: 2, ...}
    """
    schedule = {}

    if epochs_per_unfreeze is None:
        # Distribute unfreezing evenly across epochs
        epochs_per_unfreeze = max(1, num_epochs // num_groups)

    for group_idx in range(num_groups):
        epoch = group_idx * epochs_per_unfreeze
        if epoch < num_epochs:
            schedule[epoch] = group_idx

    # Ensure all groups are unfrozen by the last epoch
    schedule[num_epochs - 1] = num_groups - 1

    return schedule


def apply_gradual_unfreeze(model, model_name, epoch, num_epochs,
                           epochs_per_unfreeze=None, verbose=True):
    """
    Apply gradual unfreezing for the current epoch.

    Parameters
    ----------
    model : nn.Module
        The model to apply unfreezing to.
    model_name : str
        Name of the model architecture.
    epoch : int
        Current epoch number (0-indexed).
    num_epochs : int
        Total number of training epochs.
    epochs_per_unfreeze : int, optional
        Number of epochs between unfreezing groups.
    verbose : bool
        Whether to print unfreezing information.

    Returns
    -------
    bool
        True if a new group was unfrozen this epoch.
    """
    layer_groups = get_layer_groups(model, model_name)
    schedule = get_unfreeze_schedule(
        len(layer_groups), num_epochs, epochs_per_unfreeze)

    # Check if we should unfreeze a new group this epoch
    if epoch in schedule:
        group_idx = schedule[epoch]
        unfreeze_up_to_group(layer_groups, group_idx)

        if verbose:
            print(f"\n{'='*60}")
            print(
                f"Epoch {epoch}: Unfreezing up to group {group_idx}/{len(layer_groups)-1}")
            print(f"Groups 0-{group_idx} are now trainable")

            # Count trainable parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel()
                                   for p in model.parameters() if p.requires_grad)
            print(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
                  f"({100 * trainable_params / total_params:.1f}%)")
            print(f"{'='*60}\n")

        return True

    return False
