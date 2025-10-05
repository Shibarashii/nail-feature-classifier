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
