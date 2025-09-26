# src/models/get_model.py
from src.models.testnet import (
    get_from_scratch_model as testnet_scratch,
    get_baseline_model as testnet_baseline,
    get_full_finetune_model as testnet_full,
    get_gradual_unfreeze_model as testnet_gradual
)

from src.models.vgg16 import (
    get_from_scratch_model as vgg16_scratch,
    get_baseline_model as vgg16_baseline,
    get_full_finetune_model as vgg16_full,
    get_gradual_unfreeze_model as vgg16_gradual
)

from src.models.resnet50 import (
    get_from_scratch_model as resnet50_scratch,
    get_baseline_model as resnet50_baseline,
    get_full_finetune_model as resnet50_full,
    get_gradual_unfreeze_model as resnet50_gradual
)

from src.models.regnety16gf import (
    get_from_scratch_model as regnety16gf_scratch,
    get_baseline_model as regnety16gf_baseline,
    get_full_finetune_model as regnety16gf_full,
    get_gradual_unfreeze_model as regnety16gf_gradual
)

from src.models.efficientnetv2s import (
    get_from_scratch_model as efficientnetv2s_scratch,
    get_baseline_model as efficientnetv2s_baseline,
    get_full_finetune_model as efficientnetv2s_full,
    get_gradual_unfreeze_model as efficientnetv2s_gradual
)
from src.models.swinv2t import (
    get_from_scratch_model as swinv2t_scratch,
    get_baseline_model as swinv2t_baseline,
    get_full_finetune_model as swinv2t_full,
    get_gradual_unfreeze_model as swinv2t_gradual
)
from src.models.convnexttiny import (
    get_from_scratch_model as convnexttiny_scratch,
    get_baseline_model as convnexttiny_baseline,
    get_full_finetune_model as convnexttiny_full,
    get_gradual_unfreeze_model as convnexttiny_gradual
)


def get_model(model_name: str, strategy: str, num_classes: int):
    model_map = {
        "testnet": {
            "scratch": testnet_scratch,
            "baseline": testnet_baseline,
            "full_finetune": testnet_full,
            "gradual_unfreeze": testnet_gradual
        },
        "vgg16": {
            "scratch": vgg16_scratch,
            "baseline": vgg16_baseline,
            "full_finetune": vgg16_full,
            "gradual_unfreeze": vgg16_gradual
        },
        "resnet50": {
            "scratch": resnet50_scratch,
            "baseline": resnet50_baseline,
            "full_finetune": resnet50_full,
            "gradual_unfreeze": resnet50_gradual
        },
        "regnety16gf": {
            "scratch": regnety16gf_scratch,
            "baseline": regnety16gf_baseline,
            "full_finetune": regnety16gf_full,
            "gradual_unfreeze": regnety16gf_gradual
        },
        "efficientnetv2s": {
            "scratch": efficientnetv2s_scratch,
            "baseline": efficientnetv2s_baseline,
            "full_finetune": efficientnetv2s_full,
            "gradual_unfreeze": efficientnetv2s_gradual
        },
        "swinv2t": {
            "scratch": swinv2t_scratch,
            "baseline": swinv2t_baseline,
            "full_finetune": swinv2t_full,
            "gradual_unfreeze": swinv2t_gradual
        },
        "convnexttiny": {
            "scratch": convnexttiny_scratch,
            "baseline": convnexttiny_baseline,
            "full_finetune": convnexttiny_full,
            "gradual_unfreeze": convnexttiny_gradual
        }
    }

    if model_name not in model_map:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(model_map.keys())}")
    if strategy not in model_map[model_name]:
        raise ValueError(
            f"Unknown strategy '{strategy}' for model '{model_name}'. Available: {list(model_map[model_name].keys())}")

    return model_map[model_name][strategy](num_classes=num_classes)
