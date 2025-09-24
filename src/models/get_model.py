from src.models.efficientnetv2s import (
    get_from_scratch_model as efficientnetv2s_scratch,
    get_baseline_model as efficientnetv2s_baseline,
    get_full_finetune_model as efficientnetv2s_full,
    get_gradual_unfreeze_model as efficientnetv2s_gradual
)

from src.models.testnet import (
    get_from_scratch_model as testnet_scratch,
    get_baseline_model as testnet_baseline,
    get_full_finetune_model as testnet_full,
    get_gradual_unfreeze_model as testnet_gradual
)


def get_model(model_name: str, strategy: str, num_classes: int):
    model_map = {
        "testnet": {
            "scratch": testnet_scratch,
            "baseline": testnet_baseline,
            "full_finetune": testnet_full,
            "gradual_unfreeze": testnet_gradual
        },
        "efficientnetv2s": {
            "scratch": efficientnetv2s_scratch,
            "baseline": efficientnetv2s_baseline,
            "full_finetune": efficientnetv2s_full,
            "gradual_unfreeze": efficientnetv2s_gradual
        },
    }

    if model_name not in model_map:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(model_map.keys())}")
    if strategy not in model_map[model_name]:
        raise ValueError(
            f"Unknown strategy '{strategy}' for model '{model_name}'. Available: {list(model_map[model_name].keys())}")

    return model_map[model_name][strategy](num_classes=num_classes)
