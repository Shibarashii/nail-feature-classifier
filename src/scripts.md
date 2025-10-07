# How to use scripts

## `train.py`

This script trains a PyTorch model on the fingernail dataset using different training strategies.

### Usage Usage

```bash
python -m src.train --model swinv2t --strategy gradual_unfreeze --config gradual_unfreeze.yaml
```

## `eval.py`

This script evaluates a trained model on a dataset split (test, val, or train) and generates metrics and visualizations.

### Example Usage

```bash
python -m src.eval --model_path "PATH/TO/MODEL.pth" --model efficientnetv2s --strategy full_finetune
```

## `download_models.py`

Downloads the models from the hugging face repository

### Example Usage

```bash
python -m src.utils.download_models
```
