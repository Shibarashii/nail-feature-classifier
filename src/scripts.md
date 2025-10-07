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
python -m src.eval PATH/TO/MODEL_FOLDER
```

## `download_models.py`

Downloads the models from the hugging face repository

### Example Usage

```bash
python -m src.utils.download_models
```

## `predict.py`

Predicts an input image and visualizes the features using grad cam

### Example usage

```bash
python -m src.predict.py PATH/TO/MODEL.pth PATH/TO/IMAGE
```
