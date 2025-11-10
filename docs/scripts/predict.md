# predict.py

This script predicts nail diseases from images using trained deep learning models. It can also generate Grad-CAM visualizations and infer possible systemic diseases.

## Features

- **Nail Feature Classification**: Predict nail conditions using trained deep learning models
- **CAM Visualization**: Generate interpretable heatmaps showing which regions influenced the prediction
- **Disease Inference**: Infer potential systemic diseases based on nail features using Bayesian inference
- **Multi-Model Support**: Run predictions across multiple models simultaneously
- **Comprehensive Reports**: Automatically generated reports with predictions, visualizations, and disease probabilities

## Usage

### Most Common Usage

These scripts are the most commonly used scripts and uses the default input and output directories.

> Note that the default path for the images to be predicted is in `data/`

#### 1. Predicts an image using all the models and creates saliency maps using all available cams.

```bash
python -m src.predict 1.png --all-models --all-cams
```

#### 2. Predicts an image using all models, creates saliency maps using all available cams, and predicts systemic diseases.

```bash
python -m src.predict 1.png --all-models --all-cams --sex male --age 60
```

#### 3. Predicts an image using a single model

```bash
python -m src.predict 1.png efficientnetv2s --all-cams --sex male --age 60
```

## Command Line Arguments

### Required Arguments

| Argument | Description                                | Example                 |
| -------- | ------------------------------------------ | ----------------------- |
| image    | Image filename (indata/directory)          | 6.png                   |
| model    | Model name (optional if using--all-models) | swinv2t,efficientnetv2s |

### Optional Arguments

| Argument     | Type   | Description                                         | Example             |
| ------------ | ------ | --------------------------------------------------- | ------------------- |
| --all-cams   | flag   | Generate all CAM visualization methods              | --all-cams          |
| --all-models | flag   | Run prediction with all available models            | --all-models        |
| --sex        | string | Patient sex for disease inference (maleorfemale)    | --sex male          |
| --age        | float  | Patient age for disease inference                   | --age 60            |
| --save-dir   | string | Directory to save outputs (default:src/predictions) | --save-dir results/ |

### Legacy Arguments

Only use these when you want to explicitly set image path or model path

| Argument     | Description                            |
| ------------ | -------------------------------------- |
| --image-path | Full path to input image               |
| --model-path | Full path to model weights (.pth file) |

## Available Models

The system automatically detects models in src/best_models/:

- efficientnetv2s - EfficientNetV2-Small
- resnet50 - ResNet-50
- swinv2t - Swin Transformer V2-Tiny
- vgg16 - VGG-16
- convnexttiny - ConvNeXt-Tiny

## Help

For full help and additional examples:

```bash
python -m src.predict --help
```
