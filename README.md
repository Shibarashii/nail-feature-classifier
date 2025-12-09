<a id="readme-top"></a>

<br />
<div align="center">
  <h1 align="center">Nail Feature Classifier</h1>
  <p align="center">
    PyTorch project for nail feature & disease classification with multiple model architectures, training strategies, and CAM visualizations.
  </p>
</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#built-with">Built With</a></li>
    <li><a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#scripts">Scripts</a></li>
    <li><a href="#configuration">Configuration</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>

## About The Project

Nail Feature Classifier trains and evaluates image classification models on nail images (diseases / conditions). It includes model factories, training strategies (scratch / baseline / full fine-tune / gradual unfreeze), evaluation, Grad-CAM interpretability, utilities for reproducibility, and comparison tooling.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

Major components and libraries used:

- `Python`
- `torch`
- `torchvision`
- `torchmetrics`
- `pyyaml`
- other common ML utilities (see `requirements.txt`)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started

Follow these steps to run experiments locally.

### Prerequisites

- Python 3.8+
- CUDA-enabled GPU recommended for training
- git

### Installation

1. Clone the repo

```bash
git clone https://github.com/Shibarashii/nail-feature-classifier.git
cd nail-feature-classifier
```

2. Create and activate a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Prepare data (ImageFolder-style):

```
data/Nail Classification/
  train/<class>/*.jpg
  valid/<class>/*.jpg
  test/<class>/*.jpg
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

Most scripts live under `src/`. Recommended execution from project root using the package invocation (this ensures imports resolve):

- Download reference / pretrained best models (utility script in utils):

```bash
python -m src.utils.download_models
```

- Train:

```bash
python -m src.train --model <model_name> --strategy <strategy> --config <config.yaml>
# e.g.
python -m src.train --model convnexttiny --strategy gradual_unfreeze --config baseline.yaml
```

- Evaluate a checkpoint:

```bash
python -m src.eval --model <model_name> --checkpoint <path/to/best_model.pth> --config <config.yaml>
# e.g.
python -m src.eval --model convnexttiny --checkpoint src/best_models/convnexttiny/best_model.pth --config eval.yaml
```

- Predict / visualize CAMs for an image:

```bash
python -m src.predict /path/to/image.jpg --model <model_name> [--all-cams] [--all-models]
# e.g.
python -m src.predict tests/images/sample.jpg --model efficientnetv2s --all-cams
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Scripts

All main scripts are located in `src/`. Each script contains CLI help and docstrings — use `-h` to inspect available flags.

- src/train.py  
  Orchestrates dataset creation, model selection, optimizer, scheduler, metrics, and training strategies. Common flags:
  --model (e.g., resnet50, efficientnetv2s, convnexttiny)  
  --strategy (scratch, baseline, full_finetune, gradual_unfreeze)  
  --config (YAML under src/configs/, e.g., baseline.yaml, test.yaml)

- src/eval.py  
  Runs evaluation using a model checkpoint, computes metrics, and writes outputs under `src/output/` (or `src/best_models/.../evaluation/`).

- src/predict.py  
  Single-image or batch inference with options to generate CAM visualizations and per-model predictions.

- src/engine.py  
  Core train loop and utilities (train_model). Typically used by train.py; not invoked directly by most users.

- src/compare_results.py  
  Aggregates experiment outputs to produce comparison CSV/JSON summaries under `src/output/`.

- src/utils/download_models.py  
  Utility to fetch pretrained / baseline artifacts into `src/best_models/`. Run from project root:
  `python -m src.utils.download_models` (use `-h` to list options).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Configuration

Configs are YAML files under `src/configs/`. They define defaults (batch size, lr, epochs), scheduler params, and other experiment settings. Pass `--config <name>.yaml` to scripts to select behavior.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## License

Distributed under MIT License. See `LICENSE.txt` for more info.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
