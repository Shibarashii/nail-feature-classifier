# Nail Feature Classifier

A PyTorch-based repository for classifying nail features and diseases. Supports multiple architectures, training strategies (scratch, baseline, full fine-tune, gradual unfreeze), evaluation, and CAM visualizations.

## Quick links

- Code: `src/`
- Data: `data/`
- Pretrained / best models: `src/best_models/`
- Outputs / predictions: `src/output/`, `src/predictions/`
- Configs: `src/configs/`
- Entrypoints / scripts: see "Scripts" below

## Quickstart

1. Create and activate a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

2. Prepare data with ImageFolder-style layout:

```
data/Nail Classification/
  train/<class>/*.jpg
  valid/<class>/*.jpg
  test/<class>/*.jpg
```

3. Choose or create a config under `src/configs/` (e.g., `baseline.yaml`, `gradual_unfreeze.yaml`, `test.yaml`).

4. (Optional) Download reference/best models:

```bash
python -m src.utils.download_models
```

5. Train:

```bash
python -m src.train --model convnexttiny --strategy gradual_unfreeze --config baseline.yaml
```

6. Evaluate:

```bash
python -m src.eval --model convnexttiny --checkpoint src/best_models/convnexttiny/best_model.pth --config eval.yaml
```

7. Predict / visualize CAMs:

```bash
python -m src.predict path/to/image.jpg --model efficientnetv2s --all-cams
```

## Notes

- Run scripts from project root with `python -m src.<script>`. Running from root ensures package imports (`src.*`) resolve correctly.
- Each script contains CLI help and docstrings providing details about flags and behavior — use `-h` to inspect.
- Configs control most hyperparameters and scheduler settings. See `src/configs/` for examples.
- Outputs and experiment artifacts are written to `src/output/`, `src/predictions/`, and `src/best_models/`.
