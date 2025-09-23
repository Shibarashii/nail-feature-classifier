import torch
import torchvision
import torchmetrics
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch import nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from PIL import Image

from timeit import default_timer as timer
from tqdm.auto import tqdm
from pathlib import Path
from typing import Union
import json
import os

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from collections import Counter


def get_root_dir():
    return Path(__file__).resolve().parent.parent.parent  # src/utils -> root


def get_class_distribution(dataset, subset_name):
    if isinstance(dataset, Subset):
        # Access the underlying dataset's targets using indices
        targets = [dataset.dataset.samples[i][1] for i in dataset.indices]
    else:
        # Regular ImageFolder
        targets = [label for _, label in dataset.samples]

    class_counts = Counter(targets)

    print(f"\nClass distribution in {subset_name}:")
    for class_idx, count in sorted(class_counts.items()):
        class_name = dataset.dataset.classes[class_idx] if isinstance(
            dataset, Subset) else dataset.classes[class_idx]
        print(f"{class_name:25}: {count}")


def plot_images(data_dir: Path,
                transforms: T = None,
                random_seed: int = None):
    random.seed(random_seed)

    image_path_list = list(data_dir.glob("*/*/*.jpg"))
    fig = plt.figure(figsize=(10, 7))
    rows, cols = 3, 3

    for i in range(1, rows * cols + 1):
        random_img_path = random.choice(image_path_list)
        img_class = random_img_path.parent.stem
        fig.add_subplot(rows, cols, i)
        random_img = Image.open(random_img_path)

        if transforms:
            transformed_img = transforms(random_img)
            plt.imshow(transformed_img.permute(1, 2, 0))
        else:
            plt.imshow(random_img)
        plt.title(f"{img_class} ({random_img.height}, {random_img.width})")
        plt.axis(False)


def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               criterion: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device):
    """Performs training with model trying to learn on `data_loader`"""

    train_loss, train_acc = 0, 0

    model.train()
    for batch, (X_train, y_train) in enumerate(data_loader):

        # Put data on target device
        X_train, y_train = X_train.to(device), y_train.to(device)

        # Forward pass (outputs the raw logits from the model)
        y_pred = model(X_train)

        loss = criterion(y_pred, y_train)
        train_loss += loss
        train_acc += accuracy_fn(y_train, y_pred.argmax(dim=1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            print(
                f"Looked at {batch * len(X_train)} / {len(data_loader.dataset)} samples")

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    return train_loss, train_acc


def valid_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               criterion: torch.nn.Module,
               accuracy_fn,
               device: torch.device):
    """Performs validation with model trying to learn on `data_loader`"""

    valid_loss, valid_acc = 0, 0

    model.eval()
    with torch.inference_mode():
        for X_valid, y_valid in data_loader:
            X_valid, y_valid = X_valid.to(device), y_valid.to(device)

            valid_pred = model(X_valid)

            valid_loss += criterion(valid_pred, y_valid)
            valid_acc += accuracy_fn(y_valid, valid_pred.argmax(dim=1))

        valid_loss /= len(data_loader)
        valid_acc /= len(data_loader)
    return valid_loss, valid_acc


def train_model(epochs: int,
                model: torch.nn.Module,
                train_dataloader: torch.utils.data.DataLoader,
                valid_dataloader: torch.utils.data.DataLoader,
                criterion: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                accuracy_fn,
                device: torch.device,
                scheduler=None,
                scheduler_mode: str = "epoch"):

    results = {"train_loss": [],
               "train_acc": [],
               "valid_loss": [],
               "valid_acc": []}

    start_time = timer()

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model, train_dataloader, criterion, optimizer, accuracy_fn, device)
        val_loss, val_acc = valid_step(
            model, valid_dataloader, criterion, accuracy_fn, device)

        print(f"Epoch: {epoch} | Train loss: {train_loss:.4f}, Train accuracy : {train_acc:.4f} | Valid loss: {val_loss:.4f}, Valid accuracy: {val_acc:.4f}")

        results["train_loss"].append(train_loss.item())
        results["train_acc"].append(train_acc.item())
        results["valid_loss"].append(val_loss.item())
        results["valid_acc"].append(val_acc.item())

        # Scheduler step depending on mode
        if scheduler:
            if scheduler_mode == "plateau":
                scheduler.step(val_loss)  # needs validation loss
            elif scheduler_mode == "epoch":
                scheduler.step()

    end_time = timer()
    training_time = end_time - start_time

    print(f"Total time on {device}: {training_time:.3f}")

    return results


def save_results(dir_name: str, file_name: str, results):
    results_dir = Path(dir_name)
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f"{file_name}.json"

    # Load existing results if file exists
    if results_file.exists():
        with open(results_file, "r") as f:
            all_results = json.load(f)
    else:
        all_results = {}

    # Generate new run key (e.g., "run_1", "run_2", etc.)
    run_id = f"run_{len(all_results) + 1}"
    all_results[run_id] = results

    # Save updated results
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=4)

    return results_file


def load_results(dir_name: str, file_name: str, run: Union[str, int] = "latest") -> pd.DataFrame:
    """
    Load a specific training run from a results file and return it as a DataFrame.

    Args:
        dir_name (str): Directory containing the results file.
        file_name (str): Name of the JSON file (without `.json`).
        run (str or int): Which run to load. Options:
                          - "latest" (default): most recent run
                          - "all": returns the entire dictionary
                          - int: specific run number, e.g., 2 → "run_2"

    Returns:
        pd.DataFrame: DataFrame of the selected run (or full dict if run='all')
    """
    path = Path(dir_name) / f"{file_name}.json"

    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")

    with open(path, "r") as f:
        all_runs = json.load(f)

    if run == "all":
        return all_runs  # return raw dict
    elif run == "latest":
        run_key = max(all_runs, key=lambda k: int(k.split("_")[1]))
    else:
        run_key = f"run_{run}"

    return pd.DataFrame(all_runs[run_key])


def predict_compare(test_data: torchvision.datasets.ImageFolder,
                    model: torch.nn.Module,
                    device: torch.device,
                    class_names,
                    random_seed: int = None):
    """
    Makes random predictions and compares it to the ground truth.
    The results will be plotted.
    """
    if random_seed:
        random.seed(random_seed)

    test_samples = []
    test_labels = []

    for sample, label in random.sample(list(test_data), k=9):
        test_samples.append(sample)
        test_labels.append(label)

    # Make predictions
    pred_probs = []

    model.eval()
    with torch.inference_mode():
        for sample in test_samples:
            sample = torch.unsqueeze(sample, dim=0).to(
                device)  # Add batch dimension
            logit = model(sample)
            pred = torch.softmax(logit.squeeze(), dim=0)
            pred_probs.append(pred.cpu())

    pred_probs = torch.stack(pred_probs)
    pred_classes = torch.argmax(pred_probs, dim=1)

    # Plotting and Comparing prediction to ground truth
    plt.figure(figsize=(10, 7))

    rows, cols = 3, 3

    for i, sample in enumerate(test_samples):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(sample.squeeze(dim=0).permute(1, 2, 0), cmap="gray")

        pred_label = class_names[pred_classes[i]]
        truth_label = class_names[test_labels[i]]
        title_text = f"Pred: {pred_label} | Truth: {truth_label}"

        if pred_label == truth_label:
            # green text if prediction is correct
            plt.title(title_text, fontsize=10, c="g")
        else:
            # red text if prediction is incorrect
            plt.title(title_text, fontsize=10, c="r")

        plt.axis(False)


def make_predictions(model: torch.nn.Module,
                     test_dataloader: torch.utils.data.DataLoader,
                     test_data: torchvision.datasets.ImageFolder,
                     device: torch.device):
    """Makes predictions and returns `y_preds` and `y_true`"""
    y_preds = []

    model.eval()
    with torch.inference_mode():
        for X, y, in tqdm(test_dataloader, desc="Making predictions"):
            X, y = X.to(device), y.to(device)
            y_logits = model(X)
            y_pred = torch.softmax(y_logits.squeeze(), dim=1).argmax(dim=1)
            y_preds.append(y_pred.cpu())

    y_preds = torch.cat(y_preds)
    y_true = torch.tensor(test_data.targets)

    return y_preds, y_true


def __get_gradcam_config(model_name: str, model: torch.nn.Module):
    if "efficientnet" in model_name:
        return model.features[-1], None
    elif "resnet" in model_name:
        return model.layer4[-1], None
    elif "vgg" in model_name:
        return model.features[-1], None
    elif "regnet" in model_name:
        return model.trunk_output, None
    elif "swin" in model_name:
        def reshape_transform(tensor, height=7, width=7):
            result = tensor.reshape(tensor.size(
                0), height, width, tensor.size(2))
            result = result.permute(0, 3, 1, 2)
            return result

        target_layer = model.features[3].SwinTransformerBlockV2[-1].norm2
        return target_layer, reshape_transform
    else:
        raise ValueError(f"No Grad-CAM config for model '{model_name}'")


def make_single_prediction(model: torch.nn.Module,
                           image_path: Path,
                           class_names: list,
                           model_name: str = "Model",
                           transforms: T.Compose = None,
                           device: torch.device = torch.device("cpu"),
                           show_confidence: bool = False):
    if transforms is None:
        transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    image = Image.open(image_path).convert("RGB")
    image_resized = image.resize((224, 224))
    original_image = np.array(image_resized).astype(np.float32) / 255.0
    image_tensor = transforms(image).unsqueeze(0).to(device)

    model.eval().to(device)
    with torch.inference_mode():
        output = model(image_tensor)
        pred_prob = torch.softmax(output, dim=1).squeeze(0)
        pred_class = pred_prob.argmax().item()

    # Grad-CAM
    try:
        target_layer, reshape_transform = __get_gradcam_config(
            model_name.lower(), model)
        cam = GradCAM(model=model,
                      target_layers=[target_layer],
                      reshape_transform=reshape_transform)
        grayscale_cam = cam(input_tensor=image_tensor,
                            targets=[ClassifierOutputTarget(pred_class)])[0]
        cam_img = show_cam_on_image(
            original_image, grayscale_cam, use_rgb=True)
    except Exception as e:
        print(f"Grad-CAM failed: {e}")
        cam_img = original_image  # fallback

    if show_confidence:
        print("\n", model_name)
        print(f"Predicted class index: {class_names[pred_class]}")
        print("Confidence values:")
        for i, prob in enumerate(pred_prob):
            print(f"{class_names[i]}: {prob * 100:.2f}%")

    return pred_prob, pred_class, cam_img
