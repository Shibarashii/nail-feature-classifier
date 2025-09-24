# src/train.py
import os
import torch
from src.utils.class_weight import get_class_weight
from src.utils.helpers import get_root_dir
import yaml
from src.data.dataloaders import create_dataloaders
from src.data.transforms import get_train_transforms, get_test_transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
from src.engine import train_model
import argparse
from src.models.get_model import get_model
from src.utils.seed import set_seed
from src.utils.save import save_experiment_outputs
from torchmetrics import Accuracy


def run_model(model: nn.Module, model_name: str, strategy: str):
    """
    Train the given model and save the best model weights and training history.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to train.
    device : torch.device
        Device to train on (cuda or cpu).
    model_name : str
        Name of the model (used for saving).
    strategy : str
        Training strategy (scratch, baseline, full_finetune, gradual_unfreeze).
    """
    model = model.to(device)

    # Loss, optimizer, scheduler
    CRITERION = torch.nn.CrossEntropyLoss(
        weight=get_class_weight(device=device))
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )

    accuracy_fn = Accuracy(
        task='multiclass', num_classes=num_classes).to(device)
    scheduler_params = config.get("scheduler_params", {})
    scheduler = ReduceLROnPlateau(optimizer, **scheduler_params)

    # TRAINING THE MODEL
    best_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=CRITERION,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        accuracy_fn=accuracy_fn,
        num_epochs=NUM_EPOCHS,
        patience=5,
        print_summary=True
    )

    # SAVING THE MODEL
    save_experiment_outputs(
        best_model=best_model,
        history=history,
        model_name=model_name,
        strategy=strategy,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model with a given strategy"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (efficientnetv2s, resnet50, etc.)")
    parser.add_argument("--strategy", type=str, required=True,
                        help="Training strategy (scratch, baseline, full_finetune, gradual_unfreeze)")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="YAML config file to use (default: config.yaml)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load YAML config
    config_path = get_root_dir() / "src" / "configs" / args.config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Hyperparameters from config
    NUM_EPOCHS = config["defaults"]["num_epochs"]
    BATCH_SIZE = config["defaults"]["batch_size"]
    SEED = config["defaults"]["seed"]
    LEARNING_RATE = float(config["defaults"]["lr"])

    # Determine if this is a test run based on config filename
    test_run = args.config.lower().startswith("test")

    set_seed(SEED)

    train_loader, val_loader, test_loader, class_names, class_to_idx, num_classes = create_dataloaders(
        train_transform=get_train_transforms(test_run=test_run),
        test_transform=get_test_transforms(test_run=test_run),
        batch_size=BATCH_SIZE,
        num_workers=os.cpu_count(),
        test_run=test_run
    )

    # Get model based on CLI args
    model = get_model(args.model, args.strategy, num_classes=num_classes)

    # Run training
    run_model(model, args.model, args.strategy)
