# src/experiments.py
import numpy as np
import os
import torch
from utils.class_weight import get_class_weight
from utils.helpers import get_root_dir
import yaml
from models.efficientnetv2s import get_from_scratch_model as efficientnetv2s_scratch
from models.efficientnetv2s import get_baseline_model as efficientnetv2s_baseline
from models.efficientnetv2s import get_full_finetune_model as efficientnetv2s_full
from models.efficientnetv2s import get_gradual_unfreeze_model as efficientnetv2s_gradual
from data.dataloaders import create_dataloaders
from data.transforms import get_train_transforms, get_test_transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
from engine import train_model
import argparse
from models.efficientnetv2s import get_baseline_model as efficientnetv2s_baseline
from models.get_model import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load YAML config
config_path = get_root_dir() / "configs" / "config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Hyperparameters
SEED = config["defaults"]["seed"]
NUM_EPOCHS = config["defaults"]["num_epochs"]
BATCH_SIZE = config["defaults"]["batch_size"]
LEARNING_RATE = config["defaults"]["lr"]


train_loader, val_loader, test_loader, class_names, class_to_idx, num_classes = create_dataloaders(
    transform=get_train_transforms(),
    batch_size=BATCH_SIZE,
    num_workers=os.cpu_count())


def run_model(model: nn.Module, device: torch.device):
    model = model.to(device)

    # Loss, optimizer, scheduler
    CRITERION = torch.nn.CrossEntropyLoss(get_class_weight(device=device))
    optimizer = torch.optim.AdamW(
        # only trainable params
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
    )

    scheduler_params = config.get("scheduler_params", {})
    scheduler = ReduceLROnPlateau(optimizer, **scheduler_params)

    best_model, history = train_model(model=model,
                                      train_loader=train_loader,
                                      val_loader=val_loader,
                                      criterion=CRITERION,
                                      optimizer=optimizer,
                                      scheduler=scheduler,
                                      device=device,
                                      num_epochs=NUM_EPOCHS,
                                      patience=5,
                                      print_summary=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model with a given strategy")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (efficientnetv2s, resnet50, etc.)")
    parser.add_argument("--strategy", type=str, required=True,
                        help="Training strategy (scratch, baseline, full_finetune, gradual_unfreeze)")
    args = parser.parse_args()

    # Get model based on CLI args
    model = get_model(args.model, args.strategy, num_classes=num_classes)

    # Run training
    run_model(model, device)
