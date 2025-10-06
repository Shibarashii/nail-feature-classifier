# src/engine.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from time import time
from torchsummary import summary
from typing import Tuple
from torchmetrics import Metric
from tqdm.auto import tqdm
import copy
from src.utils.unfreeze import (
    unfreeze_layer_group,
    should_unfreeze_next_group,
    print_trainable_parameters
)


def train_step(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    accuracy_fn: Metric,
    device: torch.device,
) -> Tuple[float, float]:
    """Training step."""

    model.train()
    running_loss, running_acc = 0.0, 0.0

    for inputs, targets in tqdm(data_loader, desc="Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += accuracy_fn(targets, outputs.argmax(dim=1))
    avg_loss = running_loss / len(data_loader)
    avg_acc = running_acc / len(data_loader)
    return avg_loss, avg_acc


def val_step(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    accuracy_fn: Metric,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluates the model on the validation set."""

    model.eval()
    running_loss, running_acc = 0.0, 0.0

    with torch.inference_mode():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            running_acc += accuracy_fn(targets, outputs.argmax(dim=1))

    avg_loss = running_loss / len(data_loader)
    avg_acc = running_acc / len(data_loader)
    return avg_loss, avg_acc


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.ReduceLROnPlateau,
    device: torch.device,
    accuracy_fn: Metric,
    num_epochs: int = 50,
    patience: int = 5,
    print_summary: bool = True,
    layer_groups: list = None,
    unfreeze_schedule: list = None,
    scheduler_params: dict = None
):
    """
    Train a PyTorch model with early stopping and learning rate scheduling.
    Supports gradual unfreezing of layer groups.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to be trained.
    train_loader : DataLoader
        DataLoader for the training dataset.
    val_loader : DataLoader
        DataLoader for the validation dataset.
    criterion : nn.Module
        Loss function (e.g., nn.CrossEntropyLoss).
    optimizer : torch.optim.Optimizer
        Optimizer for updating model parameters (e.g., AdamW, SGD).
    scheduler : torch.optim.lr_scheduler.ReduceLROnPlateau
        Learning rate scheduler that reduces LR when a metric has stopped improving.
    device : torch.device
        Device to train the model on (`cuda` or `cpu`).
    accuracy_fn : Metric
        Metric function for calculating accuracy.
    num_epochs : int, optional
        Maximum number of training epochs (default is 50).
    patience : int, optional
        Number of epochs with no improvement on validation loss before early stopping (default is 5).
    print_summary : bool, optional
        Whether to print a model summary before training starts (default is True).
    layer_groups : list, optional
        List of layer groups for gradual unfreezing (None for standard training).
    unfreeze_schedule : list, optional
        Schedule of epochs at which to unfreeze each layer group.
    scheduler_params : dict, optional
        Parameters for recreating the scheduler after unfreezing.

    Returns
    -------
    model : nn.Module
        The trained PyTorch model with the best validation loss.
    history : list of dict
        List containing metrics for each epoch.

    Notes
    -----
    When gradual unfreezing is enabled:
    - The scheduler state is reset when a new layer group is unfrozen
    - This gives the model patience to adapt to newly trainable parameters
    - The learning rate is preserved across unfreezing events
    - Early stopping patience is NOT reset on unfreezing
    """
    if print_summary:
        sample_input, _ = next(iter(train_loader))
        summary(model, input_size=sample_input.shape[1:], device=str(device))

    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = []
    best_model_wts = copy.deepcopy(model.state_dict())

    # Initialize gradual unfreezing tracking
    current_unfrozen_group = 0 if layer_groups else None
    gradual_unfreeze_enabled = layer_groups is not None and unfreeze_schedule is not None

    for epoch in range(1, num_epochs + 1):
        # Handle gradual unfreezing
        if gradual_unfreeze_enabled:
            if should_unfreeze_next_group(epoch, unfreeze_schedule, current_unfrozen_group):
                current_unfrozen_group += 1
                unfreeze_layer_group(layer_groups[current_unfrozen_group])

                print(f"\n{'='*60}")
                print(
                    f"Unfreezing layer group {current_unfrozen_group} at epoch {epoch}")
                print_trainable_parameters(model)

                # Preserve current learning rate and weight decay
                current_lr = optimizer.param_groups[0]['lr']
                current_wd = optimizer.param_groups[0].get('weight_decay', 0)

                print(f"  Current LR: {current_lr:.6f}")
                print(f"  Weight decay: {current_wd:.6f}")

                # Reinitialize optimizer with newly unfrozen parameters
                optimizer = type(optimizer)(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=current_lr,
                    weight_decay=current_wd
                )

                # CRITICAL: Reinitialize scheduler with new optimizer
                # This resets the scheduler's internal state (patience counter, best value)
                # giving the model time to adapt to newly unfrozen parameters
                if scheduler_params is not None:
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, **scheduler_params
                    )
                    print(f"✓ Scheduler reinitialized with new optimizer")
                    print(f"  Scheduler will start fresh patience tracking")
                else:
                    # Fallback with default parameters
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        mode='min',
                        factor=0.5,
                        patience=5,
                        threshold=0.0001,
                        cooldown=1,
                        min_lr=1e-6
                    )
                    print(f"⚠ Scheduler reinitialized with default parameters")

                print(f"{'='*60}\n")

        start_time = time()

        # Train & validate
        train_loss, train_acc = train_step(
            model, train_loader, criterion, optimizer, accuracy_fn, device)
        val_loss, val_acc = val_step(
            model, val_loader, criterion, accuracy_fn, device)

        # Scheduler step (monitors validation loss)
        scheduler.step(val_loss)

        # Epoch timing and learning rate
        epoch_time = time() - start_time
        current_lr = optimizer.param_groups[0]['lr']

        # Save epoch metrics
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc.item() if hasattr(train_acc, 'item') else train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc.item() if hasattr(val_acc, 'item') else val_acc,
            'lr': current_lr,
            'time': epoch_time,
            'unfrozen_group': current_unfrozen_group if gradual_unfreeze_enabled else None
        })

        print(f"Epoch [{epoch}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.6f} | Time: {epoch_time:.2f}s")

        # Save best model and update early stopping counter
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            print(f"Best validation loss: {best_val_loss:.4f}")
            break

    # Load best weights before returning
    model.load_state_dict(best_model_wts)
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"{'='*60}\n")

    return model, history
