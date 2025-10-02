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
    patience: int = 5,         # Early stopping patience
    print_summary: bool = True
):
    """
    Train a PyTorch model with early stopping and learning rate scheduling.

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
    num_epochs : int, optional
        Maximum number of training epochs (default is 50).
    patience : int, optional
        Number of epochs with no improvement on validation loss before early stopping (default is 5).
    print_summary : bool, optional
        Whether to print a model summary before training starts (default is True).

    Returns
    -------
    model : nn.Module
        The trained PyTorch model with the best validation loss.
    history : list of dict
        List containing metrics for each epoch, including:
        - 'epoch': Epoch number
        - 'train_loss': Training loss
        - 'train_acc': Training accuracy
        - 'val_loss': Validation loss
        - 'val_acc': Validation accuracy
        - 'lr': Current learning rate
        - 'time': Epoch duration in seconds

    Notes
    -----
    - Early stopping is triggered if the validation loss does not improve for `patience` consecutive epochs.
    - The best model (lowest validation loss) is saved to "best_model.pth".
    - The learning rate scheduler is stepped after each validation epoch using the current validation loss.
    """
    if print_summary:
        sample_input, _ = next(iter(train_loader))
        summary(model, input_size=sample_input.shape[1:], device=str(device))

    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = []
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(1, num_epochs + 1):
        start_time = time()

        # Train & validate
        train_loss, train_acc = train_step(
            model, train_loader, criterion, optimizer, accuracy_fn, device)
        val_loss, val_acc = val_step(
            model, val_loader, criterion, accuracy_fn, device)

        # Scheduler step (Reduce LR if val_loss plateaus)
        scheduler.step(val_loss)

        # Epoch timing and learning rate
        epoch_time = time() - start_time
        current_lr = optimizer.param_groups[0]['lr']

        # Save epoch metrics
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': current_lr,
            'time': epoch_time
        })

        print(f"Epoch [{epoch}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.6f} | Time: {epoch_time:.2f}s")

        # Save best model and reset patience counter
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    # Load best weights before returning
    model.load_state_dict(best_model_wts)
    return model, history
