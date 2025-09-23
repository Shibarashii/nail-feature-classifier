import torch
from timeit import default_timer as timer
from tqdm import tqdm


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
