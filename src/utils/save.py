from datetime import datetime
from utils.helpers import get_root_dir
import json
import torch
from pathlib import Path


def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
      model: A target PyTorch model to save.
      target_dir: A directory for saving the model to.
      model_name: A filename for the saved model. Should include
        either ".pth" or ".pt" as the file extension.

    Example usage:
      save_model(model=model_0,
                 target_dir="models",
                 model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                          exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(
        ".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
               f=model_save_path)


def save_history(history: list,
                 target_dir: str,
                 filename: str = "history.json"):
    """Saves training history to a target directory as a JSON file.

    Args:
        history: A list of dictionaries containing epoch metrics.
        target_dir: Directory where the history will be saved.
        filename: Name of the JSON file (default "history.json").

    Example usage:
        save_history(history=history_list,
                     target_dir="saved_models/efficientnetv2s/baseline",
                     filename="history.json")
    """
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    if not filename.endswith(".json"):
        filename += ".json"

    history_save_path = target_dir_path / filename

    print(f"[INFO] Saving history to: {history_save_path}")
    with open(history_save_path, "w") as f:
        json.dump(history, f, indent=4)


def save_experiment_outputs(
    best_model: torch.nn.Module,
    history: dict,
    model_name: str,
    strategy: str,
    num_epochs: int,
    base_dir: str = "outputs",
    use_timestamp: bool = True,
) -> Path:
    """
    Save the best model weights and training history.

    Args:
        best_model (torch.nn.Module): The trained model.
        history (dict): Training history.
        model_name (str): Name of the model.
        strategy (str): Training strategy.
        num_epochs (int): Number of epochs trained.
        base_dir (str, optional): Base directory to save outputs. Defaults to "outputs".
        use_timestamp (bool, optional): Whether to append a timestamp to the save dir.

    Returns:
        Path: Path to the save directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if use_timestamp else ""
    save_dir = get_root_dir() / base_dir / model_name / strategy / str(num_epochs)

    if use_timestamp:
        save_dir = save_dir / timestamp

    save_dir.mkdir(parents=True, exist_ok=True)

    # Save best model
    model_path = save_dir / "best_model.pth"
    torch.save(best_model.state_dict(), model_path)
    print(f"[INFO] Saved best model to {model_path}")

    # Save training history
    history_path = save_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"[INFO] Saved training history to {history_path}")

    return save_dir
