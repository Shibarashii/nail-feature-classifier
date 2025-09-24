from datetime import datetime
from src.utils.helpers import get_root_dir
import json
import torch
from pathlib import Path


def _tensor_to_python(obj):
    """Recursively convert tensors to Python scalars so they're JSON serializable."""
    if isinstance(obj, torch.Tensor):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _tensor_to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_tensor_to_python(v) for v in obj]
    return obj


def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory."""
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(
        ".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)


def save_history(history: list,
                 target_dir: str,
                 filename: str = "history.json"):
    """Saves training history to a target directory as a JSON file."""
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    if not filename.endswith(".json"):
        filename += ".json"

    history_save_path = target_dir_path / filename

    safe_history = _tensor_to_python(history)
    print(f"[INFO] Saving history to: {history_save_path}")
    with open(history_save_path, "w") as f:
        json.dump(safe_history, f, indent=4)


def save_experiment_outputs(
    best_model: torch.nn.Module,
    history: dict,
    model_name: str,
    strategy: str,
    base_dir: str = "output",
    use_timestamp: bool = True,
) -> Path:
    """Save the best model weights and training history."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if use_timestamp else ""
    save_dir = get_root_dir() / "src" / base_dir / model_name / \
        strategy

    if use_timestamp:
        save_dir = save_dir / timestamp

    save_dir.mkdir(parents=True, exist_ok=True)

    # Save best model
    model_path = save_dir / "best_model.pth"
    torch.save(best_model.state_dict(), model_path)
    print(f"[INFO] Saved best model to {model_path}")

    # Save training history
    history_path = save_dir / "history.json"
    safe_history = _tensor_to_python(history)
    with open(history_path, "w") as f:
        json.dump(safe_history, f, indent=4)
    print(f"[INFO] Saved training history to {history_path}")

    return save_dir
