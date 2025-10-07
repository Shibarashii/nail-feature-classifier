# src/utils/helpers.py
from pathlib import Path
from src.models.get_model import get_model
import torch
import json


def get_root_dir():
    return Path(__file__).resolve().parent.parent.parent  # src/utils -> root


def load_model(model_path: str, model_name: str, strategy: str, num_classes: int, device: torch.device):
    """
    Load a trained model from checkpoint.

    Parameters
    ----------
    model_path : str
        Path to the model checkpoint (.pth file).
    model_name : str
        Name of the model architecture.
    strategy : str
        Training strategy used.
    num_classes : int
        Number of output classes.
    device : torch.device
        Device to load the model on.

    Returns
    -------
    model : torch.nn.Module
        Loaded model in evaluation mode.
    """
    # Get model architecture
    model = get_model(model_name, strategy, num_classes)

    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded from: {model_path}")
    return model


def load_history(history_path: str):
    """
    Load training history from JSON file.

    Parameters
    ----------
    history_path : str
        Path to the history JSON file.

    Returns
    -------
    history : list of dict
        Training history containing epoch metrics.
    """
    with open(history_path, 'r') as f:
        history = json.load(f)
    print(f"✓ Training history loaded from: {history_path}")
    return history
