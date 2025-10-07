# src/eval.py
"""
Evaluation script for trained models.
Computes comprehensive metrics for classification tasks including medical diagnostics.
"""
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import yaml
from tqdm.auto import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
)
import json

from src.utils.helpers import get_root_dir
from src.data.dataloaders import create_dataloaders
from src.data.transforms import get_test_transforms
from src.models.get_model import get_model
from src.utils.seed import set_seed


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


def get_predictions(model, dataloader, device):
    """
    Get predictions and true labels from the model.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.
    dataloader : DataLoader
        DataLoader for the dataset.
    device : torch.device
        Device to run inference on.

    Returns
    -------
    all_preds : np.ndarray
        Predicted class labels.
    all_probs : np.ndarray
        Predicted probabilities for each class.
    all_labels : np.ndarray
        True class labels.
    """
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.inference_mode():
        for inputs, labels in tqdm(dataloader, desc="Evaluating", unit="batch"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    return all_preds, all_probs, all_labels


def compute_ml_metrics(y_true, y_pred, class_names):
    """
    Compute standard machine learning classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    class_names : list
        List of class names.

    Returns
    -------
    metrics : dict
        Dictionary containing all computed metrics.
    """
    metrics = {}

    # Overall metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision_macro'] = precision_score(
        y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(
        y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(
        y_true, y_pred, average='macro', zero_division=0)

    metrics['precision_weighted'] = precision_score(
        y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall_weighted'] = recall_score(
        y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_weighted'] = f1_score(
        y_true, y_pred, average='weighted', zero_division=0)

    # Per-class metrics
    precision_per_class = precision_score(
        y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(
        y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    metrics['per_class'] = {}
    for i, class_name in enumerate(class_names):
        metrics['per_class'][class_name] = {
            'precision': float(precision_per_class[i]),
            'recall': float(recall_per_class[i]),
            'f1_score': float(f1_per_class[i])
        }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()

    return metrics


def compute_medical_metrics(y_true, y_pred, class_names):
    """
    Compute medical diagnostic metrics (binary or multi-class one-vs-rest).

    For binary classification:
    - Sensitivity (Recall/TPR): TP / (TP + FN)
    - Specificity (TNR): TN / (TN + FP)
    - PPV (Precision): TP / (TP + FP)
    - NPV: TN / (TN + FN)

    For multi-class: Compute metrics for each class vs rest.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    class_names : list
        List of class names.

    Returns
    -------
    medical_metrics : dict
        Dictionary containing medical diagnostic metrics.
    """
    medical_metrics = {}
    cm = confusion_matrix(y_true, y_pred)
    num_classes = len(class_names)

    if num_classes == 2:
        # Binary classification
        tn, fp, fn, tp = cm.ravel()

        medical_metrics['sensitivity'] = tp / \
            (tp + fn) if (tp + fn) > 0 else 0.0
        medical_metrics['specificity'] = tn / \
            (tn + fp) if (tn + fp) > 0 else 0.0
        medical_metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        medical_metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0

        # Additional metrics
        medical_metrics['true_positives'] = int(tp)
        medical_metrics['true_negatives'] = int(tn)
        medical_metrics['false_positives'] = int(fp)
        medical_metrics['false_negatives'] = int(fn)

    else:
        # Multi-class: one-vs-rest for each class
        medical_metrics['per_class'] = {}

        for i, class_name in enumerate(class_names):
            # Convert to binary: class i vs all others
            y_true_binary = (y_true == i).astype(int)
            y_pred_binary = (y_pred == i).astype(int)

            tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
            tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
            fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
            fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

            medical_metrics['per_class'][class_name] = {
                'sensitivity': float(sensitivity),
                'specificity': float(specificity),
                'ppv': float(ppv),
                'npv': float(npv),
                'true_positives': int(tp),
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn)
            }

        # Compute macro-averaged metrics
        sensitivities = [medical_metrics['per_class']
                         [cn]['sensitivity'] for cn in class_names]
        specificities = [medical_metrics['per_class']
                         [cn]['specificity'] for cn in class_names]
        ppvs = [medical_metrics['per_class'][cn]['ppv'] for cn in class_names]
        npvs = [medical_metrics['per_class'][cn]['npv'] for cn in class_names]

        medical_metrics['macro_avg'] = {
            'sensitivity': float(np.mean(sensitivities)),
            'specificity': float(np.mean(specificities)),
            'ppv': float(np.mean(ppvs)),
            'npv': float(np.mean(npvs))
        }

    return medical_metrics


def plot_confusion_matrix(cm, class_names, save_path):
    """
    Plot and save confusion matrix.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix.
    class_names : list
        List of class names.
    save_path : str
        Path to save the plot.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved to: {save_path}")


def plot_normalized_confusion_matrix(cm, class_names, save_path):
    """
    Plot and save normalized confusion matrix (percentages).

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix.
    class_names : list
        List of class names.
    save_path : str
        Path to save the plot.
    """
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Percentage'})
    plt.title('Normalized Confusion Matrix', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Normalized confusion matrix saved to: {save_path}")


def plot_per_class_metrics(metrics, class_names, save_path):
    """
    Plot per-class precision, recall, and F1-score.

    Parameters
    ----------
    metrics : dict
        Dictionary containing per-class metrics.
    class_names : list
        List of class names.
    save_path : str
        Path to save the plot.
    """
    precision = [metrics['per_class'][cn]['precision'] for cn in class_names]
    recall = [metrics['per_class'][cn]['recall'] for cn in class_names]
    f1 = [metrics['per_class'][cn]['f1_score'] for cn in class_names]

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)

    ax.set_xlabel('Classes', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Metrics', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Per-class metrics plot saved to: {save_path}")


def plot_roc_curves(y_true, y_probs, class_names, save_path):
    """
    Plot ROC curves for each class (one-vs-rest).

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_probs : np.ndarray
        Predicted probabilities for each class.
    class_names : list
        List of class names.
    save_path : str
        Path to save the plot.
    """
    plt.figure(figsize=(10, 8))

    for i, class_name in enumerate(class_names):
        # Convert to binary: class i vs rest
        y_true_binary = (y_true == i).astype(int)
        y_score = y_probs[:, i]

        fpr, tpr, _ = roc_curve(y_true_binary, y_score)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves (One-vs-Rest)', fontsize=16, pad=20)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ ROC curves saved to: {save_path}")


def plot_training_history(history, save_dir):
    """
    Plot training and validation loss, accuracy, and learning rate over epochs.
    Adds a universal plot with all metrics combined.

    Parameters
    ----------
    history : list of dicts
        Training log, each dict containing epoch, train_loss, val_loss, train_acc, val_acc, lr
    save_dir : str or Path
        Directory to save plots
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]
    train_acc = [h['train_acc'] for h in history]
    val_acc = [h['val_acc'] for h in history]
    lrs = [h['lr'] for h in history]

    # Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Train Loss', marker='o', markersize=3)
    plt.plot(epochs, val_loss, label='Val Loss', marker='o', markersize=3)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'loss_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Loss curve saved to: {save_dir / 'loss_curve.png'}")

    # Accuracy plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_acc, label='Train Accuracy',
             marker='o', markersize=3)
    plt.plot(epochs, val_acc, label='Val Accuracy', marker='o', markersize=3)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Training and Validation Accuracy', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'accuracy_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Accuracy curve saved to: {save_dir / 'accuracy_curve.png'}")

    # Learning rate plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, lrs, label='Learning Rate',
             marker='o', markersize=3, color='purple')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Learning Rate Schedule', fontsize=14)
    plt.yscale('log')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'lr_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Learning rate curve saved to: {save_dir / 'lr_curve.png'}")

    # Universal training history with subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

    # Loss subplot
    axs[0].plot(epochs, train_loss, label='Train Loss',
                marker='o', markersize=3)
    axs[0].plot(epochs, val_loss, label='Val Loss', marker='o', markersize=3)
    axs[0].set_ylabel('Loss', fontsize=12)
    axs[0].set_title('Training and Validation Loss', fontsize=14)
    axs[0].legend()
    axs[0].grid(alpha=0.3)

    # Accuracy subplot
    axs[1].plot(epochs, train_acc, label='Train Accuracy',
                marker='x', markersize=3)
    axs[1].plot(epochs, val_acc, label='Val Accuracy',
                marker='x', markersize=3)
    axs[1].set_ylabel('Accuracy', fontsize=12)
    axs[1].set_title('Training and Validation Accuracy', fontsize=14)
    axs[1].legend()
    axs[1].grid(alpha=0.3)

    # Learning rate subplot
    axs[2].plot(epochs, lrs, label='Learning Rate',
                marker='s', markersize=3, color='purple')
    axs[2].set_xlabel('Epoch', fontsize=12)
    axs[2].set_ylabel('Learning Rate', fontsize=12)
    axs[2].set_title('Learning Rate Schedule', fontsize=14)
    axs[2].set_yscale('log')  # Keep log scale to show small changes clearly
    axs[2].legend()
    axs[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'training_history.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print(
        f"✓ Universal training history saved to: {save_dir / 'training_history.png'}")


def save_metrics_to_json(metrics, save_path):
    """
    Save metrics dictionary to JSON file.

    Parameters
    ----------
    metrics : dict
        Dictionary containing all metrics.
    save_path : str
        Path to save the JSON file.
    """
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"✓ Metrics saved to: {save_path}")


def print_summary(ml_metrics, medical_metrics, class_names):
    """
    Print a formatted summary of all metrics.

    Parameters
    ----------
    ml_metrics : dict
        Machine learning metrics.
    medical_metrics : dict
        Medical diagnostic metrics.
    class_names : list
        List of class names.
    """
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)

    print("\n📊 Overall Machine Learning Metrics:")
    print(f"  Accuracy:           {ml_metrics['accuracy']:.4f}")
    print(f"  Precision (macro):  {ml_metrics['precision_macro']:.4f}")
    print(f"  Recall (macro):     {ml_metrics['recall_macro']:.4f}")
    print(f"  F1-Score (macro):   {ml_metrics['f1_macro']:.4f}")
    print(f"\n  Precision (weighted): {ml_metrics['precision_weighted']:.4f}")
    print(f"  Recall (weighted):    {ml_metrics['recall_weighted']:.4f}")
    print(f"  F1-Score (weighted):  {ml_metrics['f1_weighted']:.4f}")

    if 'macro_avg' in medical_metrics:
        print("\n🏥 Overall Medical Diagnostic Metrics (Macro-Averaged):")
        print(
            f"  Sensitivity (Recall): {medical_metrics['macro_avg']['sensitivity']:.4f}")
        print(
            f"  Specificity:          {medical_metrics['macro_avg']['specificity']:.4f}")
        print(
            f"  PPV (Precision):      {medical_metrics['macro_avg']['ppv']:.4f}")
        print(
            f"  NPV:                  {medical_metrics['macro_avg']['npv']:.4f}")
    else:
        print("\n🏥 Medical Diagnostic Metrics (Binary):")
        print(f"  Sensitivity (Recall): {medical_metrics['sensitivity']:.4f}")
        print(f"  Specificity:          {medical_metrics['specificity']:.4f}")
        print(f"  PPV (Precision):      {medical_metrics['ppv']:.4f}")
        print(f"  NPV:                  {medical_metrics['npv']:.4f}")

    print("\n📈 Per-Class Performance:")
    print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 60)
    for class_name in class_names:
        p = ml_metrics['per_class'][class_name]['precision']
        r = ml_metrics['per_class'][class_name]['recall']
        f1 = ml_metrics['per_class'][class_name]['f1_score']
        print(f"{class_name:<20} {p:<12.4f} {r:<12.4f} {f1:<12.4f}")

    print("\n" + "="*70)


def evaluate_model(
    experiment_path: str,
    dataset_split: str = 'test'
):
    """
    Complete evaluation pipeline for a trained model.

    Parameters
    ----------
    experiment_path : str
        Path to the experiment directory containing best_model.pth and history.json
    model_name : str
        Name of the model architecture.
    strategy : str
        Training strategy used.
    dataset_split : str
        Which dataset split to evaluate on ('test', 'val', 'train').
    """
    # Setup paths
    experiment_dir = Path(experiment_path)
    model_path = experiment_dir / "best_model.pth"
    history_path = experiment_dir / "history.json"
    output_dir = experiment_dir / "evaluation"

    # Validate paths
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not history_path.exists():
        raise FileNotFoundError(f"History not found: {history_path}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Experiment directory: {experiment_dir}")

    # Load YAML config
    config_path = get_root_dir() / "src" / "configs" / "eval.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    BATCH_SIZE = config["defaults"]["batch_size"]
    SEED = config["defaults"]["seed"]
    test_run = Path(config_path).stem.lower().startswith("test")

    set_seed(SEED)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n📁 Loading dataset...")
    train_loader, val_loader, test_loader, _, class_to_idx, num_classes = create_dataloaders(
        train_transform=get_test_transforms(test_run=test_run),
        test_transform=get_test_transforms(test_run=test_run),
        batch_size=BATCH_SIZE,
        num_workers=os.cpu_count(),
        test_run=test_run
    )

    class_names = [
        "Melanonychia",
        "Beau's Lines",
        "Blue Nail",
        "Clubbing",
        "Healthy Nail",
        "Koilonychia",
        "Muehrcke's Lines",
        "Onychogryphosis",
        "Pitting",
        "Terry's Nails"
    ]

    # Select dataloader based on split
    if dataset_split == 'test':
        dataloader = test_loader
    elif dataset_split == 'val':
        dataloader = val_loader
    elif dataset_split == 'train':
        dataloader = train_loader
    else:
        raise ValueError(
            f"Invalid dataset_split: {dataset_split}. Choose from 'test', 'val', 'train'.")

    print(f"✓ Evaluating on {dataset_split} set")
    print(f"✓ Number of classes: {num_classes}")
    print(f"✓ Class names: {class_names}")

    # Load model
    print(f"\n🔧 Loading model...")
    # Automatically infer model_name and strategy from folder structure
    try:
        strategy = experiment_dir.parent.name
        model_name = experiment_dir.parent.parent.name
    except IndexError:
        raise ValueError(
            f"Cannot infer model_name and strategy from path: {experiment_path}")

    model = load_model(str(model_path), model_name,
                       strategy, num_classes, device)

    # Load training history
    print(f"\n📜 Loading training history...")
    history = load_history(str(history_path))

    # Get predictions
    print(f"\n🔮 Generating predictions...")
    y_pred, y_probs, y_true = get_predictions(model, dataloader, device)
    print(f"✓ Processed {len(y_true)} samples")

    # Compute metrics
    print(f"\n📊 Computing metrics...")
    ml_metrics = compute_ml_metrics(y_true, y_pred, class_names)
    medical_metrics = compute_medical_metrics(y_true, y_pred, class_names)

    # Load training history
    history = load_history(str(history_path))

    # Find epoch with lowest validation loss
    min_val_loss_epoch = min(history, key=lambda x: x['val_loss'])
    lowest_val_loss = min_val_loss_epoch['val_loss']
    epoch_of_lowest_val_loss = min_val_loss_epoch['epoch']

    # Combine all metrics (including lowest_val_loss)
    all_metrics = {
        'model_name': model_name,
        'strategy': strategy,
        'dataset_split': dataset_split,
        'num_samples': int(len(y_true)),
        'num_classes': int(num_classes),
        'class_names': class_names,
        'lowest_val_loss': float(lowest_val_loss),
        'epoch_of_lowest_val_loss': int(epoch_of_lowest_val_loss),
        'ml_metrics': ml_metrics,
        'medical_metrics': medical_metrics,
    }

    # Save metrics
    metrics_path = output_dir / 'metrics.json'
    save_metrics_to_json(all_metrics, str(metrics_path))

    # Generate plots
    print(f"\n📈 Generating visualizations...")

    # Training history plots
    plot_training_history(history, output_dir)

    # Evaluation plots
    cm = np.array(ml_metrics['confusion_matrix'])

    plot_confusion_matrix(
        cm, class_names,
        str(output_dir / 'confusion_matrix.png')
    )

    plot_normalized_confusion_matrix(
        cm, class_names,
        str(output_dir / 'confusion_matrix_normalized.png')
    )

    plot_per_class_metrics(
        ml_metrics, class_names,
        str(output_dir / 'per_class_metrics.png')
    )

    plot_roc_curves(
        y_true, y_probs, class_names,
        str(output_dir / 'roc_curves.png')
    )

    # Print summary
    print_summary(ml_metrics, medical_metrics, class_names)

    print(f"\n✅ Evaluation complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python -m src.eval --path outputs/swinv2t_gradual_unfreeze --model swinv2t --strategy gradual_unfreeze
  python -m src.eval --path outputs/resnet50_baseline --model resnet50 --strategy baseline --split val
        """
    )
    parser.add_argument("path", type=str,
                        help="Path to experiment directory (containing best_model.pth and history.json)")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val", "train"],
                        help="Dataset split to evaluate on (default: test)")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all models under the given path recursively")

    args = parser.parse_args()

    path = Path(args.path)

    if args.all:
        # Find all subdirectories with best_model.pth and history.json
        experiment_dirs = []
        for subdir in path.rglob("*"):
            if subdir.is_dir() and (subdir / "best_model.pth").exists() and (subdir / "history.json").exists():
                experiment_dirs.append(subdir)

        if not experiment_dirs:
            print(
                f"❌ No experiments with best_model.pth + history.json found under {path}")
        else:
            print(
                f"✅ Found {len(experiment_dirs)} experiments. Evaluating all...")
            for exp_dir in experiment_dirs:
                print(f"\n🔹 Evaluating {exp_dir}")
                evaluate_model(experiment_path=str(
                    exp_dir), dataset_split=args.split)
    else:
        # Single experiment evaluation
        evaluate_model(
            experiment_path=str(path),
            dataset_split=args.split
        )
