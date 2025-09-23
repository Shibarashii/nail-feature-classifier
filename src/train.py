import argparse
import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy

from utils.helpers import train_model, save_results

# Import modular functions
from utils.data_utils import create_dataloaders
from transforms import train_transforms, val_test_transforms
import torchvision.models as models

# ------------------------------
# 1. Argument Parser
# ------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Nail Feature Classifier")
    parser.add_argument("--model", type=str,
                        default="efficientnetv2s", help="Model backbone")
    parser.add_argument("--batch_size", type=int,
                        default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--data_dir", type=str,
                        default="data", help="Path to dataset")
    parser.add_argument("--output_dir", type=str,
                        default="outputs", help="Where to save results")
    return parser.parse_args()

# ------------------------------
# 2. Model Factory
# ------------------------------


def build_model(model_name, num_classes):
    if model_name == "efficientnetv2s":
        weights = models.EfficientNet_V2_S_Weights.DEFAULT
        model = models.efficientnet_v2_s(weights=weights)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, num_classes)

    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "vgg16":
        weights = models.VGG16_Weights.DEFAULT
        model = models.vgg16(weights=weights)
        model.classifier[6] = nn.Linear(
            model.classifier[6].in_features, num_classes)

    else:
        raise ValueError(f"Model {model_name} not supported")

    return model

# ------------------------------
# 3. Main Training Script
# ------------------------------


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)

    # ------------------------------
    # Get dataloaders using modular helper
    # ------------------------------
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        train_transform=train_transforms,
        val_transform=val_test_transforms,
        test_transform=val_test_transforms
    )
    num_classes = len(class_names)

    # ------------------------------
    # Build model
    # ------------------------------
    model = build_model(args.model, num_classes).to(device)

    # ------------------------------
    # Training setup
    # ------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5)
    accuracy_fn = MulticlassAccuracy(num_classes=num_classes).to(device)

    # ------------------------------
    # Train the model
    # ------------------------------
    results = train_model(
        epochs=args.epochs,
        model=model,
        train_dataloader=train_loader,
        valid_dataloader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        accuracy_fn=accuracy_fn,
        device=device,
        scheduler=scheduler
    )

    # ------------------------------
    # Save results
    # ------------------------------
    save_results(args.output_dir, args.model, results)
    print("Training complete!")


if __name__ == "__main__":
    main()
