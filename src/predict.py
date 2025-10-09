# src/predict.py
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
from src.utils.interpretability import generate_gradcam
from src.data.transforms import get_test_transforms
from src.utils.helpers import load_model
from src.data.dataloaders import get_class_names
import argparse
import matplotlib.pyplot as plt


def preprocess_image(image_path):
    """Preprocess image for model input."""
    transform = get_test_transforms()
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)
    return img, img_tensor


def predict_single(model_path, image_path, device, generate_gradcam_viz=True, model_name=None):
    """
    Perform single image prediction with optional Grad-CAM generation.

    Args:
        model_path (str/Path): Path to the model weights (.pth file)
        image_path (str/Path): Path to the input image
        device (torch.device): Device to run inference on
        generate_gradcam_viz (bool): Whether to generate Grad-CAM visualization
        model_name (str, optional): Model name (if not provided, inferred from path)

    Returns:
        dict: Dictionary containing:
            - pred_class (str): Predicted class name
            - confidence (float): Prediction confidence
            - probs (torch.Tensor): Full probability distribution
            - pred_idx (int): Predicted class index
            - model_name (str): Name of the model
            - strategy (str): Training strategy
            - orig_img (PIL.Image): Original image (if gradcam enabled)
            - gradcam_img (numpy.ndarray): Grad-CAM visualization (if gradcam enabled)
    """
    model_path = Path(model_path)

    # Validate paths
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Infer model name and strategy from path if not provided
    if model_name is None:
        try:
            # Check if path matches: src/best_models/{model_name}/best_model.pth
            if model_path.parent.parent.name == "best_models":
                model_name = model_path.parent.name
                strategy = "best_model"  # For best_models, we use a generic strategy
            else:
                # Legacy path: src/output/{model_name}/{strategy}/{timestamp}/best_model.pth
                strategy = model_path.parent.parent.name
                model_name = model_path.parent.parent.parent.name
        except IndexError:
            raise ValueError(
                f"Cannot infer model_name and strategy from path: {model_path}")
    else:
        # If model_name is provided, use generic strategy
        strategy = "best_model"

    # Load classes
    class_names = get_class_names()
    num_classes = len(class_names)

    # Load model
    model = load_model(
        str(model_path),
        model_name,
        "scratch",  # must be scratch para gumana nang maayos yung model
        num_classes,
        device
    )

    # Preprocess image
    img, img_tensor = preprocess_image(image_path)
    img_tensor = img_tensor.to(device)

    # Inference
    model.eval()
    with torch.inference_mode():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()

    pred_class = class_names[pred_idx]
    confidence = probs[0][pred_idx].item()

    # Prepare result dictionary
    result = {
        "pred_class": pred_class,
        "confidence": confidence,
        "probs": probs,
        "pred_idx": pred_idx,
        "model_name": model_name,
        "strategy": strategy,
        "class_names": class_names
    }

    # Generate Grad-CAM if requested
    if generate_gradcam_viz and model_name:
        orig_img, gradcam_img = generate_gradcam(
            model, model_name, pred_idx, img_tensor, img, device
        )
        result["orig_img"] = orig_img
        result["gradcam_img"] = gradcam_img

    return result


def visualize_prediction(result, image_path, save_dir="src/predictions"):
    """
    Visualize prediction results with Grad-CAM.

    Args:
        result (dict): Result dictionary from predict_single()
        image_path (str/Path): Path to the original image (for filename)
        save_dir (str/Path): Directory to save visualization

    Returns:
        Path: Path to saved visualization
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if "gradcam_img" not in result:
        raise ValueError("Result dictionary does not contain Grad-CAM visualization. "
                         "Call predict_single() with generate_gradcam_viz=True")

    # Create visualization
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Main title for both images
    fig.suptitle(
        f"{result['model_name'].upper()}_{result['strategy'].upper()}\n"
        f"{result['pred_class']}\n{result['confidence'] * 100:.2f}%",
        fontsize=16,
        fontweight="bold",
        y=1.05
    )

    # Original Image
    axs[0].imshow(result["orig_img"])
    axs[0].axis("off")
    axs[0].set_title("Original Image", fontsize=14)

    # Grad-CAM Visualization
    axs[1].imshow(result["gradcam_img"])
    axs[1].axis("off")
    axs[1].set_title("Grad-CAM", fontsize=14)

    # Save the visualization
    save_path = save_dir / \
        f"prediction_{Path(image_path).stem}_{result['model_name']}.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    return save_path


def resolve_paths(image_name, model_name):
    """
    Resolve simple names to full paths.

    Args:
        image_name (str): Image filename (e.g., 'image.png')
        model_name (str): Model name (e.g., 'efficientnetv2s', 'swinv2t')

    Returns:
        tuple: (image_path, model_path) as Path objects

    Raises:
        FileNotFoundError: If paths cannot be resolved
    """
    # Resolve image path
    image_path = Path("data") / image_name
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Resolve model path
    model_path = Path("src/best_models") / model_name / "best_model.pth"
    if not model_path.exists():
        # List available models for helpful error message
        best_models_dir = Path("src/best_models")
        if best_models_dir.exists():
            available = [d.name for d in best_models_dir.iterdir()
                         if d.is_dir()]
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"Available models: {', '.join(available)}"
            )
        else:
            raise FileNotFoundError(
                f"Model directory not found: {best_models_dir}")

    return image_path, model_path


def main(image_path=None, model_path=None, image_name=None, model_name=None, save_dir="src/predictions"):
    """
    Main function for prediction with visualization and printing.

    Args:
        image_path (str/Path): Full path to the input image (old method)
        model_path (str/Path): Full path to the model weights (old method)
        image_name (str): Simple image filename (new method)
        model_name (str): Simple model name (new method)
        save_dir (str/Path): Directory to save outputs
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n🔧 Loading model...")

    # Resolve paths based on which arguments were provided
    if image_name and model_name:
        # New simplified method
        image_path, model_path = resolve_paths(image_name, model_name)
    elif not (image_path and model_path):
        raise ValueError(
            "Must provide either:\n"
            "  - image_name and model_name (simplified), or\n"
            "  - image_path and model_path (full paths)"
        )

    # Get prediction results
    result = predict_single(
        model_path=model_path,
        image_path=image_path,
        device=device,
        generate_gradcam_viz=True
    )

    # Print prediction
    print(
        f"[PREDICTION] {result['pred_class']} ({result['confidence']*100:.2f}%)")

    # Generate and save visualization
    if "gradcam_img" in result:
        print("🔍 Generating Grad-CAM...")
        save_path = visualize_prediction(result, image_path, save_dir)
        print(f"✅ Visualization saved to: {save_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict and optionally visualize Grad-CAM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simplified usage (recommended):
  python -m src.predict image.png efficientnetv2s
  python -m src.predict nail_photo.jpg swinv2t
  
  # Full path usage (legacy):
  python -m src.predict --image-path data/image.png --model-path src/best_models/efficientnetv2s/best_model.pth
        """
    )

    # Positional arguments for simplified usage
    parser.add_argument("image", nargs="?", type=str,
                        help="Image filename (in data/ directory)")
    parser.add_argument("model", nargs="?", type=str,
                        help="Model name (e.g., efficientnetv2s, swinv2t, resnet50)")

    # Optional arguments for full path usage (legacy)
    parser.add_argument("--image-path", type=str,
                        help="Full path to the input image")
    parser.add_argument("--model-path", type=str,
                        help="Full path to the model weights (.pth file)")

    parser.add_argument("--save-dir", type=str,
                        default="src/predictions", help="Directory to save outputs")

    args = parser.parse_args()

    # Determine which mode to use
    if args.image and args.model:
        # Simplified mode
        main(
            image_name=args.image,
            model_name=args.model,
            save_dir=args.save_dir
        )
    elif args.image_path and args.model_path:
        # Legacy full path mode
        main(
            image_path=args.image_path,
            model_path=args.model_path,
            save_dir=args.save_dir
        )
    else:
        parser.error(
            "Must provide either:\n"
            "  - IMAGE MODEL (simplified), or\n"
            "  - --image-path and --model-path (full paths)\n"
            "\nRun with -h for examples."
        )
