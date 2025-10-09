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


def predict_single(model_path, image_path, device, generate_gradcam_viz=True):
    """
    Perform single image prediction with optional Grad-CAM generation.

    Args:
        model_path (str/Path): Path to the model weights (.pth file)
        image_path (str/Path): Path to the input image
        device (torch.device): Device to run inference on
        generate_gradcam_viz (bool): Whether to generate Grad-CAM visualization

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

    # Infer model name and strategy from path
    try:
        strategy = model_path.parent.parent.name
        model_name = model_path.parent.parent.parent.name
    except IndexError:
        raise ValueError(
            f"Cannot infer model_name and strategy from path: {model_path}")

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


def main(model_path, image_path, save_dir="src/predictions"):
    """
    Main function for prediction with visualization and printing.

    Args:
        model_path (str/Path): Path to the model weights
        image_path (str/Path): Path to the input image
        save_dir (str/Path): Directory to save outputs
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n🔧 Loading model...")

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
        description="Predict and optionally visualize Grad-CAM")
    parser.add_argument("image_path", type=str,
                        help="Path to the input image")
    parser.add_argument("model_path", type=str,
                        help="Path to the model weights (.pth file)")
    parser.add_argument("--save_dir", type=str,
                        default="src/predictions", help="Directory to save outputs")
    args = parser.parse_args()

    main(args.model_path, args.image_path, save_dir=args.save_dir)
