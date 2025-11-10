# src/predict.py
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
from src.utils.interpretability import generate_gradcam, generate_all_cams
from src.data.transforms import get_test_transforms
from src.utils.helpers import load_model
from src.data.dataloaders import get_class_names
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
from numpy.linalg import pinv
import json


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
                strategy = "best_model"
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
        "scratch",
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
        "class_names": class_names,
        "model": model,
        "img_tensor": img_tensor,
        "img": img
    }

    # Generate Grad-CAM if requested
    if generate_gradcam_viz and model_name:
        orig_img, gradcam_img = generate_gradcam(
            model, model_name, pred_idx, img_tensor, img, device
        )
        result["orig_img"] = orig_img
        result["gradcam_img"] = gradcam_img

    return result


def predict_with_all_cams(model_path, image_path, device, model_name=None):
    """
    Predict and generate all available CAM visualizations.

    Args:
        model_path (str/Path): Path to model weights
        image_path (str/Path): Path to input image
        device (torch.device): Device to run on
        model_name (str, optional): Model name

    Returns:
        dict: Contains prediction results and all CAM visualizations
    """
    # First get prediction without gradcam
    result = predict_single(model_path, image_path, device,
                            generate_gradcam_viz=False, model_name=model_name)

    print(
        f"\n🎨 Generating all CAM visualizations for {result['model_name']}...")

    # Generate all CAMs
    cam_results = generate_all_cams(
        result['model'],
        result['model_name'],
        result['pred_idx'],
        result['img_tensor'],
        result['img'],
        device
    )

    result['all_cams'] = cam_results
    return result


def predict_all_models(image_path, device, models_dir="src/best_models", generate_all_vis=False):
    """
    Predict using all available models.

    Args:
        image_path (str/Path): Path to input image
        device (torch.device): Device to run on
        models_dir (str/Path): Directory containing model folders
        generate_all_vis (bool): If True, generate all CAM types; if False, just best one

    Returns:
        dict: Dictionary mapping model_name -> prediction results
    """
    models_dir = Path(models_dir)
    results = {}

    # Find all available models
    available_models = [d.name for d in models_dir.iterdir()
                        if d.is_dir() and (d / "best_model.pth").exists()]

    if not available_models:
        raise FileNotFoundError(f"No models found in {models_dir}")

    print(
        f"\n🤖 Found {len(available_models)} models: {', '.join(available_models)}")
    print("=" * 80)

    for model_name in available_models:
        print(f"\n📊 Predicting with {model_name.upper()}...")
        model_path = models_dir / model_name / "best_model.pth"

        try:
            if generate_all_vis:
                result = predict_with_all_cams(
                    model_path, image_path, device, model_name)
            else:
                result = predict_single(model_path, image_path, device,
                                        generate_gradcam_viz=True, model_name=model_name)

            results[model_name] = result
            print(
                f"   ✓ {result['pred_class']} ({result['confidence']*100:.2f}%)")

        except Exception as e:
            print(f"   ✗ Failed: {e}")
            results[model_name] = {"error": str(e)}

    print("\n" + "=" * 80)
    return results


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


def load_confusion_matrix(model_path):
    """
    Load confusion matrix from model's evaluation metrics.

    Args:
        model_path (str/Path): Path to model weights file

    Returns:
        np.ndarray: Confusion matrix, or None if not found
    """
    model_path = Path(model_path)

    # Try to find metrics.json
    # For best_models: src/best_models/{model}/evaluation/metrics.json
    # For output: src/output/{model}/{strategy}/{timestamp}/evaluation/metrics.json

    if model_path.parent.parent.name == "best_models":
        # best_models structure
        metrics_path = model_path.parent / "evaluation" / "metrics.json"
    else:
        # output structure
        metrics_path = model_path.parent / "evaluation" / "metrics.json"

    if not metrics_path.exists():
        print(f"⚠️  Warning: metrics.json not found at {metrics_path}")
        return None

    try:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

        # CHECK BOTH LOCATIONS: root level and nested in ml_metrics
        if 'confusion_matrix' in metrics:
            cm = np.array(metrics['confusion_matrix'])
            return cm
        elif 'ml_metrics' in metrics and 'confusion_matrix' in metrics['ml_metrics']:
            # ⭐ ADD THIS: Handle nested structure from eval.py
            cm = np.array(metrics['ml_metrics']['confusion_matrix'])
            return cm
        else:
            print(f"⚠️  Warning: confusion_matrix not found in metrics.json")
            return None

    except Exception as e:
        print(f"⚠️  Warning: Failed to load confusion matrix: {e}")
        return None


def infer_diseases(result, model_path, sex=None, age=None, csv_path="data/Disease Statistics/StatisticalDataset.csv"):
    """
    Infer potential systemic diseases based on nail feature predictions.

    Args:
        result (dict): Prediction result with probs and class_names
        model_path (str/Path): Path to model weights (for loading confusion matrix)
        sex (str, optional): 'male' or 'female'
        age (float, optional): Age of the patient
        csv_path (str): Path to the statistical dataset CSV

    Returns:
        dict: Disease probabilities and metadata
    """
    # Load CSV
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Statistical dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df.fillna(value=pd.NA)

    # Parse into dictionary
    feature_to_diseases = defaultdict(list)
    for _, row in df.iterrows():
        f = row['Nail Feature']
        d = row['Associated Disease/Condition']
        p_fd = row['P(Nail | Disease) 0-1%']
        p_d = row['P(Disease) 0-1%']

        if pd.isna(p_d):
            if d == 'No systemic disease':
                p_d = 1.0
            else:
                continue

        p_female = row['P(Disease) Sex_Female 0-1'] if not pd.isna(
            row['P(Disease) Sex_Female 0-1']) else None
        p_male = row['P(Disease) Sex_Male 0-1'] if not pd.isna(
            row['P(Disease) Sex_Male 0-1']) else None
        age_mean = row['Age (Mean)'] if not pd.isna(
            row['Age (Mean)']) else None
        age_low = row['Age_Low'] if not pd.isna(row['Age_Low']) else None
        age_high = row['Age_High'] if not pd.isna(row['Age_High']) else None

        feature_to_diseases[f].append({
            'disease': d,
            'p_fd': p_fd,
            'p_d': p_d,
            'p_female': p_female,
            'p_male': p_male,
            'age_low': age_low,
            'age_high': age_high,
            'age_mean': age_mean
        })

    # Get model predictions (already in 0-1 range)
    confidence = {result['class_names'][i]: float(result['probs'][0][i])
                  for i in range(len(result['class_names']))}

    # Load confusion matrix from model's evaluation
    confusion_matrix = load_confusion_matrix(model_path)

    # Apply confusion matrix calibration if available
    if confusion_matrix is not None:
        try:
            # Normalize confusion matrix
            row_sums = confusion_matrix.sum(axis=1, keepdims=True)
            # Avoid division by zero
            row_sums = np.where(row_sums == 0, 1, row_sums)
            conf = confusion_matrix / row_sums

            # Get prediction vector in correct order
            q = np.array([confidence.get(label, 0.0)
                         for label in get_class_names()])

            # Adjust predictions using confusion matrix calibration
            adjusted_p = pinv(conf) @ q
            adjusted_p = np.maximum(adjusted_p, 0.0)
            if adjusted_p.sum() > 0:
                adjusted_p /= adjusted_p.sum()

            adjusted_confidence = {
                get_class_names()[i]: adjusted_p[i] for i in range(len(get_class_names()))}
            print("   ✓ Applied confusion matrix calibration")
        except Exception as e:
            print(f"   ⚠️  Calibration failed: {e}, using raw predictions")
            adjusted_confidence = confidence
    else:
        print("   ⚠️  No confusion matrix found, using raw predictions")
        adjusted_confidence = confidence

    # Compute disease posteriors
    disease_to_post = defaultdict(float)

    for f, entries in feature_to_diseases.items():
        p_f_image = adjusted_confidence.get(f, 0)
        if p_f_image == 0:
            continue

        unnorm = {}
        sum_unnorm = 0.0

        for entry in entries:
            d = entry['disease']
            p_fd = entry['p_fd']
            if pd.isna(p_fd) or p_fd == 0:
                continue

            p_d = entry['p_d']
            p_female = entry['p_female']
            p_male = entry['p_male']

            # Calculate effective prior based on sex
            if p_female is None and p_male is None:
                effective_p_d = p_d
            else:
                if sex and sex.lower() == 'female':
                    p_sex = p_female if p_female is not None else 0.0
                elif sex and sex.lower() == 'male':
                    p_sex = p_male if p_male is not None else 0.0
                else:
                    # No sex provided, use average
                    if p_female is not None and p_male is not None:
                        p_sex = (p_female + p_male) / 2
                    elif p_female is not None:
                        p_sex = p_female
                    elif p_male is not None:
                        p_sex = p_male
                    else:
                        p_sex = 0.0

                # Determine if p_sex is conditional probability
                if p_female is not None and p_male is not None:
                    sum_p = p_female + p_male
                    is_p_sex_d = abs(sum_p - 1) < 0.05
                else:
                    is_p_sex_d = False

                if is_p_sex_d:
                    effective_p_d = p_d * p_sex
                else:
                    effective_p_d = p_sex

            if effective_p_d == 0:
                continue

            # Age adjustment
            p_age_d = 1.0
            low = entry['age_low']
            high = entry['age_high']

            if age is not None and low is not None and high is not None:
                if high > low:
                    if low <= age <= high:
                        p_age_d = 1.0 / (high - low)
                    else:
                        p_age_d = 0.0
                else:
                    if age == low:
                        p_age_d = 1.0
                    else:
                        p_age_d = 0.0

            if p_age_d == 0:
                continue

            effective_prior = effective_p_d * p_age_d
            unnorm_d = p_fd * effective_prior
            unnorm[d] = unnorm_d
            sum_unnorm += unnorm_d

        if sum_unnorm > 0:
            for d, u in unnorm.items():
                p_d_f = u / sum_unnorm
                disease_to_post[d] += p_d_f * p_f_image

    # Normalize final posteriors
    total_post = sum(disease_to_post.values())
    if total_post > 0:
        for d in disease_to_post:
            disease_to_post[d] /= total_post

    # Sort by probability
    sorted_diseases = sorted(disease_to_post.items(),
                             key=lambda x: x[1], reverse=True)

    return {
        'diseases': sorted_diseases,
        'adjusted_confidence': adjusted_confidence,
        'raw_confidence': confidence,
        'sex': sex,
        'age': age,
        'calibration_applied': confusion_matrix is not None
    }


def save_disease_inference(inference_result, output_folder):
    """
    Save disease inference results to files.

    Args:
        inference_result (dict): Result from infer_diseases()
        output_folder (Path): Folder to save results
    """
    # Save JSON
    json_data = {
        'patient_info': {
            'sex': inference_result.get('sex'),
            'age': inference_result.get('age')
        },
        'disease_probabilities': [
            {'disease': d, 'probability': float(p)}
            for d, p in inference_result['diseases']
        ],
        'adjusted_nail_features': {
            k: float(v) for k, v in inference_result['adjusted_confidence'].items()
        },
        'raw_predictions': {
            k: float(v) for k, v in inference_result['raw_confidence'].items()
        }
    }

    json_path = output_folder / 'disease_inference.json'
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)

    # Save text report
    report_path = output_folder / 'disease_report.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("SYSTEMIC DISEASE INFERENCE REPORT\n")
        f.write("=" * 70 + "\n\n")

        if inference_result.get('sex') or inference_result.get('age'):
            f.write("Patient Information:\n")
            if inference_result.get('sex'):
                f.write(f"  Sex: {inference_result['sex'].capitalize()}\n")
            if inference_result.get('age'):
                f.write(f"  Age: {inference_result['age']} years\n")
            f.write("\n")

        f.write("=" * 70 + "\n")
        f.write("POTENTIAL SYSTEMIC DISEASES (Ranked by Probability)\n")
        f.write("=" * 70 + "\n\n")

        for i, (disease, prob) in enumerate(inference_result['diseases'][:10], 1):
            f.write(f"{i:2d}. {disease:<50} {prob*100:>6.2f}%\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("CALIBRATED NAIL FEATURE PREDICTIONS\n")
        f.write("=" * 70 + "\n\n")

        sorted_features = sorted(
            inference_result['adjusted_confidence'].items(),
            key=lambda x: x[1],
            reverse=True
        )

        for feature, prob in sorted_features:
            if prob > 0.01:  # Only show features > 1%
                f.write(f"  {feature:<35} {prob*100:>6.2f}%\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("DISCLAIMER\n")
        f.write("=" * 70 + "\n\n")
        f.write("This inference is based on statistical associations between nail\n")
        f.write("features and systemic diseases. It is NOT a medical diagnosis.\n")
        f.write("Please consult a healthcare professional for proper evaluation.\n")

    print(f"   ✓ Disease inference report (JSON & TXT)")

    return json_path, report_path


def visualize_all_cams(result, image_path, save_dir="src/predictions"):
    """
    Visualize all CAM methods in a 2x2 grid and save individual CAMs + metadata.

    Args:
        result (dict): Result from predict_with_all_cams()
        image_path (str/Path): Path to original image
        save_dir (str/Path): Directory to save

    Returns:
        dict: Paths to saved files
    """
    save_dir = Path(save_dir)

    if 'all_cams' not in result:
        raise ValueError(
            "Result does not contain all_cams. Use predict_with_all_cams()")

    # Create folder structure: predictions/{image_name}/{model_name}/
    image_name = Path(image_path).name
    output_folder = save_dir / image_name / result['model_name']
    output_folder.mkdir(parents=True, exist_ok=True)

    cam_results = result['all_cams']
    successful_cams = [(name, img)
                       for name, img in cam_results.items() if img is not None]

    if not successful_cams:
        print("⚠️  No CAM visualizations were successful")
        return None

    saved_paths = {}

    # Save original image
    original_path = output_folder / "original.png"
    original_img = Image.fromarray((result['img'] * 255).astype(
        np.uint8)) if isinstance(result['img'], np.ndarray) else result['img']
    original_img.save(original_path)
    saved_paths['original'] = original_path

    # Save individual CAM images
    for cam_name, cam_img in successful_cams:
        cam_filename = f"{cam_name.lower().replace(' ', '_').replace('+', 'plus')}.png"
        cam_path = output_folder / cam_filename
        Image.fromarray(cam_img).save(cam_path)
        saved_paths[cam_name] = cam_path

    # Save prediction metadata as JSON
    metadata = {
        "image_name": Path(image_path).name,
        "model_name": result['model_name'],
        "prediction": result['pred_class'],
        "confidence": float(result['confidence']),
        "all_predictions": {
            result['class_names'][i]: float(result['probs'][0][i])
            for i in range(len(result['class_names']))
        },
        "top_5_predictions": sorted(
            [(result['class_names'][i], float(result['probs'][0][i]))
             for i in range(len(result['class_names']))],
            key=lambda x: x[1],
            reverse=True
        )[:5],
        "cam_methods_generated": [name for name, _ in successful_cams],
        "timestamp": str(Path(image_path).stat().st_mtime)
    }

    metadata_path = output_folder / "prediction_results.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    saved_paths['metadata'] = metadata_path

    # Create 2x2 grid visualization
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    axs = axs.flatten()

    # Main title
    fig.suptitle(
        f"{result['model_name'].upper()}\n"
        f"{result['pred_class']} ({result['confidence']*100:.2f}%)",
        fontsize=20,
        fontweight="bold",
        y=0.98
    )

    # Original image (top-left)
    axs[0].imshow(result['img'] if isinstance(
        result['img'], np.ndarray) else np.array(result['img']))
    axs[0].axis('off')
    axs[0].set_title("Original Image", fontsize=14, fontweight='bold', pad=10)

    # CAM visualizations (remaining 3 positions)
    for idx, (cam_name, cam_img) in enumerate(successful_cams[:3], 1):
        axs[idx].imshow(cam_img)
        axs[idx].axis('off')
        axs[idx].set_title(cam_name, fontsize=14, fontweight='bold', pad=10)

    # Hide unused subplot if less than 3 CAMs
    if len(successful_cams) < 3:
        for idx in range(len(successful_cams) + 1, 4):
            axs[idx].axis('off')

    plt.tight_layout()

    # Save combined visualization
    combined_path = output_folder / "all_cams_combined.png"
    plt.savefig(combined_path, dpi=200, bbox_inches='tight')
    plt.close()
    saved_paths['combined'] = combined_path

    # Create a summary text file
    summary_path = output_folder / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"{'='*60}\n")
        f.write(f"PREDICTION SUMMARY\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Image: {Path(image_path).name}\n")
        f.write(f"Model: {result['model_name'].upper()}\n")
        f.write(f"Prediction: {result['pred_class']}\n")
        f.write(f"Confidence: {result['confidence']*100:.2f}%\n\n")
        f.write(f"{'='*60}\n")
        f.write(f"TOP 5 PREDICTIONS\n")
        f.write(f"{'='*60}\n\n")
        for i, (class_name, prob) in enumerate(metadata['top_5_predictions'], 1):
            f.write(f"{i}. {class_name:<30} {prob*100:>6.2f}%\n")
        f.write(f"\n{'='*60}\n")
        f.write(f"CAM METHODS GENERATED\n")
        f.write(f"{'='*60}\n\n")
        for cam_name in metadata['cam_methods_generated']:
            f.write(f"  ✓ {cam_name}\n")
        f.write(f"\n{'='*60}\n")
        f.write(f"OUTPUT FILES\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Folder: {output_folder.relative_to(save_dir)}/\n")
        f.write(f"  - original.png\n")
        for cam_name in metadata['cam_methods_generated']:
            filename = cam_name.lower().replace(' ', '_').replace('+', 'plus')
            f.write(f"  - {filename}.png\n")
        f.write(f"  - all_cams_combined.png\n")
        f.write(f"  - prediction_results.json\n")
        f.write(f"  - summary.txt\n")

    saved_paths['summary'] = summary_path

    print(f"\n📁 Results saved to: {output_folder.relative_to(save_dir)}")
    print(f"   ✓ Original image")
    print(f"   ✓ {len(successful_cams)} individual CAM visualizations")
    print(f"   ✓ Combined 2x2 grid visualization")
    print(f"   ✓ Prediction metadata (JSON)")
    print(f"   ✓ Summary text report")

    return saved_paths


def visualize_all_models(results, image_path, save_dir="src/predictions"):
    """
    Create comparison visualization for all models.

    Args:
        results (dict): Results from predict_all_models()
        image_path (str/Path): Path to original image
        save_dir (str/Path): Directory to save

    Returns:
        Path: Path to saved visualization
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Filter successful results
    successful = {name: res for name, res in results.items()
                  if 'error' not in res and 'gradcam_img' in res}

    if not successful:
        print("⚠️  No successful predictions to visualize")
        return None

    n_models = len(successful)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    # Create figure
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows))
    if n_models == 1:
        axs = np.array([axs])
    axs = axs.flatten()

    fig.suptitle(
        f"Model Comparison\n{Path(image_path).name}",
        fontsize=20,
        fontweight="bold",
        y=0.98
    )

    for idx, (model_name, result) in enumerate(successful.items()):
        # Create subplot with 2 images side by side
        ax = axs[idx]
        ax.axis('off')

        # Combine original and gradcam horizontally
        orig_img = result['orig_img']
        cam_img = result['gradcam_img'] / 255.0  # Normalize if needed

        # Create side-by-side image
        combined = np.hstack([orig_img, cam_img])

        ax.imshow(combined)
        ax.set_title(
            f"{model_name.upper()}\n"
            f"{result['pred_class']} ({result['confidence']*100:.1f}%)",
            fontsize=14,
            fontweight='bold'
        )

    # Hide extra subplots
    for idx in range(n_models, len(axs)):
        axs[idx].axis('off')

    plt.tight_layout()

    # Save
    save_path = save_dir / f"all_models_{Path(image_path).stem}.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
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


def main(image_path=None, model_path=None, image_name=None, model_name=None,
         save_dir="src/predictions", all_cams=False, all_models=False):
    """
    Main function for prediction with visualization and printing.

    Args:
        image_path (str/Path): Full path to the input image (old method)
        model_path (str/Path): Full path to the model weights (old method)
        image_name (str): Simple image filename (new method)
        model_name (str): Simple model name (new method)
        save_dir (str/Path): Directory to save outputs
        all_cams (bool): Generate all CAM visualizations
        all_models (bool): Use all available models
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve paths based on which arguments were provided
    if image_name:
        image_path = Path("data") / image_name
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

    if not image_path:
        raise ValueError("Must provide image_path or image_name")

    # Mode 1: All models
    if all_models:
        print(f"\n🚀 Running prediction with ALL models...")
        results = predict_all_models(
            image_path, device, generate_all_vis=all_cams)

        if all_cams:
            # Save individual all-cams visualizations
            for model_name, result in results.items():
                if 'error' not in result and 'all_cams' in result:
                    save_path = visualize_all_cams(
                        result, image_path, save_dir)
                    if save_path:
                        print(f"✅ All CAMs for {model_name}: {save_path}")
        else:
            # Save comparison visualization
            save_path = visualize_all_models(results, image_path, save_dir)
            if save_path:
                print(f"\n✅ Model comparison saved to: {save_path}")

        if args.sex or args.age:
            print("\n🧬 Computing disease inference for all models...")
            image_name_clean = Path(image_path).name

            for model_name, result in results.items():
                if 'error' not in result:
                    model_path = Path("src/best_models") / \
                        model_name / "best_model.pth"
                    inference_result = infer_diseases(
                        result, model_path, sex=args.sex, age=args.age)

                    output_folder = Path(save_dir) / \
                        image_name_clean / model_name
                    output_folder.mkdir(parents=True, exist_ok=True)

                    save_disease_inference(inference_result, output_folder)

            print("✅ All disease inferences saved")
        return results

    # Mode 2: Single model with all CAMs
    elif all_cams:
        if model_name:
            image_path, model_path = resolve_paths(
                image_name or Path(image_path).name, model_name)
        elif not model_path:
            raise ValueError(
                "Must provide model_name or model_path when using --all-cams")

        print(f"\n🔧 Loading model...")
        result = predict_with_all_cams(
            model_path, image_path, device, model_name)

        print(
            f"[PREDICTION] {result['pred_class']} ({result['confidence']*100:.2f}%)")

        save_path = visualize_all_cams(result, image_path, save_dir)
        if save_path:
            print(
                f"✅ All results saved to folder: {save_path.get('combined', 'N/A').parent if isinstance(save_path, dict) else save_path}")
        if args.sex or args.age:
            print("\n🧬 Computing disease inference...")
            inference_result = infer_diseases(
                result, model_path, sex=args.sex, age=args.age)

            output_folder = save_path['combined'].parent if isinstance(
                save_path, dict) else save_path.parent
            json_path, report_path = save_disease_inference(
                inference_result, output_folder)
            print(f"✅ Disease inference saved")
        return result

    # Mode 3: Single model, single CAM (original behavior)
    else:
        print(f"\n🔧 Loading model...")

        if image_name and model_name:
            image_path, model_path = resolve_paths(image_name, model_name)
        elif not (image_path and model_path):
            raise ValueError(
                "Must provide either:\n"
                "  - image_name and model_name (simplified), or\n"
                "  - image_path and model_path (full paths)"
            )

        result = predict_single(
            model_path=model_path,
            image_path=image_path,
            device=device,
            generate_gradcam_viz=True,
            model_name=model_name
        )

        print(
            f"[PREDICTION] {result['pred_class']} ({result['confidence']*100:.2f}%)")

        if "gradcam_img" in result:
            print("🔍 Generating Grad-CAM...")
            save_path = visualize_prediction(result, image_path, save_dir)
            print(f"✅ Visualization saved to: {save_path}")

        if args.sex or args.age:
            print("\n🧬 Computing disease inference...")
            inference_result = infer_diseases(
                result,
                model_path,
                sex=args.sex,
                age=args.age
            )

            # Create output folder matching visualize_all_cams structure
            image_name_clean = Path(image_path).name
            output_folder = Path(save_dir) / \
                image_name_clean / result['model_name']
            output_folder.mkdir(parents=True, exist_ok=True)

            json_path, report_path = save_disease_inference(
                inference_result, output_folder)
            print(
                f"✅ Disease inference saved: {json_path.relative_to(save_dir)}")
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict and visualize with CAM methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single model, single CAM (default):
  python -m src.predict image.png efficientnetv2s
  
  # Single model, all CAM methods:
  python -m src.predict image.png swinv2t --all-cams
  
  # All models, single CAM each:
  python -m src.predict image.png --all-models
  
  # All models, all CAM methods (comprehensive):
  python -m src.predict image.png --all-models --all-cams
  
  # Legacy full path:
  python -m src.predict --image-path data/image.png --model-path src/best_models/resnet50/best_model.pth
        """
    )

    # Positional arguments
    parser.add_argument("image", nargs="?", type=str,
                        help="Image filename (in data/ directory)")
    parser.add_argument("model", nargs="?", type=str,
                        help="Model name (e.g., efficientnetv2s, swinv2t)")

    # Optional arguments for full path usage (legacy)
    parser.add_argument("--image-path", type=str,
                        help="Full path to the input image")
    parser.add_argument("--model-path", type=str,
                        help="Full path to the model weights (.pth file)")

    parser.add_argument("--save-dir", type=str,
                        default="src/predictions",
                        help="Directory to save outputs")

    # New feature flags
    parser.add_argument("--all-cams", action="store_true",
                        help="Generate all CAM visualization methods")
    parser.add_argument("--all-models", action="store_true",
                        help="Run prediction with all available models")

    # Disease inference parameters
    parser.add_argument("--sex", type=str, choices=['male', 'female'],
                        help="Patient sex for disease inference (male/female)")
    parser.add_argument("--age", type=float,
                        help="Patient age for disease inference")

    args = parser.parse_args()

    # Determine mode
    if args.all_models:
        # All models mode - only needs image
        if args.image:
            main(image_name=args.image, save_dir=args.save_dir,
                 all_cams=args.all_cams, all_models=True)
        elif args.image_path:
            main(image_path=args.image_path, save_dir=args.save_dir,
                 all_cams=args.all_cams, all_models=True)
        else:
            parser.error("--all-models requires either IMAGE or --image-path")

    elif args.image and args.model:
        # Simplified mode
        main(image_name=args.image, model_name=args.model,
             save_dir=args.save_dir, all_cams=args.all_cams)

    elif args.image_path and args.model_path:
        # Legacy full path mode
        main(image_path=args.image_path, model_path=args.model_path,
             save_dir=args.save_dir, all_cams=args.all_cams)

    else:
        parser.error(
            "Must provide either:\n"
            "  - IMAGE MODEL (simplified), or\n"
            "  - IMAGE --all-models (all models), or\n"
            "  - --image-path and --model-path (full paths)\n"
            "\nRun with -h for examples."
        )
