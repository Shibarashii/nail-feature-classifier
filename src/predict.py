# src/predict.py
"""
Nail disease classification and visualization system.
Provides prediction, CAM visualization, and disease inference capabilities.
"""
import torch
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from collections import defaultdict
from numpy.linalg import pinv

from src.utils.interpretability import generate_gradcam, generate_all_cams
from src.data.transforms import get_test_transforms
from src.utils.helpers import load_model
from src.data.dataloaders import get_class_names


# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

def preprocess_image(image_path):
    """Load and preprocess image for model input."""
    transform = get_test_transforms()
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)
    return img, img_tensor


# ============================================================================
# MODEL EVALUATION HELPERS
# ============================================================================

def load_confusion_matrix(model_path):
    """
    Load confusion matrix from model's evaluation metrics.

    Returns:
        np.ndarray or None: Confusion matrix if found, None otherwise
    """
    model_path = Path(model_path)

    # Determine metrics.json path based on directory structure
    if model_path.parent.parent.name == "best_models":
        metrics_path = model_path.parent / "evaluation" / "metrics.json"
    else:
        metrics_path = model_path.parent / "evaluation" / "metrics.json"

    if not metrics_path.exists():
        print(f"⚠️  Warning: metrics.json not found at {metrics_path}")
        return None

    try:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

        # Check both root level and nested structure
        if 'confusion_matrix' in metrics:
            return np.array(metrics['confusion_matrix'])
        elif 'ml_metrics' in metrics and 'confusion_matrix' in metrics['ml_metrics']:
            return np.array(metrics['ml_metrics']['confusion_matrix'])
        else:
            print(f"⚠️  Warning: confusion_matrix not found in metrics.json")
            return None

    except Exception as e:
        print(f"⚠️  Warning: Failed to load confusion matrix: {e}")
        return None


# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def predict_single(model_path, image_path, device, generate_gradcam_viz=True, model_name=None):
    """
    Perform single image prediction with optional Grad-CAM visualization.

    Returns:
        dict: Prediction results including class, confidence, probabilities, etc.
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Infer model name and strategy from path
    if model_name is None:
        if model_path.parent.parent.name == "best_models":
            model_name = model_path.parent.name
            strategy = "best_model"
        else:
            strategy = model_path.parent.parent.name
            model_name = model_path.parent.parent.parent.name
    else:
        strategy = "best_model"

    # Load model and classes
    class_names = get_class_names()
    num_classes = len(class_names)

    model = load_model(str(model_path), model_name,
                       "scratch", num_classes, device)

    # Preprocess and run inference
    img, img_tensor = preprocess_image(image_path)
    img_tensor = img_tensor.to(device)

    model.eval()
    with torch.inference_mode():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()

    pred_class = class_names[pred_idx]
    confidence = probs[0][pred_idx].item()

    # Prepare result
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
        "img": img,
        "model_path": str(model_path)
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
    """Generate all CAM visualizations for a single prediction."""
    result = predict_single(model_path, image_path, device,
                            generate_gradcam_viz=False, model_name=model_name)

    print(
        f"\n🎨 Generating all CAM visualizations for {result['model_name']}...")

    cam_results = generate_all_cams(
        result['model'], result['model_name'], result['pred_idx'],
        result['img_tensor'], result['img'], device
    )

    result['all_cams'] = cam_results
    return result


def predict_all_models(image_path, device, models_dir="src/best_models", generate_all_vis=False):
    """Run prediction across all available models."""
    models_dir = Path(models_dir)

    # Find all available models
    available_models = [
        d.name for d in models_dir.iterdir()
        if d.is_dir() and (d / "best_model.pth").exists()
    ]

    if not available_models:
        raise FileNotFoundError(f"No models found in {models_dir}")

    print(
        f"\n🤖 Found {len(available_models)} models: {', '.join(available_models)}")
    print("=" * 80)

    results = {}
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


# ============================================================================
# DISEASE INFERENCE
# ============================================================================

def infer_diseases(result, model_path, sex=None, age=None,
                   csv_path="data/Disease Statistics/StatisticalDataset.csv"):
    """
    Infer systemic diseases based on nail feature predictions using Bayesian inference.

    Args:
        result: Prediction result dictionary
        model_path: Path to model for loading confusion matrix
        sex: 'male' or 'female' (optional)
        age: Patient age in years (optional)
        csv_path: Path to statistical dataset

    Returns:
        dict: Disease probabilities and metadata
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Statistical dataset not found: {csv_path}")

    # Load and parse disease associations
    df = pd.read_csv(csv_path).fillna(value=pd.NA)
    feature_to_diseases = _parse_disease_associations(df)

    # Get model predictions
    confidence = {
        result['class_names'][i]: float(result['probs'][0][i])
        for i in range(len(result['class_names']))
    }

    # Apply confusion matrix calibration if available
    adjusted_confidence = _apply_calibration(confidence, model_path)

    # Compute disease posteriors using Bayesian inference
    disease_probabilities = _compute_disease_posteriors(
        feature_to_diseases, adjusted_confidence, sex, age
    )

    return {
        'diseases': sorted(disease_probabilities.items(), key=lambda x: x[1], reverse=True),
        'adjusted_confidence': adjusted_confidence,
        'raw_confidence': confidence,
        'sex': sex,
        'age': age,
        'calibration_applied': load_confusion_matrix(model_path) is not None
    }


def _parse_disease_associations(df):
    """Parse CSV into disease association dictionary."""
    feature_to_diseases = defaultdict(list)

    for _, row in df.iterrows():
        nail_feature = row['Nail Feature']
        disease = row['Associated Disease/Condition']
        p_fd = row['P(Nail | Disease) 0-1%']
        p_d = row['P(Disease) 0-1%']

        if pd.isna(p_d):
            if disease == 'No systemic disease':
                p_d = 1.0
            else:
                continue

        feature_to_diseases[nail_feature].append({
            'disease': disease,
            'p_fd': p_fd,
            'p_d': p_d,
            'p_female': row['P(Disease) Sex_Female 0-1'] if not pd.isna(row['P(Disease) Sex_Female 0-1']) else None,
            'p_male': row['P(Disease) Sex_Male 0-1'] if not pd.isna(row['P(Disease) Sex_Male 0-1']) else None,
            'age_low': row['Age_Low'] if not pd.isna(row['Age_Low']) else None,
            'age_high': row['Age_High'] if not pd.isna(row['Age_High']) else None,
        })

    return feature_to_diseases


def _apply_calibration(confidence, model_path):
    """Apply confusion matrix calibration to predictions."""
    confusion_matrix = load_confusion_matrix(model_path)

    if confusion_matrix is None:
        print("   ⚠️  No confusion matrix found, using raw predictions")
        return confidence

    try:
        # Normalize confusion matrix
        row_sums = confusion_matrix.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        conf = confusion_matrix / row_sums

        # Get prediction vector
        class_names = get_class_names()
        q = np.array([confidence.get(label, 0.0) for label in class_names])

        # Apply calibration
        adjusted_p = pinv(conf) @ q
        adjusted_p = np.maximum(adjusted_p, 0.0)
        if adjusted_p.sum() > 0:
            adjusted_p /= adjusted_p.sum()

        print("   ✓ Applied confusion matrix calibration")
        return {class_names[i]: adjusted_p[i] for i in range(len(class_names))}

    except Exception as e:
        print(f"   ⚠️  Calibration failed: {e}, using raw predictions")
        return confidence


def _compute_disease_posteriors(feature_to_diseases, adjusted_confidence, sex, age):
    """Compute disease posterior probabilities using Bayesian inference."""
    disease_to_post = defaultdict(float)

    for feature, entries in feature_to_diseases.items():
        p_f_image = adjusted_confidence.get(feature, 0)
        if p_f_image == 0:
            continue

        unnorm = {}
        sum_unnorm = 0.0

        for entry in entries:
            disease = entry['disease']
            p_fd = entry['p_fd']

            if pd.isna(p_fd) or p_fd == 0:
                continue

            # Calculate effective prior
            effective_p_d = _calculate_effective_prior(entry, sex)
            if effective_p_d == 0:
                continue

            # Apply age adjustment
            p_age_d = _calculate_age_adjustment(entry, age)
            if p_age_d == 0:
                continue

            # Compute unnormalized posterior
            effective_prior = effective_p_d * p_age_d
            unnorm[disease] = p_fd * effective_prior
            sum_unnorm += unnorm[disease]

        # Normalize and accumulate
        if sum_unnorm > 0:
            for disease, u in unnorm.items():
                p_d_f = u / sum_unnorm
                disease_to_post[disease] += p_d_f * p_f_image

    # Final normalization
    total_post = sum(disease_to_post.values())
    if total_post > 0:
        for disease in disease_to_post:
            disease_to_post[disease] /= total_post

    return disease_to_post


def _calculate_effective_prior(entry, sex):
    """Calculate effective disease prior based on sex."""
    p_d = entry['p_d']
    p_female = entry['p_female']
    p_male = entry['p_male']

    if p_female is None and p_male is None:
        return p_d

    # Determine sex-specific probability
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
        is_conditional = abs((p_female + p_male) - 1) < 0.05
    else:
        is_conditional = False

    return p_d * p_sex if is_conditional else p_sex


def _calculate_age_adjustment(entry, age):
    """Calculate age-based probability adjustment."""
    if age is None:
        return 1.0

    age_low = entry['age_low']
    age_high = entry['age_high']

    if age_low is None or age_high is None:
        return 1.0

    if age_high > age_low:
        return 1.0 / (age_high - age_low) if age_low <= age <= age_high else 0.0
    else:
        return 1.0 if age == age_low else 0.0


# ============================================================================
# VISUALIZATION AND SAVING
# ============================================================================

def save_disease_inference(inference_result, output_folder):
    """Save disease inference results to JSON only (report is now combined)."""
    output_folder = Path(output_folder)
    json_path = _save_disease_inference_json(inference_result, output_folder)
    print(f"   ✓ Disease inference (JSON)")
    return json_path


def _save_disease_inference_json(inference_result, output_folder):
    """Save disease inference to JSON file."""
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
        },
        'calibration_applied': inference_result.get('calibration_applied', False)
    }

    json_path = output_folder / 'disease_inference.json'
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)

    return json_path


def _create_comprehensive_report(result, output_folder, successful_cams, image_path, save_dir, inference_result=None):
    """Create comprehensive report combining prediction summary and disease inference."""
    report_path = output_folder / "report.txt"

    with open(report_path, 'w') as f:
        # Header
        f.write("=" * 70 + "\n")
        f.write("NAIL DISEASE CLASSIFICATION REPORT\n")
        f.write("=" * 70 + "\n\n")

        # Image and Model Info
        f.write("IMAGE AND MODEL INFORMATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Image: {Path(image_path).name}\n")
        f.write(f"Model: {result['model_name'].upper()}\n")
        f.write(f"Strategy: {result['strategy']}\n\n")

        # Prediction Results
        f.write("=" * 70 + "\n")
        f.write("PREDICTION RESULTS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Primary Prediction: {result['pred_class']}\n")
        f.write(f"Confidence: {result['confidence']*100:.2f}%\n\n")

        # Top 5 Predictions
        f.write("-" * 70 + "\n")
        f.write("Top 5 Nail Feature Predictions:\n")
        f.write("-" * 70 + "\n\n")

        top_5 = sorted(
            [(result['class_names'][i], result['probs'][0][i].item())
             for i in range(len(result['class_names']))],
            key=lambda x: x[1], reverse=True
        )[:5]

        for i, (class_name, prob) in enumerate(top_5, 1):
            f.write(f"  {i}. {class_name:<35} {prob*100:>6.2f}%\n")

        # CAM Visualization
        f.write("\n" + "-" * 70 + "\n")
        f.write("CAM Visualization Methods Generated:\n")
        f.write("-" * 70 + "\n\n")
        for cam_name, _ in successful_cams:
            f.write(f"  ✓ {cam_name}\n")

        # Disease Inference Section (if available)
        if inference_result:
            f.write("\n\n" + "=" * 70 + "\n")
            f.write("SYSTEMIC DISEASE INFERENCE\n")
            f.write("=" * 70 + "\n\n")

            # Patient Info
            if inference_result.get('sex') or inference_result.get('age'):
                f.write("Patient Information:\n")
                f.write("-" * 70 + "\n")
                if inference_result.get('sex'):
                    f.write(f"  Sex: {inference_result['sex'].capitalize()}\n")
                if inference_result.get('age'):
                    f.write(f"  Age: {inference_result['age']} years\n")
                f.write("\n")

            # Calibration Status
            if inference_result.get('calibration_applied'):
                f.write(
                    "Note: Predictions were calibrated using confusion matrix.\n\n")

            # Disease Probabilities
            f.write("-" * 70 + "\n")
            f.write("Potential Systemic Diseases (Ranked by Probability):\n")
            f.write("-" * 70 + "\n\n")

            for i, (disease, prob) in enumerate(inference_result['diseases'][:15], 1):
                f.write(f"  {i:2d}. {disease:<50} {prob*100:>6.2f}%\n")

            # Calibrated Features
            if inference_result.get('calibration_applied'):
                f.write("\n" + "-" * 70 + "\n")
                f.write("Calibrated Nail Feature Predictions:\n")
                f.write("-" * 70 + "\n\n")

                sorted_features = sorted(
                    inference_result['adjusted_confidence'].items(),
                    key=lambda x: x[1], reverse=True
                )

                for feature, prob in sorted_features:
                    if prob > 0.01:
                        f.write(f"  {feature:<35} {prob*100:>6.2f}%\n")

            # Medical Disclaimer
            f.write("\n\n" + "=" * 70 + "\n")
            f.write("IMPORTANT MEDICAL DISCLAIMER\n")
            f.write("=" * 70 + "\n\n")
            f.write(
                "This disease inference is based on statistical associations between\n")
            f.write(
                "nail features and systemic diseases from medical literature.\n\n")
            f.write("⚠️  THIS IS NOT A MEDICAL DIAGNOSIS ⚠️\n\n")
            f.write("This tool is for informational and research purposes only.\n")
            f.write("Always consult qualified healthcare professionals for proper\n")
            f.write("medical evaluation, diagnosis, and treatment.\n")

        # Output Files
        f.write("\n\n" + "=" * 70 + "\n")
        f.write("OUTPUT FILES\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Location: {output_folder.relative_to(save_dir)}/\n\n")
        f.write("Files:\n")
        f.write("  - original.png                  (Original input image)\n")
        for cam_name, _ in successful_cams:
            filename = cam_name.lower().replace(' ', '_').replace('+', 'plus')
            f.write(
                f"  - {filename}.png{' ' * (28 - len(filename))}(CAM visualization)\n")
        f.write("  - all_cams_combined.png         (2x2 grid visualization)\n")
        f.write("  - prediction_results.json       (Structured prediction data)\n")
        if inference_result:
            f.write("  - disease_inference.json        (Disease probability data)\n")
        f.write("  - report.txt                    (This comprehensive report)\n")

    return report_path


def visualize_prediction(result, image_path, save_dir="src/predictions"):
    """Create and save single Grad-CAM visualization."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if "gradcam_img" not in result:
        raise ValueError("Result does not contain Grad-CAM visualization")

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    fig.suptitle(
        f"{result['model_name'].upper()}_{result['strategy'].upper()}\n"
        f"{result['pred_class']}\n{result['confidence'] * 100:.2f}%",
        fontsize=16, fontweight="bold", y=1.05
    )

    axs[0].imshow(result["orig_img"])
    axs[0].axis("off")
    axs[0].set_title("Original Image", fontsize=14)

    axs[1].imshow(result["gradcam_img"])
    axs[1].axis("off")
    axs[1].set_title("Grad-CAM", fontsize=14)

    save_path = save_dir / \
        f"prediction_{Path(image_path).stem}_{result['model_name']}.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    return save_path


def visualize_all_cams(result, image_path, save_dir="src/predictions", sex=None, age=None):
    """
    Create comprehensive visualization with all CAM methods and save all outputs.

    Creates:
        - 2x2 grid visualization
        - Individual CAM images
        - Prediction metadata JSON
        - Comprehensive report (prediction + disease inference)
    """
    save_dir = Path(save_dir)

    if 'all_cams' not in result:
        raise ValueError("Result does not contain all_cams")

    # Setup output folder
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
    original_img = (Image.fromarray((result['img'] * 255).astype(np.uint8))
                    if isinstance(result['img'], np.ndarray) else result['img'])
    original_img.save(original_path)
    saved_paths['original'] = original_path

    # Save individual CAM images
    for cam_name, cam_img in successful_cams:
        cam_filename = cam_name.lower().replace(' ', '_').replace('+', 'plus') + '.png'
        cam_path = output_folder / cam_filename
        Image.fromarray(cam_img).save(cam_path)
        saved_paths[cam_name] = cam_path

    # Save metadata JSON
    _save_prediction_metadata(result, output_folder,
                              successful_cams, image_path)
    saved_paths['metadata'] = output_folder / "prediction_results.json"

    # Create 2x2 grid visualization
    _create_cam_grid(result, successful_cams, output_folder)
    saved_paths['combined'] = output_folder / "all_cams_combined.png"

    # Disease inference (if requested)
    inference_result = None
    if sex or age:
        try:
            print(f"\n🏥 Generating disease inference...")
            inference_result = infer_diseases(
                result, result['model_path'], sex=sex, age=age)

            # Save disease inference JSON
            json_path = _save_disease_inference_json(
                inference_result, output_folder)
            saved_paths['disease_inference_json'] = json_path

            if len(inference_result['diseases']) > 0:
                print(f"\n   Top 5 Potential Diseases:")
                for i, (disease, prob) in enumerate(inference_result['diseases'][:5], 1):
                    print(f"      {i}. {disease}: {prob*100:.2f}%")
        except Exception as e:
            print(f"   ⚠️  Disease inference failed: {e}")

    # Create comprehensive report (prediction + disease inference combined)
    report_path = _create_comprehensive_report(
        result, output_folder, successful_cams, image_path, save_dir, inference_result
    )
    saved_paths['report'] = report_path

    print(f"\n📁 Results saved to: {output_folder.relative_to(save_dir)}")
    print(f"   ✓ Original image")
    print(f"   ✓ {len(successful_cams)} individual CAM visualizations")
    print(f"   ✓ Combined 2x2 grid visualization")
    print(f"   ✓ Prediction metadata (JSON)")
    print(f"   ✓ Comprehensive report (TXT)")
    if inference_result:
        print(f"   ✓ Disease inference (JSON)")

    return saved_paths


def _save_prediction_metadata(result, output_folder, successful_cams, image_path):
    """Save prediction metadata to JSON."""
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
            key=lambda x: x[1], reverse=True
        )[:5],
        "cam_methods_generated": [name for name, _ in successful_cams],
    }

    with open(output_folder / "prediction_results.json", 'w') as f:
        json.dump(metadata, f, indent=2)


def _create_cam_grid(result, successful_cams, output_folder):
    """Create 2x2 grid visualization of CAM methods."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    axs = axs.flatten()

    fig.suptitle(
        f"{result['model_name'].upper()}\n"
        f"{result['pred_class']} ({result['confidence']*100:.2f}%)",
        fontsize=20, fontweight="bold", y=0.98
    )

    # Original image
    axs[0].imshow(result['img'] if isinstance(
        result['img'], np.ndarray) else np.array(result['img']))
    axs[0].axis('off')
    axs[0].set_title("Original Image", fontsize=14, fontweight='bold', pad=10)

    # CAM visualizations
    for idx, (cam_name, cam_img) in enumerate(successful_cams[:3], 1):
        axs[idx].imshow(cam_img)
        axs[idx].axis('off')
        axs[idx].set_title(cam_name, fontsize=14, fontweight='bold', pad=10)

    # Hide unused subplots
    if len(successful_cams) < 3:
        for idx in range(len(successful_cams) + 1, 4):
            axs[idx].axis('off')

    plt.tight_layout()
    plt.savefig(output_folder / "all_cams_combined.png",
                dpi=200, bbox_inches='tight')
    plt.close()


def visualize_all_models(results, image_path, save_dir="src/predictions"):
    """Create comparison visualization across all models."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    successful = {
        name: res for name, res in results.items()
        if 'error' not in res and 'gradcam_img' in res
    }

    if not successful:
        print("⚠️  No successful predictions to visualize")
        return None

    n_models = len(successful)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows))
    if n_models == 1:
        axs = np.array([axs])
    axs = axs.flatten()

    fig.suptitle(f"Model Comparison\n{Path(image_path).name}",
                 fontsize=20, fontweight="bold", y=0.98)

    for idx, (model_name, result) in enumerate(successful.items()):
        ax = axs[idx]
        ax.axis('off')

        # Combine original and gradcam side by side
        orig_img = result['orig_img']
        cam_img = result['gradcam_img'] / 255.0
        combined = np.hstack([orig_img, cam_img])

        ax.imshow(combined)
        ax.set_title(
            f"{model_name.upper()}\n"
            f"{result['pred_class']} ({result['confidence']*100:.1f}%)",
            fontsize=14, fontweight='bold'
        )

    # Hide extra subplots
    for idx in range(n_models, len(axs)):
        axs[idx].axis('off')

    plt.tight_layout()

    save_path = save_dir / f"all_models_{Path(image_path).stem}.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    return save_path


# ============================================================================
# PATH RESOLUTION
# ============================================================================

def resolve_paths(image_name, model_name):
    """Resolve simple names to full paths."""
    image_path = Path("data") / image_name
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model_path = Path("src/best_models") / model_name / "best_model.pth"
    if not model_path.exists():
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


# ============================================================================
# MAIN EXECUTION LOGIC
# ============================================================================

def run_single_model_prediction(image_path, model_path, model_name, save_dir,
                                all_cams, sex, age):
    """Execute single model prediction workflow."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n🔧 Loading model...")

    if all_cams:
        # Mode: Single model with all CAM methods
        result = predict_with_all_cams(
            model_path, image_path, device, model_name)
        print(
            f"[PREDICTION] {result['pred_class']} ({result['confidence']*100:.2f}%)")

        save_paths = visualize_all_cams(
            result, image_path, save_dir, sex=sex, age=age)
        if save_paths:
            folder = save_paths.get('combined', Path('.')).parent
            print(f"✅ All results saved to: {folder}")
    else:
        # Mode: Single model with single CAM
        result = predict_single(model_path, image_path, device,
                                generate_gradcam_viz=True, model_name=model_name)
        print(
            f"[PREDICTION] {result['pred_class']} ({result['confidence']*100:.2f}%)")

        if "gradcam_img" in result:
            print("🔍 Generating Grad-CAM...")
            save_path = visualize_prediction(result, image_path, save_dir)
            print(f"✅ Visualization saved to: {save_path}")

        if sex or age:
            print("\n🏥 Generating disease inference...")
            inference_result = infer_diseases(
                result, model_path, sex=sex, age=age)

            image_name = Path(image_path).name
            output_folder = Path(save_dir) / image_name / result['model_name']
            output_folder.mkdir(parents=True, exist_ok=True)

            json_path = save_disease_inference(inference_result, output_folder)

            # FIX: Create proper successful_cams list
            successful_cams = [
                ('Grad-CAM', result.get('gradcam_img'))] if 'gradcam_img' in result else []

            # Create comprehensive report with proper CAM data
            _create_comprehensive_report(
                result, output_folder, successful_cams, image_path, save_dir, inference_result
            )

            print(
                f"✅ Disease inference saved: {json_path.relative_to(save_dir)}")

    return result


def run_all_models_prediction(image_path, save_dir, all_cams, sex, age):
    """Execute all models prediction workflow."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n🚀 Running prediction with ALL models...")
    results = predict_all_models(image_path, device, generate_all_vis=all_cams)

    if all_cams:
        # Save individual all-cams visualizations for each model
        # Disease inference is handled inside visualize_all_cams()
        for model_name, result in results.items():
            if 'error' not in result and 'all_cams' in result:
                visualize_all_cams(result, image_path,
                                   save_dir, sex=sex, age=age)
    else:
        # Save comparison visualization
        save_path = visualize_all_models(results, image_path, save_dir)
        if save_path:
            print(f"\n✅ Model comparison saved to: {save_path}")

        # Disease inference for single CAM mode only
        if sex or age:
            print("\n🏥 Generating disease inference for all models...")
            image_name = Path(image_path).name

            for model_name, result in results.items():
                if 'error' not in result:
                    model_path = Path("src/best_models") / \
                        model_name / "best_model.pth"

                    try:
                        inference_result = infer_diseases(
                            result, model_path, sex=sex, age=age)
                        output_folder = Path(save_dir) / \
                            image_name / model_name
                        output_folder.mkdir(parents=True, exist_ok=True)

                        save_disease_inference(inference_result, output_folder)

                        # FIX: Use proper CAM data for single CAM mode
                        successful_cams = [
                            ('Grad-CAM', result.get('gradcam_img'))] if 'gradcam_img' in result else []
                        _create_comprehensive_report(
                            result, output_folder, successful_cams, image_path, save_dir, inference_result
                        )
                    except Exception as e:
                        print(f"   ⚠️  Failed for {model_name}: {e}")

            print("✅ All disease inferences saved")

    return results


def main(image_path=None, model_path=None, image_name=None, model_name=None,
         save_dir="src/predictions", all_cams=False, all_models=False,
         sex=None, age=None):
    """
    Main entry point for prediction system.

    Args:
        image_path: Full path to image (legacy)
        model_path: Full path to model (legacy)
        image_name: Simple image filename
        model_name: Simple model name
        save_dir: Output directory
        all_cams: Generate all CAM visualizations
        all_models: Run all available models
        sex: Patient sex for disease inference
        age: Patient age for disease inference
    """
    # Resolve image path
    if image_name:
        image_path = Path("data") / image_name
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

    if not image_path:
        raise ValueError("Must provide image_path or image_name")

    # Execute appropriate workflow
    if all_models:
        return run_all_models_prediction(image_path, save_dir, all_cams, sex, age)
    else:
        # Resolve model path if needed
        if model_name:
            image_path, model_path = resolve_paths(
                image_name or Path(image_path).name, model_name
            )
        elif not model_path:
            raise ValueError("Must provide model_name or model_path")

        return run_single_model_prediction(
            image_path, model_path, model_name, save_dir, all_cams, sex, age
        )


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Nail disease classification with CAM visualization and disease inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single model, single CAM:
  python -m src.predict image.png efficientnetv2s
  
  # Single model, all CAM methods:
  python -m src.predict image.png swinv2t --all-cams
  
  # All models, single CAM each:
  python -m src.predict image.png --all-models
  
  # All models, all CAM methods:
  python -m src.predict image.png --all-models --all-cams
  
  # With disease inference:
  python -m src.predict image.png swinv2t --all-cams --sex female --age 45
  
  # Legacy full path:
  python -m src.predict --image-path data/image.png --model-path src/best_models/resnet50/best_model.pth
        """
    )

    # Positional arguments (simplified usage)
    parser.add_argument("image", nargs="?", type=str,
                        help="Image filename (in data/ directory)")
    parser.add_argument("model", nargs="?", type=str,
                        help="Model name (e.g., efficientnetv2s, swinv2t)")

    # Optional arguments (legacy full path usage)
    parser.add_argument("--image-path", type=str,
                        help="Full path to the input image")
    parser.add_argument("--model-path", type=str,
                        help="Full path to model weights (.pth file)")

    parser.add_argument("--save-dir", type=str, default="src/predictions",
                        help="Directory to save outputs")

    # Feature flags
    parser.add_argument("--all-cams", action="store_true",
                        help="Generate all CAM visualization methods")
    parser.add_argument("--all-models", action="store_true",
                        help="Run prediction with all available models")

    # Disease inference parameters
    parser.add_argument("--sex", type=str, choices=['male', 'female'],
                        help="Patient sex for disease inference")
    parser.add_argument("--age", type=float,
                        help="Patient age for disease inference")

    args = parser.parse_args()

    # Route to appropriate execution mode
    try:
        if args.all_models:
            if args.image:
                main(image_name=args.image, save_dir=args.save_dir,
                     all_cams=args.all_cams, all_models=True,
                     sex=args.sex, age=args.age)
            elif args.image_path:
                main(image_path=args.image_path, save_dir=args.save_dir,
                     all_cams=args.all_cams, all_models=True,
                     sex=args.sex, age=args.age)
            else:
                parser.error(
                    "--all-models requires either IMAGE or --image-path")

        elif args.image and args.model:
            main(image_name=args.image, model_name=args.model,
                 save_dir=args.save_dir, all_cams=args.all_cams,
                 sex=args.sex, age=args.age)

        elif args.image_path and args.model_path:
            main(image_path=args.image_path, model_path=args.model_path,
                 save_dir=args.save_dir, all_cams=args.all_cams,
                 sex=args.sex, age=args.age)

        else:
            parser.error(
                "Must provide either:\n"
                "  - IMAGE MODEL (simplified), or\n"
                "  - IMAGE --all-models (all models), or\n"
                "  - --image-path and --model-path (full paths)\n"
                "\nRun with -h for examples."
            )

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
