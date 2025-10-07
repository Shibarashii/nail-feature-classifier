# src/utils/interpretability.py
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def get_gradcam_config(model_name: str, model):
    model_name = model_name.lower()
    if "efficientnet" in model_name:
        return model.features[-1], None
    elif "resnet" in model_name:
        return model.layer4[-1], None
    elif "vgg" in model_name:
        return model.features[-1], None
    elif "regnet" in model_name:
        return model.trunk_output, None
    elif "swin" in model_name:
        def reshape_transform(tensor, height=7, width=7):
            result = tensor.reshape(tensor.size(
                0), height, width, tensor.size(2))
            result = result.permute(0, 3, 1, 2)
            return result
        target_layer = model.features[3].SwinTransformerBlockV2[-1].norm2
        return target_layer, reshape_transform
    elif "convnext" in model_name:
        return model.features[-1][-1].block[-1].layer_scale_2, None
    else:
        raise ValueError(f"No Grad-CAM config for model '{model_name}'")


def generate_gradcam(model, model_name: str, pred_idx, img_tensor, original_image, device):
    """
    Generates and returns a Grad-CAM visualization.

    Returns:
        np.ndarray: RGB image with Grad-CAM overlay (values 0-255)
    """
    # Prepare the original image (convert to float32 and normalize)
    rgb_img = np.array(original_image.resize(
        (224, 224)), dtype=np.float32) / 255.0

    try:
        # Get target layer and reshape function
        target_layer, reshape_transform = get_gradcam_config(
            model_name.lower(), model)

        # Initialize GradCAM
        cam = GradCAM(
            model=model,
            target_layers=[target_layer],
            reshape_transform=reshape_transform,
        )
        # Run Grad-CAM with respect to the predicted class
        grayscale_cam = cam(
            input_tensor=img_tensor,
            targets=[ClassifierOutputTarget(pred_idx)]
        )[0]
        # Overlay CAM and return as RGB image
        cam_img = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    except Exception as e:
        print(f"Grad-CAM failed: {e}")
        cam_img = rgb_img  # fallback

    # Ensure model is in eval mode
    model.eval()

    return rgb_img, cam_img
