# src/utils/interpretability.py
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def get_gradcam_config(model_name: str, model):
    """
    Returns the target layer and optional reshape transform for Grad-CAM.
    """
    model_name = model_name.lower()

    if "efficientnet" in model_name:
        return model.features[-1], None

    elif "resnet" in model_name:
        return model.layer4[-1], None

    elif "vgg" in model_name:
        return model.features[-1], None

    elif "regnet" in model_name:
        return model.trunk_output, None

    elif "swinv2" in model_name or "swin_v2" in model_name:
        # SwinV2 specific configuration
        def reshape_transform_swinv2(tensor):
            # Handle different tensor formats
            if len(tensor.shape) == 3:  # (B, N, C)
                B, N, C = tensor.shape
                H = W = int(np.sqrt(N))
                # Transpose to (B, C, N) first
                result = tensor.transpose(1, 2)  # (B, C, N)
                # Then reshape to (B, C, H, W)
                result = result.reshape(B, C, H, W)
                return result
            elif len(tensor.shape) == 4:
                # Check if it's (B, H, W, C) or (B, C, H, W)
                B, dim1, dim2, dim3 = tensor.shape

                # If dim3 is large (e.g., 768), it's likely (B, H, W, C)
                if dim3 > dim1 and dim3 > dim2:
                    # (B, H, W, C) -> (B, C, H, W)
                    result = tensor.permute(0, 3, 1, 2)
                    return result
                else:
                    # Already (B, C, H, W)
                    return tensor
            else:
                raise ValueError(f"Unexpected tensor shape: {tensor.shape}")

        # Find the best target layer - use an earlier layer for better localization
        target_layer = None

        # Try multiple possible paths
        try:
            if hasattr(model, 'features'):
                # torchvision style - use second-to-last stage for better features
                # features[-1] is the last stage, features[-2] might be better
                if len(model.features) > 1:
                    # Try the last transformer block in the second-to-last stage
                    second_last_stage = model.features[-2]
                    if hasattr(second_last_stage, 'blocks') or hasattr(second_last_stage, 'SwinTransformerBlockV2'):
                        if hasattr(second_last_stage, 'SwinTransformerBlockV2'):
                            target_layer = second_last_stage.SwinTransformerBlockV2[-1]
                        elif hasattr(second_last_stage, 'blocks'):
                            target_layer = second_last_stage.blocks[-1]

                # Fallback to last stage if second-to-last didn't work
                if target_layer is None:
                    last_stage = model.features[-1]
                    if hasattr(last_stage, 'blocks') or hasattr(last_stage, 'SwinTransformerBlockV2'):
                        if hasattr(last_stage, 'SwinTransformerBlockV2'):
                            target_layer = last_stage.SwinTransformerBlockV2[-1]
                        elif hasattr(last_stage, 'blocks'):
                            target_layer = last_stage.blocks[-1]
                    else:
                        target_layer = model.features[-1]

            elif hasattr(model, 'layers'):
                # timm style - also try second-to-last
                if len(model.layers) > 1:
                    target_layer = model.layers[-2].blocks[-1]
                else:
                    target_layer = model.layers[-1].blocks[-1]
        except Exception as e:
            print(f"Warning: Could not find optimal layer structure: {e}")

        # Fallback: find last normalization or the deepest conv/linear layer
        if target_layer is None:
            for name, module in reversed(list(model.named_modules())):
                if isinstance(module, (torch.nn.LayerNorm, torch.nn.Conv2d, torch.nn.Linear)):
                    target_layer = module
                    print(f"Using fallback layer: {name}")
                    break

        if target_layer is None:
            raise ValueError("Could not find suitable target layer for SwinV2")

        return target_layer, reshape_transform_swinv2

    elif "swin" in model_name:
        # Original Swin Transformer
        def reshape_transform_swin(tensor, height=7, width=7):
            if len(tensor.shape) == 3:
                result = tensor.reshape(tensor.size(
                    0), height, width, tensor.size(2))
                result = result.permute(0, 3, 1, 2)
                return result
            return tensor

        try:
            if hasattr(model, 'features'):
                target_layer = model.features[-1][-1] if isinstance(
                    model.features[-1], torch.nn.Sequential) else model.features[-1]
            else:
                target_layer = model.layers[-1].blocks[-1]
        except:
            if hasattr(model, 'features'):
                target_layer = model.features[-1]
            else:
                target_layer = model.layers[-1]

        return target_layer, reshape_transform_swin

    elif "convnext" in model_name:
        # ConvNeXt specific configuration - use the last stage
        if hasattr(model, 'features'):
            # torchvision ConvNeXt: use entire last stage for better visualization
            # Use stage before normalization/pooling
            target_layer = model.features[-2]
        elif hasattr(model, 'stages'):
            # timm ConvNeXt
            target_layer = model.stages[-1]
        else:
            # Final fallback: use the last convolutional layer
            target_layer = None
            for name, module in reversed(list(model.named_modules())):
                if isinstance(module, torch.nn.Conv2d):
                    target_layer = module
                    break
            if target_layer is None:
                raise ValueError(
                    "Could not find suitable target layer for ConvNeXt")

        return target_layer, None

    else:
        raise ValueError(f"No Grad-CAM config for model '{model_name}'")


def generate_feature_map_visualization(model, model_name: str, img_tensor, original_image):
    """
    Generate visualization from the last feature maps.
    This is a simple but effective method for any model.
    """
    rgb_img = np.array(original_image.resize(
        (224, 224)), dtype=np.float32) / 255.0

    try:
        model.eval()
        activations = []

        def hook_fn(module, input, output):
            activations.append(output.detach())

        # Get the target layer
        target_layer, _ = get_gradcam_config(model_name.lower(), model)
        hook = target_layer.register_forward_hook(hook_fn)

        # Forward pass
        with torch.no_grad():
            _ = model(img_tensor)

        hook.remove()

        if len(activations) > 0:
            activation = activations[0]

            # Handle different shapes
            if len(activation.shape) == 3:  # (B, N, C)
                B, N, C = activation.shape
                H = W = int(np.sqrt(N))
                # Reshape: (B, N, C) -> (B, H, W, C) -> (B, C, H, W)
                activation = activation.reshape(B, H, W, C).permute(0, 3, 1, 2)
            elif len(activation.shape) == 4:  # (B, C, H, W) or (B, H, W, C)
                pass
            else:
                raise ValueError(
                    f"Unexpected activation shape: {activation.shape}")

            # Average across channels to get spatial importance
            heatmap = activation[0].mean(dim=0).cpu().numpy()

            # Normalize
            heatmap = (heatmap - heatmap.min()) / \
                (heatmap.max() - heatmap.min() + 1e-8)

            # Resize to image size
            heatmap_resized = cv2.resize(heatmap, (224, 224))

            # Apply colormap
            cam_img = show_cam_on_image(rgb_img, heatmap_resized, use_rgb=True)

            return rgb_img, cam_img
        else:
            raise ValueError("No activations captured")

    except Exception as e:
        print(f"✗ Feature map visualization failed: {e}")
        return None, None


def generate_attention_rollout(model, img_tensor, original_image):
    """
    Generate attention rollout visualization for transformer models.
    Enhanced to handle various Swin implementations.
    """
    rgb_img = np.array(original_image.resize(
        (224, 224)), dtype=np.float32) / 255.0

    try:
        model.eval()
        attentions = []

        # Hook to capture attention weights
        def hook_fn(module, input, output):
            # Try to extract attention from different output formats
            if isinstance(output, tuple):
                for item in output:
                    if item is not None and isinstance(item, torch.Tensor):
                        # Check if this looks like an attention matrix
                        if len(item.shape) == 4 and item.shape[2] == item.shape[3]:
                            attentions.append(item.detach().cpu())
                            break
            elif isinstance(output, torch.Tensor):
                # Check if this is an attention matrix
                if len(output.shape) == 4 and output.shape[2] == output.shape[3]:
                    attentions.append(output.detach().cpu())

        hooks = []
        # Register hooks more broadly
        for name, module in model.named_modules():
            # Look for attention-related modules
            if any(keyword in name.lower() for keyword in ['attn', 'attention', 'self_attn']):
                if 'drop' not in name.lower() and 'proj' not in name.lower():
                    hooks.append(module.register_forward_hook(hook_fn))

        # Forward pass
        with torch.no_grad():
            _ = model(img_tensor)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        if len(attentions) > 0:
            # Use the last attention map
            attention = attentions[-1][0]  # First batch item

            # Average across heads
            if len(attention.shape) == 3:
                heatmap = attention.mean(dim=0).numpy()
            else:
                heatmap = attention.mean(dim=0).mean(dim=0).numpy()

            # Normalize
            heatmap = (heatmap - heatmap.min()) / \
                (heatmap.max() - heatmap.min() + 1e-8)

            # Resize to square if needed
            if heatmap.shape[0] != heatmap.shape[1]:
                size = int(np.sqrt(heatmap.shape[0]))
                heatmap = heatmap[:size*size].reshape(size, size)

            # Resize to image size
            heatmap = cv2.resize(heatmap, (224, 224))

            cam_img = show_cam_on_image(rgb_img, heatmap, use_rgb=True)
            return rgb_img, cam_img
        else:
            raise ValueError("No attention weights captured")

    except Exception as e:
        print(f"✗ Attention rollout failed: {e}")
        return None, None


def generate_gradcam_plusplus(model, model_name: str, pred_idx, img_tensor, original_image, device):
    """
    Generate GradCAM++ visualization - faster than ScoreCAM, more robust than GradCAM.
    """
    rgb_img = np.array(original_image.resize(
        (224, 224)), dtype=np.float32) / 255.0

    try:
        target_layer, reshape_transform = get_gradcam_config(
            model_name.lower(), model)

        cam = GradCAMPlusPlus(
            model=model,
            target_layers=[target_layer],
            reshape_transform=reshape_transform,
        )

        grayscale_cam = cam(
            input_tensor=img_tensor,
            targets=[ClassifierOutputTarget(pred_idx)]
        )[0]

        cam_img = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        return rgb_img, cam_img

    except Exception as e:
        print(f"✗ GradCAM++ failed: {e}")
        return None, None


def generate_scorecam(model, model_name: str, pred_idx, img_tensor, original_image, device):
    """
    Generate ScoreCAM visualization - more reliable for transformers but slower.
    """
    rgb_img = np.array(original_image.resize(
        (224, 224)), dtype=np.float32) / 255.0

    try:
        target_layer, reshape_transform = get_gradcam_config(
            model_name.lower(), model)

        cam = ScoreCAM(
            model=model,
            target_layers=[target_layer],
            reshape_transform=reshape_transform,
        )

        grayscale_cam = cam(
            input_tensor=img_tensor,
            targets=[ClassifierOutputTarget(pred_idx)]
        )[0]

        cam_img = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        return rgb_img, cam_img

    except Exception as e:
        print(f"✗ ScoreCAM failed: {e}")
        return None, None


def generate_ablationcam(model, model_name: str, pred_idx, img_tensor, original_image, device):
    """
    Generate AblationCAM visualization - gradient-free method.
    """
    rgb_img = np.array(original_image.resize(
        (224, 224)), dtype=np.float32) / 255.0

    try:
        target_layer, reshape_transform = get_gradcam_config(
            model_name.lower(), model)

        cam = AblationCAM(
            model=model,
            target_layers=[target_layer],
            reshape_transform=reshape_transform,
            ablation_layer=torch.nn.Identity()
        )

        grayscale_cam = cam(
            input_tensor=img_tensor,
            targets=[ClassifierOutputTarget(pred_idx)]
        )[0]

        cam_img = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        return rgb_img, cam_img

    except Exception as e:
        print(f"✗ AblationCAM failed: {e}")
        return None, None


def generate_layercam(model, model_name: str, pred_idx, img_tensor, original_image, device):
    """
    Generate LayerCAM visualization - fast gradient-based method.
    """
    from pytorch_grad_cam import LayerCAM

    rgb_img = np.array(original_image.resize(
        (224, 224)), dtype=np.float32) / 255.0

    try:
        target_layer, reshape_transform = get_gradcam_config(
            model_name.lower(), model)

        cam = LayerCAM(
            model=model,
            target_layers=[target_layer],
            reshape_transform=reshape_transform,
        )

        grayscale_cam = cam(
            input_tensor=img_tensor,
            targets=[ClassifierOutputTarget(pred_idx)]
        )[0]

        cam_img = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        return rgb_img, cam_img

    except Exception as e:
        print(f"✗ LayerCAM failed: {e}")
        return None, None


def generate_all_cams(model, model_name: str, pred_idx, img_tensor, original_image, device):
    """
    Generate all available CAM visualizations for comprehensive analysis.

    Args:
        model: PyTorch model
        model_name (str): Name of the model
        pred_idx (int): Predicted class index
        img_tensor (torch.Tensor): Input image tensor
        original_image (PIL.Image): Original image
        device: Device to run on

    Returns:
        dict: Dictionary mapping method name -> visualization image
    """
    rgb_img = np.array(original_image.resize(
        (224, 224)), dtype=np.float32) / 255.0
    model.eval()

    # Only fast gradient-based methods
    methods = [
        ("GradCAM", lambda: generate_gradcam_standard(
            model, model_name, pred_idx, img_tensor, original_image, device)),
        ("GradCAM++", lambda: generate_gradcam_plusplus(model,
         model_name, pred_idx, img_tensor, original_image, device)),
        ("LayerCAM", lambda: generate_layercam(model, model_name,
         pred_idx, img_tensor, original_image, device)),
    ]

    results = {}
    print(f"\n{'Method':<20} {'Status':<10}")
    print("-" * 30)

    for method_name, method_func in methods:
        try:
            _, cam_img = method_func()
            if cam_img is not None:
                results[method_name] = cam_img
                print(f"{method_name:<20} ✓")
            else:
                results[method_name] = None
                print(f"{method_name:<20} ✗ (returned None)")
        except Exception as e:
            results[method_name] = None
            print(f"{method_name:<20} ✗ ({str(e)[:30]}...)")

    print("-" * 30)
    successful = sum(1 for v in results.values() if v is not None)
    print(
        f"Successfully generated {successful}/{len(methods)} visualizations\n")

    return results


def generate_gradcam_standard(model, model_name: str, pred_idx, img_tensor, original_image, device):
    """
    Generate standard GradCAM visualization.
    """
    rgb_img = np.array(original_image.resize(
        (224, 224)), dtype=np.float32) / 255.0

    try:
        target_layer, reshape_transform = get_gradcam_config(
            model_name.lower(), model)

        cam = GradCAM(
            model=model,
            target_layers=[target_layer],
            reshape_transform=reshape_transform,
        )

        grayscale_cam = cam(
            input_tensor=img_tensor,
            targets=[ClassifierOutputTarget(pred_idx)]
        )[0]

        cam_img = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        return rgb_img, cam_img

    except Exception as e:
        return None, None


def generate_gradcam(model, model_name: str, pred_idx, img_tensor, original_image, device):
    """
    Generates and returns a Grad-CAM visualization with fallback methods.
    For SwinV2, tries multiple methods in order of reliability.

    Returns:
        tuple: (rgb_img, cam_img) - Original image and CAM overlay
    """
    rgb_img = np.array(original_image.resize(
        (224, 224)), dtype=np.float32) / 255.0
    model.eval()

    # For SwinV2, try multiple methods
    if "swinv2" in model_name.lower() or "swin_v2" in model_name.lower():
        print("🔍 Generating Grad-CAM for SwinV2...")

        # Ordered by speed and reliability - all fast gradient-based methods
        methods = [
            ("GradCAM++", lambda: generate_gradcam_plusplus(model,
             model_name, pred_idx, img_tensor, original_image, device)),
            ("LayerCAM", lambda: generate_layercam(model, model_name,
             pred_idx, img_tensor, original_image, device)),
            ("GradCAM", lambda: generate_gradcam_standard(
                model, model_name, pred_idx, img_tensor, original_image, device)),
        ]

        for method_name, method_func in methods:
            try:
                result_rgb, result_cam = method_func()
                if result_cam is not None and not np.array_equal(result_rgb, result_cam / 255.0):
                    print(f"✅ {method_name} visualization successful")
                    return result_rgb, result_cam
            except Exception as e:
                print(f"⚠️  {method_name} failed, trying next method...")

        print("\n⚠️  All methods failed, returning original image...")

    # Standard GradCAM for other models or as final fallback
    try:
        target_layer, reshape_transform = get_gradcam_config(
            model_name.lower(), model)

        cam = GradCAM(
            model=model,
            target_layers=[target_layer],
            reshape_transform=reshape_transform,
        )

        grayscale_cam = cam(
            input_tensor=img_tensor,
            targets=[ClassifierOutputTarget(pred_idx)]
        )[0]

        cam_img = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    except Exception as e:
        print(f"✗ All Grad-CAM methods failed: {e}")
        cam_img = (rgb_img * 255).astype(np.uint8)  # Return original image

    return rgb_img, cam_img
