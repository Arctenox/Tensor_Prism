import torch
import copy
from enum import Enum

# Define enums for cleaner and safer dropdown/option handling in ComfyUI
class CheckpointPrecision(str, Enum):
    """Defines the floating-point precision for tensor operations."""
    FP16 = 'fp16'
    FP32 = 'fp32'

class EnhancementMethod(str, Enum):
    """Defines the method used for tensor enhancement."""
    LINEAR = 'linear'
    ATTENTION = 'attention'

class TargetModule(str, Enum):
    """Defines the target modules within the checkpoint for enhancement."""
    UNET = 'unet'
    VAE = 'vae'
    TEXT_ENCODERS = 'text_encoders'
    ALL = 'all'

# Helper functions to identify keys belonging to specific model components
def is_unet_key(key: str) -> bool:
    """Checks if a given key string likely belongs to a UNet model."""
    key_lower = key.lower()
    return 'unet' in key_lower or 'model.diffusion_model' in key_lower

def is_vae_key(key: str) -> bool:
    """Checks if a given key string likely belongs to a VAE model."""
    key_lower = key.lower()
    return 'vae' in key_lower or 'autoencoder' in key_lower

def is_text_encoder_key(key: str) -> bool:
    """Checks if a given key string likely belongs to a Text Encoder model."""
    key_lower = key.lower()
    return 'clip' in key_lower or 'text_encoder' in key_lower or 'cond_stage' in key_lower


def clamp_tensor_stats(tensor: torch.Tensor, max_abs_threshold: float = 1.0) -> torch.Tensor:
    """
    Prevents runaway values in a tensor by softly clamping via tanh scaling if its
    maximum absolute value exceeds a threshold. Applied only to floating-point tensors.
    """
    if not tensor.is_floating_point():
        return tensor

    device = tensor.device
    max_val = tensor.abs().max()
    if max_val > max_abs_threshold:
        epsilon = torch.finfo(tensor.dtype).eps
        scale = max_abs_threshold / (max_val + epsilon)
        return (tensor * scale).to(device)
    return tensor


def apply_linear_smoothing(tensor: torch.Tensor, strength: float) -> torch.Tensor:
    """
    Applies a simple exponential moving average toward the tensor's mean.
    Effective when `strength` is greater than 0.
    """
    if strength <= 0.0:
        return tensor
    device = tensor.device
    mean_value = tensor.mean().to(device)
    return (tensor * (1.0 - strength) + mean_value * strength).to(device)


def apply_linear_sharpen(tensor: torch.Tensor, strength: float) -> torch.Tensor:
    """
    Applies an unsharp-like effect by boosting deviations from the tensor's mean.
    Effective when `strength` is greater than 0.
    """
    if strength <= 0.0:
        return tensor
    device = tensor.device
    mean_value = tensor.mean().to(device)
    return (tensor + (tensor - mean_value) * strength).to(device)


def attention_refinement_stub(tensor: torch.Tensor, strength: float) -> torch.Tensor:
    """
    Placeholder for attention-based refinement. This function emulates an attention-like
    effect by applying a guided non-linear boost on high-magnitude elements,
    followed by subtle smoothing to prevent harsh artifacts.
    """
    if strength <= 0.0:
        return tensor

    device = tensor.device
    magnitudes = tensor.abs()
    percentile_threshold = torch.quantile(magnitudes.view(-1), 0.75).to(device)
    mask = (magnitudes >= percentile_threshold).to(tensor.dtype).to(device)
    boosted_tensor = tensor + (tensor * mask * strength * 0.75)
    return apply_linear_smoothing(boosted_tensor, min(0.12 * strength, 0.5))


def apply_quality_boost(tensor: torch.Tensor, strength: float) -> torch.Tensor:
    """
    Applies a multi-stage subtle enhancement: local contrast-like scaling + gentle
    non-linear sharpening. Effective when `strength` is greater than 0.
    """
    if strength <= 0.0:
        return tensor
    device = tensor.device
    mean_value = tensor.mean().to(device)
    boosted_tensor = (tensor - mean_value) * (1.0 + 0.6 * strength) + mean_value
    boosted_tensor = boosted_tensor * (1.0 + 0.12 * strength * (boosted_tensor - mean_value))
    return boosted_tensor.to(device)


def apply_adaptive_overbake_limiter(
    original_tensor: torch.Tensor,
    modified_tensor: torch.Tensor,
    max_relative_increase: float = 0.12
) -> torch.Tensor:
    """
    Compares the modified tensor to the original. If the global mean absolute
    change exceeds `max_relative_increase`, the modification is scaled back
    to avoid overbaking.
    """
    device = original_tensor.device
    with torch.no_grad():
        epsilon = torch.finfo(original_tensor.dtype).eps
        original_mean_abs = original_tensor.abs().mean().item() + epsilon
        modified_mean_abs = modified_tensor.abs().mean().item() + epsilon

        relative_change = (modified_mean_abs - original_mean_abs) / original_mean_abs

        if relative_change <= max_relative_increase:
            return modified_tensor.to(device)

        scale_back_factor = 1.0 - (relative_change - max_relative_increase) / (relative_change + epsilon)
        scale_back_factor = max(0.0, scale_back_factor)

        return (original_tensor + (modified_tensor - original_tensor) * scale_back_factor).to(device)


class ModelEnhancerTensorPrism:
    """
    ComfyUI custom node to enhance Stable Diffusion XL checkpoints.
    Applies various processing steps to selected tensors within the model's state_dict
    to improve aspects like smoothing, sharpening, and overall quality.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint": ("MODEL",),
                "smoothing": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sharpening": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01}),
                "quality_boost": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 1.0, "step": 0.01}),
                "blend_strength": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "precision": (
                    [p.value for p in CheckpointPrecision],
                    {"default": CheckpointPrecision.FP16.value}
                ),
                "method": (
                    [m.value for m in EnhancementMethod],
                    {"default": EnhancementMethod.LINEAR.value}
                ),
                "modules_to_enhance": (
                    [mod.value for mod in TargetModule],
                    {"default": TargetModule.UNET.value}
                ),
                "adaptive_overbake_prevention": ("BOOLEAN", {"default": True}),
                "attention_iterations": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("enhanced_checkpoint",)
    FUNCTION = "enhance_checkpoint"
    CATEGORY = "checkpoint/enhancement"

    def __init__(self):
        pass

    def _should_process_key(self, key: str, target_modules) -> bool:
        """
        Determines if a given tensor key should be processed based on the
        selected target modules.
        """
        if isinstance(target_modules, str):
            target_modules = [target_modules]
        
        target_modules_set = {m.lower() for m in target_modules}

        if TargetModule.ALL.value in target_modules_set:
            return True
        if is_unet_key(key) and TargetModule.UNET.value in target_modules_set:
            return True
        if is_vae_key(key) and TargetModule.VAE.value in target_modules_set:
            return True
        if is_text_encoder_key(key) and (TargetModule.TEXT_ENCODERS.value in target_modules_set or 'textenc' in target_modules_set or 'text' in target_modules_set):
            return True
        return False

    def _cast_tensor_to_precision(self, tensor: torch.Tensor, precision: CheckpointPrecision) -> torch.Tensor:
        """Casts a floating-point tensor to the specified precision."""
        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            return tensor

        device = tensor.device
        if precision == CheckpointPrecision.FP16:
            return tensor.half().to(device)
        elif precision == CheckpointPrecision.FP32:
            return tensor.float().to(device)
        return tensor.to(device)

    def _enhance_single_tensor(
        self,
        original_tensor: torch.Tensor,
        smoothing_strength: float,
        sharpening_strength: float,
        quality_boost_strength: float,
        enhancement_method: EnhancementMethod,
        adaptive_overbake_enabled: bool,
        attention_iterations: int,
        target_precision: CheckpointPrecision
    ) -> torch.Tensor:
        """Applies the enhancement pipeline to a single tensor."""

        device = original_tensor.device
        processed_tensor = original_tensor.clone().to(device)

        processed_tensor = self._cast_tensor_to_precision(processed_tensor, target_precision)

        if enhancement_method == EnhancementMethod.LINEAR:
            if smoothing_strength > 0.0:
                processed_tensor = apply_linear_smoothing(processed_tensor, smoothing_strength)
            if sharpening_strength > 0.0:
                processed_tensor = apply_linear_sharpen(processed_tensor, sharpening_strength)
        elif enhancement_method == EnhancementMethod.ATTENTION:
            for _ in range(max(1, attention_iterations)):
                combined_strength = (sharpening_strength + smoothing_strength) * 0.7
                processed_tensor = attention_refinement_stub(processed_tensor, combined_strength)

        if quality_boost_strength > 0.0:
            processed_tensor = apply_quality_boost(processed_tensor, quality_boost_strength)

        max_abs_ref_val = original_tensor.abs().max().item()
        processed_tensor = clamp_tensor_stats(processed_tensor, max_abs_threshold=max_abs_ref_val * 3.0 + 1e-6)

        if adaptive_overbake_enabled:
            processed_tensor = apply_adaptive_overbake_limiter(
                original_tensor, processed_tensor, max_relative_increase=0.16
            )

        return processed_tensor.to(device)

    def enhance_checkpoint(
        self,
        checkpoint,
        smoothing: float,
        sharpening: float,
        quality_boost: float,
        blend_strength: float,
        precision: str,
        method: str,
        modules_to_enhance,
        adaptive_overbake_prevention: bool,
        attention_iterations: int
    ) -> tuple:
        """
        Main function to process and enhance a ComfyUI checkpoint (MODEL object).
        """
        if checkpoint is None:
            raise ValueError("Input checkpoint cannot be None.")
        
        # ComfyUI MODEL object - work with it directly, don't deep copy
        if not hasattr(checkpoint, 'model'):
            raise TypeError("Input checkpoint must be a ComfyUI MODEL object.")

        # Convert string inputs to Enum types
        try:
            target_precision = CheckpointPrecision(precision)
            enhancement_method = EnhancementMethod(method)
        except ValueError as e:
            raise ValueError(f"Invalid enum value provided for precision or method: {e}")

        # Clone the model using ComfyUI's clone method (this is safe)
        enhanced_model = checkpoint.clone()

        # Get a reference to the actual model weights
        model_sd = enhanced_model.model.state_dict()

        # Process tensors in-place on the cloned model
        with torch.no_grad():
            for key in list(model_sd.keys()):
                try:
                    value = model_sd[key]
                    
                    if self._should_process_key(key, modules_to_enhance) and isinstance(value, torch.Tensor) and value.is_floating_point():
                        original_tensor = value
                        device = original_tensor.device

                        # Apply enhancement pipeline
                        enhanced_tensor = self._enhance_single_tensor(
                            original_tensor=original_tensor,
                            smoothing_strength=smoothing,
                            sharpening_strength=sharpening,
                            quality_boost_strength=quality_boost,
                            enhancement_method=enhancement_method,
                            adaptive_overbake_enabled=adaptive_overbake_prevention,
                            attention_iterations=attention_iterations,
                            target_precision=target_precision
                        )

                        enhanced_tensor = enhanced_tensor.to(device)

                        # Blend original with enhanced
                        blended_tensor = original_tensor * (1.0 - blend_strength) + enhanced_tensor * blend_strength

                        # Final clamp
                        final_max_abs_threshold = max(1.0, original_tensor.abs().max().item() * 2.0)
                        final_tensor = clamp_tensor_stats(blended_tensor, max_abs_threshold=final_max_abs_threshold).to(device)
                        
                        # Update the tensor in the state dict
                        model_sd[key] = final_tensor

                except Exception as e:
                    print(f"Warning: Model Enhancer failed to process tensor '{key}'. Keeping original value. Error: {e}")
                    continue

        return (enhanced_model,)


# ComfyUI Node Class Mappings for registration
NODE_CLASS_MAPPINGS = {
    "ModelEnhancerTensorPrism": ModelEnhancerTensorPrism
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelEnhancerTensorPrism": "Model Enhancer (Tensor Prism)"
}
