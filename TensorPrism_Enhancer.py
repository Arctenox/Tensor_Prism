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
    TEXT_ENCODERS = 'text_encoders' # Renamed from 'textenc' for clarity
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
        # Use torch.finfo for robust epsilon value based on tensor dtype
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
    For a real implementation, this would involve invoking attention maps or
    cross-attention reweighting using model internals.
    Effective when `strength` is greater than 0.
    """
    if strength <= 0.0:
        return tensor

    device = tensor.device
    # Emulate attention emphasis by applying a guided non-linear boost on high-magnitude elements
    magnitudes = tensor.abs()
    # Find the 75th percentile of magnitudes to identify "important" features
    percentile_threshold = torch.quantile(magnitudes.view(-1), 0.75).to(device)
    # Create a mask for elements above the threshold
    mask = (magnitudes >= percentile_threshold).to(tensor.dtype).to(device)
    # Boost these elements, then apply subtle smoothing to prevent harsh artifacts
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
    # Increase local contrast around the mean
    boosted_tensor = (tensor - mean_value) * (1.0 + 0.6 * strength) + mean_value
    # Apply gentle non-linear sharpening based on deviation from mean
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
    to avoid overbaking. Prevents excessively strong modifications.
    """
    device = original_tensor.device
    with torch.no_grad():
        # Add epsilon to prevent division by zero for tensors with all zeros
        epsilon = torch.finfo(original_tensor.dtype).eps
        original_mean_abs = original_tensor.abs().mean().item() + epsilon
        modified_mean_abs = modified_tensor.abs().mean().item() + epsilon

        relative_change = (modified_mean_abs - original_mean_abs) / original_mean_abs

        if relative_change <= max_relative_increase:
            return modified_tensor.to(device)

        # Calculate a scale factor to bring the relative change down to the limit
        # Ensure scale_back_factor is not negative
        scale_back_factor = 1.0 - (relative_change - max_relative_increase) / (relative_change + epsilon)
        scale_back_factor = max(0.0, scale_back_factor)

        # Blend the original with the modified based on the scale_back_factor
        return (original_tensor + (modified_tensor - original_tensor) * scale_back_factor).to(device)


class ModelEnhancerTensorPrism:
    """
    ComfyUI custom node to enhance Stable Diffusion XL checkpoints.
    Applies various processing steps to selected tensors within the model's state_dict
    to improve aspects like smoothing, sharpening, and overall quality.
    """

    # Define the input types for the ComfyUI node UI
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint": ("MODEL",),  # Input checkpoint (MODEL object)
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

    # Define the output types for the ComfyUI node UI
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("enhanced_checkpoint",)
    FUNCTION = "enhance_checkpoint"
    CATEGORY = "checkpoint/enhancement" # Or a suitable category

    def __init__(self):
        # ComfyUI nodes typically don't need a complex __init__ if all parameters
        # are passed via the INPUT_TYPES to the functional method.
        pass

    def _should_process_key(self, key: str, target_modules) -> bool:
        """
        Determines if a given tensor key should be processed based on the
        selected target modules. Now handles both single strings and lists.
        """
        # Handle both single string and list inputs
        if isinstance(target_modules, str):
            target_modules = [target_modules]
        
        # Convert target_modules to a set of lowercased strings for efficient lookup
        target_modules_set = {m.lower() for m in target_modules}

        if TargetModule.ALL.value in target_modules_set:
            return True
        if is_unet_key(key) and TargetModule.UNET.value in target_modules_set:
            return True
        if is_vae_key(key) and TargetModule.VAE.value in target_modules_set:
            return True
        # Check for both new enum value and original string values for compatibility
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
        return tensor.to(device) # Should not happen if enum is used correctly

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

        # Ensure we operate on a clone and on the correct device
        device = original_tensor.device
        processed_tensor = original_tensor.clone().to(device)

        # Cast precision upfront for operations
        processed_tensor = self._cast_tensor_to_precision(processed_tensor, target_precision)

        # Apply chosen enhancement method
        if enhancement_method == EnhancementMethod.LINEAR:
            if smoothing_strength > 0.0:
                processed_tensor = apply_linear_smoothing(processed_tensor, smoothing_strength)
            if sharpening_strength > 0.0:
                processed_tensor = apply_linear_sharpen(processed_tensor, sharpening_strength)
        elif enhancement_method == EnhancementMethod.ATTENTION:
            # Run attention-based refinement for specified iterations
            for _ in range(max(1, attention_iterations)):
                combined_strength = (sharpening_strength + smoothing_strength) * 0.7
                processed_tensor = attention_refinement_stub(processed_tensor, combined_strength)

        # Apply overall quality boost
        if quality_boost_strength > 0.0:
            processed_tensor = apply_quality_boost(processed_tensor, quality_boost_strength)

        # Clamp extreme values softly, relative to original max absolute value
        # Using a factor (e.g., 3.0) to allow for growth but prevent explosion
        max_abs_ref_val = original_tensor.abs().max().item()
        # Adding a small epsilon to max_abs_ref_val in case it's zero
        processed_tensor = clamp_tensor_stats(processed_tensor, max_abs_threshold=max_abs_ref_val * 3.0 + 1e-6)

        # Apply adaptive overbake limiter if enabled
        if adaptive_overbake_enabled:
            # Use a slightly higher max_relative_increase for the internal processing step
            # to allow for more aggressive enhancement before final blending.
            processed_tensor = apply_adaptive_overbake_limiter(
                original_tensor, processed_tensor, max_relative_increase=0.16
            )

        return processed_tensor.to(device)

    def enhance_checkpoint(
        self,
        checkpoint,  # Changed from dict to accept MODEL objects
        smoothing: float,
        sharpening: float,
        quality_boost: float,
        blend_strength: float,
        precision: str, # Will be enum value string from ComfyUI
        method: str,    # Will be enum value string from ComfyUI
        modules_to_enhance,  # Can be string or list[str]
        adaptive_overbake_prevention: bool,
        attention_iterations: int
    ) -> tuple:  # Changed return type
        """
        Main function to process and enhance a ComfyUI checkpoint (MODEL object).
        This method is called by ComfyUI when the node executes.
        """
        if checkpoint is None:
            raise ValueError("Input checkpoint cannot be None.")
        
        # Extract state_dict from ComfyUI MODEL object
        if hasattr(checkpoint, 'model') and hasattr(checkpoint.model, 'state_dict'):
            # ComfyUI MODEL object - extract the state_dict
            state_dict = checkpoint.model.state_dict()
            model_wrapper = checkpoint
        elif isinstance(checkpoint, dict):
            # Already a state_dict (for backwards compatibility)
            state_dict = checkpoint
            model_wrapper = None
        else:
            raise TypeError("Input checkpoint must be a ComfyUI MODEL object or dictionary-like state_dict.")

        # Convert string inputs from ComfyUI to Enum types for type safety and clarity
        try:
            target_precision = CheckpointPrecision(precision)
            enhancement_method = EnhancementMethod(method)
        except ValueError as e:
            raise ValueError(f"Invalid enum value provided for precision or method: {e}")

        # Deep copy the state_dict to ensure no in-place modification of the original
        # This can be memory-intensive for very large checkpoints, but ensures safety.
        enhanced_state_dict = copy.deepcopy(state_dict)

        # Iterate over all keys in the state_dict to apply enhancements
        for key, value in enhanced_state_dict.items():
            try:
                # Process only PyTorch tensors that are floating-point and match selected modules
                if self._should_process_key(key, modules_to_enhance) and isinstance(value, torch.Tensor) and value.is_floating_point():
                    original_tensor = state_dict[key] # Reference the original tensor from the extracted state_dict
                    device = original_tensor.device

                    # Apply the full enhancement pipeline to the current tensor
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

                    # Ensure enhanced tensor is on correct device
                    enhanced_tensor = enhanced_tensor.to(device)

                    # Blend the original tensor with the enhanced tensor based on blend_strength
                    blended_tensor = original_tensor * (1.0 - blend_strength) + enhanced_tensor * blend_strength

                    # Apply a final, robust clamp to the blended tensor to maintain numeric stability.
                    # The max_abs_threshold is set to be at least 1.0 or twice the original's max abs.
                    final_max_abs_threshold = max(1.0, original_tensor.abs().max().item() * 2.0)
                    enhanced_state_dict[key] = clamp_tensor_stats(blended_tensor, max_abs_threshold=final_max_abs_threshold).to(device)
                else:
                    # If not a tensor, not floating point, or not selected for processing,
                    # ensure the original value is preserved (even though deepcopy usually handles this).
                    enhanced_state_dict[key] = value

            except Exception as e:
                # Log the error and revert to the original tensor for this specific key
                # This prevents a single problematic tensor from crashing the entire node.
                print(f"Warning: Model Enhancer failed to process tensor '{key}'. Keeping original value. Error: {e}")
                enhanced_state_dict[key] = state_dict[key] # Ensure original is used if processing failed

        # Return the enhanced model in ComfyUI format
        if model_wrapper is not None:
            # Create a new model wrapper with the enhanced state_dict
            enhanced_model = copy.deepcopy(model_wrapper)
            enhanced_model.model.load_state_dict(enhanced_state_dict, strict=False)
            return (enhanced_model,)
        else:
            # For backwards compatibility, return the state_dict
            return (enhanced_state_dict,)

# ComfyUI Node Class Mappings for registration
NODE_CLASS_MAPPINGS = {
    "ModelEnhancerTensorPrism": ModelEnhancerTensorPrism
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelEnhancerTensorPrism": "Model Enhancer (Tensor Prism)"
}