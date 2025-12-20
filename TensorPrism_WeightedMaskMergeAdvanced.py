import torch
import numpy as np
import comfy.utils

class TensorPrism_WeightedMaskMerge:
    """
    Advanced weighted mask merge with sophisticated blending modes and per-layer control
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_A": ("MODEL",),
                "model_B": ("MODEL",),
                "mask": ("MASK",),
                "merge_ratio": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01,
                    "help": "Global merge strength multiplier"
                }),
                "blend_mode": (["linear", "sigmoid", "cosine", "exponential", "logarithmic", "smoothstep"], {
                    "default": "linear"
                }),
            },
            "optional": {
                "curve_power": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1,
                    "help": "Curve intensity for non-linear modes"
                }),
                "preserve_extremes": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Preserve 0/1",
                    "label_off": "Full Blend"
                }),
                "noise_injection": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.1, "step": 0.001,
                    "help": "Add subtle noise to break symmetry"
                }),
                "layer_scaling": (["uniform", "depth_progressive", "shallow_bias", "deep_bias"], {
                    "default": "uniform"
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("merged_model",)
    FUNCTION = "merge_models"
    CATEGORY = "Tensor_Prism/Advanced"

    def apply_blend_curve(self, value, mode, power):
        """Apply sophisticated blending curves"""
        if mode == "linear":
            return value
        elif mode == "sigmoid":
            # Smooth S-curve
            return 1.0 / (1.0 + np.exp(-power * (value - 0.5) * 10))
        elif mode == "cosine":
            # Smooth cosine interpolation
            return (1.0 - np.cos(value * np.pi)) * 0.5
        elif mode == "exponential":
            # Exponential curve
            if value < 0.5:
                return 0.5 * np.power(2.0 * value, power)
            else:
                return 1.0 - 0.5 * np.power(2.0 * (1.0 - value), power)
        elif mode == "logarithmic":
            # Logarithmic curve
            epsilon = 1e-6
            return np.log(value + epsilon) / np.log(1.0 + epsilon)
        elif mode == "smoothstep":
            # Smoothstep interpolation
            value = np.clip(value, 0.0, 1.0)
            return value * value * (3.0 - 2.0 * value)
        return value

    def get_layer_scale(self, layer_num, total_layers, scaling_mode):
        """Calculate layer-specific scaling factor"""
        if total_layers <= 1:
            return 1.0
            
        position = layer_num / (total_layers - 1)
        
        if scaling_mode == "uniform":
            return 1.0
        elif scaling_mode == "depth_progressive":
            # Increase influence in deeper layers
            return 0.5 + 0.5 * position
        elif scaling_mode == "shallow_bias":
            # More influence in shallow layers
            return 1.5 - 0.5 * position
        elif scaling_mode == "deep_bias":
            # More influence in deep layers
            return 0.5 + position
        return 1.0

    def extract_layer_number(self, key):
        """Extract layer number from parameter name"""
        import re
        patterns = [
            r"layers\.(\d+)\.",
            r"blocks\.(\d+)\.",
            r"h\.(\d+)\.",
            r"layer\.(\d+)\.",
            r"encoder\.layer\.(\d+)\.",
            r"decoder\.layer\.(\d+)\.",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, key)
            if match:
                return int(match.group(1))
        return None

    def merge_models(self, model_A, model_B, mask, merge_ratio, blend_mode,
                    curve_power=1.0, preserve_extremes=False, noise_injection=0.0,
                    layer_scaling="uniform"):
        """
        Advanced tensor merge with sophisticated blending and DEVICE SAFETY
        """
        print(f"\n--- Weighted Mask Merge (Advanced) ---")
        print(f"  Blend Mode: {blend_mode}")
        print(f"  Merge Ratio: {merge_ratio}")
        print(f"  Curve Power: {curve_power}")
        print(f"  Layer Scaling: {layer_scaling}")
        
        merged_model = model_A.clone()
        state_dict_A = model_A.model.state_dict()
        state_dict_B = model_B.model.state_dict()
        
        # Handle mask types
        if isinstance(mask, dict) and "mask_dict" in mask:
            mask_dict = mask["mask_dict"]
        else:
            # Convert tensor mask to dict
            mask_dict = {}
            for key in state_dict_A.keys():
                mask_dict[key] = 1.0
        
        # Analyze layer structure
        layer_numbers = {}
        max_layer = 0
        for key in state_dict_A.keys():
            layer_num = self.extract_layer_number(key)
            if layer_num is not None:
                layer_numbers[key] = layer_num
                max_layer = max(max_layer, layer_num)
        
        total_layers = max_layer + 1 if max_layer > 0 else 1
        print(f"  Detected layers: {total_layers}")
        
        patches = {}
        processed_count = 0
        
        for key in state_dict_A.keys():
            if key in state_dict_B and key in mask_dict:
                weight_A = state_dict_A[key]
                weight_B = state_dict_B[key]
                
                if isinstance(weight_A, torch.Tensor) and isinstance(weight_B, torch.Tensor):
                    if weight_A.shape == weight_B.shape:
                        # DEVICE FIX: Ensure same device for all operations
                        device = weight_A.device
                        weight_B = weight_B.to(device)
                        
                        # Get base mask value
                        mask_value = float(mask_dict[key])
                        
                        # Apply blending curve
                        blend_factor = self.apply_blend_curve(mask_value, blend_mode, curve_power)
                        
                        # Apply layer scaling
                        if key in layer_numbers:
                            layer_scale = self.get_layer_scale(
                                layer_numbers[key], total_layers, layer_scaling
                            )
                            blend_factor *= layer_scale
                        
                        # Apply global merge ratio
                        blend_factor *= merge_ratio
                        
                        # Preserve extremes if requested
                        if preserve_extremes:
                            if mask_value < 0.05:
                                blend_factor = 0.0
                            elif mask_value > 0.95:
                                blend_factor = merge_ratio
                        
                        # Clip to valid range
                        blend_factor = np.clip(blend_factor, 0.0, 2.0)
                        
                        # Perform merge - ALL ON SAME DEVICE
                        merged_weight = weight_A * (1.0 - blend_factor) + weight_B * blend_factor
                        
                        # Add noise if requested - ON SAME DEVICE
                        if noise_injection > 0:
                            noise = torch.randn_like(merged_weight, device=device) * noise_injection * merged_weight.abs().mean()
                            merged_weight = merged_weight + noise
                        
                        # Ensure result stays on original device
                        merged_weight = merged_weight.to(device)
                        
                        # Only create patch if there's actual change
                        if blend_factor > 0.001:
                            # DEVICE FIX: Move to CPU for patch storage
                            patches[key] = ((merged_weight - weight_A).cpu(),)
                            processed_count += 1
        
        if patches:
            merged_model.add_patches(patches, 1.0)
        
        print(f"  Processed {processed_count} tensors")
        print(f"--- Weighted Mask Merge completed ---\n")
        
        return (merged_model,)


NODE_CLASS_MAPPINGS = {
    "TensorPrism_WeightedMaskMerge": TensorPrism_WeightedMaskMerge
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TensorPrism_WeightedTensorMerge": "Weighted Tensor Merge (Tensor Prism)"
}