"""
TensorPrism Masking System
===========================

Comprehensive masking system for selective model merging.
Includes mask generation, filtering, blending, and application.

Author: Arctenox
Version: 1.0.0
License: GPL-3.0
"""

import torch
import numpy as np
import re
import gc
import psutil
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# ==================== UTILITY FUNCTIONS ====================

def is_unet_key(key: str) -> bool:
    """Check if key belongs to UNet"""
    key_lower = key.lower()
    return 'unet' in key_lower or 'model.diffusion_model' in key_lower

def is_vae_key(key: str) -> bool:
    """Check if key belongs to VAE"""
    key_lower = key.lower()
    return 'vae' in key_lower or 'autoencoder' in key_lower or 'first_stage_model' in key_lower

def is_text_encoder_key(key: str) -> bool:
    """Check if key belongs to text encoder"""
    key_lower = key.lower()
    return 'clip' in key_lower or 'text_encoder' in key_lower or 'cond_stage' in key_lower

def get_unet_component_type(param_name: str) -> str:
    """Identify UNet component type"""
    param_lower = param_name.lower()
    if 'time_embed' in param_lower:
        return 'time_embed'
    elif 'input_blocks' in param_lower:
        return 'input_blocks'
    elif 'middle_block' in param_lower:
        return 'middle_block'
    elif 'output_blocks' in param_lower:
        return 'output_blocks'
    elif param_lower.endswith('.out.weight') or param_lower.endswith('.out.bias'):
        return 'out'
    return 'other_unet'

def get_memory_info() -> Tuple[float, float]:
    """Get current memory usage and available memory in GB"""
    memory = psutil.virtual_memory()
    used_gb = (memory.total - memory.available) / (1024**3)
    available_gb = memory.available / (1024**3)
    return used_gb, available_gb

def estimate_dict_memory_gb(dict_size: int) -> float:
    """Estimate memory usage of a dictionary with float values in GB"""
    bytes_per_entry = 100
    return (dict_size * bytes_per_entry) / (1024**3)


# ==================== MASK GENERATOR ====================

class TensorPrism_ModelMaskGenerator:
    """Generate masks with various strategies"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_type": ([
                    "layer_based",
                    "block_based", 
                    "attention_only",
                    "feedforward_only",
                    "custom_pattern",
                    "random_sparse",
                    "depth_gradient"
                ],),
                "intensity": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05
                }),
                "reference_model": ("MODEL",),
            },
            "optional": {
                "layer_start": ("INT", {
                    "default": 0, "min": 0, "max": 50, "step": 1
                }),
                "layer_end": ("INT", {
                    "default": -1, "min": -1, "max": 50, "step": 1
                }),
                "gradient_direction": ([
                    "shallow_to_deep",
                    "deep_to_shallow",
                    "center_out",
                    "edges_in"
                ],),
                "sparsity": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05
                }),
                "custom_pattern": ("STRING", {
                    "default": "attn,mlp.fc1"
                }),
                "falloff": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05
                }),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "generate_mask"
    CATEGORY = "Tensor_Prism/Mask"

    def analyze_model_structure(self, state_dict):
        """Analyze model structure"""
        layer_info = {
            "layers": {},
            "total_layers": 0,
            "layer_names": list(state_dict.keys())
        }
        
        layer_patterns = [
            r"layers\.(\d+)\.",
            r"blocks\.(\d+)\.",
            r"h\.(\d+)\.",
            r"layer\.(\d+)\.",
            r"encoder\.layer\.(\d+)\.",
            r"decoder\.layer\.(\d+)\.",
        ]
        
        for name in state_dict.keys():
            layer_num = None
            for pattern in layer_patterns:
                match = re.search(pattern, name)
                if match:
                    layer_num = int(match.group(1))
                    break
            
            if layer_num is not None:
                if layer_num not in layer_info["layers"]:
                    layer_info["layers"][layer_num] = []
                layer_info["layers"][layer_num].append(name)
                layer_info["total_layers"] = max(layer_info["total_layers"], layer_num + 1)
        
        return layer_info
    
    def get_layer_number(self, param_name):
        """Extract layer number"""
        layer_patterns = [
            r"layers\.(\d+)\.",
            r"blocks\.(\d+)\.",
            r"h\.(\d+)\.",
            r"layer\.(\d+)\.",
            r"encoder\.layer\.(\d+)\.",
            r"decoder\.layer\.(\d+)\.",
        ]
        
        for pattern in layer_patterns:
            match = re.search(pattern, param_name)
            if match:
                return int(match.group(1))
        return None

    def create_layer_range_mask(self, layer_info, start, end, intensity):
        """Create mask for layer range"""
        mask = {}
        if end == -1:
            end = layer_info["total_layers"]
        
        for name in layer_info["layer_names"]:
            layer_num = self.get_layer_number(name)
            if layer_num is not None and start <= layer_num < end:
                mask[name] = intensity
            else:
                mask[name] = 0.0
        return mask
    
    def create_block_mask(self, layer_info, start, end, intensity):
        """Create mask with smooth transitions"""
        mask = {}
        if end == -1:
            end = layer_info["total_layers"]
        
        total_range = max(end - start, 1)
        
        for name in layer_info["layer_names"]:
            layer_num = self.get_layer_number(name)
            if layer_num is not None and start <= layer_num < end:
                progress = (layer_num - start) / total_range
                mask_value = intensity * (0.5 + 0.5 * np.cos(progress * np.pi))
                mask[name] = mask_value
            else:
                mask[name] = 0.0
        return mask
    
    def create_component_mask(self, layer_info, component_patterns, intensity):
        """Create mask for specific components"""
        mask = {}
        for name in layer_info["layer_names"]:
            should_mask = any(pattern.lower() in name.lower() 
                            for pattern in component_patterns)
            mask[name] = intensity if should_mask else 0.0
        return mask
    
    def create_pattern_mask(self, layer_info, patterns, intensity):
        """Create mask based on custom patterns"""
        mask = {}
        for name in layer_info["layer_names"]:
            should_mask = any(pattern.lower() in name.lower() 
                            for pattern in patterns)
            mask[name] = intensity if should_mask else 0.0
        return mask
    
    def create_random_mask(self, layer_info, sparsity, intensity):
        """Create random sparse mask"""
        mask = {}
        np.random.seed(42)
        for name in layer_info["layer_names"]:
            mask[name] = intensity if np.random.random() > sparsity else 0.0
        return mask
    
    def create_depth_gradient_mask(self, layer_info, direction, intensity, falloff):
        """Create gradient mask based on depth"""
        mask = {}
        total_layers = max(layer_info["total_layers"], 1)
        
        for name in layer_info["layer_names"]:
            layer_num = self.get_layer_number(name)
            
            if layer_num is not None:
                position = layer_num / (total_layers - 1) if total_layers > 1 else 0.5
                
                if direction == "shallow_to_deep":
                    mask_value = position
                elif direction == "deep_to_shallow":
                    mask_value = 1.0 - position
                elif direction == "center_out":
                    mask_value = 1.0 - 2.0 * abs(position - 0.5)
                else:  # edges_in
                    mask_value = 2.0 * abs(position - 0.5)
                
                if falloff > 0:
                    mask_value = np.power(mask_value, 1.0 / max(falloff, 0.01))
                
                mask[name] = mask_value * intensity
            else:
                mask[name] = intensity * 0.1
        return mask
    
    def create_uniform_mask(self, layer_info, intensity):
        """Create uniform mask"""
        return {name: intensity for name in layer_info["layer_names"]}

    def generate_mask(self, mask_type, intensity, reference_model,
                     layer_start=0, layer_end=-1, gradient_direction="shallow_to_deep",
                     sparsity=0.5, custom_pattern="attn,mlp", falloff=0.1):
        """Generate mask with model structure"""
        
        state_dict = reference_model.model.state_dict()
        layer_info = self.analyze_model_structure(state_dict)
        
        if mask_type == "layer_based":
            mask = self.create_layer_range_mask(layer_info, layer_start, layer_end, intensity)
        elif mask_type == "block_based":
            mask = self.create_block_mask(layer_info, layer_start, layer_end, intensity)
        elif mask_type == "attention_only":
            mask = self.create_component_mask(layer_info, ["attn", "attention", "self_attn"], intensity)
        elif mask_type == "feedforward_only":
            mask = self.create_component_mask(layer_info, ["mlp", "fc", "feedforward", "ffn"], intensity)
        elif mask_type == "custom_pattern":
            patterns = [p.strip() for p in custom_pattern.split(",")]
            mask = self.create_pattern_mask(layer_info, patterns, intensity)
        elif mask_type == "random_sparse":
            mask = self.create_random_mask(layer_info, sparsity, intensity)
        elif mask_type == "depth_gradient":
            mask = self.create_depth_gradient_mask(layer_info, gradient_direction, intensity, falloff)
        else:
            mask = self.create_uniform_mask(layer_info, intensity)
        
        return ({
            "mask_dict": mask,
            "mask_type": mask_type,
            "intensity": intensity,
            "layer_info": layer_info
        },)


# ==================== KEY FILTER ====================

class TensorPrism_ModelKeyFilter:
    """Memory-efficient model key filter"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "filter_mode": (["Include", "Exclude"], {"default": "Include"}),
                "target_components": ([
                    "All", "UNet", "VAE", "Text Encoders",
                    "Time Embeddings", "Input Blocks", "Middle Block",
                    "Output Blocks", "Final UNet Output Layer", "Custom Pattern"
                ], {"default": "UNet"}),
                "default_value": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "round": 0.001
                }),
                "target_value": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "round": 0.001
                }),
                "memory_limit_gb": ("FLOAT", {
                    "default": 2.0, "min": 0.5, "max": 16.0, "step": 0.1,
                    "round": 0.1
                }),
            },
            "optional": {
                "custom_pattern": ("STRING", {
                    "default": "attn,resnets",
                    "multiline": True
                }),
                "exact_match_custom": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("filtered_mask",)
    FUNCTION = "filter_keys_to_mask"
    CATEGORY = "Tensor_Prism/Mask"

    def create_key_batches(self, all_keys: List[str], memory_limit_gb: float) -> List[List[str]]:
        """Create batches of keys within memory limit"""
        batches = []
        current_batch = []
        max_keys_per_batch = max(1000, int((memory_limit_gb * 1024**3) / 200))
        
        for key in all_keys:
            current_batch.append(key)
            if len(current_batch) >= max_keys_per_batch:
                batches.append(current_batch)
                current_batch = []
        
        if current_batch:
            batches.append(current_batch)
        
        return batches

    def check_key_match(self, key_name: str, target_components: str,
                       patterns: List[str], exact_match_custom: bool) -> bool:
        """Check if key matches target criteria"""
        if target_components == "All":
            return True
        elif target_components == "UNet":
            return is_unet_key(key_name) and not is_vae_key(key_name) and not is_text_encoder_key(key_name)
        elif target_components == "VAE":
            return is_vae_key(key_name)
        elif target_components == "Text Encoders":
            return is_text_encoder_key(key_name)
        elif target_components == "Time Embeddings":
            return get_unet_component_type(key_name) == 'time_embed'
        elif target_components == "Input Blocks":
            return get_unet_component_type(key_name) == 'input_blocks'
        elif target_components == "Middle Block":
            return get_unet_component_type(key_name) == 'middle_block'
        elif target_components == "Output Blocks":
            return get_unet_component_type(key_name) == 'output_blocks'
        elif target_components == "Final UNet Output Layer":
            return get_unet_component_type(key_name) == 'out'
        elif target_components == "Custom Pattern":
            key_lower = key_name.lower()
            for pattern in patterns:
                if exact_match_custom:
                    if key_lower == pattern.lower():
                        return True
                else:
                    if pattern.lower() in key_lower:
                        return True
        return False

    def process_key_batch(self, batch_keys: List[str], target_components: str,
                         filter_mode: str, default_value: float, target_value: float,
                         patterns: List[str], exact_match_custom: bool) -> Dict[str, float]:
        """Process batch of keys"""
        batch_results = {}
        
        for key in batch_keys:
            is_match = self.check_key_match(key, target_components, patterns, exact_match_custom)
            
            if filter_mode == "Include":
                batch_results[key] = target_value if is_match else default_value
            elif filter_mode == "Exclude":
                batch_results[key] = default_value if is_match else target_value
        
        return batch_results

    def filter_keys_to_mask(self, model, filter_mode, target_components,
                           default_value, target_value, memory_limit_gb=2.0,
                           custom_pattern="", exact_match_custom=False):
        
        print(f"\n--- Model Key Filter (Tensor Prism) ---")
        print(f"  Filter Mode: {filter_mode}")
        print(f"  Target: {target_components}")
        
        used_memory, available_memory = get_memory_info()
        print(f"  Memory - Used: {used_memory:.2f}GB, Available: {available_memory:.2f}GB")

        state_dict = model.model.state_dict()
        all_keys = list(state_dict.keys())
        print(f"  Total keys: {len(all_keys)}")

        patterns = [p.strip() for p in custom_pattern.split(',') if p.strip()]
        batches = self.create_key_batches(all_keys, memory_limit_gb)
        print(f"  Created {len(batches)} batches")

        mask_dict = {}
        processed_keys = 0

        for i, batch_keys in enumerate(batches):
            batch_results = self.process_key_batch(
                batch_keys, target_components, filter_mode,
                default_value, target_value, patterns, exact_match_custom
            )
            mask_dict.update(batch_results)
            processed_keys += len(batch_keys)
            
            gc.collect()
            
            if i % 5 == 0:
                progress = (processed_keys / len(all_keys)) * 100
                print(f"    Progress: {progress:.1f}%")

        # Analyze structure
        layer_info = {"layer_names": all_keys, "total_layers": 0, "layers": {}}

        mask = {
            "mask_dict": mask_dict,
            "mask_type": f"filtered_by_{target_components}_{filter_mode}",
            "intensity": float(np.mean(list(mask_dict.values()))) if mask_dict else 0.0,
            "layer_info": layer_info
        }

        final_memory, _ = get_memory_info()
        print(f"  Final memory: {final_memory:.2f}GB")
        print(f"  Mask intensity: {mask['intensity']:.4f}")
        print(f"--- Filter completed ---\n")
        
        return (mask,)


# ==================== MASK BLENDER ====================

class TensorPrism_ModelMaskBlender:
    """Blend two masks together"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_A": ("MASK",),
                "mask_B": ("MASK",),
                "blend_mode": ([
                    "Add", "Multiply", "Max", "Min",
                    "Linear Blend", "Exponential Blend"
                ], {"default": "Linear Blend"}),
                "memory_limit_gb": ("FLOAT", {
                    "default": 2.0, "min": 0.5, "max": 16.0, "step": 0.1
                }),
            },
            "optional": {
                "blend_strength": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01
                }),
                "clip_output": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("combined_mask",)
    FUNCTION = "blend_masks"
    CATEGORY = "Tensor_Prism/Mask"

"""
FIXED blend_masks function for TensorPrism_ModelMaskBlender
Replace lines 487-543 in TensorPrism_MaskSystem.py with this function
"""

def blend_masks(self, mask_A, mask_B, blend_mode, memory_limit_gb=2.0,
               blend_strength=0.5, clip_output=True):
    
    print(f"\n--- Mask Blender ---")
    print(f"  Mode: {blend_mode}, Strength: {blend_strength}")
    
    # Check if masks are dictionaries (from Model Mask Generator)
    is_dict_mask_A = isinstance(mask_A, dict) and "mask_dict" in mask_A
    is_dict_mask_B = isinstance(mask_B, dict) and "mask_dict" in mask_B
    
    # Handle dictionary masks (for model merging)
    if is_dict_mask_A and is_dict_mask_B:
        print("  Processing dictionary masks (model weights)")
        
        mask_dict_A = mask_A["mask_dict"]
        mask_dict_B = mask_B["mask_dict"]
        
        # Get all keys from both masks
        all_keys = set(mask_dict_A.keys()) | set(mask_dict_B.keys())
        print(f"  Total keys: {len(all_keys)}")
        
        # Create combined mask dictionary
        combined_mask_dict = {}
        
        for key in all_keys:
            # Get values, default to 0.0 if key doesn't exist in one mask
            value_A = mask_dict_A.get(key, 0.0)
            value_B = mask_dict_B.get(key, 0.0)
            
            # Blend based on mode
            if blend_mode == "Add":
                combined_value = value_A + value_B
            elif blend_mode == "Multiply":
                combined_value = value_A * value_B
            elif blend_mode == "Max":
                combined_value = max(value_A, value_B)
            elif blend_mode == "Min":
                combined_value = min(value_A, value_B)
            elif blend_mode == "Linear Blend":
                combined_value = value_A * (1.0 - blend_strength) + value_B * blend_strength
            elif blend_mode == "Exponential Blend":
                exp_strength = blend_strength ** 2
                combined_value = value_A * (1.0 - exp_strength) + value_B * exp_strength
            else:
                combined_value = value_A
            
            # Clip if requested
            if clip_output:
                combined_value = max(0.0, min(1.0, combined_value))
            
            combined_mask_dict[key] = combined_value
        
        # Calculate average intensity
        avg_intensity = float(np.mean(list(combined_mask_dict.values()))) if combined_mask_dict else 0.0
        
        # Create combined mask object with same structure as Model Mask Generator output
        combined_mask = {
            "mask_dict": combined_mask_dict,
            "mask_type": f"blended_{blend_mode}",
            "intensity": avg_intensity,
            "layer_info": mask_A.get("layer_info", {"layer_names": list(all_keys), "total_layers": 0, "layers": {}})
        }
        
        print(f"  Combined intensity: {avg_intensity:.4f}")
        print(f"--- Blender completed (dictionary mode) ---\n")
        
        return (combined_mask,)
    
    # Handle image/tensor masks (original functionality)
    else:
        print("  Processing image/tensor masks")
        
        # Convert to numpy
        if isinstance(mask_A, torch.Tensor):
            mask_A_np = mask_A.cpu().numpy()
        elif isinstance(mask_A, dict):
            raise TypeError("Cannot blend dictionary mask with image mask. Both masks must be the same type.")
        else:
            mask_A_np = np.array(mask_A)
            
        if isinstance(mask_B, torch.Tensor):
            mask_B_np = mask_B.cpu().numpy()
        elif isinstance(mask_B, dict):
            raise TypeError("Cannot blend dictionary mask with image mask. Both masks must be the same type.")
        else:
            mask_B_np = np.array(mask_B)

        # Resize if needed
        if mask_A_np.shape != mask_B_np.shape:
            from scipy import ndimage
            if len(mask_A_np.shape) == 3:
                mask_B_np = ndimage.zoom(mask_B_np,
                    (mask_A_np.shape[0]/mask_B_np.shape[0],
                     mask_A_np.shape[1]/mask_B_np.shape[1],
                     mask_A_np.shape[2]/mask_B_np.shape[2]))
            elif len(mask_A_np.shape) == 2:
                mask_B_np = ndimage.zoom(mask_B_np,
                    (mask_A_np.shape[0]/mask_B_np.shape[0],
                     mask_A_np.shape[1]/mask_B_np.shape[1]))

        # Blend
        if blend_mode == "Add":
            result_mask = mask_A_np + mask_B_np
        elif blend_mode == "Multiply":
            result_mask = mask_A_np * mask_B_np
        elif blend_mode == "Max":
            result_mask = np.maximum(mask_A_np, mask_B_np)
        elif blend_mode == "Min":
            result_mask = np.minimum(mask_A_np, mask_B_np)
        elif blend_mode == "Linear Blend":
            result_mask = mask_A_np * (1.0 - blend_strength) + mask_B_np * blend_strength
        elif blend_mode == "Exponential Blend":
            exp_strength = blend_strength ** 2
            result_mask = mask_A_np * (1.0 - exp_strength) + mask_B_np * exp_strength
        else:
            result_mask = mask_A_np
        
        if clip_output:
            result_mask = np.clip(result_mask, 0.0, 1.0)

        result_tensor = torch.from_numpy(result_mask).float()
        gc.collect()

        print(f"  Result shape: {result_tensor.shape}")
        print(f"--- Blender completed (image mode) ---\n")
        
        return (result_tensor,)


# ==================== WEIGHTED MASK MERGE ====================

class TensorPrism_WeightedMaskMerge:
    """Apply mask to merge two models"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_A": ("MODEL",),
                "model_B": ("MODEL",),
                "mask": ("MASK",),
                "merge_ratio": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("merged_model",)
    FUNCTION = "merge_models"
    CATEGORY = "Tensor_Prism/Mask"

    def merge_models(self, model_A, model_B, mask, merge_ratio):
        """Merge models using mask with device safety"""
        
        merged_model = model_A.clone()
        state_dict_A = model_A.model.state_dict()
        state_dict_B = model_B.model.state_dict()
        
        mask_dict = mask["mask_dict"]
        patches = {}
        
        for key in state_dict_A.keys():
            if key in state_dict_B and key in mask_dict:
                weight_A = state_dict_A[key]
                weight_B = state_dict_B[key]
                
                if isinstance(weight_A, torch.Tensor) and isinstance(weight_B, torch.Tensor):
                    if weight_A.shape == weight_B.shape:
                        device = weight_A.device
                        weight_B = weight_B.to(device)
                        
                        mask_value = mask_dict[key] * merge_ratio
                        merged_weight = weight_A * (1 - mask_value) + weight_B * mask_value
                        merged_weight = merged_weight.to(device)
                        
                        if mask_value > 0:
                            patches[key] = ((merged_weight - weight_A).cpu(),)
        
        if patches:
            merged_model.add_patches(patches, 1.0)
        
        return (merged_model,)


# ==================== ADVANCED WEIGHTED MERGE ====================

class TensorPrism_WeightedMaskMergeAdvanced:
    """Advanced merge with sophisticated blending"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_A": ("MODEL",),
                "model_B": ("MODEL",),
                "mask": ("MASK",),
                "merge_ratio": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01
                }),
                "blend_mode": ([
                    "linear", "sigmoid", "cosine",
                    "exponential", "logarithmic", "smoothstep"
                ], {"default": "linear"}),
            },
            "optional": {
                "curve_power": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1
                }),
                "preserve_extremes": ("BOOLEAN", {"default": False}),
                "noise_injection": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.1, "step": 0.001
                }),
                "layer_scaling": ([
                    "uniform", "depth_progressive",
                    "shallow_bias", "deep_bias"
                ], {"default": "uniform"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("merged_model",)
    FUNCTION = "merge_models"
    CATEGORY = "Tensor_Prism/Mask"

    def apply_blend_curve(self, value, mode, power):
        """Apply blending curves"""
        if mode == "linear":
            return value
        elif mode == "sigmoid":
            return 1.0 / (1.0 + np.exp(-power * (value - 0.5) * 10))
        elif mode == "cosine":
            return (1.0 - np.cos(value * np.pi)) * 0.5
        elif mode == "exponential":
            if value < 0.5:
                return 0.5 * np.power(2.0 * value, power)
            else:
                return 1.0 - 0.5 * np.power(2.0 * (1.0 - value), power)
        elif mode == "logarithmic":
            epsilon = 1e-6
            return np.log(value + epsilon) / np.log(1.0 + epsilon)
        elif mode == "smoothstep":
            value = np.clip(value, 0.0, 1.0)
            return value * value * (3.0 - 2.0 * value)
        return value

    def get_layer_scale(self, layer_num, total_layers, scaling_mode):
        """Calculate layer-specific scaling"""
        if total_layers <= 1:
            return 1.0
            
        position = layer_num / (total_layers - 1)
        
        if scaling_mode == "uniform":
            return 1.0
        elif scaling_mode == "depth_progressive":
            return 0.5 + 0.5 * position
        elif scaling_mode == "shallow_bias":
            return 1.5 - 0.5 * position
        elif scaling_mode == "deep_bias":
            return 0.5 + position
        return 1.0

    def extract_layer_number(self, key):
        """Extract layer number from key"""
        patterns = [
            r"layers\.(\d+)\.", r"blocks\.(\d+)\.", r"h\.(\d+)\.",
            r"layer\.(\d+)\.", r"encoder\.layer\.(\d+)\.",
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
        """Advanced merge with blending"""
        
        print(f"\n--- Advanced Weighted Merge ---")
        print(f"  Mode: {blend_mode}, Ratio: {merge_ratio}")
        
        merged_model = model_A.clone()
        state_dict_A = model_A.model.state_dict()
        state_dict_B = model_B.model.state_dict()
        
        if isinstance(mask, dict) and "mask_dict" in mask:
            mask_dict = mask["mask_dict"]
        else:
            mask_dict = {key: 1.0 for key in state_dict_A.keys()}
        
        # Analyze layers
        layer_numbers = {}
        max_layer = 0
        for key in state_dict_A.keys():
            layer_num = self.extract_layer_number(key)
            if layer_num is not None:
                layer_numbers[key] = layer_num
                max_layer = max(max_layer, layer_num)
        
        total_layers = max_layer + 1 if max_layer > 0 else 1
        
        patches = {}
        processed_count = 0
        
        for key in state_dict_A.keys():
            if key in state_dict_B and key in mask_dict:
                weight_A = state_dict_A[key]
                weight_B = state_dict_B[key]
                
                if isinstance(weight_A, torch.Tensor) and isinstance(weight_B, torch.Tensor):
                    if weight_A.shape == weight_B.shape:
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
                        
                        # Perform merge - all on same device
                        merged_weight = weight_A * (1.0 - blend_factor) + weight_B * blend_factor
                        
                        # Add noise if requested
                        if noise_injection > 0:
                            noise = torch.randn_like(merged_weight, device=device)
                            noise *= noise_injection * merged_weight.abs().mean()
                            merged_weight = merged_weight + noise
                        
                        merged_weight = merged_weight.to(device)
                        
                        # Create patch if there's change
                        if blend_factor > 0.001:
                            patches[key] = ((merged_weight - weight_A).cpu(),)
                            processed_count += 1
        
        if patches:
            merged_model.add_patches(patches, 1.0)
        
        print(f"  Processed {processed_count} tensors")
        print(f"--- Advanced merge completed ---\n")
        
        return (merged_model,)


# ==================== NODE REGISTRATION ====================

NODE_CLASS_MAPPINGS = {
    "TensorPrism_ModelMaskGenerator": TensorPrism_ModelMaskGenerator,
    "TensorPrism_ModelKeyFilter": TensorPrism_ModelKeyFilter,
    "TensorPrism_ModelMaskBlender": TensorPrism_ModelMaskBlender,
    "TensorPrism_WeightedMaskMerge": TensorPrism_WeightedMaskMerge,
    "TensorPrism_WeightedMaskMergeAdvanced": TensorPrism_WeightedMaskMergeAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TensorPrism_ModelMaskGenerator": "Model Mask Generator (Tensor Prism)",
    "TensorPrism_ModelKeyFilter": "Model Key Filter (Tensor Prism)",
    "TensorPrism_ModelMaskBlender": "Mask Blender (Tensor Prism)",
    "TensorPrism_WeightedMaskMerge": "Weighted Mask Merge (Tensor Prism)",
    "TensorPrism_WeightedMaskMergeAdvanced": "Advanced Weighted Mask Merge (Tensor Prism)",
}
