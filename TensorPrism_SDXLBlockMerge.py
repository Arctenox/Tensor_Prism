import torch
import copy
import gc
import psutil
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

class SDXLBlockMergeTensorPrism:
    """
    FIXED VERSION: Properly applies merged weights with device safety
    """
    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "model_A": ("MODEL", {}),
                "model_B": ("MODEL", {}),
                "merge_method": (["Linear Interpolation", "Add Difference", "TIES-Merging (Simplified)"],),
                "default_unet_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "memory_limit_gb": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 64.0, "step": 0.5, "round": 0.1}),
            },
            "optional": {
                "model_C": ("MODEL", {}),
                "ties_global_alpha_A": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "ties_global_alpha_B": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "rescale_output_magnitudes": ("BOOLEAN", {"default": False}),
                "iterations": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
                "a_delta_factor": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01, "round": 0.001}),
                "b_delta_factor": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01, "round": 0.001}),
                
                "out_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "time_embed_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "label_emb_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
            }
        }

        # Add block-specific ratios
        for i in range(12):
            inputs["optional"][f"input_block_{i:02d}_ratio"] = ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001})
            inputs["optional"][f"output_block_{i:02d}_ratio"] = ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001})
        
        for i in range(3):
            inputs["optional"][f"middle_block_{i:02d}_ratio"] = ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001})

        return inputs

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "process"
    CATEGORY = "Tensor_Prism/Merge"

    def process(self, model_A, model_B, merge_method, default_unet_ratio, memory_limit_gb=8.0,
                model_C=None, **kwargs):
        """
        FIXED: Creates patches with proper device handling
        """
        print(f"\n--- SDXL Block Merge (Tensor Prism - Device Fixed) ---")
        
        # Validate inputs
        if merge_method in ["Add Difference", "TIES-Merging (Simplified)"] and model_C is None:
            raise ValueError(f"Model C is required for '{merge_method}' merge method.")

        # Get state dictionaries  
        unet_sd_A = model_A.model.state_dict()
        unet_sd_B = model_B.model.state_dict()
        unet_sd_C = model_C.model.state_dict() if model_C else None

        # Build ratio mapping
        key_to_ratio_map = self._build_ratio_mapping(unet_sd_A.keys(), default_unet_ratio, kwargs)
        
        # Create patches with device safety
        patches = {}
        
        print("Processing tensor merging with device safety...")
        for key in unet_sd_A.keys():
            if key not in unet_sd_B:
                continue
                
            param_A = unet_sd_A[key]
            ratio = key_to_ratio_map.get(key, default_unet_ratio)
            
            # Skip if ratio is 0
            if ratio == 0.0:
                continue
            
            # DEVICE FIX: Ensure tensors on same device
            device = param_A.device
            param_B = unet_sd_B[key].to(device, param_A.dtype)
                
            merged_param = None
            
            try:
                if merge_method == "Linear Interpolation":
                    merged_param = param_A * (1.0 - ratio) + param_B * ratio
                    
                elif merge_method == "Add Difference":
                    if unet_sd_C and key in unet_sd_C:
                        param_C = unet_sd_C[key].to(device, param_A.dtype)
                        delta_A = (param_A - param_C) * ratio * kwargs.get('a_delta_factor', 1.0)
                        delta_B = (param_B - param_C) * (1.0 - ratio) * kwargs.get('b_delta_factor', 1.0)
                        merged_param = param_C + delta_A + delta_B
                    else:
                        merged_param = param_A * (1.0 - ratio) + param_B * ratio
                        
                elif merge_method == "TIES-Merging (Simplified)":
                    if unet_sd_C and key in unet_sd_C:
                        param_C = unet_sd_C[key].to(device, param_A.dtype)
                        alpha_A = kwargs.get('ties_global_alpha_A', 0.5) * ratio
                        alpha_B = kwargs.get('ties_global_alpha_B', 0.5) * (1.0 - ratio)
                        
                        if kwargs.get('rescale_output_magnitudes', False):
                            total = alpha_A + alpha_B
                            if total > 0:
                                alpha_A /= total
                                alpha_B /= total
                        
                        delta_A = (param_A - param_C) * alpha_A * kwargs.get('a_delta_factor', 1.0)
                        delta_B = (param_B - param_C) * alpha_B * kwargs.get('b_delta_factor', 1.0)
                        merged_param = param_C + delta_A + delta_B
                    else:
                        merged_param = param_A * (1.0 - ratio) + param_B * ratio
                
                if merged_param is not None:
                    # Ensure result on correct device
                    merged_param = merged_param.to(device)
                    
                    # Store as patch (difference from original)
                    patch_diff = merged_param - param_A
                    if not torch.allclose(patch_diff, torch.zeros_like(patch_diff), atol=1e-8):
                        patches[key] = (patch_diff.cpu(),)  # Store patches on CPU
                        
            except Exception as e:
                print(f"Warning: Failed to process {key}: {e}")
                continue
        
        print(f"Created {len(patches)} patches")
        
        # Clone model and apply patches
        merged_model = model_A.clone()
        if patches:
            merged_model.add_patches(patches, 1.0)
        
        print("--- SDXL Block Merge completed ---\n")
        return (merged_model,)
    
    def _build_ratio_mapping(self, keys, default_ratio, kwargs):
        """Build mapping of parameter keys to their merge ratios"""
        ratio_prefixes = {
            "time_embed.": kwargs.get("time_embed_ratio", default_ratio),
            "label_emb.": kwargs.get("label_emb_ratio", default_ratio), 
            "out.": kwargs.get("out_ratio", default_ratio),
        }
        
        # Add block-specific prefixes
        for i in range(12):
            ratio_prefixes[f"input_blocks.{i}."] = kwargs.get(f"input_block_{i:02d}_ratio", default_ratio)
            ratio_prefixes[f"output_blocks.{i}."] = kwargs.get(f"output_block_{i:02d}_ratio", default_ratio)
        
        # Middle blocks
        ratio_prefixes["middle_block.0."] = kwargs.get("middle_block_00_ratio", default_ratio)
        ratio_prefixes["middle_block.1."] = kwargs.get("middle_block_01_ratio", default_ratio)
        ratio_prefixes["middle_block.2."] = kwargs.get("middle_block_02_ratio", default_ratio)
        
        # Sort by length (longest first)
        sorted_prefixes = sorted(ratio_prefixes.items(), key=lambda x: len(x[0]), reverse=True)
        
        key_to_ratio = {}
        for key in keys:
            ratio = default_ratio
            for prefix, r in sorted_prefixes:
                if key.startswith(prefix):
                    ratio = r
                    break
            key_to_ratio[key] = ratio
            
        return key_to_ratio


NODE_CLASS_MAPPINGS = {
    "SDXL Block Merge (Tensor Prism)": SDXLBlockMergeTensorPrism
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDXL Block Merge (Tensor Prism)": "SDXL Block Merge (Tensor Prism)"
}