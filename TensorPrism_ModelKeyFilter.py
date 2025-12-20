import torch
import copy
import gc
import psutil
import re
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple

def is_unet_key(key: str) -> bool:
    key_lower = key.lower()
    return 'unet' in key_lower or 'model.diffusion_model' in key_lower

def is_vae_key(key: str) -> bool:
    key_lower = key.lower()
    return 'vae' in key_lower or 'autoencoder' in key_lower or 'first_stage_model' in key_lower

def is_text_encoder_key(key: str) -> bool:
    key_lower = key.lower()
    return 'clip' in key_lower or 'text_encoder' in key_lower or 'cond_stage' in key_lower

def _get_unet_component_type(param_name: str) -> str:
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

def _analyze_model_structure(state_dict):
    """Analyze model structure to extract layer information"""
    layer_info = {
        "layers": {},
        "total_layers": 0,
        "layer_names": list(state_dict.keys())
    }
    
    layer_patterns = [
        r"layers\.([0-9]+)",
        r"blocks\.([0-9]+)", 
        r"h\.([0-9]+)",
        r"layer\.([0-9]+)", 
        r"encoder\.layer\.([0-9]+)", 
        r"decoder\.layer\.([0-9]+)",
        r"input_blocks\.([0-9]+)", 
        r"output_blocks\.([0-9]+)"
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

MASK_TYPE = ("MASK",)

class TensorPrism_ModelKeyFilter:
    """
    Memory-efficient model key filter that processes keys in batches
    to handle large models without excessive RAM usage.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "filter_mode": (["Include", "Exclude"], {"default": "Include"}),
                "target_components": (["All", "UNet", "VAE", "Text Encoders", "Time Embeddings", "Input Blocks", "Middle Block", "Output Blocks", "Final UNet Output Layer", "Custom Pattern"], {"default": "UNet"}),
                "default_value": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001, "label": "Default Mask Value (for non-targeted keys)"}),
                "target_value": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001, "label": "Target Mask Value (for filtered keys)"}),
                "memory_limit_gb": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 16.0, "step": 0.1, "round": 0.1, "label": "Memory Limit (GB)"}),
            },
            "optional": {
                "custom_pattern": ("STRING", {"default": "attn,resnets", "multiline": True, "placeholder": "Comma-separated regex patterns or substrings, e.g., 'attn,mlp.fc1'"}),
                "exact_match_custom": ("BOOLEAN", {"default": False, "label_on": "Exact Match for Custom Patterns", "label_off": "Substring Match for Custom Patterns"}),
            }
        }

    RETURN_TYPES = MASK_TYPE
    RETURN_NAMES = ("filtered_mask",)
    FUNCTION = "filter_keys_to_mask"
    CATEGORY = "Tensor_Prism/Mask"

    @staticmethod
    def get_memory_info() -> Tuple[float, float]:
        """Get current memory usage and available memory in GB"""
        memory = psutil.virtual_memory()
        used_gb = (memory.total - memory.available) / (1024**3)
        available_gb = memory.available / (1024**3)
        return used_gb, available_gb

    @staticmethod
    def estimate_dict_memory_gb(dict_size: int) -> float:
        """Estimate memory usage of a dictionary with float values in GB"""
        # Rough estimate: key string + float value + overhead
        bytes_per_entry = 100  # Conservative estimate
        return (dict_size * bytes_per_entry) / (1024**3)

    def create_key_batches(self, all_keys: List[str], memory_limit_gb: float) -> List[List[str]]:
        """Create batches of keys that fit within memory limit"""
        batches = []
        current_batch = []
        
        # Estimate how many keys we can process per batch
        max_keys_per_batch = max(1000, int((memory_limit_gb * 1024**3) / 200))  # Conservative estimate
        
        for key in all_keys:
            current_batch.append(key)
            
            if len(current_batch) >= max_keys_per_batch:
                batches.append(current_batch)
                current_batch = []
        
        # Add final batch if not empty
        if current_batch:
            batches.append(current_batch)
        
        return batches

    def check_key_match(self, key_name: str, target_components: str, 
                       patterns: List[str], exact_match_custom: bool) -> bool:
        """Check if a key matches the target component criteria"""
        if target_components == "All":
            return True
        elif target_components == "UNet":
            return is_unet_key(key_name) and not is_vae_key(key_name) and not is_text_encoder_key(key_name)
        elif target_components == "VAE":
            return is_vae_key(key_name)
        elif target_components == "Text Encoders":
            return is_text_encoder_key(key_name)
        elif target_components == "Time Embeddings":
            return _get_unet_component_type(key_name) == 'time_embed'
        elif target_components == "Input Blocks":
            return _get_unet_component_type(key_name) == 'input_blocks'
        elif target_components == "Middle Block":
            return _get_unet_component_type(key_name) == 'middle_block'
        elif target_components == "Output Blocks":
            return _get_unet_component_type(key_name) == 'output_blocks'
        elif target_components == "Final UNet Output Layer":
            return _get_unet_component_type(key_name) == 'out'
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
        return False

    def process_key_batch(self, batch_keys: List[str], target_components: str, 
                         filter_mode: str, default_value: float, target_value: float,
                         patterns: List[str], exact_match_custom: bool) -> Dict[str, float]:
        """Process a batch of keys to create mask values"""
        batch_results = {}
        
        for key in batch_keys:
            is_match = self.check_key_match(key, target_components, patterns, exact_match_custom)
            
            if filter_mode == "Include":
                batch_results[key] = target_value if is_match else default_value
            elif filter_mode == "Exclude":
                batch_results[key] = default_value if is_match else target_value
        
        return batch_results

    def filter_keys_to_mask(self, model, filter_mode, target_components, default_value, target_value,
                           memory_limit_gb=2.0, custom_pattern="", exact_match_custom=False):
        
        print(f"\n--- Model Key Filter (Tensor Prism) ---")
        print(f"  Filter Mode: {filter_mode}")
        print(f"  Target Components: {target_components}")
        print(f"  Default Value: {default_value}, Target Value: {target_value}")
        print(f"  Memory Limit: {memory_limit_gb:.1f}GB")
        
        # Get initial memory info
        used_memory, available_memory = self.get_memory_info()
        print(f"  System Memory - Used: {used_memory:.2f}GB, Available: {available_memory:.2f}GB")

        # Get state dict and keys
        state_dict = model.model.state_dict()
        all_keys = list(state_dict.keys())
        print(f"  Total model keys to process: {len(all_keys)}")

        # Parse custom patterns
        patterns = [p.strip() for p in custom_pattern.split(',') if p.strip()]
        if target_components == "Custom Pattern" and patterns:
            print(f"  Custom patterns: {patterns}")

        # Create processing batches
        print("  Creating memory-efficient processing batches...")
        batches = self.create_key_batches(all_keys, memory_limit_gb)
        print(f"  Created {len(batches)} processing batches")

        # Process batches
        mask_dict = {}
        total_keys = len(all_keys)
        processed_keys = 0

        for i, batch_keys in enumerate(batches):
            batch_memory = self.estimate_dict_memory_gb(len(batch_keys))
            print(f"  Processing batch {i+1}/{len(batches)} ({len(batch_keys)} keys, ~{batch_memory:.3f}GB)")

            # Process this batch
            batch_results = self.process_key_batch(
                batch_keys, target_components, filter_mode, default_value, target_value,
                patterns, exact_match_custom
            )

            # Add results to mask dict
            mask_dict.update(batch_results)
            processed_keys += len(batch_keys)

            # Force garbage collection after each batch
            del batch_results
            gc.collect()

            # Progress update
            progress = (processed_keys / total_keys) * 100
            used_memory, _ = self.get_memory_info()
            print(f"    Progress: {progress:.1f}% ({processed_keys}/{total_keys}), Memory: {used_memory:.2f}GB")

        # Analyze model structure for layer info
        print("  Analyzing model structure...")
        layer_info = _analyze_model_structure(state_dict)

        # Create the model mask object
        mask = {
            "mask_dict": mask_dict,
            "mask_type": f"filtered_by_{target_components}_{filter_mode}",
            "intensity": float(np.mean(list(mask_dict.values()))) if mask_dict else 0.0,
            "layer_info": layer_info
        }

        # Final memory cleanup
        del mask_dict
        gc.collect()

        final_memory, _ = self.get_memory_info()
        print(f"  Final memory usage: {final_memory:.2f}GB")
        print(f"  Mask intensity: {mask['intensity']:.4f}")
        print(f"  Found {layer_info['total_layers']} model layers")
        print(f"--- Model Key Filter completed ---\n")
        
        return (mask,)

NODE_CLASS_MAPPINGS = {
    "TensorPrism_ModelKeyFilter": TensorPrism_ModelKeyFilter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TensorPrism_ModelKeyFilter": "Model Key Filter (Tensor Prism)",
}