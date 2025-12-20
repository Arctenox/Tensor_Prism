import copy
import gc
import psutil
from typing import Dict, List, Tuple
import numpy as np
import torch

MODEL_MASK_TYPE = ("MASK",)

class TensorPrism_ModelMaskBlender:
    """
    Memory-efficient model mask blender that processes ComfyUI mask tensors
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask_A": ("MASK",),
                "mask_B": ("MASK",),
                "blend_mode": (["Add", "Multiply", "Max", "Min", "Linear Blend", "Exponential Blend"], {"default": "Linear Blend"}),
                "memory_limit_gb": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 16.0, "step": 0.1, "round": 0.1, "label": "Memory Limit (GB)"}),
            },
            "optional": {
                "blend_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001, "label": "Blend Strength (for Linear/Exp)"}),
                "clip_output": ("BOOLEAN", {"default": True, "label_on": "Clip to [0, 1]", "label_off": "No Clipping"}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("combined_mask",)
    FUNCTION = "blend_masks"
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
        
        for i, key in enumerate(all_keys):
            current_batch.append(key)
            
            if len(current_batch) >= max_keys_per_batch:
                batches.append(current_batch)
                current_batch = []
        
        # Add final batch if not empty
        if current_batch:
            batches.append(current_batch)
        
        return batches

    def process_mask_batch(self, batch_keys: List[str], mask_A_dict: Dict[str, float], 
                          mask_B_dict: Dict[str, float], blend_mode: str, 
                          blend_strength: float, clip_output: bool) -> Dict[str, float]:
        """Process a batch of mask keys with the specified blending operation"""
        batch_results = {}
        
        for key in batch_keys:
            val_A = mask_A_dict.get(key, 0.0)
            val_B = mask_B_dict.get(key, 0.0)

            result_val = 0.0
            if blend_mode == "Add":
                result_val = val_A + val_B
            elif blend_mode == "Multiply":
                result_val = val_A * val_B
            elif blend_mode == "Max":
                result_val = max(val_A, val_B)
            elif blend_mode == "Min":
                result_val = min(val_A, val_B)
            elif blend_mode == "Linear Blend":
                result_val = val_A * (1.0 - blend_strength) + val_B * blend_strength
            elif blend_mode == "Exponential Blend":
                exp_strength = blend_strength ** 2 
                result_val = val_A * (1.0 - exp_strength) + val_B * exp_strength
            
            if clip_output:
                result_val = max(0.0, min(1.0, result_val))
            
            batch_results[key] = result_val
        
        return batch_results

    def blend_masks(self, mask_A, mask_B, blend_mode, memory_limit_gb=2.0, 
                   blend_strength=0.5, clip_output=True):
        
        print(f"\n--- Model Mask Blender (Tensor Prism) ---")
        print(f"  Blend Mode: {blend_mode}")
        print(f"  Blend Strength: {blend_strength}")
        print(f"  Memory Limit: {memory_limit_gb:.1f}GB")
        
        # Get initial memory info
        used_memory, available_memory = self.get_memory_info()
        print(f"  System Memory - Used: {used_memory:.2f}GB, Available: {available_memory:.2f}GB")

        # Convert tensors to numpy for processing
        if isinstance(mask_A, torch.Tensor):
            mask_A_np = mask_A.cpu().numpy()
        else:
            mask_A_np = np.array(mask_A)
            
        if isinstance(mask_B, torch.Tensor):
            mask_B_np = mask_B.cpu().numpy()
        else:
            mask_B_np = np.array(mask_B)

        print(f"  Mask A shape: {mask_A_np.shape}")
        print(f"  Mask B shape: {mask_B_np.shape}")

        # Ensure masks have the same shape
        if mask_A_np.shape != mask_B_np.shape:
            # Resize mask_B to match mask_A
            from scipy import ndimage
            if len(mask_A_np.shape) == 3 and len(mask_B_np.shape) == 3:
                mask_B_np = ndimage.zoom(mask_B_np, 
                    (mask_A_np.shape[0]/mask_B_np.shape[0],
                     mask_A_np.shape[1]/mask_B_np.shape[1], 
                     mask_A_np.shape[2]/mask_B_np.shape[2]))
            elif len(mask_A_np.shape) == 2 and len(mask_B_np.shape) == 2:
                mask_B_np = ndimage.zoom(mask_B_np,
                    (mask_A_np.shape[0]/mask_B_np.shape[0],
                     mask_A_np.shape[1]/mask_B_np.shape[1]))

        # Process masks based on blend mode
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

        # Final memory cleanup
        gc.collect()

        final_memory, _ = self.get_memory_info()
        print(f"  Final memory usage: {final_memory:.2f}GB")
        print(f"  Result mask shape: {result_tensor.shape}")
        print(f"--- Model Mask Blender completed ---\n")
        
        return (result_tensor,)

NODE_CLASS_MAPPINGS = {
    "TensorPrism_ModelMaskBlender": TensorPrism_ModelMaskBlender,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TensorPrism_ModelMaskBlender": "Mask Blender (Tensor Prism)",
}