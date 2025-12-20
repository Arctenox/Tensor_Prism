import torch
import numpy as np
import re

class TensorPrism_ModelMaskGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask_type": (["layer_based", "block_based", "attention_only", "feedforward_only", "custom_pattern", "random_sparse", "depth_gradient"],),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "reference_model": ("MODEL",),
            },
            "optional": {
                "layer_start": ("INT", {"default": 0, "min": 0, "max": 50, "step": 1}),
                "layer_end": ("INT", {"default": -1, "min": -1, "max": 50, "step": 1}),
                "gradient_direction": (["shallow_to_deep", "deep_to_shallow", "center_out", "edges_in"],),
                "sparsity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "custom_pattern": ("STRING", {"default": "attn,mlp.fc1"}),
                "falloff": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "generate_mask"
    CATEGORY = "Tensor Prism/Mask"

    def generate_mask(self, mask_type, intensity, reference_model, 
                          layer_start=0, layer_end=-1, gradient_direction="shallow_to_deep",
                          sparsity=0.5, custom_pattern="attn,mlp", falloff=0.1):
        """Generate masks with model structure"""
        
        state_dict = reference_model.model.state_dict()
        layer_info = self._analyze_model_structure(state_dict)
        
        if mask_type == "layer_based":
            mask = self._create_layer_range_mask(layer_info, layer_start, layer_end, intensity)
        elif mask_type == "block_based":
            mask = self._create_block_mask(layer_info, layer_start, layer_end, intensity)
        elif mask_type == "attention_only":
            mask = self._create_component_mask(layer_info, ["attn", "attention", "self_attn"], intensity)
        elif mask_type == "feedforward_only":
            mask = self._create_component_mask(layer_info, ["mlp", "fc", "feedforward", "ffn"], intensity)
        elif mask_type == "custom_pattern":
            patterns = [p.strip() for p in custom_pattern.split(",")]
            mask = self._create_pattern_mask(layer_info, patterns, intensity)
        elif mask_type == "random_sparse":
            mask = self._create_random_mask(layer_info, sparsity, intensity)
        elif mask_type == "depth_gradient":
            mask = self._create_depth_gradient_mask(layer_info, gradient_direction, intensity, falloff)
        else:
            mask = self._create_uniform_mask(layer_info, intensity)
        
        mask = {
            "mask_dict": mask,
            "mask_type": mask_type,
            "intensity": intensity,
            "layer_info": layer_info
        }
        
        return (mask,)
    
    def _analyze_model_structure(self, state_dict):
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
    
    def _create_layer_range_mask(self, layer_info, start, end, intensity):
        """Create mask for layer range"""
        mask = {}
        
        if end == -1:
            end = layer_info["total_layers"]
        
        for name in layer_info["layer_names"]:
            layer_num = self._get_layer_number(name)
            
            if layer_num is not None and start <= layer_num < end:
                mask[name] = intensity
            else:
                mask[name] = 0.0
                
        return mask
    
    def _create_block_mask(self, layer_info, start, end, intensity):
        """Create mask with smooth transitions"""
        mask = {}
        
        if end == -1:
            end = layer_info["total_layers"]
            
        total_range = max(end - start, 1)
        
        for name in layer_info["layer_names"]:
            layer_num = self._get_layer_number(name)
            
            if layer_num is not None and start <= layer_num < end:
                progress = (layer_num - start) / total_range
                mask_value = intensity * (0.5 + 0.5 * np.cos(progress * np.pi))
                mask[name] = mask_value
            else:
                mask[name] = 0.0
                
        return mask
    
    def _create_component_mask(self, layer_info, component_patterns, intensity):
        """Create mask for specific components"""
        mask = {}
        
        for name in layer_info["layer_names"]:
            should_mask = False
            
            for pattern in component_patterns:
                if pattern.lower() in name.lower():
                    should_mask = True
                    break
            
            mask[name] = intensity if should_mask else 0.0
            
        return mask
    
    def _create_pattern_mask(self, layer_info, patterns, intensity):
        """Create mask based on custom patterns"""
        mask = {}
        
        for name in layer_info["layer_names"]:
            should_mask = False
            
            for pattern in patterns:
                if pattern.lower() in name.lower():
                    should_mask = True
                    break
            
            mask[name] = intensity if should_mask else 0.0
            
        return mask
    
    def _create_random_mask(self, layer_info, sparsity, intensity):
        """Create random sparse mask"""
        mask = {}
        np.random.seed(42)
        
        for name in layer_info["layer_names"]:
            if np.random.random() > sparsity:
                mask[name] = intensity
            else:
                mask[name] = 0.0
                
        return mask
    
    def _create_depth_gradient_mask(self, layer_info, direction, intensity, falloff):
        """Create gradient mask based on depth"""
        mask = {}
        total_layers = max(layer_info["total_layers"], 1)
        
        for name in layer_info["layer_names"]:
            layer_num = self._get_layer_number(name)
            
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
    
    def _create_uniform_mask(self, layer_info, intensity):
        """Create uniform mask"""
        mask = {}
        for name in layer_info["layer_names"]:
            mask[name] = intensity
        return mask
    
    def _get_layer_number(self, param_name):
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


class TensorPrism_WeightedMaskMerge:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_A": ("MODEL",),
                "model_B": ("MODEL",),
                "mask": ("MASK",),
                "merge_ratio": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("merged_model",)
    FUNCTION = "merge_models"
    CATEGORY = "Tensor Prism/Mask"

    def merge_models(self, model_A, model_B, mask, merge_ratio):
        """Merge models using masks with device safety"""
        
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
                        # DEVICE FIX: Ensure same device
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


NODE_CLASS_MAPPINGS = {
    "TensorPrism_ModelMaskGenerator": TensorPrism_ModelMaskGenerator,
    "TensorPrism_WeightedMaskMerge": TensorPrism_WeightedMaskMerge
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TensorPrism_ModelMaskGenerator": "Model Mask Generator (Tensor Prism)",
    "TensorPrism_WeightedMaskMerge": "Weighted Mask Merge (Tensor Prism)"
}