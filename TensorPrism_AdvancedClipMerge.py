import torch
import logging
from typing import Dict, List, Tuple, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TensorPrismAdvancedClipMerge:
    """
    Advanced CLIP merge node for Tensor Prism.
    Merges CLIP-L and CLIP-G text encoders from two models with fine-grained control.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict:
        """Defines input types for CLIP merging."""
        inputs = {
            "required": {
                "clip_a": ("CLIP", {}),
                "clip_b": ("CLIP", {}),
                "merge_method": (["Linear Interpolation", "Add Difference", "Weighted Average"],),
                "default_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "clip_c": ("CLIP", {}),  # For Add Difference method
                
                # Global encoder ratios
                "clip_l_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "clip_g_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                
                # Fine-grained control
                "text_projection_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "positional_embedding_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "logit_scale_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                

                
                # Additional merge parameters
                "normalize_weights": ("BOOLEAN", {"default": False}),
                "delta_factor_a": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "delta_factor_b": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01}),
            }
        }
        
        return inputs

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("merged_clip",)
    FUNCTION = "merge_clips"
    CATEGORY = "Tensor_Prism/CLIP"

    def __init__(self):
        """Initialize the CLIP merge node."""
        pass

    def _identify_clip_type(self, key: str) -> Optional[str]:
        """Identify if a key belongs to CLIP-L or CLIP-G."""
        key_lower = key.lower()
        
        # CLIP-L patterns (OpenAI CLIP ViT-L/14 - 12 layers)
        if any(pattern in key_lower for pattern in ['clip_l', 'cond_stage_model.transformer', 'conditioner.embedders.0']):
            return 'clip_l'
        
        # CLIP-G patterns (OpenCLIP ViT-bigG - 32 layers) 
        if any(pattern in key_lower for pattern in ['clip_g', 'conditioner.embedders.1']):
            return 'clip_g'
        
        return None

    def _build_ratio_mapping(self, keys: List[str], default_ratio: float, kwargs: Dict) -> Dict[str, float]:
        """Build ratio mapping for CLIP parameters with encoder-level control."""
        key_to_ratio = {}
        
        for key in keys:
            # Start with default ratio
            ratio = default_ratio
            
            # Check for specific component ratios
            if 'text_projection' in key.lower():
                ratio = kwargs.get('text_projection_ratio', default_ratio)
            elif 'positional_embedding' in key.lower() or 'position_embedding' in key.lower():
                ratio = kwargs.get('positional_embedding_ratio', default_ratio)
            elif 'logit_scale' in key.lower():
                ratio = kwargs.get('logit_scale_ratio', default_ratio)
            else:
                # Identify CLIP type
                clip_type = self._identify_clip_type(key)
                
                if clip_type == 'clip_l':
                    ratio = kwargs.get('clip_l_ratio', default_ratio)
                elif clip_type == 'clip_g':
                    ratio = kwargs.get('clip_g_ratio', default_ratio)
            
            # Clamp ratio to valid range
            ratio = max(0.0, min(1.0, ratio))
            key_to_ratio[key] = ratio
        
        return key_to_ratio

    def _linear_interpolation(self, param_a: torch.Tensor, param_b: torch.Tensor, 
                            ratio: float) -> torch.Tensor:
        """Linear interpolation between two parameters."""
        if ratio == 0.0:
            return param_a.clone()
        elif ratio == 1.0:
            return param_b.clone()
        else:
            return param_a * (1.0 - ratio) + param_b * ratio

    def _add_difference(self, param_a: torch.Tensor, param_b: torch.Tensor, 
                       param_c: torch.Tensor, ratio: float,
                       delta_factor_a: float, delta_factor_b: float) -> torch.Tensor:
        """Add difference merge method."""
        # Calculate deltas from base model C
        delta_a = (param_a - param_c) * delta_factor_a
        delta_b = (param_b - param_c) * delta_factor_b
        
        # Weighted combination
        combined_delta = delta_a * (1.0 - ratio) + delta_b * ratio
        
        return param_c + combined_delta

    def _weighted_average(self, param_a: torch.Tensor, param_b: torch.Tensor,
                         ratio: float, normalize: bool) -> torch.Tensor:
        """Weighted average with optional normalization."""
        weight_a = 1.0 - ratio
        weight_b = ratio
        
        if normalize:
            total_weight = weight_a + weight_b
            if total_weight > 1e-8:
                weight_a /= total_weight
                weight_b /= total_weight
        
        return param_a * weight_a + param_b * weight_b

    def merge_clips(self, clip_a, clip_b, merge_method: str, default_ratio: float,
                   clip_c=None, normalize_weights: bool = False,
                   delta_factor_a: float = 1.0, delta_factor_b: float = 1.0,
                   **kwargs) -> Tuple:
        """
        Main function to merge CLIP models.
        """
        try:
            logger.info("=== Tensor Prism Advanced CLIP Merge ===")
            logger.info(f"Merge method: {merge_method}")
            logger.info(f"Default ratio: {default_ratio}")
            
            # Validate inputs
            if clip_a is None or clip_b is None:
                raise ValueError("Both CLIP A and CLIP B are required")
            
            if merge_method == "Add Difference" and clip_c is None:
                raise ValueError("CLIP C is required for Add Difference merge method")
            
            # Clone CLIP A as the base
            merged_clip = clip_a.clone()
            
            # Get state dictionaries from CLIP models
            try:
                # Access the CLIP model's parameters through patcher
                state_dict_a = clip_a.get_sd()
                state_dict_b = clip_b.get_sd()
                state_dict_c = clip_c.get_sd() if clip_c else None
            except Exception as e:
                raise RuntimeError(f"Failed to extract CLIP state dictionaries: {e}")
            
            # Find common parameters
            common_keys = set(state_dict_a.keys()) & set(state_dict_b.keys())
            if state_dict_c:
                common_keys = common_keys & set(state_dict_c.keys())
            
            common_keys = list(common_keys)
            
            if not common_keys:
                logger.warning("No common parameters found between CLIPs")
                return (clip_a,)
            
            logger.info(f"Processing {len(common_keys)} common parameters")
            
            # Build ratio mapping
            key_to_ratio_map = self._build_ratio_mapping(common_keys, default_ratio, kwargs)
            
            # Create patches dictionary
            patches = {}
            processed_count = 0
            
            for key in common_keys:
                try:
                    param_a = state_dict_a[key]
                    param_b = state_dict_b[key]
                    
                    # Skip non-tensor parameters
                    if not isinstance(param_a, torch.Tensor) or not isinstance(param_b, torch.Tensor):
                        continue
                    
                    # Shape validation
                    if param_a.shape != param_b.shape:
                        logger.warning(f"Shape mismatch for {key}, skipping")
                        continue
                    
                    # Get merge ratio for this parameter
                    ratio = key_to_ratio_map.get(key, default_ratio)
                    
                    # Skip if ratio is 0 (no merge needed)
                    if ratio == 0.0:
                        continue
                    
                    # Perform merge based on method
                    if merge_method == "Linear Interpolation":
                        merged_param = self._linear_interpolation(param_a, param_b, ratio)
                    
                    elif merge_method == "Add Difference":
                        param_c = state_dict_c[key]
                        if param_c.shape != param_a.shape:
                            logger.warning(f"Shape mismatch with CLIP C for {key}, using linear interpolation")
                            merged_param = self._linear_interpolation(param_a, param_b, ratio)
                        else:
                            merged_param = self._add_difference(param_a, param_b, param_c, ratio,
                                                               delta_factor_a, delta_factor_b)
                    
                    elif merge_method == "Weighted Average":
                        merged_param = self._weighted_average(param_a, param_b, ratio, normalize_weights)
                    
                    else:
                        logger.warning(f"Unknown merge method: {merge_method}, using linear interpolation")
                        merged_param = self._linear_interpolation(param_a, param_b, ratio)
                    
                    # Calculate patch (difference from original)
                    patch_diff = merged_param - param_a
                    
                    # Only add patch if there's a significant change
                    if not torch.allclose(patch_diff, torch.zeros_like(patch_diff), atol=1e-8):
                        patches[key] = (patch_diff,)
                        processed_count += 1
                
                except Exception as e:
                    logger.warning(f"Failed to process parameter {key}: {e}")
                    continue
            
            logger.info(f"Generated {processed_count} patches")
            
            # Apply patches to merged CLIP
            if patches:
                try:
                    merged_clip.add_patches(patches, 1.0)
                    logger.info("Patches applied successfully")
                except Exception as e:
                    logger.error(f"Failed to apply patches: {e}")
                    return (clip_a,)
            else:
                logger.warning("No patches generated, returning original CLIP A")
            
            logger.info("=== CLIP Merge Completed Successfully ===")
            return (merged_clip,)
        
        except Exception as e:
            logger.error(f"CLIP merge failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return (clip_a,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "TensorPrismAdvancedClipMerge": TensorPrismAdvancedClipMerge
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TensorPrismAdvancedClipMerge": "Advanced CLIP Merge (Tensor Prism)"
}
