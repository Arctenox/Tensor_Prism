"""
TensorPrism SDXL Block Merging
===============================

Unified SDXL block-level merging with memory optimization and granular control.
Combines standard and advanced block merging with GPU-optimized memory management.

Author: Arctenox
Version: 2.0.0
License: GPL-3.0
"""

import torch
import gc
import logging
import psutil
from typing import Dict, List, Tuple, Optional
from contextlib import contextmanager

# ComfyUI imports
try:
    import comfy.model_management
    HAS_COMFY = True
except:
    HAS_COMFY = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryManager:
    """Advanced memory management for GPU/CPU processing."""
    
    def __init__(self):
        self.cuda_available = torch.cuda.is_available()
        self.device_memory_gb = self._get_device_memory()
        self.system_memory_gb = self._get_system_memory()
        self.memory_threshold = 0.85
        
    def _get_device_memory(self) -> float:
        """Get GPU memory in GB."""
        if self.cuda_available:
            try:
                return torch.cuda.get_device_properties(0).total_memory / (1024**3)
            except:
                return 0.0
        return 0.0
    
    def _get_system_memory(self) -> float:
        """Get system RAM in GB."""
        return psutil.virtual_memory().total / (1024**3)
    
    def get_available_memory(self, device: torch.device) -> float:
        """Get currently available memory in GB."""
        if device.type == "cuda" and self.cuda_available:
            try:
                free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
                return free_memory / (1024**3)
            except:
                return 0.0
        else:
            available_memory = psutil.virtual_memory().available / (1024**3)
            return available_memory
    
    def estimate_tensor_memory(self, tensor: torch.Tensor) -> float:
        """Estimate tensor memory usage in GB."""
        if tensor is None:
            return 0.0
        try:
            element_size = tensor.element_size()
            num_elements = tensor.numel()
            return (element_size * num_elements) / (1024**3)
        except:
            return 0.0
    
    def can_fit_in_memory(self, tensor: torch.Tensor, device: torch.device) -> bool:
        """Check if tensor can fit in device memory."""
        tensor_memory = self.estimate_tensor_memory(tensor)
        available_memory = self.get_available_memory(device) * self.memory_threshold
        return tensor_memory <= available_memory
    
    def should_use_cpu_fallback(self, device: torch.device) -> bool:
        """Determine if should fallback to CPU based on memory."""
        if device.type == "cpu":
            return False
        
        available_gpu_memory = self.get_available_memory(device)
        available_cpu_memory = self.get_available_memory(torch.device("cpu"))
        
        return available_gpu_memory < 2.0 or (available_cpu_memory > available_gpu_memory * 2)
    
    @contextmanager
    def memory_context(self, device: torch.device):
        """Context manager for memory cleanup."""
        try:
            yield
        finally:
            self.cleanup_memory(device)
    
    def cleanup_memory(self, device: torch.device = None):
        """Comprehensive memory cleanup."""
        gc.collect()
        if self.cuda_available:
            try:
                if device is None or device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except:
                pass


class SDXLBlockMergeTensorPrism:
    """
    Standard SDXL block merge with device-safe patching.
    Optimized for quick merges with straightforward block control.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "model_A": ("MODEL", {}),
                "model_B": ("MODEL", {}),
                "merge_method": (["Linear Interpolation", "Add Difference", "TIES-Merging (Simplified)"],),
                "default_unet_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "memory_limit_gb": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 64.0, "step": 0.5}),
            },
            "optional": {
                "model_C": ("MODEL", {}),
                "ties_global_alpha_A": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "ties_global_alpha_B": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "rescale_output_magnitudes": ("BOOLEAN", {"default": False}),
                "a_delta_factor": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "b_delta_factor": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                
                "out_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "time_embed_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "label_emb_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

        # Add block-specific ratios
        for i in range(12):
            inputs["optional"][f"input_block_{i:02d}_ratio"] = ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01})
            inputs["optional"][f"output_block_{i:02d}_ratio"] = ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01})
        
        for i in range(3):
            inputs["optional"][f"middle_block_{i:02d}_ratio"] = ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01})

        return inputs

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "process"
    CATEGORY = "Tensor_Prism/Merge"

    def process(self, model_A, model_B, merge_method, default_unet_ratio, memory_limit_gb=8.0,
                model_C=None, **kwargs):
        """Standard SDXL block merge with device safety."""
        
        logger.info("=== SDXL Block Merge (Tensor Prism) ===")
        
        # Validate inputs
        if merge_method in ["Add Difference", "TIES-Merging (Simplified)"] and model_C is None:
            raise ValueError(f"Model C is required for '{merge_method}'")

        # Get state dictionaries  
        unet_sd_A = model_A.model.state_dict()
        unet_sd_B = model_B.model.state_dict()
        unet_sd_C = model_C.model.state_dict() if model_C else None

        # Build ratio mapping
        key_to_ratio_map = self._build_ratio_mapping(unet_sd_A.keys(), default_unet_ratio, kwargs)
        
        # Create patches with device safety
        patches = {}
        
        logger.info(f"Processing tensor merging...")
        for key in unet_sd_A.keys():
            if key not in unet_sd_B:
                continue
                
            param_A = unet_sd_A[key]
            ratio = key_to_ratio_map.get(key, default_unet_ratio)
            
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
                    merged_param = merged_param.to(device)
                    patch_diff = merged_param - param_A
                    
                    if not torch.allclose(patch_diff, torch.zeros_like(patch_diff), atol=1e-8):
                        patches[key] = (patch_diff.cpu(),)
                        
            except Exception as e:
                logger.warning(f"Failed to process {key}: {e}")
                continue
        
        logger.info(f"Created {len(patches)} patches")
        
        merged_model = model_A.clone()
        if patches:
            merged_model.add_patches(patches, 1.0)
        
        logger.info("=== SDXL Block Merge completed ===")
        return (merged_model,)
    
    def _build_ratio_mapping(self, keys, default_ratio, kwargs):
        """Build mapping of parameter keys to their merge ratios."""
        ratio_prefixes = {
            "time_embed.": kwargs.get("time_embed_ratio", default_ratio),
            "label_emb.": kwargs.get("label_emb_ratio", default_ratio), 
            "out.": kwargs.get("out_ratio", default_ratio),
        }
        
        for i in range(12):
            ratio_prefixes[f"input_blocks.{i}."] = kwargs.get(f"input_block_{i:02d}_ratio", default_ratio)
            ratio_prefixes[f"output_blocks.{i}."] = kwargs.get(f"output_block_{i:02d}_ratio", default_ratio)
        
        ratio_prefixes["middle_block.0."] = kwargs.get("middle_block_00_ratio", default_ratio)
        ratio_prefixes["middle_block.1."] = kwargs.get("middle_block_01_ratio", default_ratio)
        ratio_prefixes["middle_block.2."] = kwargs.get("middle_block_02_ratio", default_ratio)
        
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


class SDXLAdvancedBlockMergeTensorPrism:
    """
    Advanced SDXL block merge with memory optimization for any GPU size.
    Includes batch processing, precision control, and aggressive memory management.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "model_A": ("MODEL", {}),
                "model_B": ("MODEL", {}),
                "merge_method": (["Linear Interpolation", "Add Difference", "TIES-Merging (Simplified)"],),
                "default_unet_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "memory_limit_gb": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 64.0, "step": 0.5}),
                "force_cpu": ("BOOLEAN", {"default": False}),
                "batch_size": ("INT", {"default": 50, "min": 1, "max": 500, "step": 10}),
                "auto_memory_management": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "model_C": ("MODEL", {}),
                "ties_global_alpha_A": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "ties_global_alpha_B": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "rescale_output_magnitudes": ("BOOLEAN", {"default": False}),
                "iterations": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
                "a_delta_factor": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "b_delta_factor": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "precision_mode": (["auto", "fp16", "fp32"], {"default": "auto"}),
                "aggressive_cleanup": ("BOOLEAN", {"default": True}),
                
                "out_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "time_embed_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "label_emb_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

        for i in range(9):
            inputs["optional"][f"input_block_{i:02d}_ratio"] = ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01})
            inputs["optional"][f"output_block_{i:02d}_ratio"] = ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01})
        
        for i in range(3):
            inputs["optional"][f"middle_block_{i:02d}_ratio"] = ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01})

        return inputs

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "process"
    CATEGORY = "Tensor_Prism/Merge"

    def __init__(self):
        self.memory_manager = MemoryManager()
        self.device = self._get_optimal_device()
        self.cuda_available = torch.cuda.is_available()
        self.precision_dtype = torch.float32
        self.processing_device = self.device
        
        logger.info(f"Initialized - Device: {self.device}, GPU Memory: {self.memory_manager.device_memory_gb:.1f}GB")

    def _get_optimal_device(self) -> torch.device:
        """Get optimal device based on available memory."""
        if HAS_COMFY:
            return comfy.model_management.get_torch_device()
        
        if torch.cuda.is_available():
            gpu_memory = self.memory_manager.device_memory_gb
            if gpu_memory >= 6.0:
                return torch.device("cuda")
            else:
                logger.warning(f"GPU has only {gpu_memory:.1f}GB memory, may need CPU fallback")
                return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def _determine_precision(self, precision_mode: str, device: torch.device) -> torch.dtype:
        """Determine optimal precision based on memory and device."""
        if precision_mode == "fp32":
            return torch.float32
        elif precision_mode == "fp16":
            if device.type == "cuda":
                return torch.float16
            else:
                logger.warning("FP16 not supported on CPU, using FP32")
                return torch.float32
        else:
            if device.type == "cuda":
                gpu_memory = self.memory_manager.device_memory_gb
                if gpu_memory <= 12.0:
                    return torch.float16
            return torch.float32

    def _adaptive_batch_size(self, total_params: int, device: torch.device, batch_size: int) -> int:
        """Adaptively determine batch size based on available memory."""
        available_memory = self.memory_manager.get_available_memory(device)
        
        if device.type == "cuda" and available_memory < 4.0:
            return min(batch_size, max(1, total_params // 20))
        elif available_memory < 8.0:
            return min(batch_size, max(1, total_params // 10))
        else:
            return batch_size

    def _safe_to_device_optimized(self, tensor: torch.Tensor, target_device: torch.device, 
                                target_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Optimized tensor device transfer with memory checking."""
        try:
            if not self.memory_manager.can_fit_in_memory(tensor, target_device):
                if target_device.type != "cpu":
                    logger.warning(f"Tensor too large for {target_device}, using CPU")
                    target_device = torch.device("cpu")
            
            if target_dtype is not None:
                return tensor.to(device=target_device, dtype=target_dtype)
            else:
                return tensor.to(device=target_device)
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"OOM during transfer to {target_device}, falling back to CPU")
                self.memory_manager.cleanup_memory(target_device)
                return tensor.to(device=torch.device("cpu"), dtype=target_dtype)
            else:
                raise e

    def _process_parameters_in_batches(self, param_keys: List[str], state_dicts: Dict, 
                                     merge_params: Dict, batch_size: int) -> Dict:
        """Process parameters in memory-efficient batches."""
        patches = {}
        total_params = len(param_keys)
        processed = 0
        
        effective_batch_size = self._adaptive_batch_size(total_params, self.processing_device, batch_size)
        logger.info(f"Processing {total_params} parameters in batches of {effective_batch_size}")
        
        for i in range(0, total_params, effective_batch_size):
            batch_keys = param_keys[i:i + effective_batch_size]
            batch_patches = {}
            
            with self.memory_manager.memory_context(self.processing_device):
                for key in batch_keys:
                    try:
                        patch = self._process_single_parameter(key, state_dicts, merge_params)
                        if patch is not None:
                            batch_patches[key] = patch
                            processed += 1
                    except Exception as e:
                        logger.warning(f"Failed to process parameter {key}: {e}")
                        continue
                
                patches.update(batch_patches)
                
                if merge_params.get('aggressive_cleanup', True):
                    self.memory_manager.cleanup_memory(self.processing_device)
                
                if i % (effective_batch_size * 5) == 0:
                    progress = (i + len(batch_keys)) / total_params * 100
                    available_memory = self.memory_manager.get_available_memory(self.processing_device)
                    logger.info(f"Progress: {progress:.1f}% ({processed} patches), Available memory: {available_memory:.1f}GB")
        
        return patches

    def _process_single_parameter(self, key: str, state_dicts: Dict, merge_params: Dict) -> Optional[Tuple]:
        """Process a single parameter with optimized memory usage."""
        try:
            unet_state_dict_A = state_dicts['A']
            unet_state_dict_B = state_dicts['B']
            unet_state_dict_C = state_dicts.get('C')
            
            if key not in unet_state_dict_B:
                return None
            
            param_A = unet_state_dict_A[key]
            
            if not isinstance(param_A, torch.Tensor) or param_A.numel() == 0:
                return None
            
            param_A_proc = self._safe_to_device_optimized(param_A, self.processing_device, self.precision_dtype)
            param_B_proc = self._safe_to_device_optimized(unet_state_dict_B[key], self.processing_device, self.precision_dtype)
            
            if param_A_proc.shape != param_B_proc.shape:
                return None
            
            merge_ratio = merge_params['key_to_ratio_map'].get(key, merge_params['default_unet_ratio'])
            
            if merge_ratio == 0.0:
                return None
            
            merge_method = merge_params['merge_method']
            
            if merge_method == "Linear Interpolation":
                merged_param = self._linear_interpolation(param_A_proc, param_B_proc, merge_ratio)
            elif merge_method == "Add Difference":
                merged_param = self._add_difference(param_A_proc, param_B_proc, unet_state_dict_C, 
                                                   key, merge_ratio, merge_params)
            elif merge_method == "TIES-Merging (Simplified)":
                merged_param = self._ties_merging(param_A_proc, param_B_proc, unet_state_dict_C, 
                                                 key, merge_ratio, merge_params)
            else:
                return None
            
            if merged_param is None:
                return None
            
            patch_diff = merged_param - param_A_proc
            
            if torch.allclose(patch_diff, torch.zeros_like(patch_diff), atol=1e-8, rtol=1e-6):
                return None
            
            patch_diff = patch_diff.cpu()
            if param_A.dtype != patch_diff.dtype:
                patch_diff = patch_diff.to(dtype=param_A.dtype)
            
            return (patch_diff,)
            
        except Exception as e:
            logger.error(f"Error processing parameter {key}: {e}")
            return None

    def _linear_interpolation(self, param_A: torch.Tensor, param_B: torch.Tensor, 
                             merge_ratio: float) -> Optional[torch.Tensor]:
        """Memory-optimized linear interpolation."""
        try:
            if merge_ratio == 0.0:
                return param_A
            elif merge_ratio == 1.0:
                return param_B
            else:
                if param_A.is_contiguous():
                    result = param_A * (1.0 - merge_ratio)
                    result.add_(param_B, alpha=merge_ratio)
                    return result
                else:
                    return param_A * (1.0 - merge_ratio) + param_B * merge_ratio
        except Exception as e:
            logger.error(f"Linear interpolation failed: {e}")
            return None

    def _add_difference(self, param_A: torch.Tensor, param_B: torch.Tensor, 
                       unet_state_dict_C: Optional[Dict], key: str, 
                       merge_ratio: float, merge_params: Dict) -> Optional[torch.Tensor]:
        """Memory-optimized add difference method."""
        try:
            if unet_state_dict_C and key in unet_state_dict_C:
                param_C_proc = self._safe_to_device_optimized(unet_state_dict_C[key], self.processing_device, self.precision_dtype)
                
                if param_A.shape != param_C_proc.shape:
                    return self._linear_interpolation(param_A, param_B, merge_ratio)
                
                a_delta_factor = max(-2.0, min(2.0, merge_params.get('a_delta_factor', 1.0)))
                b_delta_factor = max(-2.0, min(2.0, merge_params.get('b_delta_factor', 1.0)))
                
                delta_A = (param_A - param_C_proc) * (merge_ratio * a_delta_factor)
                delta_B = (param_B - param_C_proc) * ((1.0 - merge_ratio) * b_delta_factor)
                
                result = param_C_proc + delta_A + delta_B
                return result
            else:
                return self._linear_interpolation(param_A, param_B, merge_ratio)
        except Exception as e:
            logger.error(f"Add difference failed: {e}")
            return self._linear_interpolation(param_A, param_B, merge_ratio)

    def _ties_merging(self, param_A: torch.Tensor, param_B: torch.Tensor, 
                     unet_state_dict_C: Optional[Dict], key: str, 
                     merge_ratio: float, merge_params: Dict) -> Optional[torch.Tensor]:
        """Memory-optimized TIES merging."""
        try:
            if unet_state_dict_C and key in unet_state_dict_C:
                param_C_proc = self._safe_to_device_optimized(unet_state_dict_C[key], self.processing_device, self.precision_dtype)
                
                if param_A.shape != param_C_proc.shape:
                    return self._linear_interpolation(param_A, param_B, merge_ratio)
                
                alpha_A = merge_params.get('ties_global_alpha_A', 0.5) * merge_ratio
                alpha_B = merge_params.get('ties_global_alpha_B', 0.5) * (1.0 - merge_ratio)
                
                if merge_params.get('rescale_output_magnitudes', False):
                    total_alpha = alpha_A + alpha_B
                    if total_alpha > 1e-8:
                        alpha_A /= total_alpha
                        alpha_B /= total_alpha
                
                a_delta_factor = max(-2.0, min(2.0, merge_params.get('a_delta_factor', 1.0)))
                b_delta_factor = max(-2.0, min(2.0, merge_params.get('b_delta_factor', 1.0)))
                
                delta_A = (param_A - param_C_proc) * (alpha_A * a_delta_factor)
                delta_B = (param_B - param_C_proc) * (alpha_B * b_delta_factor)
                
                result = param_C_proc + delta_A + delta_B
                return result
            else:
                return self._linear_interpolation(param_A, param_B, merge_ratio)
        except Exception as e:
            logger.error(f"TIES merging failed: {e}")
            return self._linear_interpolation(param_A, param_B, merge_ratio)

    def process(self, model_A, model_B, merge_method: str, default_unet_ratio: float, 
                memory_limit_gb: float = 8.0, force_cpu: bool = False, batch_size: int = 50,
                auto_memory_management: bool = True, precision_mode: str = "auto",
                aggressive_cleanup: bool = True, model_C = None, **kwargs) -> Tuple:
        """Main processing function optimized for any GPU size."""
        try:
            logger.info("=== SDXL Advanced Block Merge (Tensor Prism) ===")
            
            if force_cpu or self.memory_manager.should_use_cpu_fallback(self.device):
                self.processing_device = torch.device("cpu")
                logger.info("Using CPU for processing")
            else:
                self.processing_device = self.device
                logger.info(f"Using {self.processing_device} for processing")
            
            self.precision_dtype = self._determine_precision(precision_mode, self.processing_device)
            logger.info(f"Using precision: {self.precision_dtype}")
            
            if merge_method in ["Add Difference", "TIES-Merging (Simplified)"] and model_C is None:
                raise ValueError(f"Model C is required for '{merge_method}'")
            
            try:
                unet_state_dict_A = model_A.model.state_dict()
                unet_state_dict_B = model_B.model.state_dict()
                unet_state_dict_C = model_C.model.state_dict() if model_C else None
            except Exception as e:
                raise RuntimeError(f"Failed to extract state dictionaries: {e}")
            
            common_keys = list(set(unet_state_dict_A.keys()) & set(unet_state_dict_B.keys()))
            if not common_keys:
                raise ValueError("No compatible parameters found")
            
            logger.info(f"Processing {len(common_keys)} compatible parameters")
            
            key_to_ratio_map = self._build_ratio_mapping(common_keys, default_unet_ratio, kwargs)
            
            merge_params = {
                'merge_method': merge_method,
                'default_unet_ratio': default_unet_ratio,
                'key_to_ratio_map': key_to_ratio_map,
                'aggressive_cleanup': aggressive_cleanup,
                **kwargs
            }
            
            state_dicts = {
                'A': unet_state_dict_A,
                'B': unet_state_dict_B,
                'C': unet_state_dict_C
            }
            
            with self.memory_manager.memory_context(self.processing_device):
                patches = self._process_parameters_in_batches(common_keys, state_dicts, merge_params, batch_size)
            
            logger.info(f"Generated {len(patches)} patches successfully")
            
            try:
                merged_model = model_A.clone()
                if patches:
                    merged_model.add_patches(patches, 1.0)
                    logger.info("Patches applied successfully")
                else:
                    logger.warning("No patches generated, returning original model A")
                
            except Exception as e:
                logger.error(f"Failed to apply patches: {e}")
                raise RuntimeError(f"Patch application failed: {e}")
            
            self.memory_manager.cleanup_memory()
            
            if self.processing_device.type == "cuda":
                final_memory = self.memory_manager.get_available_memory(self.processing_device)
                logger.info(f"Final GPU memory available: {final_memory:.1f}GB")
            
            logger.info("=== SDXL Advanced Block Merge Completed Successfully ===")
            return (merged_model,)
            
        except Exception as e:
            logger.error(f"Model merge failed: {e}")
            self.memory_manager.cleanup_memory()
            return (model_A,)

    def _build_ratio_mapping(self, keys: List[str], default_ratio: float, kwargs: Dict) -> Dict[str, float]:
        """Build ratio mapping with validation."""
        default_ratio = max(0.0, min(1.0, default_ratio))
        
        ratio_prefixes = {
            "time_embed.": max(0.0, min(1.0, kwargs.get("time_embed_ratio", default_ratio))),
            "label_emb.": max(0.0, min(1.0, kwargs.get("label_emb_ratio", default_ratio))),
            "out.": max(0.0, min(1.0, kwargs.get("out_ratio", default_ratio))),
        }
        
        for i in range(9):
            input_ratio = max(0.0, min(1.0, kwargs.get(f"input_block_{i:02d}_ratio", default_ratio)))
            output_ratio = max(0.0, min(1.0, kwargs.get(f"output_block_{i:02d}_ratio", default_ratio)))
            ratio_prefixes[f"input_blocks.{i}."] = input_ratio
            ratio_prefixes[f"output_blocks.{i}."] = output_ratio
        
        for i in range(3):
            middle_ratio = max(0.0, min(1.0, kwargs.get(f"middle_block_{i:02d}_ratio", default_ratio)))
            ratio_prefixes[f"middle_block.{i}."] = middle_ratio
        
        sorted_prefixes = sorted(ratio_prefixes.items(), key=lambda x: len(x[0]), reverse=True)
        
        key_to_ratio = {}
        for key in keys:
            ratio = default_ratio
            for prefix, prefix_ratio in sorted_prefixes:
                if key.startswith(prefix):
                    ratio = prefix_ratio
                    break
            key_to_ratio[key] = ratio
        
        return key_to_ratio


# Node registration
NODE_CLASS_MAPPINGS = {
    "SDXLBlockMergeTensorPrism": SDXLBlockMergeTensorPrism,
    "SDXLAdvancedBlockMergeTensorPrism": SDXLAdvancedBlockMergeTensorPrism
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDXLBlockMergeTensorPrism": "SDXL Block Merge (Tensor Prism)",
    "SDXLAdvancedBlockMergeTensorPrism": "SDXL Advanced Block Merge (Tensor Prism)"
}