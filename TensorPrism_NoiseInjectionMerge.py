"""
TensorPrism Noise Injection Merge
==================================

Advanced noise injection during model merging to escape local optima and
discover emergent capabilities. Supports multiple noise patterns with minimal
overhead and memory impact.

Author: Arctenox
Version: 1.7.0
License: GPL-3.0
"""

import torch
import math
import numpy as np
from typing import Dict, Any, Optional, Tuple
import comfy.model_management as mm
import gc

class NoisePattern:
    """Enum-like class for noise pattern types"""
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"
    STRUCTURED = "structured"
    LAYER_SCALED = "layer_scaled"
    ADAPTIVE = "adaptive"
    PERLIN = "perlin"
    GRADIENT = "gradient"

class TensorPrism_NoiseInjectionMerge:
    """
    Inject controlled noise during model merging to escape local optima
    and discover emergent capabilities with minimal overhead.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_A": ("MODEL", {"tooltip": "First model to merge"}),
                "model_B": ("MODEL", {"tooltip": "Second model to merge"}),
                "merge_ratio": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Base merge ratio (0=full A, 1=full B)"
                }),
                "noise_strength": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.005,
                    "tooltip": "Overall noise injection strength"
                }),
                "noise_pattern": ([
                    "gaussian",
                    "uniform", 
                    "structured",
                    "layer_scaled",
                    "adaptive",
                    "perlin",
                    "gradient"
                ], {
                    "default": "gaussian",
                    "tooltip": "Type of noise pattern to inject"
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Random seed for reproducibility"
                }),
            },
            "optional": {
                "layer_scaling_factor": ("FLOAT", {
                    "default": 1.5,
                    "min": 0.5,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Scale factor for layer-based noise (deeper=more noise)"
                }),
                "adaptive_threshold": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Similarity threshold for adaptive noise"
                }),
                "noise_decay": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.5,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Decay factor per layer (1.0=no decay)"
                }),
                "perlin_frequency": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.5,
                    "max": 10.0,
                    "step": 0.5,
                    "tooltip": "Frequency for Perlin noise"
                }),
                "focus_attention": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Focus noise on attention layers"
                }),
                "focus_mlp": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Focus noise on MLP layers"
                }),
                "preserve_norms": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Preserve tensor norms after injection"
                }),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("merged_model",)
    FUNCTION = "merge_with_noise"
    CATEGORY = "Tensor_Prism/Merge"
    
    def __init__(self):
        self.device = mm.get_torch_device()
        self.noise_generators = {
            "gaussian": self._generate_gaussian_noise,
            "uniform": self._generate_uniform_noise,
            "structured": self._generate_structured_noise,
            "layer_scaled": self._generate_layer_scaled_noise,
            "adaptive": self._generate_adaptive_noise,
            "perlin": self._generate_perlin_noise,
            "gradient": self._generate_gradient_noise,
        }
    
    def _clear_memory(self):
        """Minimal memory cleanup"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def _is_target_layer(self, key: str, focus_attention: bool, focus_mlp: bool) -> bool:
        """Determine if this layer should receive focused noise"""
        key_lower = key.lower()
        
        if focus_attention and any(x in key_lower for x in ['attn', 'attention', 'to_q', 'to_k', 'to_v', 'to_out']):
            return True
        
        if focus_mlp and any(x in key_lower for x in ['mlp', 'fc', 'feedforward', 'ff']):
            return True
        
        # If no focus specified, target all layers
        if not focus_attention and not focus_mlp:
            return True
        
        return False
    
    def _get_layer_depth(self, key: str) -> float:
        """Estimate layer depth (0.0=early, 1.0=late) for scaling"""
        key_lower = key.lower()
        
        # Try to extract block numbers
        if 'input_blocks' in key_lower:
            try:
                block_num = int(''.join(filter(str.isdigit, key.split('input_blocks')[1].split('.')[1])))
                return block_num / 12.0  # SDXL has ~12 input blocks
            except:
                return 0.0
        elif 'middle_block' in key_lower:
            return 0.5
        elif 'output_blocks' in key_lower:
            try:
                block_num = int(''.join(filter(str.isdigit, key.split('output_blocks')[1].split('.')[1])))
                return 0.5 + (block_num / 12.0) * 0.5
            except:
                return 0.75
        
        return 0.5  # Default middle
    
    def _generate_gaussian_noise(self, shape: torch.Size, device: torch.device, 
                                 generator: torch.Generator, **kwargs) -> torch.Tensor:
        """Standard Gaussian noise"""
        return torch.randn(shape, device=device, generator=generator, dtype=torch.float32)
    
    def _generate_uniform_noise(self, shape: torch.Size, device: torch.device,
                               generator: torch.Generator, **kwargs) -> torch.Tensor:
        """Uniform noise in [-1, 1]"""
        return torch.rand(shape, device=device, generator=generator, dtype=torch.float32) * 2.0 - 1.0
    
    def _generate_structured_noise(self, shape: torch.Size, device: torch.device,
                                   generator: torch.Generator, **kwargs) -> torch.Tensor:
        """Structured noise with block patterns"""
        noise = torch.randn(shape, device=device, generator=generator, dtype=torch.float32)
        
        # Add structured component (low-frequency patterns)
        if len(shape) >= 2:
            # Create coarse noise and upsample
            coarse_shape = [max(1, s // 4) for s in shape]
            coarse_noise = torch.randn(coarse_shape, device=device, generator=generator, dtype=torch.float32)
            
            # Simple repeat-based upsampling for structured patterns
            for dim in range(len(shape)):
                repeats = [1] * len(coarse_noise.shape)
                repeats[dim] = min(4, shape[dim] // coarse_shape[dim]) if coarse_shape[dim] > 0 else 1
                coarse_noise = coarse_noise.repeat(*repeats)
            
            # Trim to exact shape
            slices = tuple(slice(0, s) for s in shape)
            coarse_noise = coarse_noise[slices]
            
            noise = 0.7 * noise + 0.3 * coarse_noise
        
        return noise
    
    def _generate_layer_scaled_noise(self, shape: torch.Size, device: torch.device,
                                     generator: torch.Generator, layer_depth: float = 0.5,
                                     scaling_factor: float = 1.5, **kwargs) -> torch.Tensor:
        """Noise scaled by layer depth - deeper layers get more noise"""
        base_noise = torch.randn(shape, device=device, generator=generator, dtype=torch.float32)
        
        # Scale by depth (deeper = more noise)
        depth_scale = 1.0 + (layer_depth * (scaling_factor - 1.0))
        return base_noise * depth_scale
    
    def _generate_adaptive_noise(self, shape: torch.Size, device: torch.device,
                                generator: torch.Generator, similarity: float = 0.5,
                                threshold: float = 0.3, **kwargs) -> torch.Tensor:
        """Adaptive noise based on tensor similarity - high similarity = more noise"""
        base_noise = torch.randn(shape, device=device, generator=generator, dtype=torch.float32)
        
        # Scale noise inversely with similarity (high similarity = exploration needed)
        if similarity > threshold:
            # High similarity - inject more noise to escape local optima
            noise_scale = 1.0 + (similarity - threshold) * 2.0
        else:
            # Low similarity - less noise (models already different)
            noise_scale = 0.5 + (similarity / threshold) * 0.5
        
        return base_noise * noise_scale
    
    def _generate_perlin_noise(self, shape: torch.Size, device: torch.device,
                              generator: torch.Generator, frequency: float = 2.0, **kwargs) -> torch.Tensor:
        """Perlin-like smooth noise (simplified version)"""
        # Generate base noise
        noise = torch.randn(shape, device=device, generator=generator, dtype=torch.float32)
        
        # Apply smoothing through low-pass filtering
        if len(shape) >= 2 and shape[0] > 1 and shape[1] > 1:
            # Simple averaging kernel for smoothing
            kernel_size = max(3, int(frequency))
            noise_padded = torch.nn.functional.pad(noise, (kernel_size//2,) * (len(shape) * 2), mode='reflect')
            
            # Apply averaging
            if len(shape) == 2:
                noise = torch.nn.functional.avg_pool2d(
                    noise_padded.unsqueeze(0).unsqueeze(0),
                    kernel_size=kernel_size,
                    stride=1,
                    padding=0
                ).squeeze(0).squeeze(0)
        
        return noise
    
    def _generate_gradient_noise(self, shape: torch.Size, device: torch.device,
                                generator: torch.Generator, **kwargs) -> torch.Tensor:
        """Gradient-like noise with directional structure"""
        noise = torch.randn(shape, device=device, generator=generator, dtype=torch.float32)
        
        # Add gradient component
        if len(shape) >= 2:
            # Create gradient along first dimension
            gradient = torch.linspace(-1, 1, shape[0], device=device, dtype=torch.float32)
            for _ in range(len(shape) - 1):
                gradient = gradient.unsqueeze(-1)
            gradient = gradient.expand(*shape)
            
            noise = 0.7 * noise + 0.3 * gradient
        
        return noise
    
    def _calculate_similarity(self, tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> float:
        """Calculate cosine similarity between tensors"""
        if tensor_a.numel() == 0 or tensor_b.numel() == 0:
            return 0.5
        
        try:
            flat_a = tensor_a.flatten().float()
            flat_b = tensor_b.flatten().float()
            
            norm_a = torch.norm(flat_a)
            norm_b = torch.norm(flat_b)
            
            if norm_a < 1e-8 or norm_b < 1e-8:
                return 0.5
            
            similarity = torch.dot(flat_a, flat_b) / (norm_a * norm_b)
            return similarity.item()
        except:
            return 0.5
    
    def _inject_noise_to_tensor(
        self,
        tensor: torch.Tensor,
        noise_pattern: str,
        noise_strength: float,
        generator: torch.Generator,
        layer_depth: float = 0.5,
        similarity: float = 0.5,
        **kwargs
    ) -> torch.Tensor:
        """Inject noise into a tensor with specified pattern"""
        if noise_strength <= 0.0 or tensor.numel() == 0:
            return tensor
        
        device = tensor.device
        dtype = tensor.dtype
        
        # Generate noise on the same device
        noise_generator = self.noise_generators.get(noise_pattern, self._generate_gaussian_noise)
        noise = noise_generator(
            tensor.shape,
            device,
            generator,
            layer_depth=layer_depth,
            similarity=similarity,
            **kwargs
        )
        
        # Scale noise
        tensor_std = tensor.std() if tensor.numel() > 1 else 1.0
        if tensor_std < 1e-8:
            tensor_std = 1.0
        
        scaled_noise = noise * noise_strength * tensor_std
        
        # Inject noise
        noisy_tensor = tensor + scaled_noise.to(dtype)
        
        # Preserve norms if requested
        if kwargs.get('preserve_norms', True):
            original_norm = torch.norm(tensor)
            noisy_norm = torch.norm(noisy_tensor)
            
            if noisy_norm > 1e-8:
                noisy_tensor = noisy_tensor * (original_norm / noisy_norm)
        
        return noisy_tensor
    
    def merge_with_noise(
        self,
        model_A,
        model_B,
        merge_ratio: float,
        noise_strength: float,
        noise_pattern: str,
        seed: int,
        layer_scaling_factor: float = 1.5,
        adaptive_threshold: float = 0.3,
        noise_decay: float = 0.9,
        perlin_frequency: float = 2.0,
        focus_attention: bool = True,
        focus_mlp: bool = False,
        preserve_norms: bool = True,
    ):
        """
        Main merge function with noise injection
        """
        print(f"\n[TensorPrism] Noise Injection Merge")
        print(f"[TensorPrism] Pattern: {noise_pattern}, Strength: {noise_strength:.4f}")
        print(f"[TensorPrism] Merge Ratio: {merge_ratio:.4f}, Seed: {seed}")
        
        # Set random seed for reproducibility
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        
        # Clone model A as base
        merged_model = model_A.clone()
        
        # Get state dicts
        state_dict_a = model_A.model.state_dict()
        state_dict_b = model_B.model.state_dict()
        
        patches = {}
        processed_count = 0
        noisy_count = 0
        
        # Process each key
        for key in state_dict_a.keys():
            if key not in state_dict_b:
                continue
            
            try:
                tensor_a = state_dict_a[key]
                tensor_b = state_dict_b[key]
                
                # Only process float tensors
                if not isinstance(tensor_a, torch.Tensor) or not tensor_a.is_floating_point():
                    continue
                
                # Check if this is a target layer
                is_target = self._is_target_layer(key, focus_attention, focus_mlp)
                
                # Get layer depth for scaling
                layer_depth = self._get_layer_depth(key)
                
                # Apply decay based on depth
                effective_noise_strength = noise_strength * (noise_decay ** layer_depth)
                
                # Move to device
                tensor_a = tensor_a.to(self.device)
                tensor_b = tensor_b.to(self.device)
                
                # Basic merge
                merged_tensor = (1.0 - merge_ratio) * tensor_a + merge_ratio * tensor_b
                
                # Inject noise if target layer
                if is_target and effective_noise_strength > 0.0:
                    # Calculate similarity for adaptive noise
                    similarity = self._calculate_similarity(tensor_a, tensor_b)
                    
                    # Inject noise
                    merged_tensor = self._inject_noise_to_tensor(
                        merged_tensor,
                        noise_pattern,
                        effective_noise_strength,
                        generator,
                        layer_depth=layer_depth,
                        similarity=similarity,
                        scaling_factor=layer_scaling_factor,
                        threshold=adaptive_threshold,
                        frequency=perlin_frequency,
                        preserve_norms=preserve_norms,
                    )
                    noisy_count += 1
                
                # Calculate patch (difference from original)
                patch_diff = merged_tensor - tensor_a
                
                if not torch.allclose(patch_diff, torch.zeros_like(patch_diff), atol=1e-8):
                    patches[key] = (patch_diff.cpu(),)
                    processed_count += 1
                
            except Exception as e:
                print(f"[TensorPrism] Warning: Failed to process {key}: {e}")
                continue
        
        # Apply patches
        if patches:
            merged_model.add_patches(patches, 1.0)
            print(f"[TensorPrism] Processed {processed_count} tensors")
            print(f"[TensorPrism] Injected noise into {noisy_count} target layers")
        else:
            print(f"[TensorPrism] Warning: No patches created")
        
        self._clear_memory()
        
        print(f"[TensorPrism] Merge with noise injection completed successfully\n")
        return (merged_model,)


# ==================== NODE REGISTRATION ====================

NODE_CLASS_MAPPINGS = {
    "TensorPrism_NoiseInjectionMerge": TensorPrism_NoiseInjectionMerge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TensorPrism_NoiseInjectionMerge": "Noise Injection Merge (Tensor Prism)",
}
