"""
TensorPrism Core Merge Operations
==================================
Consolidates: MainMerge, LayeredBlend, and Prism (spectral merging)

Author: Arctenox
Version: 1.6.5
License: GPL-3.0
"""

import torch
import math
import torch.fft
import comfy.model_management

# ==================== MAIN MERGE NODE ====================

class TensorPrism_MainMerge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_A": ("MODEL",),
                "model_B": ("MODEL",),
                "merge_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "method": (["linear", "slerp", "cosine", "directional", "frequency", "stochastic"],),
            },
            "optional": {
                "random_seed": ("INT", {"default": 42, "min": 0}),
                "stochastic_prob": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0}),
                "freq_low_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "freq_high_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("merged_model",)
    FUNCTION = "merge_models"
    CATEGORY = "Tensor_Prism/Core"

    def __init__(self):
        self.device = self._get_optimal_device()

    def _get_optimal_device(self):
        try:
            return comfy.model_management.get_torch_device()
        except:
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")

    def _to_device(self, tensor: torch.Tensor, target_device: torch.device = None) -> torch.Tensor:
        if target_device is None:
            target_device = self.device
        if tensor.device != target_device:
            return tensor.to(target_device)
        return tensor

    @staticmethod
    def _blend_linear(t1, t2, alpha):
        device = t1.device
        t2 = t2.to(device)
        return (1 - alpha) * t1 + alpha * t2

    def _blend_slerp(self, t1, t2, alpha):
        if t1.numel() == 0 or t2.numel() == 0:
            return t1
        
        device = t1.device
        t2 = t2.to(device)
        
        t1_flat, t2_flat = t1.view(-1), t2.view(-1)
        t1_norm = torch.nn.functional.normalize(t1_flat, dim=0, eps=1e-8)
        t2_norm = torch.nn.functional.normalize(t2_flat, dim=0, eps=1e-8)
        
        dot = torch.clamp(torch.dot(t1_norm, t2_norm), -1.0 + 1e-7, 1.0 - 1e-7)
        
        if abs(dot.item()) > 0.9995:
            return self._blend_linear(t1, t2, alpha)
        
        theta = torch.acos(dot)
        sin_theta = torch.sin(theta)
        
        if sin_theta.abs() < 1e-6:
            return self._blend_linear(t1, t2, alpha)
        
        w1 = torch.sin((1 - alpha) * theta) / sin_theta
        w2 = torch.sin(alpha * theta) / sin_theta
        
        result = w1 * t1_norm + w2 * t2_norm
        original_norm = torch.norm(t1_flat) * (1 - alpha) + torch.norm(t2_flat) * alpha
        result = result * original_norm
        
        return result.view(t1.shape).to(device)

    @staticmethod
    def _blend_cosine(t1, t2, alpha):
        device = t1.device
        t2 = t2.to(device)
        mu = (1 - math.cos(alpha * math.pi)) / 2
        return (1 - mu) * t1 + mu * t2

    @staticmethod
    def _blend_directional(t1, t2, alpha):
        device = t1.device
        t2 = t2.to(device)
        return t1 + (t2 - t1) * alpha

    def _blend_frequency(self, t1, t2, low_ratio, high_ratio):
        device = t1.device
        t2 = t2.to(device)
        
        if t1.numel() < 4 or t2.numel() < 4:
            return self._blend_linear(t1, t2, (low_ratio + high_ratio) / 2)
        
        try:
            orig_shape = t1.shape
            if t1.dim() < 2:
                t1_fft = t1.view(-1, 1)
                t2_fft = t2.view(-1, 1)
            else:
                t1_fft = t1
                t2_fft = t2
            
            fA = torch.fft.fft2(t1_fft.float())
            fB = torch.fft.fft2(t2_fft.float())
            
            h, w = fA.shape[-2:]
            cy, cx = h // 2, w // 2
            
            y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
            
            dist = torch.sqrt((y - cy).float()**2 + (x - cx).float()**2)
            max_dist = min(cy, cx)
            
            if max_dist > 0:
                normalized_dist = dist / max_dist
                low_freq_mask = (normalized_dist < 0.3).float()
                high_freq_mask = (normalized_dist >= 0.3).float()
            else:
                low_freq_mask = torch.ones_like(dist)
                high_freq_mask = torch.zeros_like(dist)
            
            fA_low = fA * low_freq_mask.unsqueeze(0) if fA.dim() > 2 else fA * low_freq_mask
            fB_low = fB * low_freq_mask.unsqueeze(0) if fB.dim() > 2 else fB * low_freq_mask
            fA_high = fA * high_freq_mask.unsqueeze(0) if fA.dim() > 2 else fA * high_freq_mask
            fB_high = fB * high_freq_mask.unsqueeze(0) if fB.dim() > 2 else fB * high_freq_mask
            
            merged_low = fA_low * (1 - low_ratio) + fB_low * low_ratio
            merged_high = fA_high * (1 - high_ratio) + fB_high * high_ratio
            merged_freq = merged_low + merged_high
            
            result = torch.fft.ifft2(merged_freq).real
            
            return result.view(orig_shape).to(t1.dtype).to(device)
            
        except Exception as e:
            return self._blend_linear(t1, t2, (low_ratio + high_ratio) / 2)

    def _blend_stochastic(self, t1, t2, alpha, prob, seed):
        device = t1.device
        t2 = t2.to(device)
        
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        
        mask = torch.rand(t1.shape, generator=generator, device=device) < prob
        
        base_blend = self._blend_linear(t1, t2, alpha)
        stochastic_component = torch.where(mask, t2, t1)
        
        return (base_blend * (1 - prob) + stochastic_component * prob).to(device)

    def merge_models(self, model_A, model_B, merge_ratio, method,
                     random_seed=42, stochastic_prob=0.1,
                     freq_low_ratio=0.5, freq_high_ratio=0.5):
        
        merged_model = model_A.clone()
        
        state_dict_A = model_A.model.state_dict()
        state_dict_B = model_B.model.state_dict()
        
        patches = {}
        
        for key in state_dict_A.keys():
            if key in state_dict_B:
                t1 = state_dict_A[key]
                t2 = state_dict_B[key]

                if isinstance(t1, torch.Tensor) and isinstance(t2, torch.Tensor):
                    if t1.shape == t2.shape:
                        try:
                            device = t1.device
                            t2 = t2.to(device)
                            
                            if method == "linear":
                                merged_tensor = self._blend_linear(t1, t2, merge_ratio)
                            elif method == "slerp":
                                merged_tensor = self._blend_slerp(t1, t2, merge_ratio)
                            elif method == "cosine":
                                merged_tensor = self._blend_cosine(t1, t2, merge_ratio)
                            elif method == "directional":
                                merged_tensor = self._blend_directional(t1, t2, merge_ratio)
                            elif method == "frequency":
                                merged_tensor = self._blend_frequency(t1, t2, freq_low_ratio, freq_high_ratio)
                            elif method == "stochastic":
                                merged_tensor = self._blend_stochastic(t1, t2, merge_ratio, stochastic_prob, random_seed)
                            else:
                                merged_tensor = self._blend_linear(t1, t2, merge_ratio)
                            
                            merged_tensor = merged_tensor.to(device)
                            
                            diff = merged_tensor - t1
                            if torch.abs(diff).max() > 1e-8:
                                patches[key] = (diff.cpu(),)
                                
                        except Exception as e:
                            print(f"Warning: Failed to merge parameter {key} with method {method}: {e}")
                            continue
        
        if patches:
            merged_model.add_patches(patches, 1.0)
        
        return (merged_model,)


# ==================== LAYERED BLEND NODE ====================

class TensorPrism_LayeredBlend:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_A": ("MODEL",),
                "model_B": ("MODEL",),
                "text_encoder_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "text_encoder_method": (["linear", "slerp", "cosine"],),
                "unet_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "unet_method": (["linear", "slerp", "cosine"],),
                "time_embed_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "input_blocks_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "middle_block_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "output_blocks_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "out_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
            },
            "optional": {
                "vae_A": ("VAE",),
                "vae_B": ("VAE",),
                "vae_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "vae_method": (["linear", "slerp", "cosine"],),
            }
        }

    RETURN_TYPES = ("MODEL", "VAE",)
    RETURN_NAMES = ("merged_model", "merged_vae",)
    FUNCTION = "blend_models"
    CATEGORY = "Tensor_Prism/Core"

    def __init__(self):
        self.device = self._get_optimal_device()

    def _get_optimal_device(self):
        try:
            return comfy.model_management.get_torch_device()
        except:
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")

    @staticmethod
    def _blend_linear(t1: torch.Tensor, t2: torch.Tensor, alpha: float) -> torch.Tensor:
        device = t1.device
        t2 = t2.to(device)
        return (1 - alpha) * t1 + alpha * t2

    @staticmethod
    def _blend_slerp(t1: torch.Tensor, t2: torch.Tensor, alpha: float) -> torch.Tensor:
        if t1.numel() == 0 or t2.numel() == 0:
            return t1
        
        device = t1.device
        t2 = t2.to(device)
            
        t1_flat = t1.view(-1)
        t2_flat = t2.view(-1)
        
        t1_norm = torch.nn.functional.normalize(t1_flat, dim=0, eps=1e-8)
        t2_norm = torch.nn.functional.normalize(t2_flat, dim=0, eps=1e-8)
        
        dot = torch.clamp(torch.dot(t1_norm, t2_norm), -1.0 + 1e-7, 1.0 - 1e-7)
        
        if abs(dot.item()) > 0.9995:
            return TensorPrism_LayeredBlend._blend_linear(t1, t2, alpha)
        
        theta = torch.acos(dot)
        sin_theta = torch.sin(theta)
        
        if sin_theta.abs() < 1e-6:
            return TensorPrism_LayeredBlend._blend_linear(t1, t2, alpha)
        
        w1 = torch.sin((1 - alpha) * theta) / sin_theta
        w2 = torch.sin(alpha * theta) / sin_theta
        
        result = w1 * t1_norm + w2 * t2_norm
        
        original_norm = torch.norm(t1_flat) * (1 - alpha) + torch.norm(t2_flat) * alpha
        result = result * original_norm
        
        return result.view(t1.shape).to(device)

    @staticmethod
    def _blend_cosine(t1: torch.Tensor, t2: torch.Tensor, alpha: float) -> torch.Tensor:
        device = t1.device
        t2 = t2.to(device)
        mu = (1 - math.cos(alpha * math.pi)) / 2
        return (1 - mu) * t1 + mu * t2

    def _get_component_type(self, param_name: str) -> str:
        param_lower = param_name.lower()
        
        if any(pattern in param_lower for pattern in ['cond_stage_model', 'text_encoder', 'clip', 'transformer.text_model']):
            return 'text_encoder'
        
        if any(pattern in param_lower for pattern in ['model.diffusion_model', 'unet']):
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
            else:
                return 'unet'
        
        if any(pattern in param_lower for pattern in ['first_stage_model', 'vae', 'decoder', 'encoder']):
            return 'vae'
        
        return 'unet'

    def _apply_blend_method(self, t1: torch.Tensor, t2: torch.Tensor, strength: float, method: str) -> torch.Tensor:
        device = t1.device
        t2 = t2.to(device)
        
        if method == "linear":
            return self._blend_linear(t1, t2, strength)
        elif method == "slerp":
            return self._blend_slerp(t1, t2, strength)
        elif method == "cosine":
            return self._blend_cosine(t1, t2, strength)
        else:
            return self._blend_linear(t1, t2, strength)

    def blend_models(self, model_A, model_B,
                     text_encoder_strength, text_encoder_method,
                     unet_strength, unet_method,
                     time_embed_strength, input_blocks_strength,
                     middle_block_strength, output_blocks_strength, out_strength,
                     vae_A=None, vae_B=None,
                     vae_strength=0.5, vae_method="linear"):
        
        merged_model = model_A.clone()
        
        state_dict_A = model_A.model.state_dict()
        state_dict_B = model_B.model.state_dict()
        
        patches = {}
        
        component_strengths = {
            'text_encoder': (text_encoder_strength, text_encoder_method),
            'time_embed': (time_embed_strength, unet_method),
            'input_blocks': (input_blocks_strength, unet_method),
            'middle_block': (middle_block_strength, unet_method),
            'output_blocks': (output_blocks_strength, unet_method),
            'out': (out_strength, unet_method),
            'unet': (unet_strength, unet_method),
            'vae': (0.0, "linear")
        }
        
        for key in state_dict_A.keys():
            if key in state_dict_B:
                t1 = state_dict_A[key]
                t2 = state_dict_B[key]

                if isinstance(t1, torch.Tensor) and isinstance(t2, torch.Tensor):
                    if t1.shape == t2.shape:
                        try:
                            device = t1.device
                            t2 = t2.to(device)
                            
                            component_type = self._get_component_type(key)
                            
                            strength, method = component_strengths.get(component_type, (unet_strength, unet_method))
                            
                            if strength == 0:
                                continue
                            
                            merged_tensor = self._apply_blend_method(t1, t2, strength, method)
                            merged_tensor = merged_tensor.to(device)
                            
                            diff = merged_tensor - t1
                            if torch.abs(diff).max() > 1e-8:
                                patches[key] = (diff.cpu(),)
                                
                        except Exception as e:
                            print(f"Warning: Failed to blend parameter {key}: {e}")
                            continue
        
        if patches:
            merged_model.add_patches(patches, 1.0)
        
        merged_vae = None
        if vae_A is not None and vae_B is not None:
            try:
                merged_vae = self._blend_vaes(vae_A, vae_B, vae_strength, vae_method)
            except Exception as e:
                print(f"Warning: Failed to blend VAEs: {e}")
                merged_vae = vae_A
        
        return (merged_model, merged_vae)

    def _blend_vaes(self, vae_A, vae_B, strength: float, method: str):
        if strength == 0:
            return vae_A
        if strength == 1:
            return vae_B
            
        merged_vae = vae_A.clone() if hasattr(vae_A, 'clone') else vae_A
        
        try:
            if hasattr(vae_A, 'first_stage_model') and hasattr(vae_B, 'first_stage_model'):
                state_dict_A = vae_A.first_stage_model.state_dict()
                state_dict_B = vae_B.first_stage_model.state_dict()
                
                vae_patches = {}
                
                for key in state_dict_A.keys():
                    if key in state_dict_B:
                        t1 = state_dict_A[key]
                        t2 = state_dict_B[key]
                        
                        if isinstance(t1, torch.Tensor) and isinstance(t2, torch.Tensor):
                            if t1.shape == t2.shape:
                                device = t1.device
                                t2 = t2.to(device)
                                
                                merged_tensor = self._apply_blend_method(t1, t2, strength, method)
                                merged_tensor = merged_tensor.to(device)
                                
                                diff = merged_tensor - t1
                                
                                if torch.abs(diff).max() > 1e-8:
                                    vae_patches[key] = (diff.cpu(),)
                
                if vae_patches and hasattr(merged_vae, 'add_patches'):
                    merged_vae.add_patches(vae_patches, 1.0)
                    
        except Exception as e:
            print(f"VAE blending failed, using VAE A: {e}")
            return vae_A
            
        return merged_vae


# ==================== PRISM (SPECTRAL) NODE ====================

class TensorPrism_FastPrism:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_A": ("MODEL",),
                "model_B": ("MODEL",),
                "merge_method": ([
                    "spectral_blend", "frequency_bands", "magnitude_weighted", 
                    "adaptive_mix", "harmonic_merge"
                ],),
                "strength": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01
                }),
            },
            "optional": {
                "low_freq_bias": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "high_freq_bias": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "spectral_precision": (["fast", "balanced", "precise"], {"default": "fast"}),
                "layer_selective": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("merged_model",)
    FUNCTION = "fast_prism_merge"
    CATEGORY = "Tensor_Prism/Core"

    def fast_prism_merge(self, model_A, model_B, merge_method, strength,
                        low_freq_bias=0.7, high_freq_bias=0.3, 
                        spectral_precision="fast", layer_selective=False):
        
        merged_model = model_A.clone()
        
        state_dict_A = model_A.model.state_dict()
        state_dict_B = model_B.model.state_dict()
        
        patches = {}
        
        for key in state_dict_A.keys():
            if key not in state_dict_B:
                continue
                
            tensor_A = state_dict_A[key]
            tensor_B = state_dict_B[key]
            
            if not (isinstance(tensor_A, torch.Tensor) and isinstance(tensor_B, torch.Tensor)):
                continue
                
            if tensor_A.shape != tensor_B.shape:
                continue
                
            if tensor_A.numel() < 10:
                continue
            
            device = tensor_A.device
            tensor_B = tensor_B.to(device)
            
            if layer_selective:
                layer_strength = self._get_layer_strength(key, strength)
            else:
                layer_strength = strength
            
            try:
                merged_tensor = self._fast_spectral_merge(
                    tensor_A, tensor_B, merge_method, layer_strength,
                    low_freq_bias, high_freq_bias, spectral_precision
                )
                
                merged_tensor = merged_tensor.to(device)
                
                diff = merged_tensor - tensor_A
                if torch.abs(diff).max() > 1e-6:
                    patches[key] = (diff.cpu(),)
                    
            except Exception as e:
                merged_tensor = tensor_A * (1 - layer_strength) + tensor_B * layer_strength
                merged_tensor = merged_tensor.to(device)
                diff = merged_tensor - tensor_A
                if torch.abs(diff).max() > 1e-6:
                    patches[key] = (diff.cpu(),)
        
        if patches:
            merged_model.add_patches(patches, 1.0)
        
        return (merged_model,)

    def _get_layer_strength(self, layer_name, base_strength):
        layer_name = layer_name.lower()
        
        if any(x in layer_name for x in ['attention', 'attn']):
            return base_strength * 0.8
        elif any(x in layer_name for x in ['norm', 'ln']):
            return base_strength * 0.6
        elif any(x in layer_name for x in ['embed', 'pos']):
            return base_strength * 0.4
        elif any(x in layer_name for x in ['output', 'head']):
            return base_strength * 0.9
        else:
            return base_strength

    def _fast_spectral_merge(self, tensor_A, tensor_B, method, strength, 
                            low_bias, high_bias, precision):
        
        device = tensor_A.device
        tensor_B = tensor_B.to(device)
        
        if method == "spectral_blend":
            return self._spectral_blend_fast(tensor_A, tensor_B, strength, low_bias, high_bias)
        elif method == "frequency_bands":
            return self._frequency_bands_fast(tensor_A, tensor_B, strength, low_bias, high_bias, precision)
        elif method == "magnitude_weighted":
            return self._magnitude_weighted_fast(tensor_A, tensor_B, strength)
        elif method == "adaptive_mix":
            return self._adaptive_mix_fast(tensor_A, tensor_B, strength)
        elif method == "harmonic_merge":
            return self._harmonic_merge_fast(tensor_A, tensor_B, strength, low_bias, high_bias)
        else:
            return tensor_A * (1 - strength) + tensor_B * strength

    def _spectral_blend_fast(self, tensor_A, tensor_B, strength, low_bias, high_bias):
        device = tensor_A.device
        tensor_B = tensor_B.to(device)
        
        mag_A = torch.abs(tensor_A)
        mag_B = torch.abs(tensor_B)
        
        threshold = (mag_A.mean() + mag_B.mean()) / 2
        
        low_freq_mask = (mag_A + mag_B) > threshold
        high_freq_mask = ~low_freq_mask
        
        result = torch.zeros_like(tensor_A, device=device)
        result[low_freq_mask] = (tensor_A[low_freq_mask] * (1 - strength * low_bias) + 
                                tensor_B[low_freq_mask] * (strength * low_bias))
        result[high_freq_mask] = (tensor_A[high_freq_mask] * (1 - strength * high_bias) + 
                                 tensor_B[high_freq_mask] * (strength * high_bias))
        
        return result.to(device)

    def _frequency_bands_fast(self, tensor_A, tensor_B, strength, low_bias, high_bias, precision):
        device = tensor_A.device
        tensor_B = tensor_B.to(device)
        
        if tensor_A.dim() < 2 or precision == "fast":
            var_A = torch.var(tensor_A, dim=-1, keepdim=True)
            var_B = torch.var(tensor_B, dim=-1, keepdim=True)
            
            freq_weight = torch.sigmoid((var_A + var_B) - (var_A.mean() + var_B.mean()))
            
            effective_bias = low_bias * (1 - freq_weight) + high_bias * freq_weight
            effective_strength = strength * effective_bias
            
            return (tensor_A * (1 - effective_strength) + tensor_B * effective_strength).to(device)
        
        else:
            try:
                flat_A = tensor_A.flatten()
                flat_B = tensor_B.flatten()
                
                fft_A = torch.fft.fft(flat_A.float())
                fft_B = torch.fft.fft(flat_B.float())
                
                n = len(fft_A)
                low_cutoff = n // 4
                high_cutoff = 3 * n // 4
                
                fft_merged = fft_A.clone()
                fft_merged[:low_cutoff] = (fft_A[:low_cutoff] * (1 - strength * low_bias) + 
                                          fft_B[:low_cutoff] * (strength * low_bias))
                fft_merged[high_cutoff:] = (fft_A[high_cutoff:] * (1 - strength * high_bias) + 
                                           fft_B[high_cutoff:] * (strength * high_bias))
                fft_merged[low_cutoff:high_cutoff] = (fft_A[low_cutoff:high_cutoff] * (1 - strength) + 
                                                     fft_B[low_cutoff:high_cutoff] * strength)
                
                merged_flat = torch.fft.ifft(fft_merged).real.to(tensor_A.dtype)
                
                return merged_flat.view(tensor_A.shape).to(device)
                
            except Exception as e:
                return self._spectral_blend_fast(tensor_A, tensor_B, strength, low_bias, high_bias)

    def _magnitude_weighted_fast(self, tensor_A, tensor_B, strength):
        device = tensor_A.device
        tensor_B = tensor_B.to(device)
        
        mag_A = torch.norm(tensor_A)
        mag_B = torch.norm(tensor_B)
        
        if mag_A + mag_B > 1e-8:
            weight_B = mag_B / (mag_A + mag_B)
            effective_strength = strength * weight_B + (1 - strength) * 0.5
        else:
            effective_strength = strength
        
        return (tensor_A * (1 - effective_strength) + tensor_B * effective_strength).to(device)

    def _adaptive_mix_fast(self, tensor_A, tensor_B, strength):
        device = tensor_A.device
        tensor_B = tensor_B.to(device)
        
        flat_A = tensor_A.flatten()
        flat_B = tensor_B.flatten()
        
        similarity = torch.cosine_similarity(flat_A.unsqueeze(0), flat_B.unsqueeze(0), dim=1)
        
        adaptive_strength = strength * (1 - 0.3 * torch.abs(similarity))
        
        return (tensor_A * (1 - adaptive_strength) + tensor_B * adaptive_strength).to(device)

    def _harmonic_merge_fast(self, tensor_A, tensor_B, strength, low_bias, high_bias):
        device = tensor_A.device
        tensor_B = tensor_B.to(device)
        
        sign_A = torch.sign(tensor_A)
        sign_B = torch.sign(tensor_B)
        
        alignment = (sign_A * sign_B) > 0
        
        effective_bias = torch.where(alignment, 
                                     torch.tensor(low_bias, device=device), 
                                     torch.tensor(high_bias, device=device))
        effective_strength = strength * effective_bias
        
        return (tensor_A * (1 - effective_strength) + tensor_B * effective_strength).to(device)


# ==================== NODE REGISTRATION ====================

NODE_CLASS_MAPPINGS = {
    "TensorPrism_MainMerge": TensorPrism_MainMerge,
    "TensorPrism_LayeredBlend": TensorPrism_LayeredBlend,
    "TensorPrism_Prism": TensorPrism_FastPrism
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TensorPrism_MainMerge": "Main Merge (Tensor Prism)",
    "TensorPrism_LayeredBlend": "Layered Blend (Tensor Prism)",
    "TensorPrism_Prism": "Prism (Tensor Prism)"
}