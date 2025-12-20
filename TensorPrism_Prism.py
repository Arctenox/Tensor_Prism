import torch
import math
import comfy.model_management

class TensorPrism_FastPrism:
    """
    Fast Prism - Streamlined spectral merging with device mismatch fixes
    """

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
    CATEGORY = "Tensor Prism/Core"

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
            
            # DEVICE FIX: Ensure tensors are on the same device
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
                
                # Ensure result is on correct device
                merged_tensor = merged_tensor.to(device)
                
                diff = merged_tensor - tensor_A
                if torch.abs(diff).max() > 1e-6:
                    patches[key] = (diff.cpu(),)  # Store patches on CPU
                    
            except Exception as e:
                # Silent fallback to linear merge
                merged_tensor = tensor_A * (1 - layer_strength) + tensor_B * layer_strength
                merged_tensor = merged_tensor.to(device)
                diff = merged_tensor - tensor_A
                if torch.abs(diff).max() > 1e-6:
                    patches[key] = (diff.cpu(),)
        
        if patches:
            merged_model.add_patches(patches, 1.0)
        
        return (merged_model,)

    def _get_layer_strength(self, layer_name, base_strength):
        """Get layer-specific strength multipliers"""
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
        """Fast spectral merging methods with device safety"""
        
        # Ensure same device
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
        """Fast spectral blend with device safety"""
        
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
        """Fast frequency band separation with device safety"""
        
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
                
                # Keep FFT on same device
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
        """Magnitude-weighted merging with device safety"""
        
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
        """Adaptive mixing with device safety"""
        
        device = tensor_A.device
        tensor_B = tensor_B.to(device)
        
        flat_A = tensor_A.flatten()
        flat_B = tensor_B.flatten()
        
        similarity = torch.cosine_similarity(flat_A.unsqueeze(0), flat_B.unsqueeze(0), dim=1)
        
        adaptive_strength = strength * (1 - 0.3 * torch.abs(similarity))
        
        return (tensor_A * (1 - adaptive_strength) + tensor_B * adaptive_strength).to(device)

    def _harmonic_merge_fast(self, tensor_A, tensor_B, strength, low_bias, high_bias):
        """Harmonic merging with device safety"""
        
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


NODE_CLASS_MAPPINGS = {
    "TensorPrism_Prism": TensorPrism_FastPrism
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TensorPrism_Prism": "Prism (Tensor Prism)"
}