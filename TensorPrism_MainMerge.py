import torch
import math
import torch.fft
import comfy.model_management

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
    CATEGORY = "Tensor Prism/Core"

    def __init__(self):
        """Initialize with device management"""
        self.device = self._get_optimal_device()

    def _get_optimal_device(self):
        """Get optimal device for processing"""
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
        """Safely move tensor to target device"""
        if target_device is None:
            target_device = self.device
        
        if tensor.device != target_device:
            return tensor.to(target_device)
        return tensor

    @staticmethod
    def _blend_linear(t1, t2, alpha):
        """Linear interpolation with device safety"""
        device = t1.device
        t2 = t2.to(device)
        return (1 - alpha) * t1 + alpha * t2

    def _blend_slerp(self, t1, t2, alpha):
        """Spherical linear interpolation with device safety"""
        if t1.numel() == 0 or t2.numel() == 0:
            return t1
        
        device = t1.device
        t2 = t2.to(device)
        
        t1_flat, t2_flat = t1.view(-1), t2.view(-1)
        
        # Normalize vectors
        t1_norm = torch.nn.functional.normalize(t1_flat, dim=0, eps=1e-8)
        t2_norm = torch.nn.functional.normalize(t2_flat, dim=0, eps=1e-8)
        
        # Calculate dot product and clamp
        dot = torch.clamp(torch.dot(t1_norm, t2_norm), -1.0 + 1e-7, 1.0 - 1e-7)
        
        # Handle near-parallel vectors
        if abs(dot.item()) > 0.9995:
            return self._blend_linear(t1, t2, alpha)
        
        theta = torch.acos(dot)
        sin_theta = torch.sin(theta)
        
        if sin_theta.abs() < 1e-6:
            return self._blend_linear(t1, t2, alpha)
        
        # SLERP formula
        w1 = torch.sin((1 - alpha) * theta) / sin_theta
        w2 = torch.sin(alpha * theta) / sin_theta
        
        result = w1 * t1_norm + w2 * t2_norm
        
        # Restore original magnitude
        original_norm = torch.norm(t1_flat) * (1 - alpha) + torch.norm(t2_flat) * alpha
        result = result * original_norm
        
        return result.view(t1.shape).to(device)

    @staticmethod
    def _blend_cosine(t1, t2, alpha):
        """Cosine interpolation with device safety"""
        device = t1.device
        t2 = t2.to(device)
        mu = (1 - math.cos(alpha * math.pi)) / 2
        return (1 - mu) * t1 + mu * t2

    @staticmethod
    def _blend_directional(t1, t2, alpha):
        """Direct vector interpolation with device safety"""
        device = t1.device
        t2 = t2.to(device)
        return t1 + (t2 - t1) * alpha

    def _blend_frequency(self, t1, t2, low_ratio, high_ratio):
        """FFT-based frequency domain blending with device safety"""
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
            
            # Compute FFT
            fA = torch.fft.fft2(t1_fft.float())
            fB = torch.fft.fft2(t2_fft.float())
            
            h, w = fA.shape[-2:]
            cy, cx = h // 2, w // 2
            
            # Create coordinate grids on correct device
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
            
            # Blend in frequency domain
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
        """Stochastic blending with device safety"""
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
        """Merge two ComfyUI models using various interpolation methods"""
        
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
                            # DEVICE FIX: Ensure both tensors on same device
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
                            
                            # Ensure result is on correct device
                            merged_tensor = merged_tensor.to(device)
                            
                            # Only add patch if meaningful difference
                            diff = merged_tensor - t1
                            if torch.abs(diff).max() > 1e-8:
                                patches[key] = (diff.cpu(),)  # Store patches on CPU
                                
                        except Exception as e:
                            print(f"Warning: Failed to merge parameter {key} with method {method}: {e}")
                            continue
        
        if patches:
            merged_model.add_patches(patches, 1.0)
        
        return (merged_model,)


NODE_CLASS_MAPPINGS = {
    "TensorPrism_MainMerge": TensorPrism_MainMerge
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TensorPrism_MainMerge": "Main Merge (Tensor Prism)"
}