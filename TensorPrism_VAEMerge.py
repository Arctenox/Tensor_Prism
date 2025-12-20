import os
import folder_paths
from tqdm import tqdm
import torch
import torch.nn.functional as F
import safetensors.torch
import comfy.sd
import comfy.utils
import comfy.model_management as mm
from torch import nn
import gc
import numpy as np

# Advanced merge methods for VAE
class VAEMergeMethods:
    """Production-tested VAE merge methods with device handling"""
    
    @staticmethod
    def weighted_sum(a, b, alpha):
        """Standard interpolation: a * alpha + b * (1-alpha)"""
        # DEVICE FIX: Ensure same device
        device = a.device
        b = b.to(device)
        return a * alpha + b * (1 - alpha)
    
    @staticmethod
    def slerp(a, b, alpha):
        """Spherical Linear Interpolation - smooth blending"""
        # DEVICE FIX: Ensure same device
        device = a.device
        b = b.to(device)
        
        # Normalize
        a_norm = a / (torch.norm(a, dim=-1, keepdim=True) + 1e-8)
        b_norm = b / (torch.norm(b, dim=-1, keepdim=True) + 1e-8)
        
        # Compute angle
        dot = (a_norm * b_norm).sum(dim=-1, keepdim=True)
        dot = torch.clamp(dot, -1.0, 1.0)
        theta = torch.acos(dot)
        
        # Slerp interpolation
        sin_theta = torch.sin(theta)
        weight_a = torch.sin((1 - alpha) * theta) / (sin_theta + 1e-8)
        weight_b = torch.sin(alpha * theta) / (sin_theta + 1e-8)
        
        return weight_a * a + weight_b * b
    
    @staticmethod
    def gradient_merge(a, b, alpha):
        """Merge based on gradient magnitude - keeps stronger features"""
        # DEVICE FIX: Ensure same device
        device = a.device
        b = b.to(device)
        
        grad_a = torch.abs(a)
        grad_b = torch.abs(b)
        
        # Weight by gradient strength
        total_grad = grad_a + grad_b + 1e-8
        weight_a = grad_a / total_grad
        weight_b = grad_b / total_grad
        
        # Apply alpha blending to weights
        final_weight_a = weight_a * alpha + (1 - alpha) * 0.5
        final_weight_b = weight_b * (1 - alpha) + alpha * 0.5
        
        return a * final_weight_a + b * final_weight_b
    
    @staticmethod
    def adaptive_merge(a, b, alpha):
        """Adaptively merge based on tensor statistics"""
        # DEVICE FIX: Ensure same device
        device = a.device
        b = b.to(device)
        
        # Calculate local importance
        var_a = torch.var(a, dim=-1, keepdim=True)
        var_b = torch.var(b, dim=-1, keepdim=True)
        
        # Higher variance = more important
        total_var = var_a + var_b + 1e-8
        adaptive_alpha = (var_a / total_var) * alpha + (1 - alpha) * 0.5
        
        return a * adaptive_alpha + b * (1 - adaptive_alpha)
    
    @staticmethod
    def harmonic_merge(a, b, alpha):
        """Harmonic mean merge - good for preserving details"""
        # DEVICE FIX: Ensure same device
        device = a.device
        b = b.to(device)
        
        eps = 1e-8
        sign_a = torch.sign(a)
        sign_b = torch.sign(b)
        
        abs_a = torch.abs(a) + eps
        abs_b = torch.abs(b) + eps
        
        # Harmonic mean
        harmonic = 2 * (abs_a * abs_b) / (abs_a + abs_b)
        
        # Preserve signs with alpha blending
        sign = sign_a * alpha + sign_b * (1 - alpha)
        
        return harmonic * sign
    
    @staticmethod
    def tensor_sum(a, b, alpha, beta):
        """Additive merge: a * alpha + b * beta"""
        # DEVICE FIX: Ensure same device
        device = a.device
        b = b.to(device)
        return a * alpha + b * beta
    
    @staticmethod
    def geometric_mean(a, b, alpha):
        """Geometric mean merge - multiplicative blending"""
        # DEVICE FIX: Ensure same device
        device = a.device
        b = b.to(device)
        
        eps = 1e-8
        sign_a = torch.sign(a)
        sign_b = torch.sign(b)
        
        abs_a = torch.abs(a) + eps
        abs_b = torch.abs(b) + eps
        
        # Geometric mean
        geometric = torch.pow(abs_a, alpha) * torch.pow(abs_b, 1 - alpha)
        
        # Preserve signs
        sign = torch.sign(sign_a * alpha + sign_b * (1 - alpha))
        
        return geometric * sign
    
    @staticmethod
    def cosine_similarity_merge(a, b, alpha, threshold=0.7):
        """Merge based on cosine similarity - blends similar regions more"""
        # DEVICE FIX: Ensure same device
        device = a.device
        b = b.to(device)
        
        # Flatten for similarity calculation
        a_flat = a.flatten()
        b_flat = b.flatten()
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0))
        similarity_value = similarity.item()
        
        # Adjust alpha based on similarity
        if similarity_value > threshold:
            adjusted_alpha = 0.5
        else:
            adjusted_alpha = alpha
        
        return a * adjusted_alpha + b * (1 - adjusted_alpha)
    
    @staticmethod
    def percentile_merge(a, b, alpha, percentile=50):
        """Merge using percentile-based weighting"""
        # DEVICE FIX: Ensure same device
        device = a.device
        b = b.to(device)
        
        # Get percentile values
        perc_a = torch.quantile(torch.abs(a.flatten()), percentile / 100.0)
        perc_b = torch.quantile(torch.abs(b.flatten()), percentile / 100.0)
        
        # Create masks for values above percentile
        mask_a = torch.abs(a) > perc_a
        mask_b = torch.abs(b) > perc_b
        
        # Weighted merge based on masks
        result = torch.where(mask_a, a * alpha, b * (1 - alpha))
        result = torch.where(mask_b, result * 0.7 + b * 0.3, result)
        
        return result
    
    @staticmethod
    def smooth_step_merge(a, b, alpha):
        """Smooth step interpolation - smoother than linear"""
        # DEVICE FIX: Ensure same device
        device = a.device
        b = b.to(device)
        
        # Smooth step function: 3t^2 - 2t^3
        smooth_alpha = 3 * alpha**2 - 2 * alpha**3
        return a * smooth_alpha + b * (1 - smooth_alpha)


class TensorPrismVAEMerge:
    """
    Advanced VAE merging for Tensor Prism
    Production-tested methods with per-block control and device safety
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "merge_mode": ([
                    "weighted_sum",
                    "slerp",
                    "gradient_merge",
                    "adaptive_merge",
                    "harmonic_merge",
                    "tensor_sum",
                    "geometric_mean",
                    "cosine_similarity_merge",
                    "percentile_merge",
                    "smooth_step_merge",
                ],),
                "device": (["auto", "cuda", "cpu"],),
                "alpha": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Weight for VAE A (remember: 0.50 = balanced)"
                }),
                "beta": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Secondary weight (for tensor_sum mode)"
                }),
                "brightness": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Adjust output brightness"
                }),
                "contrast": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Adjust output contrast"
                }),
                "saturation": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Adjust color saturation"
                }),
                "use_block_weights": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable per-block weight control (remember 0.50 default)"
                }),
            },
            "optional": {
                "vae_a": ("VAE",),
                "vae_b": ("VAE",),
                # Encoder blocks
                "encoder_conv_in": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "encoder_block_0": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "encoder_block_1": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "encoder_block_2": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "encoder_block_3": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "encoder_mid": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "encoder_norm_out": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "encoder_conv_out": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                # Decoder blocks
                "decoder_conv_in": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "decoder_block_0": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "decoder_block_1": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "decoder_block_2": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "decoder_block_3": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "decoder_mid": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "decoder_norm_out": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "decoder_conv_out": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                # Quantization
                "quant_conv": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "post_quant_conv": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("VAE", "STRING")
    RETURN_NAMES = ("merged_vae", "merge_info")
    FUNCTION = "merge_vaes"
    CATEGORY = "TensorPrism/VAE"

    def __init__(self):
        self.merge_methods = VAEMergeMethods()

    def clear_memory(self):
        """Memory cleanup"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()

    def get_block_weight(self, key, block_weights, global_alpha):
        """Get weight for specific block based on key"""
        if not block_weights:
            return global_alpha
        
        # Match key to block weights
        for block_key, weight in block_weights.items():
            if block_key in key:
                return weight
        
        return global_alpha

    def determine_target_device(self, device_setting, sd_a, sd_b):
        """Intelligently determine target device based on where VAEs are loaded"""
        if device_setting == "cpu":
            return torch.device("cpu")
        
        # Check where the state dicts actually are
        device_a = None
        device_b = None
        
        # Sample a tensor from each to determine device
        for key in list(sd_a.keys())[:5]:  # Check first 5 tensors
            if isinstance(sd_a[key], torch.Tensor):
                device_a = sd_a[key].device
                break
        
        for key in list(sd_b.keys())[:5]:
            if isinstance(sd_b[key], torch.Tensor):
                device_b = sd_b[key].device
                break
        
        # Decision logic
        if device_setting == "cuda":
            if not torch.cuda.is_available():
                print("⚠️ CUDA requested but not available, using CPU")
                return torch.device("cpu")
            return torch.device("cuda")
        
        # Auto mode - intelligent selection
        if device_a and device_b:
            # Both on same device
            if device_a.type == device_b.type:
                if device_a.type == "cuda" and torch.cuda.is_available():
                    return torch.device("cuda")
                return torch.device("cpu")
            
            # Mixed devices - prefer CUDA if available
            if torch.cuda.is_available():
                print(f"🔀 VAEs on different devices (A: {device_a}, B: {device_b}), using CUDA for merge")
                return torch.device("cuda")
            else:
                print(f"🔀 VAEs on different devices, using CPU for merge")
                return torch.device("cpu")
        
        # Fallback
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def merge_state_dicts(self, sd_a, sd_b, alpha, beta, block_weights, mode, target_device):
        """Merge two VAE state dicts with device safety"""
        
        common_keys = set(sd_a.keys()) & set(sd_b.keys())
        
        if len(common_keys) == 0:
            raise ValueError("VAEs have no common keys - incompatible models")
        
        merged_sd = {}
        
        print(f"🔧 Merging {len(common_keys)} layers with '{mode}' method on {target_device}")
        
        for key in tqdm(common_keys, desc="Merging VAE"):
            current_alpha = self.get_block_weight(key, block_weights, alpha)
            
            tensor_a = sd_a[key]
            tensor_b = sd_b[key]
            
            # DEVICE FIX: Move both tensors to target device
            if isinstance(tensor_a, torch.Tensor):
                tensor_a = tensor_a.to(target_device)
            if isinstance(tensor_b, torch.Tensor):
                tensor_b = tensor_b.to(target_device)
            
            try:
                # Apply merge method - methods handle their own device safety
                if mode == "weighted_sum":
                    merged_sd[key] = self.merge_methods.weighted_sum(tensor_a, tensor_b, current_alpha)
                
                elif mode == "slerp":
                    merged_sd[key] = self.merge_methods.slerp(tensor_a, tensor_b, current_alpha)
                
                elif mode == "gradient_merge":
                    merged_sd[key] = self.merge_methods.gradient_merge(tensor_a, tensor_b, current_alpha)
                
                elif mode == "adaptive_merge":
                    merged_sd[key] = self.merge_methods.adaptive_merge(tensor_a, tensor_b, current_alpha)
                
                elif mode == "harmonic_merge":
                    merged_sd[key] = self.merge_methods.harmonic_merge(tensor_a, tensor_b, current_alpha)
                
                elif mode == "tensor_sum":
                    merged_sd[key] = self.merge_methods.tensor_sum(tensor_a, tensor_b, current_alpha, beta)
                
                elif mode == "geometric_mean":
                    merged_sd[key] = self.merge_methods.geometric_mean(tensor_a, tensor_b, current_alpha)
                
                elif mode == "cosine_similarity_merge":
                    merged_sd[key] = self.merge_methods.cosine_similarity_merge(tensor_a, tensor_b, current_alpha)
                
                elif mode == "percentile_merge":
                    merged_sd[key] = self.merge_methods.percentile_merge(tensor_a, tensor_b, current_alpha)
                
                elif mode == "smooth_step_merge":
                    merged_sd[key] = self.merge_methods.smooth_step_merge(tensor_a, tensor_b, current_alpha)
                
                else:
                    merged_sd[key] = self.merge_methods.weighted_sum(tensor_a, tensor_b, current_alpha)
            
            except Exception as e:
                print(f"⚠️ Failed to merge {key}: {e}, using weighted_sum fallback")
                merged_sd[key] = self.merge_methods.weighted_sum(tensor_a, tensor_b, current_alpha)
        
        # Copy remaining keys from A
        for key in sd_a.keys():
            if key not in merged_sd:
                tensor = sd_a[key]
                if isinstance(tensor, torch.Tensor):
                    merged_sd[key] = tensor.to(target_device)
                else:
                    merged_sd[key] = tensor
        
        return merged_sd

    def apply_color_adjustments(self, state_dict, brightness, contrast, saturation):
        """Apply color adjustments to decoder output"""
        
        # Brightness adjustment
        if "decoder.conv_out.bias" in state_dict and brightness != 0.0:
            bias = state_dict["decoder.conv_out.bias"]
            if isinstance(bias, nn.Parameter):
                bias = bias.data
            state_dict["decoder.conv_out.bias"] = bias + brightness
        
        # Contrast adjustment
        if "decoder.conv_out.weight" in state_dict and contrast != 0.0:
            weight = state_dict["decoder.conv_out.weight"]
            if isinstance(weight, nn.Parameter):
                weight = weight.data
            state_dict["decoder.conv_out.weight"] = weight * (1.0 + contrast / 10.0)
        
        # Saturation adjustment
        if "decoder.conv_out.weight" in state_dict and saturation != 0.0:
            weight = state_dict["decoder.conv_out.weight"]
            if isinstance(weight, nn.Parameter):
                weight = weight.data.clone()
            else:
                weight = weight.clone()
            
            if weight.shape[0] >= 3:
                sat_factor = 1.0 + saturation
                weight[:3] = weight[:3] * sat_factor
                state_dict["decoder.conv_out.weight"] = weight
        
        return state_dict

    def merge_vaes(self, merge_mode, device, alpha, beta, brightness, contrast, saturation,
                   use_block_weights, vae_a=None, vae_b=None, **kwargs):
        
        print("=" * 70)
        print("🎨 TENSOR PRISM VAE MERGE")
        print("=" * 70)
        
        # Validate inputs
        if vae_a is None and vae_b is None:
            raise ValueError("At least one VAE (A or B) must be provided")
        
        # Single VAE pass-through
        if vae_a is None:
            print("⚠️  Only VAE B provided, returning unchanged")
            return (vae_b, "Single VAE pass-through (VAE B)")
        
        if vae_b is None:
            print("⚠️  Only VAE A provided, returning unchanged")
            return (vae_a, "Single VAE pass-through (VAE A)")
        
        # Get state dicts
        sd_a = vae_a.first_stage_model.state_dict()
        sd_b = vae_b.first_stage_model.state_dict()
        
        # Intelligent device selection
        target_device = self.determine_target_device(device, sd_a, sd_b)
        
        print(f"\n⚙️  Configuration:")
        print(f"   Target Device: {target_device}")
        print(f"   Mode: {merge_mode}")
        print(f"   Alpha: {alpha:.2f} (remember: 0.50 = balanced)")
        print(f"   Beta: {beta:.2f}")
        print(f"   Brightness: {brightness:+.2f}")
        print(f"   Contrast: {contrast:+.2f}")
        print(f"   Saturation: {saturation:+.2f}")
        print(f"   Block weights: {'Enabled' if use_block_weights else 'Disabled'}")
        
        # Build block weights
        block_weights = None
        if use_block_weights:
            block_weights = {
                'encoder.conv_in': kwargs.get('encoder_conv_in', 0.50),
                'encoder.down.0': kwargs.get('encoder_block_0', 0.50),
                'encoder.down.1': kwargs.get('encoder_block_1', 0.50),
                'encoder.down.2': kwargs.get('encoder_block_2', 0.50),
                'encoder.down.3': kwargs.get('encoder_block_3', 0.50),
                'encoder.mid': kwargs.get('encoder_mid', 0.50),
                'encoder.norm_out': kwargs.get('encoder_norm_out', 0.50),
                'encoder.conv_out': kwargs.get('encoder_conv_out', 0.50),
                'decoder.conv_in': kwargs.get('decoder_conv_in', 0.50),
                'decoder.up.0': kwargs.get('decoder_block_0', 0.50),
                'decoder.up.1': kwargs.get('decoder_block_1', 0.50),
                'decoder.up.2': kwargs.get('decoder_block_2', 0.50),
                'decoder.up.3': kwargs.get('decoder_block_3', 0.50),
                'decoder.mid': kwargs.get('decoder_mid', 0.50),
                'decoder.norm_out': kwargs.get('decoder_norm_out', 0.50),
                'decoder.conv_out': kwargs.get('decoder_conv_out', 0.50),
                'quant_conv': kwargs.get('quant_conv', 0.50),
                'post_quant_conv': kwargs.get('post_quant_conv', 0.50)
            }
            
            print(f"   Per-block weights: Configured (default 0.50)")
        
        # Merge
        print(f"\n🔧 Starting merge...")
        merged_sd = self.merge_state_dicts(sd_a, sd_b, alpha, beta, block_weights, merge_mode, target_device)
        
        # Apply color adjustments
        if brightness != 0.0 or contrast != 0.0 or saturation != 0.0:
            print(f"🎨 Applying color adjustments...")
            merged_sd = self.apply_color_adjustments(merged_sd, brightness, contrast, saturation)
        
        # Create VAE
        print(f"📦 Creating merged VAE...")
        merged_vae = comfy.sd.VAE(sd=merged_sd)
        
        # Generate info
        info = f"VAE Merge Complete\n"
        info += f"{'='*50}\n"
        info += f"Device: {target_device}\n"
        info += f"Method: {merge_mode}\n"
        info += f"Alpha: {alpha:.3f} (0.50 = balanced)\n"
        info += f"Beta: {beta:.3f}\n"
        info += f"Color: B={brightness:+.2f}, C={contrast:+.2f}, S={saturation:+.2f}\n"
        info += f"Layers merged: {len(merged_sd)}\n"
        
        # Cleanup
        self.clear_memory()
        
        print(f"\n✅ VAE merge complete!")
        print("=" * 70)
        
        return (merged_vae, info)


NODE_CLASS_MAPPINGS = {
    "TensorPrismVAEMerge": TensorPrismVAEMerge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TensorPrismVAEMerge": "VAE Merge (Tensor Prism)",
}