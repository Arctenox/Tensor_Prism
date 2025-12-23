import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import folder_paths
import comfy.sd
import comfy.utils
import comfy.model_management as mm
import os
import gc
import json

class TensorPrism_AnalyzeModelWeights:
    """
    Step 1: Analyze two models and calculate optimal per-block weights
    Memory efficient with device safety
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_a": ("MODEL",),
                "model_b": ("MODEL",),
                "optimization_method": (["combined", "similarity", "variance", "gradient_magnitude", "entropy"],),
                "global_alpha": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Starting point for optimization"
                }),
                "optimization_strength": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "How much to trust automated optimization"
                }),
                "target_mean_weight": ("FLOAT", {
                    "default": 0.5,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Target mean weight across all blocks (-1 = auto-detect)"
                }),
                "target_std_dev": ("FLOAT", {
                    "default": -1.0,
                    "min": -1.0,
                    "max": 0.5,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Target standard deviation (-1 = auto, 0 = uniform, higher = more variation)"
                }),
                "smooth_weights": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Apply smoothing to reduce weight variance"
                }),
                "smoothing_strength": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "How much to smooth (0=none, 1=completely flatten)"
                }),
            },
            "optional": {
                "use_model_a_as_base": ("BOOLEAN", {"default": True}),
                "fp16_mode": ("BOOLEAN", {"default": True}),
                "verbose": ("BOOLEAN", {"default": True}),
                "auto_detect_optimal": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Auto-detect optimal mean weight based on model similarity"
                }),
            }
        }
    
    RETURN_TYPES = ("MERGE_RECIPE",)
    RETURN_NAMES = ("merge_recipe",)
    FUNCTION = "analyze_models"
    CATEGORY = "Tensor_Prism/Advanced"

    def __init__(self):
        self.device = mm.get_torch_device()

    def clear_memory(self):
        """Aggressive memory cleanup"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()

    def calculate_tensor_similarity(self, tensor_a, tensor_b):
        """Memory-efficient similarity with device safety - ALL CPU"""
        # DEVICE FIX: Everything on CPU
        a_cpu = tensor_a.detach().cpu().float()
        b_cpu = tensor_b.detach().cpu().float()
        
        flat_a = a_cpu.flatten()
        flat_b = b_cpu.flatten()
        
        if len(flat_a) == 0:
            return 0.5
        
        if len(flat_a) > 1000000:
            indices = torch.randperm(len(flat_a))[:1000000]
            flat_a = flat_a[indices]
            flat_b = flat_b[indices]
        
        dot_product = torch.dot(flat_a, flat_b)
        norm_a = torch.norm(flat_a)
        norm_b = torch.norm(flat_b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.5
        
        similarity = (dot_product / (norm_a * norm_b)).item()
        return (similarity + 1) / 2

    def calculate_variance_score(self, tensor_a, tensor_b):
        """Variance with device safety - ALL CPU"""
        # DEVICE FIX: Everything on CPU
        a_cpu = tensor_a.detach().cpu().float()
        b_cpu = tensor_b.detach().cpu().float()
        
        if a_cpu.numel() > 1000000:
            flat_a = a_cpu.flatten()
            flat_b = b_cpu.flatten()
            indices = torch.randperm(len(flat_a))[:1000000]
            a_cpu = flat_a[indices]
            b_cpu = flat_b[indices]
        
        var_a = torch.var(a_cpu).item()
        var_b = torch.var(b_cpu).item()
        
        total_var = var_a + var_b
        return 0.5 if total_var == 0 else var_a / total_var

    def calculate_gradient_magnitude(self, tensor_a, tensor_b):
        """Magnitude comparison with device safety - ALL CPU"""
        # DEVICE FIX: Everything on CPU
        a_cpu = tensor_a.detach().cpu().float()
        b_cpu = tensor_b.detach().cpu().float()
        
        mag_a = torch.norm(a_cpu.flatten()[:1000000] if a_cpu.numel() > 1000000 else a_cpu).item()
        mag_b = torch.norm(b_cpu.flatten()[:1000000] if b_cpu.numel() > 1000000 else b_cpu).item()
        
        total_mag = mag_a + mag_b
        return 0.5 if total_mag == 0 else mag_a / total_mag

    def calculate_entropy_score(self, tensor_a, tensor_b):
        """Entropy-based score with device safety - ALL CPU"""
        # DEVICE FIX: Everything on CPU
        a_cpu = tensor_a.detach().cpu().float()
        b_cpu = tensor_b.detach().cpu().float()
        
        flat_a = a_cpu.flatten()
        flat_b = b_cpu.flatten()
        
        if len(flat_a) > 1000000:
            indices = torch.randperm(len(flat_a))[:1000000]
            flat_a = flat_a[indices]
            flat_b = flat_b[indices]
        
        abs_a = torch.abs(flat_a)
        abs_b = torch.abs(flat_b)
        
        prob_a = abs_a / abs_a.sum() if abs_a.sum() > 0 else torch.ones_like(abs_a) / len(abs_a)
        prob_b = abs_b / abs_b.sum() if abs_b.sum() > 0 else torch.ones_like(abs_b) / len(abs_b)
        
        eps = 1e-10
        kl_div = torch.sum(prob_a * torch.log((prob_a + eps) / (prob_b + eps))).item()
        
        score = 0.5
        if kl_div > 0.1:
            score = 0.5 - min(kl_div / 10, 0.3)
        
        return max(0.1, min(0.9, score))

    def identify_blocks(self, state_dict):
        """Identify SDXL model blocks"""
        blocks = {
            "conditioner": {},
            "first_stage": {},
            "input_blocks": {},
            "middle_block": {},
            "output_blocks": {},
            "time_embed": {},
            "label_emb": {},
            "out": {}
        }
        
        for key in state_dict.keys():
            if "conditioner" in key or "cond_stage_model" in key:
                blocks["conditioner"].setdefault("conditioner", []).append(key)
            elif "first_stage_model" in key:
                blocks["first_stage"].setdefault("vae", []).append(key)
            elif "input_blocks" in key:
                block_num = key.split(".")[1] if len(key.split(".")) > 1 else "0"
                blocks["input_blocks"].setdefault(f"input_{block_num}", []).append(key)
            elif "middle_block" in key:
                blocks["middle_block"].setdefault("middle", []).append(key)
            elif "output_blocks" in key:
                block_num = key.split(".")[1] if len(key.split(".")) > 1 else "0"
                blocks["output_blocks"].setdefault(f"output_{block_num}", []).append(key)
            elif "time_embed" in key:
                blocks["time_embed"].setdefault("time", []).append(key)
            elif "label_emb" in key:
                blocks["label_emb"].setdefault("label", []).append(key)
            elif "out." in key:
                blocks["out"].setdefault("out", []).append(key)
        
        return blocks

    def optimize_block_weight(self, block_tensors_a, block_tensors_b, 
                              optimization_method, fp16_mode=True):
        """Calculate optimal weight with device safety"""
        scores = []
        
        for key in list(block_tensors_a.keys())[:10]:
            if key not in block_tensors_b:
                continue
            
            tensor_a = block_tensors_a[key]
            tensor_b = block_tensors_b[key]
            
            # DEVICE FIX: Convert to CPU early for fp16 operations
            if fp16_mode and tensor_a.dtype == torch.float32:
                tensor_a = tensor_a.cpu().half()
                tensor_b = tensor_b.cpu().half()
            
            try:
                if optimization_method == "similarity":
                    sim = self.calculate_tensor_similarity(tensor_a, tensor_b)
                    score = 0.5 + (0.5 - sim) * 0.5
                elif optimization_method == "variance":
                    score = self.calculate_variance_score(tensor_a, tensor_b)
                elif optimization_method == "gradient_magnitude":
                    score = self.calculate_gradient_magnitude(tensor_a, tensor_b)
                elif optimization_method == "entropy":
                    score = self.calculate_entropy_score(tensor_a, tensor_b)
                elif optimization_method == "combined":
                    sim_score = self.calculate_tensor_similarity(tensor_a, tensor_b)
                    var_score = self.calculate_variance_score(tensor_a, tensor_b)
                    mag_score = self.calculate_gradient_magnitude(tensor_a, tensor_b)
                    ent_score = self.calculate_entropy_score(tensor_a, tensor_b)
                    score = (sim_score * 0.2 + var_score * 0.3 + mag_score * 0.3 + ent_score * 0.2)
                
                scores.append(score)
            except Exception as e:
                print(f"Warning: {e}")
                continue
            
            self.clear_memory()
        
        return float(np.median(scores)) if scores else 0.5

    def analyze_models(self, model_a, model_b, optimization_method, global_alpha,
                      optimization_strength, target_mean_weight, target_std_dev,
                      smooth_weights, smoothing_strength, use_model_a_as_base=True, 
                      fp16_mode=True, verbose=True, auto_detect_optimal=False):
        
        print("=" * 60)
        print("🔍 ANALYZING MODELS FOR OPTIMAL MERGE")
        print("=" * 60)
        
        a_sd = model_a.model.state_dict()
        b_sd = model_b.model.state_dict()
        
        blocks = self.identify_blocks(a_sd)
        
        if verbose:
            total_blocks = sum(len(blocks[cat]) for cat in blocks)
            print(f"\n📊 Found {total_blocks} blocks to optimize")
        
        block_weights = {}
        
        if verbose:
            print(f"\n🎯 Optimizing with '{optimization_method}' method...")
        
        total = sum(len(blocks[cat]) for cat in blocks)
        current = 0
        
        for category in blocks:
            for block_id, keys in blocks[category].items():
                current += 1
                full_block_id = f"{category}_{block_id}"
                
                if verbose and current % 5 == 0:
                    print(f"  Progress: {current}/{total} blocks...")
                
                block_a = {k: a_sd[k] for k in keys if k in a_sd}
                block_b = {k: b_sd[k] for k in keys if k in b_sd}
                
                optimal_alpha = self.optimize_block_weight(
                    block_a, block_b, optimization_method, fp16_mode
                )
                
                final_alpha = (global_alpha * (1 - optimization_strength) + 
                              optimal_alpha * optimization_strength)
                final_alpha = max(0.0, min(1.0, final_alpha))
                
                block_weights[full_block_id] = final_alpha
                
                self.clear_memory()
        
        # Calculate statistics and apply adjustments
        weights_array = np.array(list(block_weights.values()))
        current_mean = weights_array.mean()
        current_std = weights_array.std()
        
        if verbose:
            print(f"\n📊 Initial statistics:")
            print(f"   Mean: {current_mean:.3f}")
            print(f"   Std Dev: {current_std:.3f}")
        
        # Auto-detect optimal mean weight if enabled
        if auto_detect_optimal or target_mean_weight < 0:
            if verbose:
                print(f"\n🤖 Auto-detecting optimal mean weight...")
            target_mean_weight = current_mean
            if verbose:
                print(f"   Auto-detected mean: {target_mean_weight:.3f}")
        
        # Apply weight smoothing
        if smooth_weights and smoothing_strength > 0:
            if verbose:
                print(f"\n🎚️ Applying weight smoothing (strength: {smoothing_strength:.2f})...")
            
            smoothed_weights = {}
            mean_weight = np.mean(list(block_weights.values()))
            
            for block_id, weight in block_weights.items():
                smoothed = weight * (1 - smoothing_strength) + mean_weight * smoothing_strength
                smoothed_weights[block_id] = smoothed
            
            block_weights = smoothed_weights
            weights_array = np.array(list(block_weights.values()))
            
            if verbose:
                print(f"   New std dev after smoothing: {weights_array.std():.3f}")
        
        # Apply target mean weight adjustment
        if abs(target_mean_weight - weights_array.mean()) > 0.001:
            offset = target_mean_weight - weights_array.mean()
            
            if verbose:
                print(f"\n🎯 Adjusting mean weight:")
                print(f"   Current: {weights_array.mean():.3f}")
                print(f"   Target: {target_mean_weight:.3f}")
                print(f"   Offset: {offset:+.3f}")
            
            adjusted_weights = {}
            for block_id, weight in block_weights.items():
                adjusted_weight = weight + offset
                adjusted_weights[block_id] = max(0.0, min(1.0, adjusted_weight))
            
            block_weights = adjusted_weights
            weights_array = np.array(list(block_weights.values()))
        
        # Apply target standard deviation adjustment
        if target_std_dev >= 0 and abs(target_std_dev - weights_array.std()) > 0.001:
            current_mean = weights_array.mean()
            current_std = weights_array.std()
            
            if verbose:
                print(f"\n📐 Adjusting standard deviation:")
                print(f"   Current std: {current_std:.3f}")
                print(f"   Target std: {target_std_dev:.3f}")
            
            if current_std > 0.001:
                adjusted_weights = {}
                for block_id, weight in block_weights.items():
                    deviation = weight - current_mean
                    scale_factor = target_std_dev / current_std
                    new_weight = current_mean + (deviation * scale_factor)
                    adjusted_weights[block_id] = max(0.0, min(1.0, new_weight))
                
                block_weights = adjusted_weights
                weights_array = np.array(list(block_weights.values()))
                
                if verbose:
                    print(f"   New std dev: {weights_array.std():.3f}")
        
        if verbose:
            print(f"\n✅ Analysis complete!")
            print(f"   Blocks optimized: {len(block_weights)}")
            print(f"   Final mean weight: {weights_array.mean():.3f}")
            print(f"   Final std dev: {weights_array.std():.3f}")
            print(f"   Range: [{weights_array.min():.3f}, {weights_array.max():.3f}]")
        
        # Create recipe
        recipe = {
            "block_weights": block_weights,
            "optimization_method": optimization_method,
            "global_alpha": global_alpha,
            "optimization_strength": optimization_strength,
            "target_mean_weight": target_mean_weight,
            "target_std_dev": target_std_dev,
            "smooth_weights": smooth_weights,
            "smoothing_strength": smoothing_strength,
            "auto_detect_optimal": auto_detect_optimal,
            "use_model_a_as_base": use_model_a_as_base,
            "stats": {
                "mean": float(weights_array.mean()),
                "std": float(weights_array.std()),
                "min": float(weights_array.min()),
                "max": float(weights_array.max()),
                "blocks": len(block_weights)
            }
        }
        
        return (recipe,)


class TensorPrism_ApplyMergeRecipe:
    """
    Step 2: Apply the merge recipe to two models with device safety
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_a": ("MODEL",),
                "model_b": ("MODEL",),
                "merge_recipe": ("MERGE_RECIPE",),
                "merge_method": (["weighted_sum", "add_difference"],),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05
                }),
            },
            "optional": {
                "fp16_mode": ("BOOLEAN", {"default": True}),
                "verbose": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("merged_model",)
    FUNCTION = "apply_merge"
    CATEGORY = "Tensor_Prism/Advanced"

    def __init__(self):
        self.device = mm.get_torch_device()

    def clear_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()

    def identify_blocks(self, state_dict):
        """Same as analyzer"""
        blocks = {
            "conditioner": {},
            "first_stage": {},
            "input_blocks": {},
            "middle_block": {},
            "output_blocks": {},
            "time_embed": {},
            "label_emb": {},
            "out": {}
        }
        
        for key in state_dict.keys():
            if "conditioner" in key or "cond_stage_model" in key:
                blocks["conditioner"].setdefault("conditioner", []).append(key)
            elif "first_stage_model" in key:
                blocks["first_stage"].setdefault("vae", []).append(key)
            elif "input_blocks" in key:
                block_num = key.split(".")[1] if len(key.split(".")) > 1 else "0"
                blocks["input_blocks"].setdefault(f"input_{block_num}", []).append(key)
            elif "middle_block" in key:
                blocks["middle_block"].setdefault("middle", []).append(key)
            elif "output_blocks" in key:
                block_num = key.split(".")[1] if len(key.split(".")) > 1 else "0"
                blocks["output_blocks"].setdefault(f"output_{block_num}", []).append(key)
            elif "time_embed" in key:
                blocks["time_embed"].setdefault("time", []).append(key)
            elif "label_emb" in key:
                blocks["label_emb"].setdefault("label", []).append(key)
            elif "out." in key:
                blocks["out"].setdefault("out", []).append(key)
        
        return blocks

    def apply_merge(self, model_a, model_b, merge_recipe, merge_method, 
                   strength, fp16_mode=True, verbose=True):
        
        print("=" * 60)
        print("🚀 APPLYING MERGE RECIPE")
        print("=" * 60)
        
        block_weights = merge_recipe["block_weights"]
        
        if verbose:
            print(f"\n📋 Recipe stats:")
            print(f"   Blocks: {merge_recipe['stats']['blocks']}")
            print(f"   Mean weight: {merge_recipe['stats']['mean']:.3f}")
            print(f"   Std dev: {merge_recipe['stats']['std']:.3f}")
            print(f"   Method: {merge_recipe['optimization_method']}")
        
        # Create patches with device safety
        merged_model = model_a.clone()
        
        a_sd = model_a.model.state_dict()
        b_sd = model_b.model.state_dict()
        
        blocks = self.identify_blocks(a_sd)
        
        patches = {}
        processed = 0
        
        if verbose:
            print("\n🔧 Applying merge recipe...")
        
        for category in blocks:
            for block_id, keys in blocks[category].items():
                full_block_id = f"{category}_{block_id}"
                alpha = block_weights.get(full_block_id, 0.5)
                
                for key in keys:
                    if key not in a_sd or key not in b_sd:
                        continue
                    
                    try:
                        # DEVICE FIX: Keep on same device throughout
                        device = a_sd[key].device
                        tensor_a = a_sd[key]
                        tensor_b = b_sd[key].to(device)  # Ensure B on same device
                        
                        if merge_method == "weighted_sum":
                            # ALL operations on same device
                            merged_tensor = tensor_a * alpha + tensor_b * (1 - alpha)
                        
                        elif merge_method == "add_difference":
                            # ALL operations on same device
                            delta = tensor_b - tensor_a
                            merged_tensor = tensor_a + delta * alpha * strength
                        
                        else:
                            merged_tensor = tensor_a * alpha + tensor_b * (1 - alpha)
                        
                        # Ensure result on correct device before diff
                        merged_tensor = merged_tensor.to(device)
                        
                        # Create patch - diff on same device, then move to CPU
                        diff = (merged_tensor - tensor_a).to(device)
                        if torch.abs(diff).max() > 1e-8:
                            patches[key] = (diff.cpu(),)  # DEVICE FIX: CPU for storage
                            processed += 1
                        
                    except Exception as e:
                        if verbose:
                            print(f"Warning: Failed to merge {key}: {e}")
                        continue
                
                if processed % 100 == 0:
                    self.clear_memory()
        
        if patches:
            merged_model.add_patches(patches, 1.0)
        
        if verbose:
            print(f"\n✅ Applied {len(patches)} patches")
            print("=" * 60)
        
        return (merged_model,)


NODE_CLASS_MAPPINGS = {
    "TensorPrism_AnalyzeModelWeights": TensorPrism_AnalyzeModelWeights,
    "TensorPrism_ApplyMergeRecipe": TensorPrism_ApplyMergeRecipe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TensorPrism_AnalyzeModelWeights": "Analyze Model Weights (Tensor Prism)",
    "TensorPrism_ApplyMergeRecipe": "Apply Merge Recipe (Tensor Prism)",
}