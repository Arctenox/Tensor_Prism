"""
TensorPrism Intelligent Tensor Selector
=======================================

Advanced tensor-by-tensor selection that evaluates multiple models and chooses
the best tensor based on quality metrics and configurable weights.

Inspired by Hyphoria's merge methodology - evaluates each tensor individually
and selects the best one based on multiple quality criteria.

Author: Arctenox
Version: 1.0.1
License: GPL-3.0
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import gc
import comfy.model_management

class TensorPrism_IntelligentTensorSelector:
    """
    Evaluates tensors from multiple models and intelligently selects the best
    ones based on configurable quality metrics and weights.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_A": ("MODEL",),
                "model_B": ("MODEL",),
                "selection_mode": ([
                    "best_quality",
                    "balanced", 
                    "conservative",
                    "aggressive"
                ], {"default": "best_quality"}),
                
                # Quality metric weights
                "magnitude_weight": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Weight for tensor magnitude (strength)"
                }),
                "variance_weight": ("FLOAT", {
                    "default": 0.25, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Weight for tensor variance (detail preservation)"
                }),
                "spectral_weight": ("FLOAT", {
                    "default": 0.25, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Weight for spectral content (frequency richness)"
                }),
                "stability_weight": ("FLOAT", {
                    "default": 0.2, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Weight for numerical stability"
                }),
                
                # Bias controls
                "model_A_bias": ("FLOAT", {
                    "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Bias toward Model A (-1 = strong B preference, +1 = strong A preference)"
                }),
                "attention_preference": ("FLOAT", {
                    "default": 0.0, "min": -0.5, "max": 0.5, "step": 0.05,
                    "tooltip": "Extra weight for attention layers"
                }),
                "output_preference": ("FLOAT", {
                    "default": 0.0, "min": -0.5, "max": 0.5, "step": 0.05,
                    "tooltip": "Extra weight for output layers"
                }),
            },
            "optional": {
                "model_C": ("MODEL",),
                "model_D": ("MODEL",),
                "enable_block_analysis": ("BOOLEAN", {"default": True}),
                "memory_efficient": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("merged_model", "selection_report")
    FUNCTION = "select_best_tensors"
    CATEGORY = "Tensor_Prism/Advanced"
    
    def __init__(self):
        self.device = comfy.model_management.get_torch_device()
    
    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Safely move tensor to working device."""
        if tensor.device != self.device:
            return tensor.to(self.device)
        return tensor
    
    def _evaluate_tensor_quality(self, tensor: torch.Tensor, 
                                 magnitude_w: float, variance_w: float,
                                 spectral_w: float, stability_w: float) -> float:
        """
        Evaluate the quality of a tensor based on multiple metrics.
        Returns a quality score (higher is better).
        """
        try:
            # Move to working device
            tensor = self._to_device(tensor)
            
            # Magnitude score - strength of the tensor
            magnitude = torch.norm(tensor).item()
            magnitude_score = np.log1p(magnitude)  # Log scale to handle large values
            
            # Variance score - detail preservation
            variance = torch.var(tensor).item()
            variance_score = np.log1p(variance)
            
            # Spectral score - frequency content richness
            if tensor.numel() > 16 and tensor.dim() >= 2:
                try:
                    flat = tensor.flatten()[:1024]  # Limit for efficiency
                    fft = torch.fft.fft(flat.float())
                    spectral_energy = torch.abs(fft).mean().item()
                    spectral_score = np.log1p(spectral_energy)
                except:
                    spectral_score = 0.0
            else:
                spectral_score = 0.0
            
            # Stability score - inverse of numerical extremes
            max_val = torch.abs(tensor).max().item()
            if max_val > 1e-8:
                stability_score = 1.0 / (1.0 + np.log1p(max_val / (torch.abs(tensor).mean().item() + 1e-8)))
            else:
                stability_score = 1.0
            
            # Weighted combination
            total_score = (
                magnitude_score * magnitude_w +
                variance_score * variance_w +
                spectral_score * spectral_w +
                stability_score * stability_w
            )
            
            return total_score
            
        except Exception as e:
            print(f"Warning: Tensor evaluation failed: {e}")
            return 0.0
    
    def _get_layer_type(self, key: str) -> str:
        """Determine the type of layer from its key."""
        key_lower = key.lower()
        if any(term in key_lower for term in ['attn', 'attention']):
            return 'attention'
        elif any(term in key_lower for term in ['output_blocks', 'out.']):
            return 'output'
        elif any(term in key_lower for term in ['input_blocks']):
            return 'input'
        elif any(term in key_lower for term in ['middle_block']):
            return 'middle'
        else:
            return 'other'
    
    def _apply_layer_bias(self, base_score: float, layer_type: str,
                         attention_pref: float, output_pref: float) -> float:
        """Apply layer-specific bias to the quality score."""
        if layer_type == 'attention':
            return base_score * (1.0 + attention_pref)
        elif layer_type == 'output':
            return base_score * (1.0 + output_pref)
        return base_score
    
    def _select_best_model(self, models: List[Tuple[str, Dict]], key: str,
                          weights: Dict, biases: Dict) -> Tuple[str, torch.Tensor]:
        """
        Select the best tensor from available models for a specific key.
        Returns (model_name, selected_tensor).
        """
        if len(models) == 0:
            raise ValueError("No models provided")
        
        # Find models that have this key
        candidates = [(name, sd[key]) for name, sd in models if key in sd]
        
        if len(candidates) == 0:
            # Return from first model if key not in others
            return models[0][0], models[0][1][key]
        
        if len(candidates) == 1:
            return candidates[0][0], candidates[0][1]
        
        # Evaluate each candidate
        scores = []
        layer_type = self._get_layer_type(key)
        
        for name, tensor in candidates:
            # Base quality score
            quality = self._evaluate_tensor_quality(
                tensor,
                weights['magnitude'],
                weights['variance'],
                weights['spectral'],
                weights['stability']
            )
            
            # Apply layer-specific bias
            quality = self._apply_layer_bias(
                quality, layer_type,
                biases['attention'],
                biases['output']
            )
            
            # Apply model bias
            if name == 'A':
                quality *= (1.0 + biases['model_A'])
            elif name == 'B':
                quality *= (1.0 - biases['model_A'])
            
            scores.append((name, tensor, quality))
        
        # Select best
        best = max(scores, key=lambda x: x[2])
        return best[0], best[1]
    
    def select_best_tensors(self, model_A, model_B, selection_mode: str,
                           magnitude_weight: float, variance_weight: float,
                           spectral_weight: float, stability_weight: float,
                           model_A_bias: float, attention_preference: float,
                           output_preference: float,
                           model_C=None, model_D=None,
                           enable_block_analysis: bool = True,
                           memory_efficient: bool = True):
        """
        Main selection function - evaluates and selects best tensors.
        """
        
        print("\n" + "="*70)
        print("🎯 INTELLIGENT TENSOR SELECTOR (Tensor Prism)")
        print("="*70)
        
        # Prepare weights based on mode
        if selection_mode == "best_quality":
            pass  # Use provided weights
        elif selection_mode == "balanced":
            magnitude_weight = variance_weight = spectral_weight = stability_weight = 0.25
        elif selection_mode == "conservative":
            magnitude_weight *= 0.8
            stability_weight *= 1.5
        elif selection_mode == "aggressive":
            magnitude_weight *= 1.3
            variance_weight *= 1.2
        
        weights = {
            'magnitude': magnitude_weight,
            'variance': variance_weight,
            'spectral': spectral_weight,
            'stability': stability_weight
        }
        
        biases = {
            'model_A': model_A_bias,
            'attention': attention_preference,
            'output': output_preference
        }
        
        print(f"\n⚙️ Configuration:")
        print(f"   Selection Mode: {selection_mode}")
        print(f"   Weights - Mag: {magnitude_weight:.2f}, Var: {variance_weight:.2f}, "
              f"Spec: {spectral_weight:.2f}, Stab: {stability_weight:.2f}")
        print(f"   Model A Bias: {model_A_bias:+.2f}")
        
        # Collect models - keep state dicts in CPU to avoid memory issues
        models = [('A', model_A.model.state_dict()), ('B', model_B.model.state_dict())]
        if model_C is not None:
            models.append(('C', model_C.model.state_dict()))
        if model_D is not None:
            models.append(('D', model_D.model.state_dict()))
        
        print(f"   Evaluating {len(models)} models")
        
        # Get all keys from first model
        base_state_dict = models[0][1]
        all_keys = list(base_state_dict.keys())
        
        print(f"\n🔍 Processing {len(all_keys)} tensors...")
        
        # Selection tracking
        selection_stats = {name: 0 for name, _ in models}
        layer_stats = {
            'attention': {name: 0 for name, _ in models},
            'output': {name: 0 for name, _ in models},
            'input': {name: 0 for name, _ in models},
            'middle': {name: 0 for name, _ in models},
            'other': {name: 0 for name, _ in models}
        }
        
        # Create patches
        patches = {}
        processed = 0
        
        for key in all_keys:
            try:
                # Select best tensor
                selected_model, selected_tensor = self._select_best_model(
                    models, key, weights, biases
                )
                
                # Track selection
                selection_stats[selected_model] += 1
                layer_type = self._get_layer_type(key)
                layer_stats[layer_type][selected_model] += 1
                
                # Create patch if not from model A
                if selected_model != 'A':
                    original = base_state_dict[key]
                    # Ensure tensors are on same device before subtraction
                    selected_tensor = self._to_device(selected_tensor)
                    original = self._to_device(original)
                    diff = selected_tensor - original
                    
                    if torch.abs(diff).max() > 1e-8:
                        # Move diff back to CPU for patches
                        patches[key] = (diff.cpu(),)
                
                processed += 1
                
                # Memory cleanup
                if memory_efficient and processed % 100 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"⚠️ Warning: Failed to process {key}: {e}")
                continue
        
        # Apply patches
        print(f"\n📦 Applying {len(patches)} patches...")
        merged_model = model_A.clone()
        if patches:
            merged_model.add_patches(patches, 1.0)
        
        # Generate report
        report = self._generate_report(selection_stats, layer_stats, len(all_keys),
                                       selection_mode, weights, biases)
        
        print("\n✅ Selection complete!")
        print("="*70 + "\n")
        
        return (merged_model, report)
    
    def _generate_report(self, selection_stats: Dict, layer_stats: Dict,
                        total_tensors: int, mode: str, weights: Dict,
                        biases: Dict) -> str:
        """Generate detailed selection report."""
        
        report = "="*60 + "\n"
        report += "INTELLIGENT TENSOR SELECTION REPORT\n"
        report += "="*60 + "\n\n"
        
        report += f"Selection Mode: {mode}\n"
        report += f"Total Tensors Processed: {total_tensors}\n\n"
        
        report += "OVERALL SELECTION STATISTICS:\n"
        report += "-"*40 + "\n"
        for model, count in sorted(selection_stats.items()):
            percentage = (count / total_tensors * 100) if total_tensors > 0 else 0
            report += f"Model {model}: {count} tensors ({percentage:.1f}%)\n"
        
        report += "\n\nLAYER-SPECIFIC BREAKDOWN:\n"
        report += "-"*40 + "\n"
        for layer_type, stats in layer_stats.items():
            layer_total = sum(stats.values())
            if layer_total > 0:
                report += f"\n{layer_type.upper()}:\n"
                for model, count in sorted(stats.items()):
                    percentage = (count / layer_total * 100) if layer_total > 0 else 0
                    report += f"  Model {model}: {count} ({percentage:.1f}%)\n"
        
        report += "\n\nQUALITY METRIC WEIGHTS:\n"
        report += "-"*40 + "\n"
        report += f"Magnitude:  {weights['magnitude']:.2f}\n"
        report += f"Variance:   {weights['variance']:.2f}\n"
        report += f"Spectral:   {weights['spectral']:.2f}\n"
        report += f"Stability:  {weights['stability']:.2f}\n"
        
        report += "\n\nBIAS SETTINGS:\n"
        report += "-"*40 + "\n"
        report += f"Model A Bias:         {biases['model_A']:+.2f}\n"
        report += f"Attention Preference: {biases['attention']:+.2f}\n"
        report += f"Output Preference:    {biases['output']:+.2f}\n"
        
        report += "\n" + "="*60 + "\n"
        
        return report


NODE_CLASS_MAPPINGS = {
    "TensorPrism_IntelligentTensorSelector": TensorPrism_IntelligentTensorSelector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TensorPrism_IntelligentTensorSelector": "Intelligent Tensor Selector (Tensor Prism)"
}