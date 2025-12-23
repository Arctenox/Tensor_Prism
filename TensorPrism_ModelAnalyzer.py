"""
TensorPrism Model Analyzer
===========================

Advanced model analysis and comparison tools for understanding models
before merging. Provides insights into model structure, statistics,
and compatibility for optimal merge planning.

Includes smart model weight analysis and recipe-based merging.

Author: Arctenox
Version: 2.0.0
License: GPL-3.0
"""

import torch
import numpy as np
import re
import gc
import json
import psutil
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import comfy.model_management as mm


# ==================== MODEL ANALYZER ====================

class TensorPrism_ModelAnalyzer:
    """
    Comprehensive model analysis providing structure, statistics, and insights
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "analysis_depth": (["Quick", "Standard", "Deep"], {
                    "default": "Standard"
                }),
            },
            "optional": {
                "compare_with": ("MODEL",),
                "show_layer_stats": ("BOOLEAN", {"default": True}),
                "show_memory_usage": ("BOOLEAN", {"default": True}),
                "show_architecture": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("analysis_report", "json_data")
    FUNCTION = "analyze_model"
    CATEGORY = "Tensor_Prism/Analysis"
    OUTPUT_NODE = True
    
    def analyze_model(self, model, analysis_depth, compare_with=None, 
                     show_layer_stats=True, show_memory_usage=True, 
                     show_architecture=True):
        """Main analysis function"""
        
        print("\n" + "="*60)
        print("🔍 TENSORPRISM MODEL ANALYZER")
        print("="*60)
        
        state_dict = model.model.state_dict()
        
        # Core analysis
        structure = self._analyze_structure(state_dict)
        stats = self._analyze_statistics(state_dict, analysis_depth)
        
        # Build report
        report_lines = []
        report_lines.append("="*60)
        report_lines.append("MODEL ANALYSIS REPORT")
        report_lines.append("="*60)
        report_lines.append("")
        
        # Architecture section
        if show_architecture:
            report_lines.extend(self._format_architecture(structure))
            report_lines.append("")
        
        # Layer statistics
        if show_layer_stats:
            report_lines.extend(self._format_layer_stats(stats))
            report_lines.append("")
        
        # Memory usage
        if show_memory_usage:
            report_lines.extend(self._format_memory_info(state_dict))
            report_lines.append("")
        
        # Comparison if provided
        if compare_with is not None:
            report_lines.extend(self._compare_models(model, compare_with, analysis_depth))
            report_lines.append("")
        
        # Recommendations
        report_lines.extend(self._generate_recommendations(structure, stats))
        
        report_lines.append("="*60)
        
        report = "\n".join(report_lines)
        
        # Create JSON data
        json_data = self._create_json_report(structure, stats)
        
        print(report)
        
        return (report, json_data)
    
    def _analyze_structure(self, state_dict) -> Dict:
        """Analyze model architecture structure"""
        structure = {
            "total_parameters": len(state_dict),
            "input_blocks": 0,
            "middle_blocks": 0,
            "output_blocks": 0,
            "time_embeds": 0,
            "attention_layers": 0,
            "resnet_blocks": 0,
            "text_encoder_params": 0,
            "unet_params": 0,
            "layer_distribution": defaultdict(int)
        }
        
        for key in state_dict.keys():
            # Count block types
            if "input_blocks" in key:
                structure["input_blocks"] += 1
            elif "middle_block" in key:
                structure["middle_blocks"] += 1
            elif "output_blocks" in key:
                structure["output_blocks"] += 1
            elif "time_embed" in key:
                structure["time_embeds"] += 1
            
            # Count layer types
            if any(x in key.lower() for x in ["attn", "attention"]):
                structure["attention_layers"] += 1
            if "resnets" in key or "resnet" in key:
                structure["resnet_blocks"] += 1
            
            # Count component types
            if any(x in key.lower() for x in ["clip", "text_encoder", "cond_stage"]):
                structure["text_encoder_params"] += 1
            elif any(x in key.lower() for x in ["unet", "diffusion_model"]):
                structure["unet_params"] += 1
            
            # Layer distribution
            layer_num = self._extract_layer_number(key)
            if layer_num is not None:
                structure["layer_distribution"][layer_num] += 1
        
        return structure
    
    def _analyze_statistics(self, state_dict, depth) -> Dict:
        """Analyze statistical properties of parameters"""
        stats = {
            "parameter_stats": {},
            "magnitude_stats": {},
            "sparsity_stats": {},
            "distribution_stats": {}
        }
        
        total_params = 0
        total_size = 0
        magnitudes = []
        sparsities = []
        
        sample_rate = {"Quick": 10, "Standard": 5, "Deep": 1}[depth]
        
        for i, (key, param) in enumerate(state_dict.items()):
            if not isinstance(param, torch.Tensor):
                continue
            
            # Basic stats for all
            param_count = param.numel()
            total_params += param_count
            total_size += param.element_size() * param_count
            
            # Detailed stats based on depth
            if i % sample_rate == 0:
                param_float = param.float()
                
                # Magnitude
                magnitude = torch.norm(param_float).item()
                magnitudes.append(magnitude)
                
                # Sparsity
                sparsity = (param_float.abs() < 1e-6).float().mean().item()
                sparsities.append(sparsity)
                
                # Store per-parameter stats for important layers
                if any(x in key for x in ["out.", "middle_block", "time_embed"]):
                    stats["parameter_stats"][key] = {
                        "shape": list(param.shape),
                        "magnitude": magnitude,
                        "mean": param_float.mean().item(),
                        "std": param_float.std().item(),
                        "min": param_float.min().item(),
                        "max": param_float.max().item(),
                        "sparsity": sparsity
                    }
        
        # Aggregate stats
        stats["magnitude_stats"] = {
            "mean": np.mean(magnitudes) if magnitudes else 0.0,
            "std": np.std(magnitudes) if magnitudes else 0.0,
            "min": np.min(magnitudes) if magnitudes else 0.0,
            "max": np.max(magnitudes) if magnitudes else 0.0
        }
        
        stats["sparsity_stats"] = {
            "mean": np.mean(sparsities) if sparsities else 0.0,
            "std": np.std(sparsities) if sparsities else 0.0,
            "min": np.min(sparsities) if sparsities else 0.0,
            "max": np.max(sparsities) if sparsities else 0.0
        }
        
        stats["total_parameters"] = total_params
        stats["total_size_mb"] = total_size / (1024 * 1024)
        
        return stats
    
    def _compare_models(self, model_A, model_B, depth) -> List[str]:
        """Compare two models for merge compatibility"""
        lines = []
        lines.append("MODEL COMPARISON")
        lines.append("-" * 60)
        
        state_dict_A = model_A.model.state_dict()
        state_dict_B = model_B.model.state_dict()
        
        # Structure comparison
        keys_A = set(state_dict_A.keys())
        keys_B = set(state_dict_B.keys())
        
        common_keys = keys_A & keys_B
        only_A = keys_A - keys_B
        only_B = keys_B - keys_A
        
        lines.append(f"Common parameters: {len(common_keys)}")
        lines.append(f"Only in Model A: {len(only_A)}")
        lines.append(f"Only in Model B: {len(only_B)}")
        
        if only_A or only_B:
            lines.append("\n⚠️  WARNING: Models have different architectures!")
            lines.append("   Merging may produce unexpected results.")
        else:
            lines.append("\n✅ Models have compatible architectures")
        
        # Parameter similarity analysis
        similarities = []
        magnitude_diffs = []
        
        sample_rate = {"Quick": 20, "Standard": 10, "Deep": 5}[depth]
        
        for i, key in enumerate(list(common_keys)):
            if i % sample_rate != 0:
                continue
                
            param_A = state_dict_A[key]
            param_B = state_dict_B[key]
            
            if isinstance(param_A, torch.Tensor) and isinstance(param_B, torch.Tensor):
                if param_A.shape == param_B.shape:
                    # Cosine similarity
                    flat_A = param_A.flatten().float()
                    flat_B = param_B.flatten().float()
                    
                    similarity = torch.cosine_similarity(
                        flat_A.unsqueeze(0), 
                        flat_B.unsqueeze(0)
                    ).item()
                    similarities.append(similarity)
                    
                    # Magnitude difference
                    mag_diff = (torch.norm(flat_B) - torch.norm(flat_A)).abs().item()
                    magnitude_diffs.append(mag_diff)
        
        if similarities:
            avg_similarity = np.mean(similarities)
            avg_mag_diff = np.mean(magnitude_diffs)
            
            lines.append(f"\nAverage parameter similarity: {avg_similarity:.4f}")
            lines.append(f"Average magnitude difference: {avg_mag_diff:.4f}")
            
            # Recommendations based on similarity
            if avg_similarity > 0.9:
                lines.append("\n💡 Models are very similar - small merge ratios recommended")
            elif avg_similarity > 0.7:
                lines.append("\n💡 Models are moderately similar - standard merging recommended")
            elif avg_similarity > 0.5:
                lines.append("\n💡 Models are quite different - careful testing recommended")
            else:
                lines.append("\n⚠️  Models are very different - experimental merge")
        
        return lines
    
    def _format_architecture(self, structure) -> List[str]:
        """Format architecture information"""
        lines = []
        lines.append("ARCHITECTURE")
        lines.append("-" * 60)
        lines.append(f"Total parameters: {structure['total_parameters']:,}")
        lines.append(f"UNet parameters: {structure['unet_params']:,}")
        lines.append(f"Text encoder parameters: {structure['text_encoder_params']:,}")
        lines.append("")
        lines.append("Block Structure:")
        lines.append(f"  Input blocks: {structure['input_blocks']}")
        lines.append(f"  Middle blocks: {structure['middle_blocks']}")
        lines.append(f"  Output blocks: {structure['output_blocks']}")
        lines.append(f"  Time embeddings: {structure['time_embeds']}")
        lines.append("")
        lines.append("Layer Types:")
        lines.append(f"  Attention layers: {structure['attention_layers']}")
        lines.append(f"  ResNet blocks: {structure['resnet_blocks']}")
        
        if structure['layer_distribution']:
            max_layer = max(structure['layer_distribution'].keys())
            lines.append(f"\nDetected depth: {max_layer + 1} layers")
        
        return lines
    
    def _format_layer_stats(self, stats) -> List[str]:
        """Format layer statistics"""
        lines = []
        lines.append("PARAMETER STATISTICS")
        lines.append("-" * 60)
        lines.append(f"Total parameters: {stats['total_parameters']:,}")
        lines.append(f"Model size: {stats['total_size_mb']:.2f} MB")
        lines.append("")
        lines.append("Magnitude Statistics:")
        lines.append(f"  Mean: {stats['magnitude_stats']['mean']:.4f}")
        lines.append(f"  Std: {stats['magnitude_stats']['std']:.4f}")
        lines.append(f"  Range: [{stats['magnitude_stats']['min']:.4f}, {stats['magnitude_stats']['max']:.4f}]")
        lines.append("")
        lines.append("Sparsity Statistics:")
        lines.append(f"  Mean: {stats['sparsity_stats']['mean']:.4%}")
        lines.append(f"  Std: {stats['sparsity_stats']['std']:.4%}")
        
        return lines
    
    def _format_memory_info(self, state_dict) -> List[str]:
        """Format memory usage information"""
        lines = []
        lines.append("MEMORY INFORMATION")
        lines.append("-" * 60)
        
        # Calculate memory by component
        memory_by_component = defaultdict(int)
        
        for key, param in state_dict.items():
            if isinstance(param, torch.Tensor):
                size = param.element_size() * param.numel()
                
                if "input_blocks" in key:
                    memory_by_component["Input Blocks"] += size
                elif "middle_block" in key:
                    memory_by_component["Middle Block"] += size
                elif "output_blocks" in key:
                    memory_by_component["Output Blocks"] += size
                elif "time_embed" in key:
                    memory_by_component["Time Embeddings"] += size
                else:
                    memory_by_component["Other"] += size
        
        for component, size in sorted(memory_by_component.items(), key=lambda x: -x[1]):
            lines.append(f"  {component}: {size / (1024*1024):.2f} MB")
        
        # System memory
        memory = psutil.virtual_memory()
        lines.append("")
        lines.append("System Memory:")
        lines.append(f"  Used: {(memory.total - memory.available) / (1024**3):.2f} GB")
        lines.append(f"  Available: {memory.available / (1024**3):.2f} GB")
        
        return lines
    
    def _generate_recommendations(self, structure, stats) -> List[str]:
        """Generate merge recommendations"""
        lines = []
        lines.append("MERGE RECOMMENDATIONS")
        lines.append("-" * 60)
        
        # Based on model size
        size_gb = stats['total_size_mb'] / 1024
        if size_gb > 10:
            lines.append("• Large model detected - use memory-efficient merge methods")
            lines.append("  Recommended: Set memory_limit_gb=8 or lower")
        
        # Based on sparsity
        avg_sparsity = stats['sparsity_stats']['mean']
        if avg_sparsity > 0.3:
            lines.append(f"• High sparsity detected ({avg_sparsity:.1%})")
            lines.append("  Recommended: Use magnitude-weighted merging")
        
        # Based on architecture
        if structure['attention_layers'] > 100:
            lines.append("• Many attention layers detected")
            lines.append("  Recommended: Use attention_bias in LayeredBlend")
        
        # General recommendations
        lines.append("\nGeneral Tips:")
        lines.append("• Start with small merge ratios (0.1-0.3) and test")
        lines.append("• Use SLERP for smoother interpolation")
        lines.append("• Consider using masks to target specific components")
        lines.append("• Test output quality frequently during experimentation")
        
        return lines
    
    def _create_json_report(self, structure, stats) -> str:
        """Create JSON formatted report for programmatic use"""
        report_data = {
            "architecture": {
                "total_parameters": structure['total_parameters'],
                "unet_params": structure['unet_params'],
                "text_encoder_params": structure['text_encoder_params'],
                "input_blocks": structure['input_blocks'],
                "middle_blocks": structure['middle_blocks'],
                "output_blocks": structure['output_blocks'],
                "attention_layers": structure['attention_layers'],
                "resnet_blocks": structure['resnet_blocks']
            },
            "statistics": {
                "total_parameters": stats['total_parameters'],
                "size_mb": stats['total_size_mb'],
                "magnitude_mean": stats['magnitude_stats']['mean'],
                "sparsity_mean": stats['sparsity_stats']['mean']
            }
        }
        
        return json.dumps(report_data, indent=2)
    
    def _extract_layer_number(self, key: str) -> Optional[int]:
        """Extract layer number from parameter key"""
        patterns = [
            r"layers\.(\d+)\.",
            r"blocks\.(\d+)\.",
            r"h\.(\d+)\.",
            r"encoder\.layer\.(\d+)\.",
            r"decoder\.layer\.(\d+)\.",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, key)
            if match:
                return int(match.group(1))
        
        return None


# ==================== MODEL COMPARATOR ====================

class TensorPrism_ModelComparator:
    """
    Quick comparison tool for determining merge compatibility
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_A": ("MODEL",),
                "model_B": ("MODEL",),
            }
        }
    
    RETURN_TYPES = ("STRING", "FLOAT", "BOOLEAN")
    RETURN_NAMES = ("comparison_report", "similarity_score", "compatible")
    FUNCTION = "compare"
    CATEGORY = "Tensor_Prism/Analysis"
    OUTPUT_NODE = True
    
    def compare(self, model_A, model_B):
        """Quick comparison of two models"""
        
        state_dict_A = model_A.model.state_dict()
        state_dict_B = model_B.model.state_dict()
        
        keys_A = set(state_dict_A.keys())
        keys_B = set(state_dict_B.keys())
        
        common_keys = keys_A & keys_B
        compatibility = len(common_keys) / max(len(keys_A), len(keys_B))
        
        # Calculate similarity
        similarities = []
        for key in list(common_keys)[:100]:  # Sample first 100
            param_A = state_dict_A[key]
            param_B = state_dict_B[key]
            
            if isinstance(param_A, torch.Tensor) and isinstance(param_B, torch.Tensor):
                if param_A.shape == param_B.shape:
                    flat_A = param_A.flatten().float()
                    flat_B = param_B.flatten().float()
                    sim = torch.cosine_similarity(
                        flat_A.unsqueeze(0), 
                        flat_B.unsqueeze(0)
                    ).item()
                    similarities.append(sim)
        
        avg_similarity = np.mean(similarities) if similarities else 0.0
        is_compatible = compatibility > 0.95 and avg_similarity > 0.1
        
        report = f"""QUICK COMPARISON REPORT
{"="*40}
Architecture Compatibility: {compatibility:.1%}
Parameter Similarity: {avg_similarity:.4f}
Compatible for Merging: {"✅ Yes" if is_compatible else "⚠️  No"}

Common Parameters: {len(common_keys)}
Model A Only: {len(keys_A - keys_B)}
Model B Only: {len(keys_B - keys_A)}

Recommendation:
"""
        
        if avg_similarity > 0.9:
            report += "Models are very similar - use small merge ratios"
        elif avg_similarity > 0.7:
            report += "Models are moderately similar - standard merging OK"
        elif avg_similarity > 0.5:
            report += "Models are different - careful testing recommended"
        else:
            report += "Models are very different - experimental only"
        
        print(report)
        
        return (report, avg_similarity, is_compatible)


# ==================== ANALYZE MODEL WEIGHTS ====================

class TensorPrism_AnalyzeModelWeights:
    """
    Analyze two models and calculate optimal per-block weights
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
    CATEGORY = "Tensor_Prism/Analysis"

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
        a_cpu = tensor_a.detach().cpu().float()
        b_cpu = tensor_b.detach().cpu().float()
        
        mag_a = torch.norm(a_cpu.flatten()[:1000000] if a_cpu.numel() > 1000000 else a_cpu).item()
        mag_b = torch.norm(b_cpu.flatten()[:1000000] if b_cpu.numel() > 1000000 else b_cpu).item()
        
        total_mag = mag_a + mag_b
        return 0.5 if total_mag == 0 else mag_a / total_mag

    def calculate_entropy_score(self, tensor_a, tensor_b):
        """Entropy-based score with device safety - ALL CPU"""
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
                print(f"\n📏 Adjusting standard deviation:")
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


# ==================== APPLY MERGE RECIPE ====================

class TensorPrism_ApplyMergeRecipe:
    """
    Apply the merge recipe to two models with device safety
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
    CATEGORY = "Tensor_Prism/Analysis"

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
                        tensor_b = b_sd[key].to(device)
                        
                        if merge_method == "weighted_sum":
                            merged_tensor = tensor_a * alpha + tensor_b * (1 - alpha)
                        
                        elif merge_method == "add_difference":
                            delta = tensor_b - tensor_a
                            merged_tensor = tensor_a + delta * alpha * strength
                        
                        else:
                            merged_tensor = tensor_a * alpha + tensor_b * (1 - alpha)
                        
                        merged_tensor = merged_tensor.to(device)
                        
                        # Create patch
                        diff = (merged_tensor - tensor_a).to(device)
                        if torch.abs(diff).max() > 1e-8:
                            patches[key] = (diff.cpu(),)
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


# ==================== NODE REGISTRATION ====================

NODE_CLASS_MAPPINGS = {
    "TensorPrism_ModelAnalyzer": TensorPrism_ModelAnalyzer,
    "TensorPrism_ModelComparator": TensorPrism_ModelComparator,
    "TensorPrism_AnalyzeModelWeights": TensorPrism_AnalyzeModelWeights,
    "TensorPrism_ApplyMergeRecipe": TensorPrism_ApplyMergeRecipe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TensorPrism_ModelAnalyzer": "Model Analyzer (Tensor Prism)",
    "TensorPrism_ModelComparator": "Model Comparator (Tensor Prism)",
    "TensorPrism_AnalyzeModelWeights": "Analyze Model Weights (Tensor Prism)",
    "TensorPrism_ApplyMergeRecipe": "Apply Merge Recipe (Tensor Prism)",
}
