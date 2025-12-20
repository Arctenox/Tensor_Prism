"""
TensorPrism Model Analyzer
===========================

Advanced model analysis and comparison tools for understanding models
before merging. Provides insights into model structure, statistics,
and compatibility for optimal merge planning.

Author: Arctenox
Version: 1.0.0
License: GPL-3.0
"""

import torch
import numpy as np
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import psutil


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
        import json
        
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


# Node registration
NODE_CLASS_MAPPINGS = {
    "TensorPrism_ModelAnalyzer": TensorPrism_ModelAnalyzer,
    "TensorPrism_ModelComparator": TensorPrism_ModelComparator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TensorPrism_ModelAnalyzer": "Model Analyzer (Tensor Prism)",
    "TensorPrism_ModelComparator": "Model Comparator (Tensor Prism)",
}
