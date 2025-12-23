"""
TensorPrism Competitive Model Selector
======================================

Multi-model tournament-style selection where models compete for each block/layer.
Implements advanced comparison metrics similar to Hyphoria's approach.

Author: Arctenox
Version: 1.0.1
License: GPL-3.0
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import gc
import comfy.model_management

class TensorPrism_CompetitiveModelSelector:
    """
    Tournament-style model selection where multiple models compete for each layer.
    The winner is chosen based on configurable criteria and block-level granularity.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_A": ("MODEL",),
                "model_B": ("MODEL",),
                "competition_mode": ([
                    "tournament",        # Best of all wins
                    "weighted_vote",     # Weighted average of top performers
                    "hybrid_blend",      # Blend top 2 performers
                    "consensus",         # All must agree on quality
                ], {"default": "tournament"}),
                
                # Block-level granularity
                "input_blocks_strategy": ([
                    "compete", "favor_A", "favor_B", "blend_50"
                ], {"default": "compete"}),
                "middle_blocks_strategy": ([
                    "compete", "favor_A", "favor_B", "blend_50"
                ], {"default": "compete"}),
                "output_blocks_strategy": ([
                    "compete", "favor_A", "favor_B", "blend_50"
                ], {"default": "compete"}),
                
                # Comparison criteria
                "detail_preservation_weight": ("FLOAT", {
                    "default": 0.35, "min": 0.0, "max": 2.0, "step": 0.05
                }),
                "coherence_weight": ("FLOAT", {
                    "default": 0.25, "min": 0.0, "max": 2.0, "step": 0.05
                }),
                "efficiency_weight": ("FLOAT", {
                    "default": 0.20, "min": 0.0, "max": 2.0, "step": 0.05
                }),
                "innovation_weight": ("FLOAT", {
                    "default": 0.20, "min": 0.0, "max": 2.0, "step": 0.05
                }),
            },
            "optional": {
                "model_C": ("MODEL",),
                "model_D": ("MODEL",),
                "model_E": ("MODEL",),
                "enable_tie_breaking": ("BOOLEAN", {"default": True}),
                "minimum_quality_threshold": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05
                }),
            }
        }
    
    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("winner_model", "competition_report")
    FUNCTION = "compete_models"
    CATEGORY = "Tensor_Prism/Advanced"
    
    def __init__(self):
        self.device = comfy.model_management.get_torch_device()
    
    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Safely move tensor to working device."""
        if tensor.device != self.device:
            return tensor.to(self.device)
        return tensor
    
    def _get_block_identifier(self, key: str) -> Tuple[str, int]:
        """
        Identify which block a parameter belongs to.
        Returns (block_type, block_number).
        """
        key_lower = key.lower()
        
        # Input blocks
        if 'input_blocks.' in key_lower:
            try:
                block_num = int(key.split('input_blocks.')[1].split('.')[0])
                return ('input', block_num)
            except:
                return ('input', -1)
        
        # Middle block
        elif 'middle_block' in key_lower:
            try:
                if 'middle_block.0' in key_lower:
                    return ('middle', 0)
                elif 'middle_block.1' in key_lower:
                    return ('middle', 1)
                elif 'middle_block.2' in key_lower:
                    return ('middle', 2)
                else:
                    return ('middle', -1)
            except:
                return ('middle', -1)
        
        # Output blocks
        elif 'output_blocks.' in key_lower:
            try:
                block_num = int(key.split('output_blocks.')[1].split('.')[0])
                return ('output', block_num)
            except:
                return ('output', -1)
        
        # Time embedding
        elif 'time_embed' in key_lower:
            return ('time_embed', 0)
        
        # Final output
        elif key_lower.endswith('.out.weight') or key_lower.endswith('.out.bias'):
            return ('final_out', 0)
        
        return ('other', -1)
    
    def _evaluate_detail_preservation(self, tensor: torch.Tensor) -> float:
        """Measure how well a tensor preserves fine details."""
        try:
            tensor = self._to_device(tensor)
            
            # High frequency content analysis
            if tensor.numel() < 4:
                return 0.0
            
            # Calculate local variance as detail metric
            if tensor.dim() >= 2:
                # For 2D+ tensors, analyze spatial variance
                variance = torch.var(tensor, dim=list(range(tensor.dim())))
                detail_score = torch.log1p(variance.mean()).item()
            else:
                # For 1D tensors, use gradient magnitude
                if tensor.numel() > 1:
                    grad = torch.diff(tensor)
                    detail_score = torch.log1p(torch.abs(grad).mean()).item()
                else:
                    detail_score = 0.0
            
            return detail_score
            
        except Exception as e:
            return 0.0
    
    def _evaluate_coherence(self, tensor: torch.Tensor) -> float:
        """Measure internal coherence and consistency."""
        try:
            tensor = self._to_device(tensor)
            
            if tensor.numel() < 4:
                return 1.0
            
            # Measure consistency via correlation
            flat = tensor.flatten()
            if len(flat) > 1000:
                flat = flat[:1000]  # Sample for efficiency
            
            # Auto-correlation as coherence metric
            mean = flat.mean()
            centered = flat - mean
            autocorr = torch.dot(centered, centered) / (torch.norm(centered) ** 2 + 1e-8)
            
            coherence_score = autocorr.item()
            return max(0.0, coherence_score)
            
        except Exception as e:
            return 0.5
    
    def _evaluate_efficiency(self, tensor: torch.Tensor) -> float:
        """Measure parameter efficiency (information density)."""
        try:
            tensor = self._to_device(tensor)
            
            if tensor.numel() == 0:
                return 0.0
            
            # Information entropy as efficiency metric
            flat = tensor.flatten()
            
            # Normalize and discretize
            normalized = (flat - flat.min()) / (flat.max() - flat.min() + 1e-8)
            discretized = (normalized * 100).long()
            
            # Count unique values
            unique_ratio = len(torch.unique(discretized)) / len(discretized)
            
            # Sparsity
            sparsity = (torch.abs(flat) < 1e-6).float().mean().item()
            
            # Combine metrics
            efficiency_score = unique_ratio * (1.0 - sparsity * 0.5)
            
            return efficiency_score
            
        except Exception as e:
            return 0.5
    
    def _evaluate_innovation(self, tensor: torch.Tensor, reference: torch.Tensor) -> float:
        """Measure how different/innovative a tensor is compared to reference."""
        try:
            tensor = self._to_device(tensor)
            reference = self._to_device(reference)
            
            if tensor.shape != reference.shape:
                return 0.5
            
            # Calculate divergence
            diff = tensor - reference
            divergence = torch.norm(diff) / (torch.norm(reference) + 1e-8)
            
            # Normalized innovation score (moderate divergence is good)
            innovation_score = min(1.0, divergence.item() * 0.5)
            
            return innovation_score
            
        except Exception as e:
            return 0.5
    
    def _compete_tensors(self, candidates: List[Tuple[str, torch.Tensor]],
                        weights: Dict, reference_tensor: Optional[torch.Tensor] = None,
                        enable_tie_breaking: bool = True) -> Tuple[str, torch.Tensor, Dict]:
        """
        Run competition between candidate tensors.
        Returns (winner_name, winner_tensor, scores_dict).
        """
        if len(candidates) == 1:
            return candidates[0][0], candidates[0][1], {}
        
        scores = []
        
        for name, tensor in candidates:
            # Calculate individual metrics
            detail_score = self._evaluate_detail_preservation(tensor)
            coherence_score = self._evaluate_coherence(tensor)
            efficiency_score = self._evaluate_efficiency(tensor)
            
            # Innovation requires reference
            if reference_tensor is not None:
                innovation_score = self._evaluate_innovation(tensor, reference_tensor)
            else:
                innovation_score = 0.5  # Neutral if no reference
            
            # Weighted total
            total_score = (
                detail_score * weights['detail'] +
                coherence_score * weights['coherence'] +
                efficiency_score * weights['efficiency'] +
                innovation_score * weights['innovation']
            )
            
            scores.append((name, tensor, total_score, {
                'detail': detail_score,
                'coherence': coherence_score,
                'efficiency': efficiency_score,
                'innovation': innovation_score,
                'total': total_score
            }))
        
        # Sort by total score
        scores.sort(key=lambda x: x[2], reverse=True)
        
        # Check for tie
        if enable_tie_breaking and len(scores) > 1:
            if abs(scores[0][2] - scores[1][2]) < 0.01:  # Close scores
                # Use detail as tie-breaker
                if scores[0][3]['detail'] < scores[1][3]['detail']:
                    scores[0], scores[1] = scores[1], scores[0]
        
        winner = scores[0]
        return winner[0], winner[1], winner[3]
    
    def _apply_block_strategy(self, block_type: str, strategy: str,
                             candidates: List[Tuple[str, torch.Tensor]],
                             weights: Dict, enable_tie_breaking: bool) -> Tuple[str, torch.Tensor]:
        """Apply the specified strategy for a block type."""
        
        if strategy == "favor_A":
            # Return Model A if available
            for name, tensor in candidates:
                if name == 'A':
                    return name, tensor
            return candidates[0][0], candidates[0][1]
        
        elif strategy == "favor_B":
            # Return Model B if available
            for name, tensor in candidates:
                if name == 'B':
                    return name, tensor
            return candidates[0][0], candidates[0][1]
        
        elif strategy == "blend_50":
            # Blend A and B if both available
            tensor_A = None
            tensor_B = None
            for name, tensor in candidates:
                if name == 'A':
                    tensor_A = tensor
                elif name == 'B':
                    tensor_B = tensor
            
            if tensor_A is not None and tensor_B is not None:
                # Ensure both on same device
                tensor_A = self._to_device(tensor_A)
                tensor_B = self._to_device(tensor_B)
                blended = (tensor_A + tensor_B) * 0.5
                return 'A+B', blended
            else:
                return candidates[0][0], candidates[0][1]
        
        else:  # compete
            winner_name, winner_tensor, _ = self._compete_tensors(
                candidates, weights, None, enable_tie_breaking
            )
            return winner_name, winner_tensor
    
    def compete_models(self, model_A, model_B, competition_mode: str,
                      input_blocks_strategy: str, middle_blocks_strategy: str,
                      output_blocks_strategy: str,
                      detail_preservation_weight: float, coherence_weight: float,
                      efficiency_weight: float, innovation_weight: float,
                      model_C=None, model_D=None, model_E=None,
                      enable_tie_breaking: bool = True,
                      minimum_quality_threshold: float = 0.0):
        """
        Main competition function - let models compete for supremacy!
        """
        
        print("\n" + "="*70)
        print("⚔️ COMPETITIVE MODEL SELECTOR (Tensor Prism)")
        print("="*70)
        
        # Prepare weights
        weights = {
            'detail': detail_preservation_weight,
            'coherence': coherence_weight,
            'efficiency': efficiency_weight,
            'innovation': innovation_weight
        }
        
        print(f"\n⚙️ Competition Setup:")
        print(f"   Mode: {competition_mode}")
        print(f"   Input Strategy: {input_blocks_strategy}")
        print(f"   Middle Strategy: {middle_blocks_strategy}")
        print(f"   Output Strategy: {output_blocks_strategy}")
        print(f"   Weights - Detail: {detail_preservation_weight:.2f}, "
              f"Coherence: {coherence_weight:.2f}")
        print(f"             Efficiency: {efficiency_weight:.2f}, "
              f"Innovation: {innovation_weight:.2f}")
        
        # Collect models - keep state dicts in CPU to avoid memory issues
        models = [('A', model_A.model.state_dict()), ('B', model_B.model.state_dict())]
        if model_C is not None:
            models.append(('C', model_C.model.state_dict()))
        if model_D is not None:
            models.append(('D', model_D.model.state_dict()))
        if model_E is not None:
            models.append(('E', model_E.model.state_dict()))
        
        print(f"   Competitors: {len(models)} models")
        
        # Get base keys
        base_sd = models[0][1]
        all_keys = list(base_sd.keys())
        
        print(f"\n🏆 Starting competition for {len(all_keys)} tensors...")
        
        # Competition tracking
        winner_stats = {name: 0 for name, _ in models}
        block_winner_stats = {
            'input': {name: 0 for name, _ in models},
            'middle': {name: 0 for name, _ in models},
            'output': {name: 0 for name, _ in models},
            'other': {name: 0 for name, _ in models}
        }
        
        # Create patches
        patches = {}
        processed = 0
        
        for key in all_keys:
            try:
                # Identify block
                block_type, block_num = self._get_block_identifier(key)
                
                # Get candidates
                candidates = [(name, sd[key]) for name, sd in models if key in sd]
                
                if len(candidates) == 0:
                    continue
                
                # Determine strategy for this block
                if block_type == 'input':
                    strategy = input_blocks_strategy
                elif block_type == 'middle':
                    strategy = middle_blocks_strategy
                elif block_type == 'output':
                    strategy = output_blocks_strategy
                else:
                    strategy = 'compete'
                
                # Apply strategy
                winner_name, winner_tensor = self._apply_block_strategy(
                    block_type, strategy, candidates, weights, enable_tie_breaking
                )
                
                # Track winner
                if winner_name in winner_stats:
                    winner_stats[winner_name] += 1
                    if block_type in block_winner_stats:
                        block_winner_stats[block_type][winner_name] += 1
                
                # Create patch if not from Model A
                if winner_name != 'A':
                    original = base_sd[key]
                    # Ensure tensors are on same device before subtraction
                    winner_tensor = self._to_device(winner_tensor)
                    original = self._to_device(original)
                    diff = winner_tensor - original
                    
                    if torch.abs(diff).max() > 1e-8:
                        # Move diff back to CPU for patches
                        patches[key] = (diff.cpu(),)
                
                processed += 1
                
                if processed % 100 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"⚠️ Error processing {key}: {e}")
                continue
        
        # Apply patches
        print(f"\n📦 Applying {len(patches)} winning patches...")
        merged_model = model_A.clone()
        if patches:
            merged_model.add_patches(patches, 1.0)
        
        # Generate report
        report = self._generate_competition_report(
            winner_stats, block_winner_stats, len(all_keys),
            competition_mode, weights
        )
        
        print("\n✅ Competition complete!")
        print("="*70 + "\n")
        
        return (merged_model, report)
    
    def _generate_competition_report(self, winner_stats: Dict,
                                    block_stats: Dict, total: int,
                                    mode: str, weights: Dict) -> str:
        """Generate competition results report."""
        
        report = "="*60 + "\n"
        report += "COMPETITIVE MODEL SELECTION REPORT\n"
        report += "="*60 + "\n\n"
        
        report += f"Competition Mode: {mode}\n"
        report += f"Total Tensors: {total}\n\n"
        
        report += "🏆 OVERALL WINNERS:\n"
        report += "-"*40 + "\n"
        for model, wins in sorted(winner_stats.items(), key=lambda x: x[1], reverse=True):
            percentage = (wins / total * 100) if total > 0 else 0
            report += f"Model {model}: {wins} wins ({percentage:.1f}%)\n"
        
        report += "\n\n📊 BLOCK-LEVEL RESULTS:\n"
        report += "-"*40 + "\n"
        for block_type, stats in block_stats.items():
            block_total = sum(stats.values())
            if block_total > 0:
                report += f"\n{block_type.upper()} BLOCKS:\n"
                for model, wins in sorted(stats.items(), key=lambda x: x[1], reverse=True):
                    percentage = (wins / block_total * 100) if block_total > 0 else 0
                    report += f"  Model {model}: {wins} ({percentage:.1f}%)\n"
        
        report += "\n\n⚖️ EVALUATION WEIGHTS:\n"
        report += "-"*40 + "\n"
        report += f"Detail Preservation: {weights['detail']:.2f}\n"
        report += f"Coherence:          {weights['coherence']:.2f}\n"
        report += f"Efficiency:         {weights['efficiency']:.2f}\n"
        report += f"Innovation:         {weights['innovation']:.2f}\n"
        
        report += "\n" + "="*60 + "\n"
        
        return report


NODE_CLASS_MAPPINGS = {
    "TensorPrism_CompetitiveModelSelector": TensorPrism_CompetitiveModelSelector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TensorPrism_CompetitiveModelSelector": "Competitive Model Selector (Tensor Prism)"
}