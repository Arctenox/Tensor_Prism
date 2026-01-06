"""
ComfyUI Tensor Prism Node Pack
===============================

Advanced model merging and enhancement nodes for ComfyUI, providing sophisticated
techniques for blending, enhancing, and manipulating Stable Diffusion models with
GPU-optimized memory management.

Author: Arctenox
Version: 1.7.0
License: GPL-3.0
Repository: https://github.com/Arctenox/Tensor_Prism

Features:
- Advanced model merging with multiple interpolation methods
- Spectral frequency-domain merging
- Granular SDXL block control
- Sophisticated masking system
- Intelligent tensor-by-tensor selection
- Competitive multi-model selection
- Smart model analysis and recipe-based merging
- Comprehensive model analysis and comparison tools
- Model enhancement capabilities
- GPU-optimized memory management
- Cross-platform compatibility (CUDA/MPS/CPU)
"""

import os
import sys
import importlib.util
from pathlib import Path

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# ==================== CORE MERGING NODES ====================
try:
    from .TensorPrism_CoreMerge import (
        TensorPrism_MainMerge,
        TensorPrism_LayeredBlend,
        TensorPrism_FastPrism
    )
except ImportError:
    TensorPrism_MainMerge = None
    TensorPrism_LayeredBlend = None
    TensorPrism_FastPrism = None

# ==================== SDXL MERGING NODES ====================
try:
    from .TensorPrism_SDXLMerge import (
        SDXLBlockMergeTensorPrism,
        SDXLAdvancedBlockMergeTensorPrism
    )
except ImportError:
    SDXLBlockMergeTensorPrism = None
    SDXLAdvancedBlockMergeTensorPrism = None

# ==================== MASKING SYSTEM NODES ====================
try:
    from .TensorPrism_MaskSystem import (
        TensorPrism_ModelMaskGenerator,
        TensorPrism_ModelKeyFilter,
        TensorPrism_ModelMaskBlender,
        TensorPrism_WeightedMaskMerge,
        TensorPrism_WeightedMaskMergeAdvanced
    )
except ImportError:
    TensorPrism_ModelMaskGenerator = None
    TensorPrism_ModelKeyFilter = None
    TensorPrism_ModelMaskBlender = None
    TensorPrism_WeightedMaskMerge = None
    TensorPrism_WeightedMaskMergeAdvanced = None

# ==================== ADVANCED SELECTION NODES ====================
try:
    from .TensorPrism_IntelligentTensorSelector import TensorPrism_IntelligentTensorSelector
except ImportError:
    TensorPrism_IntelligentTensorSelector = None

try:
    from .TensorPrism_CompetitiveModelSelector import TensorPrism_CompetitiveModelSelector
except ImportError:
    TensorPrism_CompetitiveModelSelector = None

# ==================== ENHANCEMENT NODES ====================
try:
    from .TensorPrism_Enhancer import ModelEnhancerTensorPrism
except ImportError:
    ModelEnhancerTensorPrism = None

# ==================== ANALYSIS NODES ====================
try:
    from .TensorPrism_ModelAnalyzer import (
        TensorPrism_ModelAnalyzer,
        TensorPrism_ModelComparator,
        TensorPrism_AnalyzeModelWeights,
        TensorPrism_ApplyMergeRecipe
    )
except ImportError:
    TensorPrism_ModelAnalyzer = None
    TensorPrism_ModelComparator = None
    TensorPrism_AnalyzeModelWeights = None
    TensorPrism_ApplyMergeRecipe = None

# ==================== CLIP MERGING NODES ====================
try:
    from .TensorPrism_AdvancedClipMerge import TensorPrismAdvancedClipMerge
except ImportError:
    TensorPrismAdvancedClipMerge = None

# ==================== VAE MERGING NODES ====================
try:
    from .TensorPrism_VAEMerge import TensorPrismVAEMerge
except ImportError:
    TensorPrismVAEMerge = None

# ==================== CONVERSION NODES ====================
try:
    from .TensorPrism_vpredepsilonconverter import TensorPrism_EpsilonVPredConverter
except ImportError:
    TensorPrism_EpsilonVPredConverter = None

# ==================== NOISE INJECTION MERGE ====================
try:
    from .TensorPrism_NoiseInjectionMerge import TensorPrism_NoiseInjectionMerge
except ImportError:
    TensorPrism_NoiseInjectionMerge = None

# Version info
__version__ = "1.7.0"
__author__ = "Arctenox"
__description__ = "Advanced model merging and enhancement nodes for ComfyUI"

# ==================== NODE REGISTRATION ====================
NODE_CLASS_MAPPINGS = {}

# Core merging nodes (3 nodes from TensorPrism_CoreMerge.py)
if TensorPrism_MainMerge:
    NODE_CLASS_MAPPINGS["TensorPrism_MainMerge"] = TensorPrism_MainMerge
if TensorPrism_LayeredBlend:
    NODE_CLASS_MAPPINGS["TensorPrism_LayeredBlend"] = TensorPrism_LayeredBlend
if TensorPrism_FastPrism:
    NODE_CLASS_MAPPINGS["TensorPrism_Prism"] = TensorPrism_FastPrism

# SDXL merging nodes (2 nodes from TensorPrism_SDXLMerge.py)
if SDXLBlockMergeTensorPrism:
    NODE_CLASS_MAPPINGS["SDXLBlockMergeTensorPrism"] = SDXLBlockMergeTensorPrism
if SDXLAdvancedBlockMergeTensorPrism:
    NODE_CLASS_MAPPINGS["SDXLAdvancedBlockMergeTensorPrism"] = SDXLAdvancedBlockMergeTensorPrism

# Masking system nodes (5 nodes from TensorPrism_MaskSystem.py)
if TensorPrism_ModelMaskGenerator:
    NODE_CLASS_MAPPINGS["TensorPrism_ModelMaskGenerator"] = TensorPrism_ModelMaskGenerator
if TensorPrism_ModelKeyFilter:
    NODE_CLASS_MAPPINGS["TensorPrism_ModelKeyFilter"] = TensorPrism_ModelKeyFilter
if TensorPrism_ModelMaskBlender:
    NODE_CLASS_MAPPINGS["TensorPrism_ModelMaskBlender"] = TensorPrism_ModelMaskBlender
if TensorPrism_WeightedMaskMerge:
    NODE_CLASS_MAPPINGS["TensorPrism_WeightedMaskMerge"] = TensorPrism_WeightedMaskMerge
if TensorPrism_WeightedMaskMergeAdvanced:
    NODE_CLASS_MAPPINGS["TensorPrism_WeightedMaskMergeAdvanced"] = TensorPrism_WeightedMaskMergeAdvanced

# Advanced selection nodes (2 nodes)
if TensorPrism_IntelligentTensorSelector:
    NODE_CLASS_MAPPINGS["TensorPrism_IntelligentTensorSelector"] = TensorPrism_IntelligentTensorSelector
if TensorPrism_CompetitiveModelSelector:
    NODE_CLASS_MAPPINGS["TensorPrism_CompetitiveModelSelector"] = TensorPrism_CompetitiveModelSelector

# Enhancement nodes (1 node from TensorPrism_Enhancer.py)
if ModelEnhancerTensorPrism:
    NODE_CLASS_MAPPINGS["ModelEnhancerTensorPrism"] = ModelEnhancerTensorPrism

# Analysis nodes (4 nodes from TensorPrism_ModelAnalyzer.py)
if TensorPrism_ModelAnalyzer:
    NODE_CLASS_MAPPINGS["TensorPrism_ModelAnalyzer"] = TensorPrism_ModelAnalyzer
if TensorPrism_ModelComparator:
    NODE_CLASS_MAPPINGS["TensorPrism_ModelComparator"] = TensorPrism_ModelComparator
if TensorPrism_AnalyzeModelWeights:
    NODE_CLASS_MAPPINGS["TensorPrism_AnalyzeModelWeights"] = TensorPrism_AnalyzeModelWeights
if TensorPrism_ApplyMergeRecipe:
    NODE_CLASS_MAPPINGS["TensorPrism_ApplyMergeRecipe"] = TensorPrism_ApplyMergeRecipe

# CLIP merging nodes (1 node)
if TensorPrismAdvancedClipMerge:
    NODE_CLASS_MAPPINGS["TensorPrismAdvancedClipMerge"] = TensorPrismAdvancedClipMerge

# VAE merging nodes (1 node)
if TensorPrismVAEMerge:
    NODE_CLASS_MAPPINGS["TensorPrismVAEMerge"] = TensorPrismVAEMerge

# Conversion nodes (1 node)
if TensorPrism_EpsilonVPredConverter:
    NODE_CLASS_MAPPINGS["TensorPrism_EpsilonVPredConverter"] = TensorPrism_EpsilonVPredConverter

# Noise Injection Merge (1 node)
if TensorPrism_NoiseInjectionMerge:
    NODE_CLASS_MAPPINGS["TensorPrism_NoiseInjectionMerge"] = TensorPrism_NoiseInjectionMerge

# ==================== DISPLAY NAME MAPPINGS ====================
NODE_DISPLAY_NAME_MAPPINGS = {
    # Core merging nodes
    "TensorPrism_MainMerge": "Main Merge (Tensor Prism)",
    "TensorPrism_LayeredBlend": "Layered Blend (Tensor Prism)",
    "TensorPrism_Prism": "Prism (Tensor Prism)",
    
    # SDXL merging nodes
    "SDXLBlockMergeTensorPrism": "SDXL Block Merge (Tensor Prism)",
    "SDXLAdvancedBlockMergeTensorPrism": "SDXL Advanced Block Merge (Tensor Prism)",
    
    # Masking system nodes
    "TensorPrism_ModelMaskGenerator": "Model Mask Generator (Tensor Prism)",
    "TensorPrism_ModelKeyFilter": "Model Key Filter (Tensor Prism)",
    "TensorPrism_ModelMaskBlender": "Mask Blender (Tensor Prism)",
    "TensorPrism_WeightedMaskMerge": "Weighted Mask Merge (Tensor Prism)",
    "TensorPrism_WeightedMaskMergeAdvanced": "Advanced Weighted Mask Merge (Tensor Prism)",
    
    # Advanced selection nodes
    "TensorPrism_IntelligentTensorSelector": "Intelligent Tensor Selector (Tensor Prism)",
    "TensorPrism_CompetitiveModelSelector": "Competitive Model Selector (Tensor Prism)",
    
    # Enhancement nodes
    "ModelEnhancerTensorPrism": "Model Enhancer (Tensor Prism)",
    
    # Analysis nodes
    "TensorPrism_ModelAnalyzer": "Model Analyzer (Tensor Prism)",
    "TensorPrism_ModelComparator": "Model Comparator (Tensor Prism)",
    "TensorPrism_AnalyzeModelWeights": "Analyze Model Weights (Tensor Prism)",
    "TensorPrism_ApplyMergeRecipe": "Apply Merge Recipe (Tensor Prism)",
    
    # CLIP merging
    "TensorPrismAdvancedClipMerge": "Advanced CLIP Merge (Tensor Prism)",
    
    # VAE merging
    "TensorPrismVAEMerge": "VAE Merge (Tensor Prism)",
    
    # Conversion nodes
    "TensorPrism_EpsilonVPredConverter": "Epsilon/V-Pred Converter (Tensor Prism)",
    
    # Noise Injection Merge
    "TensorPrism_NoiseInjectionMerge": "Noise Injection Merge (Tensor Prism)",
}

# ==================== CATEGORY MAPPINGS ====================
NODE_CATEGORIES = {
    # Core merging nodes
    "TensorPrism_MainMerge": "Tensor_Prism/Core",
    "TensorPrism_LayeredBlend": "Tensor_Prism/Core",
    "TensorPrism_Prism": "Tensor_Prism/Core",
    
    # SDXL merging nodes
    "SDXLBlockMergeTensorPrism": "Tensor_Prism/Merge",
    "SDXLAdvancedBlockMergeTensorPrism": "Tensor_Prism/Merge",
    
    # Masking system nodes
    "TensorPrism_ModelMaskGenerator": "Tensor_Prism/Mask",
    "TensorPrism_ModelKeyFilter": "Tensor_Prism/Mask",
    "TensorPrism_ModelMaskBlender": "Tensor_Prism/Mask",
    "TensorPrism_WeightedMaskMerge": "Tensor_Prism/Mask",
    "TensorPrism_WeightedMaskMergeAdvanced": "Tensor_Prism/Mask",
    
    # Advanced selection nodes
    "TensorPrism_IntelligentTensorSelector": "Tensor_Prism/Advanced",
    "TensorPrism_CompetitiveModelSelector": "Tensor_Prism/Advanced",
    
    # Enhancement nodes
    "ModelEnhancerTensorPrism": "Tensor_Prism/Transform",
    
    # Analysis nodes
    "TensorPrism_ModelAnalyzer": "Tensor_Prism/Analysis",
    "TensorPrism_ModelComparator": "Tensor_Prism/Analysis",
    "TensorPrism_AnalyzeModelWeights": "Tensor_Prism/Analysis",
    "TensorPrism_ApplyMergeRecipe": "Tensor_Prism/Analysis",
    
    # CLIP merging
    "TensorPrismAdvancedClipMerge": "Tensor_Prism/CLIP",
    
    # VAE merging
    "TensorPrismVAEMerge": "Tensor_Prism/VAE",
    
    # Conversion nodes
    "TensorPrism_EpsilonVPredConverter": "Tensor_Prism/Merge",
    
    # Noise Injection Merge
    "TensorPrism_NoiseInjectionMerge": "Tensor_Prism/Merge",
}

# ==================== UTILITY FUNCTIONS ====================
def check_dependencies():
    """Check if required dependencies are available."""
    required_packages = [
        ("torch", "PyTorch >= 1.12.0"),
        ("numpy", "NumPy >= 1.21.0"), 
        ("psutil", "psutil >= 5.8.0"),
    ]
    
    missing_packages = []
    
    for package_name, description in required_packages:
        try:
            importlib.import_module(package_name)
        except ImportError:
            missing_packages.append(description)
    
    if missing_packages:
        print(f"[TensorPrism] Warning: Missing dependencies:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("[TensorPrism] Some features may not work properly.")
    
    return len(missing_packages) == 0

def get_system_info():
    """Get system information for optimization."""
    try:
        import torch
        import psutil
        
        info = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
            "cpu_count": psutil.cpu_count(),
            "total_ram": round(psutil.virtual_memory().total / (1024**3), 1),
        }
        
        if info["cuda_available"]:
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_memory"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
        
        return info
    except Exception as e:
        print(f"[TensorPrism] Could not get system info: {e}")
        return {}

def print_welcome_message():
    """Print welcome message with system information."""
    print("\n" + "="*70)
    print("🎭 TensorPrism Node Pack v" + __version__)
    print("="*70)
    print(f"Total Nodes: {len(NODE_CLASS_MAPPINGS)}")
    
    # System info
    sys_info = get_system_info()
    if sys_info:
        print(f"\n📊 System Information:")
        print(f"  PyTorch: {sys_info.get('torch_version', 'Unknown')}")
        print(f"  CPU Cores: {sys_info.get('cpu_count', 'Unknown')}")
        print(f"  RAM: {sys_info.get('total_ram', 'Unknown')}GB")
        
        if sys_info.get("cuda_available"):
            print(f"  CUDA: ✓ ({sys_info.get('cuda_device_count', 0)} device(s))")
            print(f"  GPU Memory: {sys_info.get('cuda_memory', 'Unknown')}GB")
        elif sys_info.get("mps_available"):
            print(f"  MPS: ✓ (Apple Silicon)")
        else:
            print(f"  GPU: CPU fallback mode")
    
    # Node category breakdown
    print(f"\n🚀 Node Categories:")
    category_counts = {}
    for category in NODE_CATEGORIES.values():
        category_counts[category] = category_counts.get(category, 0) + 1
    
    for category in sorted(category_counts.keys()):
        print(f"  • {category}: {category_counts[category]} nodes")
    
    # Memory recommendations
    print(f"\n💡 Memory Recommendations:")
    gpu_mem = sys_info.get("cuda_memory", 0)
    if gpu_mem >= 24:
        print("  • 24GB+ GPU: Use default settings")
    elif gpu_mem >= 12:
        print("  • 12GB GPU: Set memory_limit_gb=8, use auto precision")
    elif gpu_mem >= 8:
        print("  • 8GB GPU: Set memory_limit_gb=6, CPU fallback for large merges")
    else:
        print("  • <8GB: CPU processing recommended for stability")
    
    print("="*70)
    print("✅ TensorPrism initialized successfully!")
    print("📝 v1.7.0: Advanced CLIP Merge with CLIP-L/G control")
    print("   • 3 files contain 10 core nodes (CoreMerge, SDXLMerge, MaskSystem)")
    print("   • 11 additional standalone feature nodes")
    print("   • Total: 21 production-ready nodes")
    print("="*70 + "\n")

# ==================== INITIALIZATION ====================
def __init_package():
    """Initialize the package and perform startup checks."""
    try:
        # Check dependencies
        deps_ok = check_dependencies()
        
        # Print welcome message
        print_welcome_message()
        
        if not deps_ok:
            print("[TensorPrism] ⚠️  Some dependencies missing - install for full functionality")
        
    except Exception as e:
        print(f"[TensorPrism] ❌ Initialization error: {e}")
        print("[TensorPrism] Package may not function correctly.")

# Run initialization
__init_package()

# Export for ComfyUI
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS", 
    "__version__",
    "__author__",
    "__description__"
]
