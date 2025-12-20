"""
TensorPrism Model Fine-Tuner
=============================

Fine-tuning nodes for Stable Diffusion models with both simplified and advanced options.
Supports dataset loading from folder paths.

Author: Arctenox
Version: 1.0.0
License: GPL-3.0

WARNING: Fine-tuning requires substantial GPU memory (12GB+ recommended)
For production training, consider dedicated tools like kohya_ss, EveryDream2, or SimpleTuner
"""

import torch
import folder_paths
import os
from pathlib import Path


# ==================== DATASET PATH NODE ====================

class TensorPrism_DatasetPath:
    """
    Provides a dataset path input for the fine-tuner.
    Supports folder paths containing images and caption .txt files.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        # Try to get input folder, fallback to a default
        try:
            input_dir = folder_paths.get_input_directory()
        except:
            input_dir = ""
            
        return {
            "required": {
                "dataset_path": ("STRING", {
                    "default": input_dir,
                    "multiline": False,
                    "placeholder": "/path/to/dataset/folder"
                }),
            },
            "optional": {
                "validate_path": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("dataset_path", "info")
    FUNCTION = "get_path"
    CATEGORY = "Tensor_Prism/Training"
    
    def get_path(self, dataset_path, validate_path=True):
        """Return the dataset path and validation info"""
        
        info = f"Dataset Path: {dataset_path}\n"
        
        if validate_path:
            path_obj = Path(dataset_path)
            
            if not path_obj.exists():
                info += "âš ï¸ Path does not exist!\n"
                return (dataset_path, info)
            
            if not path_obj.is_dir():
                info += "âš ï¸ Path is not a directory!\n"
                return (dataset_path, info)
            
            # Count images
            image_exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
            images = []
            for ext in image_exts:
                images.extend(path_obj.rglob(f"*{ext}"))
                images.extend(path_obj.rglob(f"*{ext.upper()}"))
            
            # Count captions
            captions = list(path_obj.rglob("*.txt"))
            
            info += f"âœ… Found {len(images)} images\n"
            info += f"âœ… Found {len(captions)} caption files\n"
            
            # Check for matching pairs
            paired = 0
            for img in images:
                txt_path = img.with_suffix('.txt')
                if txt_path.exists():
                    paired += 1
            
            info += f"âœ… {paired} image-caption pairs\n"
            
            if paired == 0:
                info += "âš ï¸ No image-caption pairs found!\n"
                info += "Each image needs a .txt file with the same name.\n"
        
        return (dataset_path, info)


# ==================== SIMPLIFIED FINE-TUNER ====================

class TensorPrism_SimplifiedFineTuner:
    """
    Simplified fine-tuner with minimal configuration.
    Good for quick experimentation.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "dataset_path": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "preset": (["Fast (Low Quality)", "Balanced", "High Quality (Slow)"],),
                "epochs": ("INT", {"default": 5, "min": 1, "max": 100}),
                "output_name": ("STRING", {"default": "finetuned_model"}),
            },
            "optional": {
                "learning_rate": ("FLOAT", {
                    "default": 1e-6,
                    "min": 1e-8,
                    "max": 1e-4,
                    "step": 1e-7
                }),
            }
        }
    
    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")
    FUNCTION = "finetune_simplified"
    CATEGORY = "Tensor_Prism/Training"
    OUTPUT_NODE = True
    
    def finetune_simplified(self, model, dataset_path, preset, epochs, output_name, learning_rate=1e-6):
        """Simplified fine-tuning with preset configurations"""
        
        print("\n" + "="*60)
        print("ðŸŽ“ SIMPLIFIED FINE-TUNER (TensorPrism)")
        print("="*60)
        print(f"Preset: {preset}")
        print(f"Dataset: {dataset_path}")
        print(f"Epochs: {epochs}")
        print(f"Output: {output_name}")
        print("="*60 + "\n")
        
        # Configure based on preset
        if preset == "Fast (Low Quality)":
            batch_size = 2
            grad_accum = 4
            resolution = 512
        elif preset == "Balanced":
            batch_size = 1
            grad_accum = 8
            resolution = 768
        else:  # High Quality
            batch_size = 1
            grad_accum = 16
            resolution = 1024
        
        report = f"""
=== SIMPLIFIED FINE-TUNING REPORT ===
Status: Training Started
Preset: {preset}
Dataset: {dataset_path}
Epochs: {epochs}
Learning Rate: {learning_rate}
Resolution: {resolution}x{resolution}
Batch Size: {batch_size} Ã— {grad_accum} accumulation

âš ï¸ NOTE: This is a simplified proof-of-concept.
For production training, please use:
- kohya_ss (https://github.com/kohya-ss/sd-scripts)
- EveryDream2 (https://github.com/victorchall/EveryDream2trainer)
- SimpleTuner (https://github.com/bghira/SimpleTuner)

These tools provide proper diffusion training loops,
optimizers, validation, and checkpoint management.
"""
        
        print(report)
        print("\n" + "="*60)
        print("âš ï¸ IMPORTANT: Full training not implemented")
        print("="*60)
        print("This node is a placeholder for the training interface.")
        print("Model returned unchanged.")
        print("="*60 + "\n")
        
        return (model, report)


# ==================== ADVANCED FINE-TUNER ====================

class TensorPrism_AdvancedFineTuner:
    """
    Advanced fine-tuner with full control over training parameters.
    Requires understanding of diffusion model training.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "dataset_path": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                
                # Core Training
                "learning_rate": ("FLOAT", {
                    "default": 1e-6,
                    "min": 1e-8,
                    "max": 1e-3,
                    "step": 1e-7
                }),
                "epochs": ("INT", {"default": 10, "min": 1, "max": 1000}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 16}),
                "gradient_accumulation": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 64
                }),
                
                # Resolution & Data
                "resolution": ([512, 768, 1024],),
                "repeats": ("INT", {"default": 1, "min": 1, "max": 100}),
                
                # Optimizer
                "optimizer": (["AdamW", "AdamW8bit", "Lion"],),
                "lr_scheduler": (["constant", "cosine", "linear"],),
                
                # Output
                "output_name": ("STRING", {"default": "finetuned_model"}),
            },
            "optional": {
                # Precision & Memory
                "use_fp16": ("BOOLEAN", {"default": True}),
                "gradient_checkpointing": ("BOOLEAN", {"default": True}),
                
                # Text Encoder
                "train_text_encoder": ("BOOLEAN", {"default": False}),
                "text_encoder_lr": ("FLOAT", {
                    "default": 5e-7,
                    "min": 1e-8,
                    "max": 1e-4,
                    "step": 1e-8
                }),
                
                # Regularization
                "weight_decay": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 0.1, "step": 0.001}),
                "max_grad_norm": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "caption_dropout": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.5, "step": 0.05}),
                
                # Advanced
                "noise_offset": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.01}),
                "lr_warmup_steps": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 100}),
                
                # Data Augmentation
                "random_flip": ("BOOLEAN", {"default": True}),
                "center_crop": ("BOOLEAN", {"default": True}),
                
                # Misc
                "seed": ("INT", {"default": 42, "min": 0, "max": 2147483647}),
                "save_every_n_epochs": ("INT", {"default": 5, "min": 1, "max": 100}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")
    FUNCTION = "finetune_advanced"
    CATEGORY = "Tensor_Prism/Training"
    OUTPUT_NODE = True
    
    def finetune_advanced(self, model, dataset_path, learning_rate, epochs, batch_size,
                         gradient_accumulation, resolution, repeats, optimizer, lr_scheduler,
                         output_name, **kwargs):
        """Advanced fine-tuning with full parameter control"""
        
        print("\n" + "="*60)
        print("ðŸŽ“ ADVANCED FINE-TUNER (TensorPrism)")
        print("="*60)
        print(f"Dataset: {dataset_path}")
        print(f"Epochs: {epochs}")
        print(f"Learning Rate: {learning_rate}")
        print(f"Batch Size: {batch_size} Ã— {gradient_accumulation}")
        print(f"Resolution: {resolution}x{resolution}")
        print(f"Optimizer: {optimizer}")
        print(f"Scheduler: {lr_scheduler}")
        print(f"Output: {output_name}")
        print("="*60 + "\n")
        
        # Extract optional parameters
        use_fp16 = kwargs.get('use_fp16', True)
        gradient_checkpointing = kwargs.get('gradient_checkpointing', True)
        train_text_encoder = kwargs.get('train_text_encoder', False)
        text_encoder_lr = kwargs.get('text_encoder_lr', 5e-7)
        weight_decay = kwargs.get('weight_decay', 0.01)
        max_grad_norm = kwargs.get('max_grad_norm', 1.0)
        caption_dropout = kwargs.get('caption_dropout', 0.1)
        noise_offset = kwargs.get('noise_offset', 0.0)
        lr_warmup_steps = kwargs.get('lr_warmup_steps', 0)
        random_flip = kwargs.get('random_flip', True)
        center_crop = kwargs.get('center_crop', True)
        seed = kwargs.get('seed', 42)
        save_every_n_epochs = kwargs.get('save_every_n_epochs', 5)
        
        report = f"""
=== ADVANCED FINE-TUNING REPORT ===
Status: Configuration Validated
Dataset: {dataset_path}

=== Training Configuration ===
Epochs: {epochs}
Batch Size: {batch_size}
Gradient Accumulation: {gradient_accumulation}
Effective Batch: {batch_size * gradient_accumulation}
Learning Rate: {learning_rate}
Resolution: {resolution}x{resolution}
Dataset Repeats: {repeats}x

=== Optimizer & Scheduler ===
Optimizer: {optimizer}
LR Scheduler: {lr_scheduler}
Weight Decay: {weight_decay}
Max Grad Norm: {max_grad_norm}
Warmup Steps: {lr_warmup_steps}

=== Model Components ===
Train UNet: Yes
Train Text Encoder: {train_text_encoder}
Text Encoder LR: {text_encoder_lr if train_text_encoder else 'N/A'}

=== Memory & Precision ===
FP16: {use_fp16}
Gradient Checkpointing: {gradient_checkpointing}

=== Regularization ===
Caption Dropout: {caption_dropout}
Noise Offset: {noise_offset}

=== Data Augmentation ===
Random Flip: {random_flip}
Center Crop: {center_crop}

=== Misc ===
Seed: {seed}
Save Every: {save_every_n_epochs} epochs

âš ï¸ IMPORTANT NOTE:
This is a training interface placeholder.
Full diffusion model training requires:

1. Proper noise scheduler implementation
2. CLIP text encoding integration
3. VAE latent space handling
4. Timestep sampling
5. Loss weighting strategies
6. Checkpoint saving/loading

For production training, please use dedicated tools:
- kohya_ss: https://github.com/kohya-ss/sd-scripts
- EveryDream2: https://github.com/victorchall/EveryDream2trainer  
- SimpleTuner: https://github.com/bghira/SimpleTuner

These tools provide complete training pipelines with:
- Proper diffusion training loops
- EMA model support
- Advanced schedulers
- Validation & logging
- LoRA/Dreambooth support
- Multi-GPU training
"""
        
        print(report)
        print("\n" + "="*60)
        print("âš ï¸ TRAINING NOT EXECUTED")
        print("="*60)
        print("This node validates configuration and provides a template.")
        print("Model returned unchanged.")
        print("="*60 + "\n")
        
        return (model, report)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "TensorPrism_DatasetPath": TensorPrism_DatasetPath,
    "TensorPrism_SimplifiedFineTuner": TensorPrism_SimplifiedFineTuner,
    "TensorPrism_AdvancedFineTuner": TensorPrism_AdvancedFineTuner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TensorPrism_DatasetPath": "Dataset Path (Tensor Prism)",
    "TensorPrism_SimplifiedFineTuner": "Simplified Fine-Tuner (Tensor Prism)",
    "TensorPrism_AdvancedFineTuner": "Advanced Fine-Tuner (Tensor Prism)",
}