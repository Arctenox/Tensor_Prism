import torch
import comfy.utils

class TensorPrism_WeightedMaskMerge:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_A": ("MODEL",),
                "model_B": ("MODEL",),
                "mask": ("MASK",),
                "merge_ratio": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "help": "0.0 = Model A only, 1.0 = Model B only"
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("merged_model",)
    FUNCTION = "merge_models"
    CATEGORY = "Tensor Prism/Mask"

    def merge_models(self, model_A, model_B, mask, merge_ratio):
        """
        Blends two models (A, B) according to a mask and merge ratio.
        Mask defines where Model B is applied. Merge ratio controls how strongly B overrides A.
        The applied mask is also stored in the output model as 'mask_used'.
        """

        merged = {}

        # Ensure mask is broadcastable
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.ndim == 3:
            mask = mask.unsqueeze(1)
        elif mask.ndim == 4 and mask.shape[1] > 1:
            mask = mask[:, :1, :, :]

        for key in model_A.keys():
            if key in model_B:
                tA = model_A[key]
                tB = model_B[key]

                if isinstance(tA, torch.Tensor) and isinstance(tB, torch.Tensor):
                    # Resize mask if needed
                    if mask.shape[2:] != tA.shape[2:]:
                        mask_resized = torch.nn.functional.interpolate(
                            mask, size=tA.shape[2:], mode='bilinear', align_corners=False
                        )
                    else:
                        mask_resized = mask

                    blend_factor = mask_resized * merge_ratio
                    merged[key] = tA * (1 - blend_factor) + tB * blend_factor
                else:
                    merged[key] = tA
            else:
                merged[key] = model_A[key]

        # Save mask into output model
        merged["mask_used"] = mask

        return (merged,)


NODE_CLASS_MAPPINGS = {
    "TensorPrism_WeightedMaskMerge": TensorPrism_WeightedMaskMerge
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TensorPrism_WeightedMaskMerge": "Weighted Mask Merge (Tensor Prism)"
}