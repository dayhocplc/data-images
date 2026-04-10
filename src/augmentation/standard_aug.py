"""
src/augmentation/standard_aug.py
M1: Basic 2D augmentation (baseline configuration A1).

Paper Section 4.4 config A1:
  Rotation ±10°, Gaussian noise σ=0.01, brightness [0.8, 1.2]
  No compression, no fairness constraint.
  Result: F1=0.506, EOD_gender=12.1%, 27.8 MB — DFZ excluded.

This is the weakest configuration — intentionally minimal to establish
the lower bound and demonstrate the need for 3D-aware augmentation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image
from torchvision import transforms


class BasicAugmentation:
    """
    M1: Basic 2D augmentation.
    Applied only to training images.

    Transforms:
      - RandomRotation ±10°
      - Gaussian noise σ=0.01
      - ColorJitter brightness [0.8, 1.2]
      - RandomHorizontalFlip (p=0.5)

    Note: These 2D transforms do NOT preserve facial biomarkers
    (inter-ocular distance, facial proportions). Rotation and flipping
    can distort structural attributes by 15–25% [Kebaili et al., 41].
    This is why 3D-aware augmentation (M7) significantly outperforms M1
    on both accuracy and fairness.
    """

    def __init__(
        self,
        rotation_deg: float = 10.0,
        noise_std: float = 0.01,
        brightness: tuple = (0.8, 1.2),
        flip_prob: float = 0.5,
    ):
        self.rotation_deg = rotation_deg
        self.noise_std    = noise_std
        self.brightness   = brightness
        self.flip_prob    = flip_prob

        self._train_transform = transforms.Compose([
            transforms.RandomRotation(rotation_deg),
            transforms.RandomHorizontalFlip(p=flip_prob),
            transforms.ColorJitter(
                brightness=(brightness[0], brightness[1]),
                contrast=0.1,
                saturation=0.1,
            ),
        ])

    def __call__(
        self,
        image: Image.Image,
        metadata: Optional[dict] = None,
    ) -> Image.Image:
        """Apply M1 augmentation to a PIL image."""
        augmented = self._train_transform(image)

        # Add Gaussian noise (applied in numpy space)
        img_np = np.array(augmented, dtype=np.float32) / 255.0
        noise  = np.random.normal(0, self.noise_std, img_np.shape)
        img_np = np.clip(img_np + noise, 0, 1)
        return Image.fromarray((img_np * 255).astype(np.uint8))


def get_inference_transform(image_size: int = 224) -> transforms.Compose:
    """
    Standard inference transform — applied to ALL images at test time.
    Identical across all 11 configurations.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
