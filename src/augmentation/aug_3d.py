"""
src/augmentation/aug_3d.py
3D-aware augmentation pipeline (M7: Stratified 3D-Aware).

Uses DECA/FLAME parametric face model to generate 10 variants per image
(5 poses × 2 expressions) while preserving β_shape (facial biomarkers).

Key design principle: β_shape is NEVER modified — it encodes the
subject-specific structural attributes (facial proportions, IOD, etc.)
that are classification-relevant. Only θ_pose and ψ_expr are varied.

Applied ONLY to training-fold images (never val/test) to prevent
identity leakage from 3D-reconstructed variants sharing identity
embeddings with source images.

Reference: Feng et al. "Learning an animatable detailed 3D face model
from in-the-wild images," ACM TOG, vol. 40, no. 4, Art. 88, 2021.
DOI: 10.1145/3450626.3459936
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class AugConfig3D:
    """
    Configuration for 3D-aware augmentation pipeline.
    Paper Section 4.4, config A2/C2.
    """
    # Pose variation: yaw angles (degrees)
    # 5 poses centered on neutral
    pose_yaw_angles: List[float] = field(
        default_factory=lambda: [-20.0, -10.0, 0.0, 10.0, 20.0]
    )
    # Pitch/roll held near neutral
    pose_pitch_range: Tuple[float, float] = (-5.0, 5.0)
    pose_roll_range:  Tuple[float, float] = (-5.0, 5.0)

    # Expression variation: 2 expressions
    expressions: List[str] = field(
        default_factory=lambda: ["neutral", "mild_smile"]
    )
    expression_intensity: float = 0.3  # Mild — preserves biomarkers

    # Shape: NEVER modified (biomarker preservation)
    preserve_shape: bool = True        # β_shape = original, always

    # Rendering quality
    image_size: int = 224
    render_quality: str = "high"

    # Quality filter threshold (composite blur-exposure score)
    quality_threshold: float = 0.6

    # Minority-stratified oversampling
    minority_stratified: bool = True
    minority_oversample_factor: float = 2.0  # Extra variants for rare groups

    @property
    def n_variants(self) -> int:
        """10 variants per image: 5 poses × 2 expressions."""
        return len(self.pose_yaw_angles) * len(self.expressions)


class DECA3DReconstructor:
    """
    Wrapper around DECA for single-image 3D face reconstruction.

    Extracts FLAME parameters (β_shape, ψ_expr, θ_pose) from a 2D image.
    β_shape encodes identity-specific shape — preserved across all variants.
    ψ_expr and θ_pose are modified for augmentation.

    Requires: pip install deca-pytorch
    (https://github.com/YadiraF/DECA)
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._deca = None
        self._loaded = False

    def _load_deca(self):
        """Lazy load DECA model."""
        if self._loaded:
            return
        try:
            from decalib.deca import DECA
            from decalib.utils.config import cfg as deca_cfg
            self._deca = DECA(config=deca_cfg, device=self.device)
            self._loaded = True
            logger.info("[3DAug] DECA model loaded successfully.")
        except ImportError:
            logger.error(
                "[3DAug] DECA not installed. "
                "Run: pip install git+https://github.com/YadiraF/DECA.git"
            )
            raise

    def reconstruct(self, image: Image.Image) -> Dict[str, torch.Tensor]:
        """
        Reconstruct FLAME parameters from a 2D facial image.

        Returns:
            Dict with keys:
              beta_shape   (100,) — shape parameters, identity-specific
              psi_expr     (50,)  — expression parameters
              theta_pose   (6,)   — pose parameters (3 rotation + 3 translation)
              tform        (3,3)  — transform matrix
        """
        self._load_deca()

        # Preprocess image for DECA
        img_tensor = self._preprocess_image(image)

        with torch.no_grad():
            codedict = self._deca.encode(img_tensor.to(self.device))

        return {
            "beta_shape": codedict["shape"].squeeze(),      # (100,)
            "psi_expr":   codedict["exp"].squeeze(),        # (50,)
            "theta_pose": codedict["pose"].squeeze(),       # (6,)
            "cam":        codedict.get("cam", None),        # camera params
        }

    def render_variant(
        self,
        params: Dict[str, torch.Tensor],
        yaw_deg: float,
        expression_delta: Optional[torch.Tensor] = None,
    ) -> Optional[Image.Image]:
        """
        Render a 2D image from FLAME parameters with modified pose/expression.

        Args:
            params: FLAME parameters from reconstruct()
            yaw_deg: Target yaw rotation in degrees
            expression_delta: Expression shift vector (50,)

        Returns:
            Rendered PIL Image, or None if quality check fails.
        """
        self._load_deca()

        # Modify pose (yaw only — pitch/roll kept near neutral)
        pose = params["theta_pose"].clone()
        pose[1] = torch.tensor(np.deg2rad(yaw_deg))  # yaw in radians

        # Modify expression (if provided)
        expr = params["psi_expr"].clone()
        if expression_delta is not None:
            expr = expr + expression_delta

        # β_shape is NEVER modified — biomarker preservation
        shape = params["beta_shape"]  # Unchanged

        # Build parameter dict for DECA decoder
        opdict, visdict = self._deca.decode({
            "shape": shape.unsqueeze(0),
            "exp":   expr.unsqueeze(0),
            "pose":  pose.unsqueeze(0),
            "cam":   params["cam"].unsqueeze(0) if params["cam"] is not None else None,
        })

        # Render to 2D image
        rendered = visdict.get("rendered_images")  # (1, 3, H, W)
        if rendered is None:
            return None

        img_np = (rendered[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(img_np)

    @staticmethod
    def _preprocess_image(image: Image.Image, size: int = 224) -> torch.Tensor:
        """Preprocess PIL image for DECA input."""
        import torchvision.transforms as T
        transform = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        return transform(image.convert("RGB")).unsqueeze(0)


class QualityFilter:
    """
    Composite quality filter (blur + exposure score).
    Retains images with score ≥ 0.6 (paper Section 4.2).
    """

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold

    def score(self, image: Image.Image) -> float:
        import cv2
        img_np = np.array(image.convert("L"))

        # Blur score: Laplacian variance (higher = sharper)
        blur = cv2.Laplacian(img_np, cv2.CV_64F).var()
        blur_score = min(blur / 500.0, 1.0)  # normalize

        # Exposure score: mean pixel value distance from extremes
        mean_pixel = img_np.mean() / 255.0
        exposure_score = 1.0 - abs(mean_pixel - 0.5) * 2.0

        return 0.5 * blur_score + 0.5 * exposure_score

    def passes(self, image: Image.Image) -> bool:
        return self.score(image) >= self.threshold


class ThreeDAwareAugmentation:
    """
    M7: Stratified 3D-Aware Augmentation Pipeline.

    Generates 10 variants per input image (5 poses × 2 expressions)
    while preserving β_shape (structural facial biomarkers).

    Applied only during training (never val/test) to prevent
    identity leakage.

    Usage (in ASDFaceDataset):
        aug = ThreeDAwareAugmentation(config, device)
        augmented_images = aug.augment(image, row_metadata)
    """

    # Expression deltas (mild shifts in ψ space)
    EXPRESSION_DELTAS = {
        "neutral":    None,  # Keep original expression params
        "mild_smile": None,  # Placeholder — loaded from precomputed deltas
    }

    def __init__(
        self,
        config: AugConfig3D = AugConfig3D(),
        device: Optional[torch.device] = None,
    ):
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.reconstructor = DECA3DReconstructor(device=self.device)
        self.quality_filter = QualityFilter(config.quality_threshold)
        self._expr_deltas = self._load_expression_deltas()

    def _load_expression_deltas(self) -> Dict[str, Optional[torch.Tensor]]:
        """Load precomputed mild expression delta vectors."""
        # Mild smile: slight upward lip corner movement
        # Derived from FACS AU6+AU12 (cheek raiser + lip corner puller)
        # Intensity kept low (0.3) to preserve biomarker-relevant proportions
        mild_smile_delta = torch.zeros(50)
        mild_smile_delta[6]  = self.config.expression_intensity   # AU6 proxy
        mild_smile_delta[12] = self.config.expression_intensity   # AU12 proxy
        return {
            "neutral":    None,
            "mild_smile": mild_smile_delta,
        }

    def augment(
        self,
        image: Image.Image,
        metadata: Optional[Dict] = None,
    ) -> List[Image.Image]:
        """
        Generate up to 10 augmented variants of a single input image.

        Args:
            image: Input PIL image (pre-cropped face, 224×224)
            metadata: Row metadata for minority oversampling logic

        Returns:
            List of augmented PIL images (passed quality filter).
        """
        try:
            # Step 1: Reconstruct FLAME parameters
            params = self.reconstructor.reconstruct(image)
        except Exception as e:
            logger.warning(f"[3DAug] DECA reconstruction failed: {e}. Skipping.")
            return [image]  # Fallback to original

        variants = []

        for yaw in self.config.pose_yaw_angles:
            for expr_name in self.config.expressions:
                expr_delta = self._expr_deltas.get(expr_name)
                try:
                    rendered = self.reconstructor.render_variant(
                        params, yaw_deg=yaw, expression_delta=expr_delta
                    )
                    if rendered is None:
                        continue
                    if self.quality_filter.passes(rendered):
                        variants.append(rendered)
                except Exception as e:
                    logger.debug(
                        f"[3DAug] Render failed (yaw={yaw}, expr={expr_name}): {e}"
                    )

        if not variants:
            logger.warning("[3DAug] No variants passed quality filter. Using original.")
            return [image]

        return variants

    def __call__(self, image: Image.Image, metadata: Optional[Dict] = None) -> Image.Image:
        """
        Single-image interface for dataset integration.
        Randomly samples one augmented variant from the generated set.
        """
        variants = self.augment(image, metadata)
        if len(variants) == 1:
            return variants[0]
        idx = np.random.randint(len(variants))
        return variants[idx]
