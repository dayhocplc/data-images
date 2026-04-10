"""
src/data/preprocessing.py
Face detection, alignment, and quality filtering pipeline.
Implements the 4-stage preprocessing described in paper Section 4.2.

Stage 1: Face detection (MediaPipe Face Mesh, confidence ≥ 0.9, pose ≤ 30°)
Stage 2: Alignment (inter-ocular axis horizontal)
Stage 3: Crop to bounding box + pad to 224×224 with neutral gray
Stage 4: Quality filter (composite blur-exposure score ≥ 0.6)

Retention rate: 2,821 / 2,936 = 96.1%
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
TARGET_SIZE    = 224
NEUTRAL_GRAY   = (128, 128, 128)    # Padding color (RGB)
MARGIN_PX      = 5                  # Bounding box margin
QUALITY_THRESH = 0.6                # Minimum composite quality score

# MediaPipe landmark indices for eye centers
# Left eye center: mean of landmarks 33, 133
# Right eye center: mean of landmarks 362, 263
LEFT_EYE_LANDMARKS  = [33, 133]
RIGHT_EYE_LANDMARKS = [362, 263]


class FacePreprocessor:
    """
    4-stage face preprocessing pipeline (paper Section 4.2).

    Usage:
        preprocessor = FacePreprocessor()
        result = preprocessor.process("path/to/image.jpg")
        if result is not None:
            processed_image = result  # PIL Image, 224×224
    """

    def __init__(
        self,
        target_size: int = TARGET_SIZE,
        margin_px: int = MARGIN_PX,
        quality_threshold: float = QUALITY_THRESH,
        min_detection_confidence: float = 0.9,
        max_pose_deviation_deg: float = 30.0,
    ):
        self.target_size          = target_size
        self.margin_px            = margin_px
        self.quality_threshold    = quality_threshold
        self.min_confidence       = min_detection_confidence
        self.max_pose_deg         = max_pose_deviation_deg

        self._mp_face_mesh = None
        self._face_mesh    = None

    def _init_mediapipe(self):
        """Lazy init MediaPipe Face Mesh."""
        if self._face_mesh is not None:
            return
        try:
            import mediapipe as mp
            self._mp_face_mesh = mp.solutions.face_mesh
            self._face_mesh = self._mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=self.min_confidence,
            )
        except ImportError:
            raise ImportError(
                "MediaPipe not installed. Run: pip install mediapipe>=0.9.0"
            )

    # ── Public API ────────────────────────────────────────────────────────

    def process(self, image_path: str | Path) -> Optional[Image.Image]:
        """
        Apply full 4-stage pipeline to an image.

        Returns:
            Preprocessed 224×224 PIL Image, or None if rejected.
        """
        img_path = Path(image_path)
        if not img_path.exists():
            logger.warning(f"Image not found: {img_path}")
            return None

        # Load image
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            logger.warning(f"Failed to load: {img_path}")
            return None
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Stage 1: Face detection
        landmarks = self._detect_landmarks(img_rgb)
        if landmarks is None:
            logger.debug(f"No face detected: {img_path.name}")
            return None

        # Check pose deviation
        pose_dev = self._estimate_pose_deviation(landmarks, img_rgb.shape)
        if pose_dev > self.max_pose_deg:
            logger.debug(
                f"Pose deviation {pose_dev:.1f}° > {self.max_pose_deg}°: {img_path.name}"
            )
            return None

        # Stage 2: Alignment (rotate so inter-ocular axis is horizontal)
        aligned = self._align_face(img_rgb, landmarks)

        # Stage 3: Crop + pad to target size
        cropped = self._crop_and_pad(aligned, landmarks)

        # Stage 4: Quality filter
        score = self._quality_score(cropped)
        if score < self.quality_threshold:
            logger.debug(
                f"Quality {score:.3f} < {self.quality_threshold}: {img_path.name}"
            )
            return None

        return Image.fromarray(cropped)

    def batch_process(
        self,
        image_paths: list,
        output_dir: Optional[Path] = None,
    ) -> dict:
        """
        Process a list of images and optionally save to output_dir.

        Returns:
            dict with "processed" (list of paths) and "rejected" (list of paths)
        """
        self._init_mediapipe()
        processed, rejected = [], []

        for img_path in image_paths:
            result = self.process(img_path)
            if result is None:
                rejected.append(str(img_path))
                continue

            if output_dir is not None:
                out_path = Path(output_dir) / Path(img_path).name
                result.save(str(out_path))
                processed.append(str(out_path))
            else:
                processed.append(str(img_path))

        logger.info(
            f"[Preprocessing] Processed: {len(processed)}/{len(image_paths)} "
            f"({len(processed)/max(len(image_paths),1)*100:.1f}%), "
            f"Rejected: {len(rejected)}"
        )
        return {"processed": processed, "rejected": rejected}

    # ── Stage 1: Face detection ───────────────────────────────────────────

    def _detect_landmarks(
        self, img_rgb: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Detect 468 facial landmarks using MediaPipe Face Mesh.

        Returns:
            (468, 2) array of (x, y) pixel coordinates, or None.
        """
        self._init_mediapipe()
        results = self._face_mesh.process(img_rgb)

        if not results.multi_face_landmarks:
            return None

        h, w = img_rgb.shape[:2]
        landmarks = results.multi_face_landmarks[0]
        points = np.array([
            [lm.x * w, lm.y * h]
            for lm in landmarks.landmark
        ], dtype=np.float32)
        return points

    # ── Stage 2: Alignment ────────────────────────────────────────────────

    def _align_face(
        self, img_rgb: np.ndarray, landmarks: np.ndarray
    ) -> np.ndarray:
        """
        Rotate image so inter-ocular axis is horizontal.
        θ_r = arctan(Δy / Δx) (paper Section 4.2)
        """
        left_eye  = landmarks[LEFT_EYE_LANDMARKS].mean(axis=0)
        right_eye = landmarks[RIGHT_EYE_LANDMARKS].mean(axis=0)

        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle_deg = math.degrees(math.atan2(dy, dx))

        h, w = img_rgb.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle_deg, scale=1.0)
        aligned = cv2.warpAffine(
            img_rgb, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        return aligned

    def _estimate_pose_deviation(
        self, landmarks: np.ndarray, shape: tuple
    ) -> float:
        """
        Estimate yaw pose deviation from facial symmetry.
        Returns approximate absolute yaw in degrees.
        """
        h, w = shape[:2]
        nose_tip   = landmarks[1]   # Nose tip landmark
        nose_bridge = landmarks[6]  # Nose bridge

        center_x = w / 2
        deviation_px = abs(nose_tip[0] - center_x)
        # Approximate conversion: 1° ≈ w/180 pixels at typical FOV
        return float(deviation_px / w * 180)

    # ── Stage 3: Crop + pad ───────────────────────────────────────────────

    def _crop_and_pad(
        self, img_rgb: np.ndarray, landmarks: np.ndarray
    ) -> np.ndarray:
        """
        Crop to facial bounding box (+ margin), pad to square,
        resize to 224×224 with neutral gray padding.
        """
        h, w = img_rgb.shape[:2]

        # Bounding box from landmarks
        x_min = max(0, int(landmarks[:, 0].min()) - self.margin_px)
        y_min = max(0, int(landmarks[:, 1].min()) - self.margin_px)
        x_max = min(w, int(landmarks[:, 0].max()) + self.margin_px)
        y_max = min(h, int(landmarks[:, 1].max()) + self.margin_px)

        face_crop = img_rgb[y_min:y_max, x_min:x_max]
        if face_crop.size == 0:
            # Fallback: return resized full image
            return cv2.resize(img_rgb, (self.target_size, self.target_size))

        # Pad to square with neutral gray
        fh, fw = face_crop.shape[:2]
        side = max(fh, fw)
        padded = np.full((side, side, 3), NEUTRAL_GRAY, dtype=np.uint8)
        y_offset = (side - fh) // 2
        x_offset = (side - fw) // 2
        padded[y_offset:y_offset+fh, x_offset:x_offset+fw] = face_crop

        # Resize to 224×224
        resized = cv2.resize(
            padded,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_LINEAR,
        )
        return resized

    # ── Stage 4: Quality filter ───────────────────────────────────────────

    def _quality_score(self, img_rgb: np.ndarray) -> float:
        """
        Composite blur-exposure quality score (paper Section 4.2).

        score = 0.5 × blur_score + 0.5 × exposure_score
        Retained if score ≥ 0.6.

        blur_score:     Laplacian variance normalized to [0,1]
        exposure_score: Distance from extreme pixel values
        """
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        # Blur: Laplacian variance
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = min(lap_var / 500.0, 1.0)

        # Exposure: distance from under/over exposure
        mean_pixel = gray.mean() / 255.0
        exposure_score = 1.0 - abs(mean_pixel - 0.5) * 2.0

        return 0.5 * blur_score + 0.5 * exposure_score


def run_preprocessing(
    data_root: str | Path,
    output_dir: str | Path,
    quality_threshold: float = QUALITY_THRESH,
) -> dict:
    """
    Preprocess all images in data_root and save to output_dir.
    Reports retention statistics.
    """
    data_root  = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = FacePreprocessor(quality_threshold=quality_threshold)
    all_images   = list(data_root.rglob("*.jpg")) + list(data_root.rglob("*.png"))

    logger.info(f"[Preprocessing] Found {len(all_images)} candidate images")
    stats = preprocessor.batch_process(all_images, output_dir=output_dir)

    retention = len(stats["processed"]) / max(len(all_images), 1)
    logger.info(
        f"[Preprocessing] Done. Retention: {len(stats['processed'])}/{len(all_images)} "
        f"= {retention*100:.1f}% (paper: 96.1%)"
    )
    return stats
