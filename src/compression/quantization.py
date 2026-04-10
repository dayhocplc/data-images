"""
src/compression/quantization.py
INT8 post-training quantization and knowledge distillation.

INT8 (B2, B4, C3):
  Post-training static INT8 quantization via TensorFlow Lite.
  Calibrated on 200-image subset of training data.
  Paper result B2: F1=0.919, 7.1 MB, 142 ms (E3).

Knowledge Distillation (B3):
  Teacher: DenseNet-121 (A2 model, 3D-augmented)
  Student: MobileNetV3-Small
  T=4, λ=0.7 soft-target weight
  Paper result B3: F1=0.891, 4.2 MB, 96 ms (E3).

Note: B3 uses default hyperparameters from Hinton et al. [10] without
tuning — a deliberate controlled-comparison design choice (Section 6.3).
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

logger = logging.getLogger(__name__)


# ── INT8 Quantization ────────────────────────────────────────────────────────

def quantize_int8(
    model: nn.Module,
    calibration_loader: DataLoader,
    device: torch.device,
    n_calibration: int = 200,
    output_path: Optional[str] = None,
) -> nn.Module:
    """
    Post-training static INT8 quantization via TFLite.
    Paper Section 4.4 configs B2, B4, C3.

    Workflow:
      1. Export PyTorch model → ONNX
      2. Convert ONNX → TFLite float32
      3. Apply INT8 quantization with representative dataset calibration
      4. Wrap TFLite interpreter as nn.Module for unified evaluation

    Args:
        model:              Trained PyTorch model
        calibration_loader: DataLoader for calibration (200 images)
        device:             CPU recommended for TFLite export
        n_calibration:      Number of calibration samples (paper: 200)
        output_path:        Optional path to save .tflite file

    Returns:
        TFLiteWrapper (nn.Module interface for evaluation)
    """
    logger.info(f"[INT8] Starting post-training quantization (n_cal={n_calibration})")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # ── Step 1: Export to ONNX ───────────────────────────────────────
        onnx_path = tmp_path / "model.onnx"
        model.eval().cpu()
        dummy = torch.randn(1, 3, 224, 224)

        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            opset_version=13,
            input_names=["image"],
            output_names=["logit"],
            dynamic_axes={
                "image": {0: "batch"},
                "logit": {0: "batch"},
            },
        )
        logger.info(f"[INT8] ONNX exported: {onnx_path}")

        # ── Step 2–3: Convert to TFLite INT8 ─────────────────────────────
        try:
            import onnx
            from onnx_tf.backend import prepare
            import tensorflow as tf

            # ONNX → TF SavedModel
            onnx_model = onnx.load(str(onnx_path))
            tf_rep = prepare(onnx_model)
            tf_model_path = tmp_path / "tf_model"
            tf_rep.export_graph(str(tf_model_path))

            # TF → TFLite INT8 with representative dataset
            converter = tf.lite.TFLiteConverter.from_saved_model(
                str(tf_model_path)
            )
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8
            ]
            converter.inference_input_type  = tf.int8
            converter.inference_output_type = tf.int8

            # Calibration dataset (200 images)
            cal_images = _collect_calibration_images(
                calibration_loader, n_calibration
            )

            def representative_dataset():
                for img in cal_images:
                    yield [img.numpy().astype(np.float32)]

            converter.representative_dataset = representative_dataset
            tflite_model = converter.convert()

        except ImportError:
            logger.warning(
                "[INT8] onnx-tf or tensorflow not found. "
                "Using PyTorch static quantization as fallback."
            )
            return _pytorch_int8_fallback(model, calibration_loader)

        # Save .tflite file
        tflite_path = output_path or str(tmp_path / "model_int8.tflite")
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)

        logger.info(
            f"[INT8] TFLite INT8 model: {Path(tflite_path).stat().st_size / 1e6:.2f} MB"
        )

    return TFLiteWrapper(tflite_path)


def _collect_calibration_images(
    loader: DataLoader, n: int = 200
) -> list:
    """Collect n images from DataLoader for INT8 calibration."""
    images = []
    for batch in loader:
        for img in batch["image"]:
            images.append(img.unsqueeze(0))
            if len(images) >= n:
                return images
    return images


def _pytorch_int8_fallback(
    model: nn.Module, calibration_loader: DataLoader
) -> nn.Module:
    """
    PyTorch static quantization fallback (when TFLite not available).
    For development/testing — production should use TFLite.
    """
    model.eval().cpu()
    model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
    torch.quantization.prepare(model, inplace=True)

    # Calibrate
    with torch.no_grad():
        n = 0
        for batch in calibration_loader:
            model(batch["image"].cpu())
            n += len(batch["image"])
            if n >= 200:
                break

    torch.quantization.convert(model, inplace=True)
    logger.info("[INT8] PyTorch static quantization applied (fallback)")
    return model


class TFLiteWrapper(nn.Module):
    """
    Wraps a TFLite interpreter as an nn.Module for unified evaluation.
    Used for INT8 quantized configurations (B2, B4, C3).
    """

    def __init__(self, tflite_path: str):
        super().__init__()
        self.tflite_path = tflite_path
        self._interpreter = None

    def _load_interpreter(self):
        if self._interpreter is not None:
            return
        import tensorflow as tf
        self._interpreter = tf.lite.Interpreter(model_path=self.tflite_path)
        self._interpreter.allocate_tensors()
        self._input_details  = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run inference through TFLite interpreter."""
        self._load_interpreter()
        outputs = []
        for img in x:
            img_np = img.unsqueeze(0).numpy()
            self._interpreter.set_tensor(
                self._input_details[0]["index"], img_np
            )
            self._interpreter.invoke()
            out = self._interpreter.get_tensor(
                self._output_details[0]["index"]
            )
            outputs.append(float(out.squeeze()))
        return torch.tensor(outputs)


# ── Knowledge Distillation ───────────────────────────────────────────────────

class KnowledgeDistillationLoss(nn.Module):
    """
    Knowledge Distillation loss (Hinton et al. [10]).

    L_KD = (1-λ)·L_CE(hard) + λ·L_KL(soft, T)

    Paper config B3: T=4, λ=0.7
    Teacher: DenseNet-121 (A2 pipeline)
    Student: MobileNetV3-Small
    """

    def __init__(self, temperature: float = 4.0, lambda_soft: float = 0.7):
        super().__init__()
        self.T = temperature
        self.lam = lambda_soft

    def forward(
        self,
        student_logits: torch.Tensor,   # (B,)
        teacher_logits: torch.Tensor,   # (B,)
        labels: torch.Tensor,           # (B,) binary
    ) -> dict:
        # Hard label loss
        l_ce = F.binary_cross_entropy_with_logits(student_logits, labels)

        # Soft label loss (KL divergence with temperature scaling)
        p_teacher = torch.sigmoid(teacher_logits / self.T)
        p_student  = torch.sigmoid(student_logits / self.T)
        # KL for binary: p_t * log(p_t / p_s)
        eps = 1e-8
        l_kl = (
            p_teacher * torch.log((p_teacher + eps) / (p_student + eps))
            + (1 - p_teacher) * torch.log(
                (1 - p_teacher + eps) / (1 - p_student + eps)
            )
        ).mean() * (self.T ** 2)

        total = (1 - self.lam) * l_ce + self.lam * l_kl

        return {"loss": total, "l_ce": l_ce.detach(), "l_kl": l_kl.detach()}


def distill_to_mobilenet(
    teacher: nn.Module,
    student: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
    device: torch.device,
    output_dir: Path,
    temperature: float = 4.0,
    lambda_soft: float = 0.7,
    max_epochs: int = 150,
    patience: int = 10,
) -> nn.Module:
    """
    Train MobileNetV3-Small student via knowledge distillation.
    Paper config B3.

    Args:
        teacher:     Pre-trained DenseNet-121 (A2 model, frozen)
        student:     MobileNetV3-Small to be trained
        temperature: Distillation temperature T (paper: T=4)
        lambda_soft: Soft-target weight λ (paper: λ=0.7)

    Returns:
        Best student model (highest val-F1)
    """
    from src.training.trainer import EarlyStopping

    logger.info(
        f"[KD] Starting distillation: T={temperature}, λ={lambda_soft}, "
        f"max_epochs={max_epochs}"
    )

    teacher.eval().to(device)
    student.train().to(device)

    # Freeze teacher
    for p in teacher.parameters():
        p.requires_grad_(False)

    kd_loss_fn = KnowledgeDistillationLoss(temperature, lambda_soft)
    optimizer  = torch.optim.AdamW(
        student.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=cfg["training"]["lr_min"]
    )
    early_stopping = EarlyStopping(patience=patience)

    from src.evaluation.metrics import evaluate

    for epoch in range(max_epochs):
        student.train()
        total_loss = 0.0
        n = 0

        for batch in train_loader:
            imgs   = batch["image"].to(device)
            labels = batch["label"].to(device)

            with torch.no_grad():
                teacher_logits = teacher(imgs)

            student_logits = student(imgs)
            loss_dict = kd_loss_fn(student_logits, teacher_logits, labels)

            optimizer.zero_grad()
            loss_dict["loss"].backward()
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss_dict["loss"].item()
            n += 1

        scheduler.step()

        # Validate
        val_m = evaluate(student, val_loader, device, n_bootstrap=0)
        stop  = early_stopping.step(val_m.f1, student)

        if epoch % 10 == 0 or stop:
            logger.info(
                f"[KD] Epoch {epoch:3d}: "
                f"loss={total_loss/max(n,1):.4f}, "
                f"val_F1={val_m.f1:.4f}, "
                f"EOD_g={val_m.eod_gender*100:.1f}%"
            )

        if stop:
            logger.info(
                f"[KD] Early stopping at epoch {epoch}. "
                f"Best val-F1: {early_stopping.best_score:.4f}"
            )
            break

    if early_stopping.best_weights:
        student.load_state_dict(early_stopping.best_weights)

    return student
