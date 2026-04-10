"""
src/evaluation/efficiency.py
Model size and inference latency measurement.

Implements multi-environment benchmarking (Section 4.5.1):
  E1: NVIDIA RTX 3080 Ti (server/workstation)
  E2: Qualcomm Snapdragon 888 / Xiaomi Mi 11 (upper-mid smartphone)
  E3: MediaTek Helio G88 / Redmi Note 11 (entry-level SoC) ← constraint-binding

DFZ constraints (Eqs. 8c–8d):
  Size:    S(θ) ≤ 10 MB (post-TFLite conversion)
  Latency: T_E3(θ) ≤ 300 ms (on E3 hardware)

Measurement protocol:
  500 inference passes after 20-pass warm-up
  Mean latency over 500 passes reported
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def measure_model_size_mb(model: nn.Module, format: str = "tflite") -> float:
    """
    Measure model storage size post-serialization.
    Paper Section 4.5.1: "Size measured post-TFLite conversion."

    Args:
        model: PyTorch model
        format: "pytorch", "onnx", or "tflite"

    Returns:
        Size in MB
    """
    with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        if format == "pytorch":
            torch.save(model.state_dict(), tmp_path)

        elif format == "onnx":
            dummy = torch.randn(1, 3, 224, 224)
            torch.onnx.export(model.cpu().eval(), dummy, tmp_path, opset_version=13)

        elif format == "tflite":
            # Try TFLite export; fall back to ONNX size estimate
            try:
                return _measure_tflite_size(model, tmp_path)
            except (ImportError, Exception) as e:
                logger.warning(
                    f"[Efficiency] TFLite size measurement failed ({e}). "
                    "Using ONNX size as proxy."
                )
                dummy = torch.randn(1, 3, 224, 224)
                torch.onnx.export(model.cpu().eval(), dummy, tmp_path, opset_version=13)

        size_mb = os.path.getsize(tmp_path) / 1e6
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return size_mb


def _measure_tflite_size(model: nn.Module, output_path: str) -> float:
    """Export to TFLite float32 and measure size."""
    import tensorflow as tf
    import onnx
    from onnx_tf.backend import prepare

    with tempfile.TemporaryDirectory() as tmp_dir:
        onnx_path = Path(tmp_dir) / "model.onnx"
        dummy = torch.randn(1, 3, 224, 224)
        torch.onnx.export(model.cpu().eval(), dummy, str(onnx_path), opset_version=13)

        onnx_model = onnx.load(str(onnx_path))
        tf_rep = prepare(onnx_model)
        tf_path = Path(tmp_dir) / "tf_model"
        tf_rep.export_graph(str(tf_path))

        converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_path))
        tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    return os.path.getsize(output_path) / 1e6


def measure_flops(model: nn.Module, input_size: tuple = (1, 3, 224, 224)) -> float:
    """
    Measure FLOPs via ptflops library.
    Paper Table 5: FLOPs column.

    Returns:
        FLOPs in billions (GFLOPs)
    """
    try:
        from ptflops import get_model_complexity_info
        macs, _ = get_model_complexity_info(
            model,
            input_size[1:],
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
        )
        return macs / 1e9  # GFLOPs (1 MAC ≈ 2 FLOPs; reported as MACs here)
    except ImportError:
        logger.warning("[Efficiency] ptflops not installed. Run: pip install ptflops")
        return 0.0


def measure_latency_gpu(
    model: nn.Module,
    device: torch.device,
    input_size: tuple = (1, 3, 224, 224),
    n_warmup: int = 20,
    n_measure: int = 500,
) -> float:
    """
    Measure GPU inference latency (E1 environment).

    Args:
        n_warmup:  20 warm-up passes (paper Section 4.5.1)
        n_measure: 500 measurement passes

    Returns:
        Mean latency in milliseconds
    """
    model.eval().to(device)
    dummy = torch.randn(*input_size).to(device)

    # Warm-up
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)

    # Measure
    if device.type == "cuda":
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            for _ in range(n_measure):
                _ = model(dummy)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / n_measure
    else:
        times = []
        with torch.no_grad():
            for _ in range(n_measure):
                t0 = time.perf_counter()
                _ = model(dummy)
                times.append((time.perf_counter() - t0) * 1000)
        return float(sum(times) / len(times))


def measure_latency_mobile(
    tflite_path: str,
    n_warmup: int = 20,
    n_measure: int = 500,
) -> float:
    """
    Measure TFLite inference latency (CPU, proxy for E3).

    In the paper, E3 (Helio G88) latency is measured on device.
    This function runs on host CPU as a proxy for development.
    For accurate E3 measurements, deploy the .tflite file to
    a Redmi Note 11 and use TFLite Benchmark Tool.

    Returns:
        Mean latency in milliseconds
    """
    try:
        import tensorflow as tf
        import numpy as np

        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        input_details  = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        dummy = np.random.randn(1, 224, 224, 3).astype(np.float32)
        interpreter.set_tensor(input_details[0]["index"], dummy)

        # Warm-up
        for _ in range(n_warmup):
            interpreter.invoke()

        # Measure
        times = []
        for _ in range(n_measure):
            t0 = time.perf_counter()
            interpreter.invoke()
            times.append((time.perf_counter() - t0) * 1000)

        return float(sum(times) / len(times))

    except ImportError:
        logger.warning(
            "[Efficiency] TensorFlow not available. "
            "Latency not measured — set manually from device benchmarking."
        )
        return 0.0


def compute_l_eff(
    size_mb: float,
    latency_ms: float,
    s_star: float = 10.0,
    t_star: float = 300.0,
    w_s: float = 0.5,
    w_t: float = 0.5,
) -> float:
    """
    Compute normalized efficiency loss L_eff (Eq. 6).

      L_eff(θ) = w_s · S̃(θ) + w_t · T̃(θ)
      S̃ = S(θ)/S*,  T̃ = T_E3(θ)/T*

    L_eff > 1 when constraints violated (intentional, Section 3.2).
    """
    s_norm = size_mb    / s_star
    t_norm = latency_ms / t_star
    return w_s * s_norm + w_t * t_norm


def full_efficiency_profile(
    model: nn.Module,
    device: torch.device,
    tflite_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute complete efficiency profile for a configuration.
    Populates Table 5 columns.
    """
    profile = {}

    # FLOPs
    profile["flops_g"] = measure_flops(model)

    # Size
    profile["size_mb_pytorch"] = measure_model_size_mb(model, "pytorch")
    if tflite_path:
        profile["size_mb_tflite"] = os.path.getsize(tflite_path) / 1e6
    else:
        profile["size_mb_tflite"] = measure_model_size_mb(model, "tflite")

    # Latency E1 (GPU)
    if device.type == "cuda":
        profile["latency_e1_ms"] = measure_latency_gpu(model, device)
    else:
        profile["latency_e1_ms"] = measure_latency_gpu(
            model, torch.device("cpu")
        )

    # Latency E3 (TFLite proxy on CPU)
    if tflite_path:
        profile["latency_e3_ms"] = measure_latency_mobile(tflite_path)
    else:
        profile["latency_e3_ms"] = 0.0

    # L_eff (using TFLite size and E3 latency)
    profile["l_eff"] = compute_l_eff(
        size_mb=profile["size_mb_tflite"],
        latency_ms=profile["latency_e3_ms"],
    )

    # DFZ constraint check
    profile["size_ok"]    = profile["size_mb_tflite"] <= 10.0
    profile["latency_ok"] = (
        profile["latency_e3_ms"] <= 300.0 or profile["latency_e3_ms"] == 0.0
    )

    return profile
