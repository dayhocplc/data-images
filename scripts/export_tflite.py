#!/usr/bin/env python3
"""
scripts/export_tflite.py
Export a trained PyTorch model to TFLite format for mobile benchmarking.

Used to measure model size (post-TFLite conversion) and enable
E3 latency measurement on MediaTek Helio G88 / Redmi Note 11.

Paper Section 4.5.1:
  "Mobile inference used TensorFlow Lite with XNNPACK delegate (float32)
   or INT8 delegate for quantized configurations."
  "Model size was measured post-conversion."

Usage:
    # Float32 (for A1, A2, B1, C2, C4)
    python scripts/export_tflite.py \\
        --checkpoint outputs/C2/C2_final.pt \\
        --output outputs/C2/C2.tflite

    # INT8 (for B2, B4, C3)
    python scripts/export_tflite.py \\
        --checkpoint outputs/B2/B2_final.pt \\
        --output outputs/B2/B2_int8.tflite \\
        --quantize int8 \\
        --calibration_dir data/processed/train

After export, deploy to E3 device and run:
    adb push C2.tflite /data/local/tmp/
    adb shell "cd /data/local/tmp && benchmark_model \\
        --graph=C2.tflite \\
        --num_threads=4 \\
        --num_runs=500 \\
        --warmup_runs=20"
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.backbone import build_model

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)


def export_tflite_float32(
    checkpoint: str,
    output: str,
    backbone: str = "densenet121",
):
    """Export float32 TFLite model."""
    model = build_model(backbone=backbone, pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)

    try:
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = Path(tmp) / "model.onnx"
            torch.onnx.export(
                model, dummy, str(onnx_path), opset_version=13,
                input_names=["image"], output_names=["logit"],
            )

            onnx_model = onnx.load(str(onnx_path))
            tf_rep = prepare(onnx_model)
            tf_path = Path(tmp) / "tf_model"
            tf_rep.export_graph(str(tf_path))

            converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_path))
            tflite_model = converter.convert()

        with open(output, "wb") as f:
            f.write(tflite_model)

        size_mb = os.path.getsize(output) / 1e6
        logger.info(f"Float32 TFLite: {output} ({size_mb:.2f} MB)")
        return size_mb

    except ImportError as e:
        logger.error(
            f"TFLite export requires onnx-tf + tensorflow. "
            f"Missing: {e}\n"
            f"Run: pip install onnx onnx-tf tensorflow"
        )
        return None


def export_tflite_int8(
    checkpoint: str,
    output: str,
    calibration_dir: str,
    n_calibration: int = 200,
    backbone: str = "densenet121",
):
    """Export INT8 quantized TFLite model."""
    from torch.utils.data import DataLoader
    from src.data.dataset import ASDFaceDataset

    model = build_model(backbone=backbone, pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()

    cal_ds = ASDFaceDataset(
        Path(calibration_dir) / "metadata_train.csv",
        split="train",
    )
    cal_loader = DataLoader(cal_ds, batch_size=1, shuffle=True)

    from src.compression.quantization import quantize_int8
    quantized = quantize_int8(
        model, cal_loader,
        device=torch.device("cpu"),
        n_calibration=n_calibration,
        output_path=output,
    )

    if os.path.exists(output):
        size_mb = os.path.getsize(output) / 1e6
        logger.info(f"INT8 TFLite: {output} ({size_mb:.2f} MB)")
        return size_mb
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",      required=True)
    parser.add_argument("--output",          required=True)
    parser.add_argument("--backbone",        default="densenet121")
    parser.add_argument("--quantize",        choices=["none", "int8"], default="none")
    parser.add_argument("--calibration_dir", default="data/processed")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    if args.quantize == "int8":
        export_tflite_int8(
            args.checkpoint, args.output,
            args.calibration_dir,
            backbone=args.backbone,
        )
    else:
        export_tflite_float32(
            args.checkpoint, args.output,
            backbone=args.backbone,
        )

    logger.info("\nDeploy to E3 device for latency measurement:")
    logger.info(f"  adb push {args.output} /data/local/tmp/")
    logger.info(
        f"  adb shell 'benchmark_model "
        f"--graph=/data/local/tmp/{Path(args.output).name} "
        f"--num_threads=4 --num_runs=500 --warmup_runs=20'"
    )


if __name__ == "__main__":
    sys.exit(main())
