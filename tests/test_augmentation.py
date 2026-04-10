"""
tests/test_augmentation.py
Unit tests for augmentation pipelines and integration.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import torch
from PIL import Image

from src.augmentation.standard_aug import BasicAugmentation
from src.augmentation.aug_3d import AugConfig3D


class TestBasicAugmentation:
    """Tests for M1 basic 2D augmentation."""

    def _random_image(self, size=(224, 224)) -> Image.Image:
        arr = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
        return Image.fromarray(arr)

    def test_output_is_pil(self):
        aug = BasicAugmentation()
        img = self._random_image()
        out = aug(img)
        assert isinstance(out, Image.Image)

    def test_output_size_preserved(self):
        aug = BasicAugmentation()
        img = self._random_image((224, 224))
        out = aug(img)
        assert out.size == (224, 224)

    def test_augmentation_varies_output(self):
        """Output should differ from input (with high probability)."""
        aug = BasicAugmentation()
        img = self._random_image()
        np.random.seed(None)

        outputs = [np.array(aug(img)) for _ in range(10)]
        all_same = all(np.array_equal(outputs[0], o) for o in outputs[1:])
        assert not all_same, "Augmentation never changes image"

    def test_pixel_range_preserved(self):
        """Output pixels should remain in [0, 255]."""
        aug = BasicAugmentation()
        img = self._random_image()
        out = np.array(aug(img))
        assert out.min() >= 0 and out.max() <= 255


class TestAugConfig3D:
    """Tests for 3D augmentation configuration."""

    def test_n_variants(self):
        cfg = AugConfig3D(
            pose_yaw_angles=[-20, -10, 0, 10, 20],
            expressions=["neutral", "mild_smile"],
        )
        assert cfg.n_variants == 10  # 5 poses × 2 expressions

    def test_shape_preserved_flag(self):
        cfg = AugConfig3D()
        assert cfg.preserve_shape is True  # β_shape must never be modified

    def test_custom_config(self):
        cfg = AugConfig3D(
            pose_yaw_angles=[-15, 0, 15],
            expressions=["neutral"],
            expression_intensity=0.2,
        )
        assert cfg.n_variants == 3   # 3 poses × 1 expression
        assert cfg.expression_intensity == 0.2


class TestDatasetIntegration:
    """Integration tests for dataset loading."""

    def test_dataset_metadata_schema(self):
        """Dataset must have required columns."""
        import pandas as pd
        from src.data.dataset import ASDFaceDataset

        # Create minimal metadata CSV
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("image_path,label,gender,ethnicity,age_group,split\n")
            for i in range(20):
                f.write(
                    f"/tmp/fake_{i}.jpg,{i%2},"
                    f"{'male' if i%2==0 else 'female'},"
                    f"white,3-4,train\n"
                )
            tmp_path = f.name

        ds = ASDFaceDataset(tmp_path, split="train")
        assert "gender_id"    in ds.df.columns
        assert "ethnicity_id" in ds.df.columns
        assert "subgroup_id"  in ds.df.columns

    def test_class_weights_sum(self):
        """Class weights should reflect class imbalance direction."""
        import tempfile
        from src.data.dataset import ASDFaceDataset

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # 3x more ASD than non-ASD
            f.write("image_path,label,gender,ethnicity,age_group,split\n")
            for i in range(30):
                label = 1 if i < 22 else 0
                f.write(f"/tmp/img_{i}.jpg,{label},male,white,3-4,train\n")
            tmp_path = f.name

        ds = ASDFaceDataset(tmp_path, split="train")
        weights = ds.get_class_weights()
        # Non-ASD (minority here) should have higher weight
        assert weights[0] > weights[1], "Minority class should have higher weight"

    def test_subgroup_indices_coverage(self):
        """All 8 gender×ethnicity combinations should have indices."""
        import tempfile
        from src.data.dataset import ASDFaceDataset

        genders     = ['male', 'female']
        ethnicities = ['white', 'asian', 'black', 'dark']

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("image_path,label,gender,ethnicity,age_group,split\n")
            for g in genders:
                for e in ethnicities:
                    for i in range(3):
                        f.write(f"/tmp/{g}_{e}_{i}.jpg,1,{g},{e},3-4,train\n")
            tmp_path = f.name

        ds = ASDFaceDataset(tmp_path, split="train")
        for g in genders:
            for e in ethnicities:
                key = f"{g}_{e}"
                assert key in ds.subgroup_indices, f"Missing subgroup: {key}"


class TestLossFunctions:
    """Tests for trilemma loss components."""

    def test_accuracy_loss_perfect(self):
        """Perfect predictions → L_acc ≈ 0."""
        from src.training.losses import AccuracyLoss
        loss_fn = AccuracyLoss()
        logits = torch.tensor([10.0, 10.0, -10.0, -10.0])
        labels = torch.tensor([1.0,  1.0,   0.0,   0.0])
        loss = loss_fn(logits, labels)
        assert loss.item() < 0.05, f"Expected near 0, got {loss.item():.4f}"

    def test_accuracy_loss_range(self):
        """L_acc ∈ [0, 1]."""
        from src.training.losses import AccuracyLoss
        loss_fn = AccuracyLoss()
        for _ in range(20):
            logits = torch.randn(32)
            labels = torch.randint(0, 2, (32,)).float()
            loss = loss_fn(logits, labels)
            assert 0 <= loss.item() <= 1.0, f"L_acc={loss.item()} out of range"

    def test_fairness_loss_zero_disparity(self):
        """Equal TPR for all groups → L_fair ≈ 0."""
        from src.training.losses import FairnessLoss
        loss_fn = FairnessLoss()
        # All correct, equal groups
        logits = torch.tensor([5.0, 5.0, -5.0, -5.0, 5.0, 5.0, -5.0, -5.0])
        labels = torch.tensor([1.0, 1.0,  0.0,  0.0, 1.0, 1.0,  0.0,  0.0])
        gender = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        ethnicity = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0])
        loss = loss_fn(logits, labels, gender, ethnicity)
        assert loss.item() < 0.1, f"Expected near 0, got {loss.item():.4f}"

    def test_trilemma_loss_weights_sum_to_one(self):
        """After update_weights, α+β+γ should equal 1.0."""
        from src.training.losses import TrilemmaLoss
        loss_fn = TrilemmaLoss()
        for alpha, beta, gamma in [(0.7, 0.2, 0.1), (0.3, 0.4, 0.3), (0.4, 0.4, 0.2)]:
            loss_fn.update_weights(alpha, beta, gamma)
            total = loss_fn.alpha + loss_fn.beta + loss_fn.gamma
            assert abs(total - 1.0) < 1e-9, f"Weights sum={total}"

    def test_efficiency_loss_constraint_violation(self):
        """L_eff > 1 when size or latency exceeds constraint."""
        from src.training.losses import EfficiencyLoss
        loss_fn = EfficiencyLoss(size_constraint_mb=10.0, latency_constraint_ms=300.0)
        # Size = 27.8 MB > 10 MB → L_eff > 1
        l_eff = loss_fn.from_measurements(size_mb=27.8, latency_ms=214.0)
        assert l_eff > 1.0, f"Expected L_eff > 1 for oversized model, got {l_eff}"
        # Size = 6.3 MB, latency = 187 ms → within constraints
        l_eff_ok = loss_fn.from_measurements(size_mb=6.3, latency_ms=187.0)
        assert l_eff_ok <= 1.0, f"Expected L_eff <= 1 for DFZ config, got {l_eff_ok}"


class TestModelBackbone:
    """Tests for DenseNet-121 backbone."""

    def test_output_shape(self):
        from src.models.backbone import TrilemmaClassifier
        model = TrilemmaClassifier(pretrained=False)
        model.eval()
        x = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4,), f"Expected (4,), got {out.shape}"

    def test_feature_shape(self):
        from src.models.backbone import TrilemmaClassifier
        model = TrilemmaClassifier(pretrained=False)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            feat = model.forward_features(x)
        assert feat.shape == (2, 1024), f"Expected (2,1024), got {feat.shape}"

    def test_parameter_count(self):
        from src.models.backbone import TrilemmaClassifier
        model = TrilemmaClassifier(pretrained=False)
        params = model.get_num_parameters()
        # DenseNet-121 ≈ 8M params
        assert 6e6 < params["total"] < 12e6, (
            f"Expected ~8M params, got {params['total']/1e6:.1f}M"
        )

    def test_mobilenet_output_shape(self):
        from src.models.backbone import MobileNetV3Student
        model = MobileNetV3Student(pretrained=False)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2,)
