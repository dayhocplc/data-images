"""
src/models/backbone.py
DenseNet-121 backbone with trilemma classification head.

Architecture selection rationale (paper Section 4.3):
  DenseNet-121 achieved highest F1 (0.727 avg across augmentation
  conditions) and best parameter efficiency (0.0909 F1/M params)
  in prior architecture comparison:
    VGG19:          143M params, 0.0045 F1/M
    ResNet-50+CBAM:  25M params, 0.0274 F1/M
    EfficientNet-B0:  5.3M params, 0.1336 F1/M
    DenseNet-121:     8.0M params, 0.0909 F1/M  ← selected

Head: global average pooling → Dropout(0.5) → sigmoid linear unit
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as tv_models


class TrilemmaClassifier(nn.Module):
    """
    DenseNet-121 with binary classification head for ASD screening.

    The head uses:
      GlobalAveragePooling → Dropout(p) → Linear(1024, 1) → Sigmoid

    Output is a single logit (before sigmoid); sigmoid is applied
    externally during loss computation and evaluation.
    """

    def __init__(
        self,
        backbone: str = "densenet121",
        pretrained: bool = True,
        dropout: float = 0.5,
        freeze_bn: bool = False,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.dropout_p = dropout

        # ── Backbone ─────────────────────────────────────────────────────
        weights = "IMAGENET1K_V1" if pretrained else None
        base = tv_models.densenet121(weights=weights)

        # Extract feature extractor (remove original classifier)
        self.features = base.features      # (B, 1024, 7, 7) for 224×224 input
        self.feature_dim = base.classifier.in_features  # 1024

        if freeze_bn:
            for m in self.features.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad_(False)

        # ── Classification head ───────────────────────────────────────────
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(self.feature_dim, 1),
            # No sigmoid here — applied in loss and at inference threshold
        )

        # Initialize head weights
        nn.init.xavier_uniform_(self.head[3].weight)
        nn.init.zeros_(self.head[3].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224) normalized image tensor

        Returns:
            logits: (B, 1) raw logits (pre-sigmoid)
        """
        features = self.features(x)
        features = torch.relu(features)    # DenseNet uses ReLU after features
        logits   = self.head(features)
        return logits.squeeze(1)           # (B,)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return feature embeddings (B, 1024) for analysis."""
        features = self.features(x)
        features = torch.relu(features)
        pooled   = nn.functional.adaptive_avg_pool2d(features, (1, 1))
        return pooled.flatten(1)

    def get_num_parameters(self) -> dict:
        total   = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


class MobileNetV3Student(nn.Module):
    """
    MobileNetV3-Small student model for knowledge distillation (B3).

    Paper Section 4.4 config B3:
      Teacher: DenseNet-121 fine-tuned on A2 pipeline (A2 model)
      Student: MobileNetV3-Small
      T=4, λ=0.7 soft-target weight
      Result: F1=0.891, 4.2 MB, 96 ms (E3)
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        base = tv_models.mobilenet_v3_small(weights=weights)

        self.features   = base.features
        self.avgpool    = base.avgpool
        in_features     = base.classifier[0].in_features

        # Replace classifier head
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x).squeeze(1)


def build_model(
    backbone: str = "densenet121",
    pretrained: bool = True,
    dropout: float = 0.5,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Factory function — returns the appropriate model for a given config.

    Args:
        backbone: "densenet121" or "mobilenet_v3_small"
        pretrained: Use ImageNet pretrained weights
        dropout: Dropout probability in classification head
        device: Target device

    Returns:
        Initialized nn.Module
    """
    if backbone == "densenet121":
        model = TrilemmaClassifier(
            backbone=backbone,
            pretrained=pretrained,
            dropout=dropout,
        )
    elif backbone == "mobilenet_v3_small":
        model = MobileNetV3Student(pretrained=pretrained, dropout=dropout)
    else:
        raise ValueError(
            f"Unknown backbone: {backbone}. "
            f"Supported: ['densenet121', 'mobilenet_v3_small']"
        )

    if device is not None:
        model = model.to(device)

    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[Model] {backbone}: "
        f"{params/1e6:.2f}M total params, "
        f"{trainable/1e6:.2f}M trainable"
    )
    return model
