"""
Small CNN for Myanmar Syllable Recognition
==========================================
Architecture follows the spirit of myMNIST-benchmark style models, scaled up
in the final layer for ~4400 classes.

Input:  grayscale image (1 x H x W), default 64x64
Output: logits over `num_classes` syllables
"""

import torch
import torch.nn as nn


class SyllableCNN(nn.Module):
    """
    Compact CNN. Around 3-4M parameters with 4413 classes -- tractable on CPU
    and trains quickly on a modest GPU.
    """

    def __init__(self, num_classes: int, img_size: int = 64, dropout: float = 0.3):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 64x64 -> 32x32
            nn.Conv2d(1,  32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4: 8x8 -> 4x4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        feat_dim = 256 * 4 * 4

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = SyllableCNN(num_classes=4413)
    n = count_parameters(m)
    print(f"SyllableCNN parameters: {n:,}  ({n/1e6:.2f} M)")
    x = torch.randn(2, 1, 64, 64)
    y = m(x)
    print(f"Input  : {tuple(x.shape)}")
    print(f"Output : {tuple(y.shape)}")
