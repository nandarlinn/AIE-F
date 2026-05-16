"""
Multi-Task CNN for Myanmar Syllable Recognition
================================================
Same CNN backbone as before, but with 7 small classification heads instead of
one giant 4413-way head. The heads predict the structural components
(base, medials, vowels, final, stack, tones, asat) of the syllable.

At inference time, we recompose the syllable from the predicted components,
or look it up in a precomputed component->syllable map for syllables that
share decompositions.

Why this works for our data:
  - Each head has 2-47 classes (instead of 4413).
  - Each head sees hundreds-to-thousands of training samples per class
    (vs only 2 per syllable for the original model).
  - The dense gradient signal at every output layer lets training converge.
"""

import torch
import torch.nn as nn


class SyllableMultiHead(nn.Module):
    """
    Backbone: same CNN feature extractor used in the original SyllableCNN.
    Then 7 small linear classification heads sharing the backbone.
    """

    def __init__(self, head_sizes: dict, img_size: int = 64, dropout: float = 0.3):
        """
        head_sizes : dict like {"base": 47, "medials": 11, "vowels": 38, ...}
                     One entry per head; value is the number of classes for
                     that head.
        """
        super().__init__()
        self.head_names = list(head_sizes.keys())

        self.features = nn.Sequential(
            nn.Conv2d(1,  32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128,128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        feat_dim = 256 * 4 * 4

        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # One head per component
        self.heads = nn.ModuleDict({
            name: nn.Linear(512, n_classes)
            for name, n_classes in head_sizes.items()
        })

    def forward(self, x):
        """Returns a dict {head_name: logits_tensor}."""
        f = self.features(x)
        f = self.shared(f)
        return {name: head(f) for name, head in self.heads.items()}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick architecture check using the head sizes from our actual syl.txt
    head_sizes = {
        "base": 47, "medials": 11, "vowels": 38,
        "final": 32, "stack": 26, "tones": 3, "asat": 2,
    }
    m = SyllableMultiHead(head_sizes)
    print(f"Parameters: {count_parameters(m):,}  ({count_parameters(m)/1e6:.2f} M)")
    x = torch.randn(4, 1, 64, 64)
    out = m(x)
    for name, t in out.items():
        print(f"  {name:>8}: logits shape {tuple(t.shape)}")
