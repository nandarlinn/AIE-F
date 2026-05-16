"""
Predict the syllable shown in a single image OR a stroke .txt file.

Examples:
    # Predict from an image already produced by convert2image.py
    python predict.py --checkpoint runs/exp1/best.pt --image image/yekyaw/5-2.png

    # Predict directly from a stroke file (auto-renders to image first)
    python predict.py --checkpoint runs/exp1/best.pt --stroke_file dataset/yekyaw/5-2.txt

Author: For Ye Kyaw Thu / LU Lab.
"""

import os
import sys
import argparse
import json

import torch
from torchvision import transforms
from PIL import Image

from model import SyllableCNN


# -------------------------------------------------------------------
# Stroke -> image (small standalone reimplementation matching convert2image.py)
# -------------------------------------------------------------------
def stroke_file_to_image(stroke_path: str, img_size: int = 64):
    from PIL import ImageDraw

    # Parse strokes
    strokes, current = [], []
    with open(stroke_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("STROKE"):
                if current:
                    strokes.append(current)
                    current = []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    current.append((float(parts[0]), float(parts[1])))
    if current:
        strokes.append(current)

    if not strokes:
        raise ValueError(f"No strokes found in {stroke_path}")

    # Normalize + scale to img_size (matches convert2image.py logic)
    padding = 10
    all_x = [p[0] for s in strokes for p in s]
    all_y = [p[1] for s in strokes for p in s]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    span = max(max_x - min_x, max_y - min_y)
    scale = (img_size - 2 * padding) / span if span > 0 else 1

    normed = [
        [((x - min_x) * scale, (y - min_y) * scale) for x, y in s]
        for s in strokes
    ]
    all_x2 = [p[0] for s in normed for p in s]
    all_y2 = [p[1] for s in normed for p in s]
    ox = (img_size - (max(all_x2) - min(all_x2))) / 2 - min(all_x2)
    oy = (img_size - (max(all_y2) - min(all_y2))) / 2 - min(all_y2)
    final = [[(x + ox, y + oy) for x, y in s] for s in normed]

    # Draw black on white grayscale
    img = Image.new("L", (img_size, img_size), 255)
    draw = ImageDraw.Draw(img)
    for stroke in final:
        if len(stroke) < 2:
            continue
        for j in range(1, len(stroke)):
            draw.line([stroke[j - 1], stroke[j]], fill=0, width=2)
    return img


# -------------------------------------------------------------------
# Predict
# -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Predict syllable from one sample")
    ap.add_argument("--checkpoint", required=True, help="Path to .pt model checkpoint")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image",       help="Path to a PNG image")
    src.add_argument("--stroke_file", help="Path to a stroke .txt file")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--labels_json",
                    help="Override labels.json path (defaults to <ckpt_dir>/labels.json)")
    args = ap.parse_args()

    # ---- Load checkpoint ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    num_classes = ckpt["num_classes"]
    img_size    = ckpt.get("img_size", 64)

    model = SyllableCNN(num_classes=num_classes, img_size=img_size).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ---- Load labels ----
    labels_json = args.labels_json or os.path.join(
        os.path.dirname(args.checkpoint), "labels.json"
    )
    with open(labels_json, "r", encoding="utf-8") as f:
        labels = json.load(f)["labels"]

    if len(labels) != num_classes:
        print(f"WARNING: label count {len(labels)} != model classes {num_classes}",
              file=sys.stderr)

    # ---- Get the image ----
    if args.image:
        img = Image.open(args.image).convert("L")
    else:
        img = stroke_file_to_image(args.stroke_file, img_size=img_size)

    # ---- Transform and predict ----
    # Must match the eval_tf used in training (invert: strokes -> 1, bg -> 0)
    class _Invert:
        def __call__(self, t):
            return 1.0 - t
    invert = _Invert()
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        invert,
    ])
    x = tf(img).unsqueeze(0).to(device)  # (1, 1, H, W)

    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1)[0]
        top_p, top_i = probs.topk(args.topk)

    print("\nTop predictions:")
    for rank, (p, i) in enumerate(zip(top_p.tolist(), top_i.tolist()), start=1):
        syl = labels[i] if i < len(labels) else "?"
        print(f"  {rank}. line {i+1:>5}  '{syl}'   prob={p*100:6.2f}%")


if __name__ == "__main__":
    main()
