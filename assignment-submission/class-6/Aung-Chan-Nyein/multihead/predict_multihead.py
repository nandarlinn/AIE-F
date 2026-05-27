"""
Predict syllable from one image or stroke .txt using the multi-head model.

The model predicts 7 structural components; we look up the matching
syllable(s) in the precomputed component-to-syllable map.

Examples:
    python predict_multihead.py --checkpoint runs/mt1/best.pt \
        --image image/AungChanNyein/100-3.png

    python predict_multihead.py --checkpoint runs/mt1/best.pt \
        --stroke_file dataset/AungChanNyein/100-3.txt
"""
import os, sys, json, argparse

import torch
from torchvision import transforms
from PIL import Image, ImageDraw

sys.path.insert(0, "recognizer")
from decompose import decompose, build_vocabularies
from model_multihead import SyllableMultiHead
from train_multihead import (
    Invert, build_component_to_syllable_map, predicted_components_to_key,
)


def stroke_file_to_image(stroke_path, img_size=64):
    strokes, current = [], []
    with open(stroke_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("STROKE"):
                if current:
                    strokes.append(current); current = []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    current.append((float(parts[0]), float(parts[1])))
    if current:
        strokes.append(current)
    if not strokes:
        raise ValueError("no strokes")

    padding = 10
    all_x = [p[0] for s in strokes for p in s]
    all_y = [p[1] for s in strokes for p in s]
    span = max(max(all_x) - min(all_x), max(all_y) - min(all_y))
    scale = (img_size - 2 * padding) / span if span > 0 else 1
    normed = [
        [((x - min(all_x)) * scale, (y - min(all_y)) * scale) for x, y in s]
        for s in strokes
    ]
    all_x2 = [p[0] for s in normed for p in s]
    all_y2 = [p[1] for s in normed for p in s]
    ox = (img_size - (max(all_x2) - min(all_x2))) / 2 - min(all_x2)
    oy = (img_size - (max(all_y2) - min(all_y2))) / 2 - min(all_y2)
    final = [[(x + ox, y + oy) for x, y in s] for s in normed]
    img = Image.new("L", (img_size, img_size), 255)
    draw = ImageDraw.Draw(img)
    for stroke in final:
        if len(stroke) < 2:
            continue
        for j in range(1, len(stroke)):
            draw.line([stroke[j - 1], stroke[j]], fill=0, width=2)
    return img


def load_vocabs_from_json(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    vocabs = {}
    for k, v in raw.items():
        # turn list back into tuple if it was a tuple originally
        out = []
        for x in v:
            if isinstance(x, list):
                out.append(tuple(x))
            else:
                out.append(x)
        vocabs[k] = out
    return vocabs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image")
    src.add_argument("--stroke_file")
    ap.add_argument("--syl_file", default="syl.txt")
    ap.add_argument("--vocabs_json", default=None,
                    help="defaults to <ckpt_dir>/vocabs.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    head_sizes = ckpt["head_sizes"]
    img_size   = ckpt["img_size"]

    vocabs_path = args.vocabs_json or os.path.join(
        os.path.dirname(args.checkpoint), "vocabs.json"
    )
    vocabs = load_vocabs_from_json(vocabs_path)

    with open(args.syl_file, "r", encoding="utf-8") as f:
        syl_labels = [ln.strip() for ln in f if ln.strip()]
    comp_map = build_component_to_syllable_map(syl_labels, vocabs)

    model = SyllableMultiHead(head_sizes, img_size=img_size).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    if args.image:
        img = Image.open(args.image).convert("L")
    else:
        img = stroke_file_to_image(args.stroke_file, img_size=img_size)

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        Invert(),
    ])
    x = tf(img).unsqueeze(0).to(device)

    head_names = list(head_sizes.keys())
    with torch.no_grad():
        logits = model(x)
        preds = {h: logits[h].argmax(dim=1).item() for h in head_names}
        # Also extract probabilities for display
        probs = {h: torch.softmax(logits[h], dim=1)[0] for h in head_names}

    print("\nPredicted components:")
    for h in head_names:
        idx = preds[h]
        val = vocabs[h][idx]
        p   = probs[h][idx].item() * 100
        # pretty-print: if val is a codepoint int, show the character
        if isinstance(val, int):
            disp = chr(val)
        elif isinstance(val, (list, tuple)):
            disp = "".join(chr(c) for c in val) if val else "<none>"
        elif val is None:
            disp = "<none>"
        else:
            disp = str(val)
        print(f"  {h:>8}: index={idx:>2}  '{disp}'   prob={p:6.2f}%")

    key = predicted_components_to_key(preds, vocabs)
    cands = comp_map.get(key, [])
    print(f"\nLookup candidates: {len(cands)}")
    if cands:
        print("Matching syllables in syl.txt:")
        for line_1b, syl in cands[:10]:
            print(f"  line {line_1b:>5}  '{syl}'")
    else:
        print("No exact match in syl.txt for predicted components.")


if __name__ == "__main__":
    main()
