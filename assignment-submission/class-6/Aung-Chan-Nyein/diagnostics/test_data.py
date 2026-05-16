"""
Diagnostic: are our images actually distinguishable across syllable classes?
Save as test_data.py and run:  python test_data.py
"""

import os
import glob
import hashlib
from collections import defaultdict

from PIL import Image
import numpy as np


def img_hash(path):
    return hashlib.md5(Image.open(path).tobytes()).hexdigest()[:10]


def main():
    # 1. How many images per class do we actually have?
    files = sorted(glob.glob("image/*/*.png"))
    print(f"Total images: {len(files)}")

    if not files:
        print("ERROR: no images found at image/*/*.png. "
              "Are you in the right folder?")
        return

    per_class = defaultdict(list)
    for f in files:
        name = os.path.basename(f)             # e.g. "1-1.png"
        try:
            line_idx = int(name.split("-")[0])
        except ValueError:
            print(f"  skipped weird filename: {f}")
            continue
        per_class[line_idx].append(f)

    class_counts = [len(v) for v in per_class.values()]
    print(f"Unique classes seen : {len(per_class)}")
    print(f"Samples/class       : min={min(class_counts)}, "
          f"max={max(class_counts)}, "
          f"avg={sum(class_counts)/len(class_counts):.2f}")

    # 2. Compare images across classes - are they actually different?
    print("\n--- One sample from selected classes ---")
    for cls in [1, 100, 1000, 4000, 4413]:
        if cls in per_class:
            f = per_class[cls][0]
            a = np.array(Image.open(f).convert("L"))
            print(f"class {cls:>4}: {os.path.basename(f):>12}  "
                  f"mean={a.mean():6.2f}  "
                  f"ink_pixels={(a < 128).sum():>4}  "
                  f"hash={img_hash(f)}")
        else:
            print(f"class {cls:>4}: MISSING")

    # 3. Are the 3 copies of one class different (good)
    #    or identical (bad - convert2image saved same image 3x)?
    for cls in [1, 100, 1000]:
        print(f"\n--- Class {cls}, all copies ---")
        for f in sorted(per_class.get(cls, [])):
            a = np.array(Image.open(f).convert("L"))
            print(f"  {os.path.basename(f):>12}  "
                  f"mean={a.mean():6.2f}  "
                  f"ink={(a < 128).sum():>4}  "
                  f"hash={img_hash(f)}")


if __name__ == "__main__":
    main()
