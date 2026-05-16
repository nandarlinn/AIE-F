================================================================
Myanmar Syllable Handwriting Recognizer
================================================================

This project trains a CNN to recognize 4,413 handwritten Myanmar
syllables from a stroke-based handwriting collector.

Two approaches are included:
  1. Single-head CNN   (one 4,413-way classifier)
  2. Multi-head CNN    (seven structural-component classifiers)


================================================================
1. Project Layout
================================================================

Syllable_HandWRecognizer/
├── syl.txt                  4,413 Myanmar syllables, one per line
├── requirements.txt         dependencies
│
├── hw_collector.py          PyQt5 desktop collector
├── server.py                Flask web collector
├── index.html
├── script.js
├── style.css
├── dataset_browser.py       sample validation tool
├── convert2image.py         stroke .txt -> .png renderer
│
├── dataset/                 raw stroke files
│   └── <username>/
│       ├── 1-1.txt
│       └── ...
│
├── image/                   rendered images
│   └── <username>/
│       ├── 1-1.png
│       └── ...
│
├── recognizer/              Approach 1: single-head CNN
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── multihead/               Approach 2: multi-head CNN
│   ├── decompose.py
│   ├── model_multihead.py
│   ├── train_multihead.py
│   ├── evaluate_multihead.py
│   └── predict_multihead.py
│
└── runs/                    training outputs (auto-created)


================================================================
2. Setup
================================================================

Python 3.8 or newer is required.

  pip install -r recognizer/requirements.txt

Dependencies: torch, torchvision, Pillow.


================================================================
3. Data Preparation
================================================================

Step 1. Collect handwriting samples using hw_collector.py or
        server.py. Files are saved in dataset/<username>/.

Step 2. Render stroke files to images:

  python convert2image.py --dataset dataset --output image --size 64

Step 3. Verify the dataset:

  python recognizer/dataset.py --image_dir image --syl_file syl.txt


================================================================
4. Approach 1: Single-Head CNN
================================================================

One CNN with a 4,413-way output head.

Train:

  python recognizer/train.py \
      --image_dir image \
      --syl_file  syl.txt \
      --epochs    120 \
      --num_workers 0 \
      --out_dir   runs/exp1

Evaluate:

  python recognizer/evaluate.py \
      --checkpoint runs/exp1/best.pt \
      --image_dir image \
      --syl_file  syl.txt \
      --out_json  runs/exp1/eval.json

Predict a single sample:

  python recognizer/predict.py \
      --checkpoint runs/exp1/best.pt \
      --image image/<username>/100-3.png

  python recognizer/predict.py \
      --checkpoint runs/exp1/best.pt \
      --stroke_file dataset/<username>/100-3.txt

Note: With 2 training samples per class across 4,413 classes,
this approach does not converge. Accuracy stays near random
(0.02% top-1). This approach is included for comparison.


================================================================
5. Approach 2: Multi-Head CNN
================================================================

The same CNN backbone with seven output heads that predict the
structural components of each syllable (base, medials, vowels,
final, stack, tones, asat). At inference time, components are
mapped back to a syllable through a lookup table.

Check the decomposer:

  python multihead/decompose.py --syl_file syl.txt

Train:

  python multihead/train_multihead.py \
      --image_dir image \
      --syl_file  syl.txt \
      --epochs    60 \
      --num_workers 0 \
      --out_dir   runs/mt1

Evaluate:

  python multihead/evaluate_multihead.py \
      --checkpoint runs/mt1/best.pt \
      --image_dir image \
      --syl_file  syl.txt \
      --out_json  runs/mt1/eval.json

Predict a single sample:

  python multihead/predict_multihead.py \
      --checkpoint runs/mt1/best.pt \
      --image image/<username>/100-3.png

  python multihead/predict_multihead.py \
      --checkpoint runs/mt1/best.pt \
      --stroke_file dataset/<username>/100-3.txt


================================================================
6. Results (single writer, copy 3 held out as test)
================================================================

Approach 1: Single-Head CNN
  Test top-1 accuracy           0.02%
  Test top-5 accuracy           0.11%
  Macro precision / recall / F1 ~0%

Approach 2: Multi-Head CNN
  Syllable lookup accuracy      90.98%
  Mean per-head top-1           98.51%
  Mean per-head macro F1        95.81%
  Mean per-head weighted F1     98.50%
  Best epoch                    55 / 60
  Total training time           ~40 minutes on CPU


================================================================
7. Output Files
================================================================

Each training run produces:

  best.pt          best checkpoint (highest test accuracy)
  last.pt          most recent checkpoint
  labels.json      syllable list
  history.json     per-epoch metrics
  vocabs.json      component vocabularies (multi-head only)


================================================================
8. Command Reference
================================================================

  # Render strokes to images
  python convert2image.py --dataset dataset --output image --size 64

  # Single-head: train, evaluate, predict
  python recognizer/train.py --epochs 120 --num_workers 0 --out_dir runs/exp1
  python recognizer/evaluate.py --checkpoint runs/exp1/best.pt --out_json runs/exp1/eval.json
  python recognizer/predict.py --checkpoint runs/exp1/best.pt --image image/<u>/<file>.png

  # Multi-head: train, evaluate, predict
  python multihead/train_multihead.py --epochs 60 --num_workers 0 --out_dir runs/mt1
  python multihead/evaluate_multihead.py --checkpoint runs/mt1/best.pt --out_json runs/mt1/eval.json
  python multihead/predict_multihead.py --checkpoint runs/mt1/best.pt --image image/<u>/<file>.png


================================================================
9. Notes
================================================================

- Use --num_workers 0 on macOS / Python 3.13 to avoid
  multiprocessing issues with DataLoader.
- The recognizer expects images at image/<username>/<line>-<copy>.png
  where <line> is the 1-based line number in syl.txt.
- The current dataset has only one writer. Cross-writer
  accuracy is expected to be lower.