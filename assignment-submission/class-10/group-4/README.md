# SCRDR Tutorial Extension — Heart and Adult Datasets

> An extension of the original SCRDR Tutorial by Dr. Ye Kyaw Thu (LU Lab., Myanmar),
> adding two new datasets and an interactive knowledge-acquisition experiment.

**Group 4** · AI Engineering (Fundamental) · May 2026

---

## 1. What This Project Adds

The original tutorial compared five datasets (Iris, Wine, Mushroom, WDBC, Titanic)
across five ML methods and SCRDR auto-induction. This project extends that work
in two directions:

1. **Two new datasets** — Heart Disease (UCI, 303 rows) and Adult Income (UCI,
   32,561 rows) — chosen to fill gaps in the original five (small medical
   data and large categorical data).
2. **Interactive SCRDR** — manually building rule trees as a "domain expert"
   using `scrdr_interactive.py`, then comparing to the auto-induction baseline.

Importantly, **no Python source code was modified**. We only changed input
flags and shell scripts, then analyzed the resulting JSON trees.

---

## 2. Repository Layout

```
.
├── README.md                          ← this file
├── classi/                            ← auto-induction (from original tutorial)
│   ├── scrdr_learner.py
│   ├── data/
│   │   ├── heart.csv                  ← NEW
│   │   └── adult.csv                  ← NEW
│   ├── results_scrdr/
│   │   ├── heart_rdr_cf.png           ← NEW
│   │   ├── heart_rdr_model.json       ← NEW
│   │   ├── adult_rdr_cf.png           ← NEW
│   │   └── adult_rdr_model.json       ← NEW
│   └── run_scrdr.sh                   ← extended with 2 new lines
│
├── ml/                                ← ML baselines (from original tutorial)
│   ├── five_ml.py
│   ├── data/
│   │   ├── heart.csv                  ← NEW
│   │   └── adult.csv                  ← NEW
│   └── run_five_ml.sh                 ← extended with 2 new entries
│
├── inter/                             ← interactive SCRDR (NEW WORK)
│   ├── scrdr_interactive.py
│   ├── data/
│   │   ├── heart.csv
│   │   ├── adult.csv
│   │   └── adult_small.csv            ← 400-row balanced sample
│   ├── heart_demo.json                ← 9-rule expert tree
│   ├── adult_demo.json                ← 8-rule expert tree
│   ├── run_inter_heart.sh             ← NEW
│   ├── run_inter_adult.sh             ← NEW
│   ├── sample_adult.py                ← NEW (balanced sampler)
│   ├── heart_tree.png                 ← NEW (tree visualization)
│   └── adult_tree.png                 ← NEW (tree visualization)
│
└── presentation/
    └── Group4_SCRDR_HeartAdultDatasets.pptx
```

---

## 3. Datasets

| Dataset    | Type        | Rows   | Target            | Source | Notes                       |
| ---------- | ----------- | ------ | ----------------- | ------ | --------------------------- |
| Iris       | Numeric     | 150    | class (3 species) | UCI    | original                    |
| Wine       | Numeric     | 178    | Class (3)         | UCI    | original                    |
| Mushroom   | Categorical | 8,124  | edible / poisonous | UCI   | original                    |
| WDBC       | Numeric     | 569    | Diagnosis (M/B)   | UCI    | original                    |
| Titanic    | Mixed       | 891    | Survived (0/1)    | Kaggle | original                    |
| **Heart**  | Mixed       | 303    | target (0/1)      | UCI    | **NEW — small medical**     |
| **Adult**  | Mixed       | 32,561 | income (>50K)     | UCI    | **NEW — large categorical** |

### Why these two?

- **Heart Disease** tests SCRDR on continuous medical features where subtle
  feature interactions matter — similar to WDBC.
- **Adult Income** tests SCRDR on a large, mostly categorical dataset and
  exposes scaling concerns for the interactive mode.

---

## 4. How to Reproduce

### 4.1 Prerequisites

- Python 3.10+
- `pandas`, `scikit-learn`, `numpy`, `matplotlib`, `seaborn` (same as
  original tutorial)
- All the original tutorial files in place

### 4.2 Place the new datasets

Copy the CSV files for Heart and Adult into both data folders:

```bash
cp heart.csv classi/data/
cp adult.csv classi/data/
cp heart.csv ml/data/
cp adult.csv ml/data/
cp heart.csv inter/data/
cp adult.csv inter/data/
```

### 4.3 Run the ML baseline

In `ml/run_five_ml.sh`, the `datasets` array now ends with two new entries:

```bash
"./data/heart.csv|target|"
"./data/adult.csv|income|"
```

Execute:

```bash
cd ml/
./run_five_ml.sh
```

### 4.4 Run SCRDR auto-induction

In `classi/run_scrdr.sh`, two new lines were appended:

```bash
time python ./scrdr_learner.py --input ./data/heart.csv --target target \
  --plot ./results_scrdr/heart_rdr_cf.png --output ./results_scrdr/heart_rdr_model.json

time python ./scrdr_learner.py --input ./data/adult.csv --target income \
  --plot ./results_scrdr/adult_rdr_cf.png --output ./results_scrdr/adult_rdr_model.json
```

Execute:

```bash
cd classi/
./run_scrdr.sh
```

### 4.5 Run interactive SCRDR (Heart)

```bash
cd inter/
./run_inter_heart.sh
```

This script first runs in **build mode** — the system reads each row of
Heart and prompts for a rule whenever it makes a mistake. Type `exit` after
~10 rules to stop. Then it automatically runs in **test mode** to produce
the classification report.

### 4.6 Run interactive SCRDR (Adult)

Adult has 32,000 rows, which is too many for interactive prompting. We
created a 400-row balanced sample first:

```bash
cd inter/
python sample_adult.py     # creates data/adult_small.csv
./run_inter_adult.sh       # builds tree on sample, tests on full data
```

The build script uses `adult_small.csv`, but the test step uses the full
`adult.csv` to check generalization.

---

## 5. Results

### 5.1 ML baseline (weighted F1)

| Dataset    | DT   | RF   | SVM  | NB   | LR   |
| ---------- | ---- | ---- | ---- | ---- | ---- |
| Iris       | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Wine       | 0.94 | 1.00 | 1.00 | 1.00 | 1.00 |
| Mushroom   | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 |
| WDBC       | 0.95 | 0.96 | 0.97 | 0.96 | 0.97 |
| Titanic    | 0.78 | 0.80 | 0.82 | 0.78 | 0.80 |
| **Adult**  | 0.81 | 0.85 | 0.84 | **0.34** | 0.84 |
| **Heart**  | 0.75 | 0.87 | 0.90 | 0.85 | 0.89 |

**Notable:** NB collapses on Adult (0.34) because the Naive Bayes
independence assumption is violated by correlated features such as
`education` and `marital-status`.

### 5.2 SCRDR auto-induction (weighted F1)

| Dataset    | Best ML | DT   | SCRDR | Gap (vs DT) |
| ---------- | ------- | ---- | ----- | ----------- |
| Iris       | 1.00    | 1.00 | 0.97  | −0.03       |
| Wine       | 1.00    | 0.94 | 0.72  | −0.22       |
| Mushroom   | 1.00    | 1.00 | 1.00  |  0.00       |
| WDBC       | 0.97    | 0.95 | 0.84  | −0.11       |
| Titanic    | 0.82    | 0.78 | 0.82  | **+0.04**   |
| **Adult**  | 0.85    | 0.81 | 0.74  | −0.07       |
| **Heart**  | 0.90    | 0.75 | 0.56  | **−0.19**   |

### 5.3 Final comparison: ML vs Auto vs Interactive

| Dataset | Best ML    | SCRDR Auto | SCRDR Interactive             |
| ------- | ---------- | ---------- | ----------------------------- |
| Heart   | 0.90 (SVM) | 0.56       | **0.73**                      |
| Adult   | 0.85 (RF)  | 0.74       | 0.73 (sample) / 0.56 (full)   |

- **Heart: +17 F1 points** from interactive over auto. Expert rules clearly help.
- **Adult: interactive on the small sample matched auto, but did not generalize
  to the full dataset.** The gap (0.73 → 0.56) is the cost of training on a
  balanced sample of an imbalanced dataset.

---

## 6. Key Findings

### 6.1 SCRDR has a predictable performance pattern

| Pattern                | Datasets                  | Why                                     |
| ---------------------- | ------------------------- | --------------------------------------- |
| SCRDR ≈ ML             | Iris, Mushroom, Titanic   | Clean rule-like structure               |
| SCRDR slightly < ML    | Adult, WDBC               | Subtle feature interactions             |
| SCRDR << ML            | Wine, Heart               | Continuous features, small effects      |

SCRDR's "one feature per rule" greedy strategy cannot combine many small
signals the way SVM and Logistic Regression do.

### 6.2 Expert rules can rescue weak auto-induction

On Heart, the auto-learner produced a 24-rule tree of depth 15 that scored 0.56.
By restarting interactive build with a strong first rule
(`oldpeak > 1.5 → Disease`), we produced a **9-rule tree of depth 6** that
scored 0.73 — a 17-point gain.

The lesson: when the first rule sits at the root of the binary tree, choosing
a rule that captures a strong, clean signal pays off across the entire tree.

### 6.3 Interactive RDR has a hidden risk: rule-entry inversion

When the script prompts about a single row, the user can pick a conclusion
that fits *that specific row* but inverts the rule's general meaning.
We observed this twice:

- **Heart:** `cp == 0 → No Disease` (entered).
  Reality: `cp == 0` means asymptomatic chest pain — a **high-risk** signal.
- **Adult:** `marital.status == Married-civ-spouse → ≤50K` (entered initially).
  Reality: married-civ-spouse is associated with **higher** income.

These mistakes survive because RDR never retracts rules. The fix is to
cross-check each rule entry against general domain knowledge, not just
the row in front of you.

### 6.4 Adult interactive results highlight a sampling pitfall

Building on a 50/50 balanced sample produced rules that worked well on
that sample (F1 = 0.73) but performed worse than auto-induction on the
full imbalanced dataset (F1 = 0.56 vs 0.74). The rules over-predicted
the minority class because that's what the training distribution
encouraged.

---

## 7. Tree Visualizations

Both interactive trees are rendered as PNG files in `inter/` using the
`draw_tree.py` script (Graphviz):

- `heart_tree.png` — 9 rules, depth 6, balanced 5/4 conclusions
- `adult_tree.png` — 8 rules, depth 7

Visual conventions:

- **Warm beige fill + green arrows** = "True (Refine)" branch
- **Cool blue-gray fill + gray arrows** = "False (Next)" branch

The last True-firing rule along the path is the prediction.

---

## 8. Lessons Learned

1. **The original SCRDR tutorial scripts are well-designed for extension.**
   Adding two new datasets required zero code changes — only flag changes
   in shell scripts.

2. **Dataset shape matters more than dataset size for SCRDR.** Mushroom
   (8k rows, categorical) takes seconds and reaches 1.00; Heart (300 rows,
   continuous) struggles even with expert rules.

3. **Interactive RDR is not strictly better than auto-induction.** It
   helps a lot on Heart (where auto-induction fails) but is risky on
   imbalanced data like Adult.

4. **Auto-induction trees can be a starting point, not a destination.**
   The 24-rule auto Heart tree contained useful information, but a
   restart with a better first rule produced a much cleaner 9-rule tree.

---

## 9. Future Work

- Try a **hybrid** approach: bootstrap the interactive tree from auto-induction,
  then refine the deepest unhelpful branches by hand.
- For Adult, try **stratified** sampling instead of balanced — keep the
  76/24 class ratio but with fewer rows, to see if it generalizes better.
- Test SCRDR on the **POS-tagging** and **tokenization** notebooks
  referenced in the original README but not yet attempted in this project.
- Quantify **rule overlap** between the auto and interactive trees on
  Heart to see how much expert knowledge actually agreed with the data.

---

## 10. Acknowledgments

This project builds directly on the SCRDR Tutorial materials by
**Dr. Ye Kyaw Thu (LU Lab., Myanmar)**. The original tutorial includes
the `scrdr_learner.py`, `scrdr_interactive.py`, `five_ml.py` scripts and
the slide deck *RDR_Intro.pdf*, all licensed under CC BY-NC-SA 4.0
for educational and R&D use. We thank the author for the clear,
well-structured starting materials that made this extension possible.

---

## 11. References

- Compton, P., & Kang, B. (2021). *Ripple-Down Rules: An Alternative
  to Machine Learning.* CRC Press.
- UCI Machine Learning Repository — Heart Disease dataset
  (https://archive.ics.uci.edu/dataset/45/heart+disease)
- UCI Machine Learning Repository — Adult dataset
  (https://archive.ics.uci.edu/dataset/2/adult)
- Dr. Ye Kyaw Thu — SCRDR Tutorial (original materials).

---

*Last updated: May 2026*
