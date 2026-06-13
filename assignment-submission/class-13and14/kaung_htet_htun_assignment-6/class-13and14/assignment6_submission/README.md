# Assignment 6 G2P SMT Submission

This is the clean submission folder for the Myanmar G2P SMT assignment.

## Notebooks

- `notebooks/assignment6_baseline_pbsmt.ipynb`  
  Baseline Moses/PBSMT training notebook. It includes data checking, `my -> ph` training, tuning, test decoding, and BLEU calculation.

- `notebooks/assignment6_bleu_improvement.ipynb`  
  BLEU improvement notebook. It includes normalization, OOV checking, LM/phrase/alignment experiments, and BLEU comparison.

## Report

- `report.md`  
  One summarized report covering both the baseline PBSMT training and BLEU improvement experiments.

## Results

- `results/baseline/`  
  Baseline trained/tuned configs, decoded test output, BLEU result, and training/tuning/decode logs.

- `results/improvement_best_6gram/`  
  Best improved model result using 6-gram target LM, including tuned config, decoded output, BLEU result, and logs.

- `results/experiments/`  
  Experiment comparison table and OOV token list.

## Final Scores

```text
Baseline BLEU = 69.59
Best BLEU = 69.85
Improvement = +0.26 BLEU
Best setting = 6-gram LM, max phrase length 7, grow-diag-final-and alignment
```
