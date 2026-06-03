# Myanmar G2P PBSMT Experiment Report

## Abstract

This experiment evaluates Phrase-Based Statistical Machine Translation (PBSMT) for Myanmar grapheme-to-phoneme conversion using the myG2P dataset. The confirmed baseline is a 5-gram Moses PBSMT system generated from the original assignment template. Additional 3-gram, 5-gram tuning, and 7-gram systems were tested to determine whether n-gram order, phrase length, pruning, MERT n-best size, or stronger search improves exact-match accuracy.

The tuning experiments produced only marginal changes. This indicates that the main remaining limitation is not PBSMT search or simple hyperparameter choice. The error analysis points instead to Myanmar G2P pronunciation variation and data coverage/preparation issues.

## Data

The experiment uses the myG2P Myanmar grapheme-to-phoneme dictionary data. The source side is Myanmar grapheme syllables (`.my`), and the target side is phoneme representation (`.ph`). The cleaned split files used by the experiment are:

```text
g2p-par/train_clean.my
g2p-par/train_clean.ph
g2p-par/dev_clean.my
g2p-par/dev_clean.ph
g2p-par/test_clean.my
g2p-par/test_clean.ph
```

## Normalization and Evaluation Preparation

The dataset was normalized with `syl-Normalizer` v0.6 and `final_syl_dictionary_13Feb2024.sorted.txt`. The cleaned test split was converted to Moses SGM format using original-style segment text handling. Apostrophes in phoneme tokens, such as `te'`, are part of the target representation and must remain unchanged.

## Experiment Setup

All systems were generated from the original assignment template:

```text
pbsmt-big-normalize/config.baseline
```

The main confirmed baseline is:

```text
pbsmt_5gram_main
```

Tuning and comparison systems:

```text
pbsmt_5gram_phrase7
pbsmt_5gram_phrase11
pbsmt_5gram_phrase13
pbsmt_5gram_less_pruned
pbsmt_5gram_nbest200
pbsmt_5gram_search10000
pbsmt_3gram_phrase5
pbsmt_3gram_phrase7
pbsmt_7gram_phrase9
pbsmt_7gram_phrase11
```

## Results

### my-ph Direction

| experiment              | direction   |   bleu_c |   nist_c |   bleu |   nist |   test_items |   exact_match_errors |   exact_match_accuracy |
|:------------------------|:------------|---------:|---------:|-------:|-------:|-------------:|---------------------:|-----------------------:|
| pbsmt_7gram_phrase11    | my-ph       |    75.8  |    1.001 |  75.8  |  1.001 |         2802 |                  998 |                64.3826 |
| pbsmt_3gram_phrase7     | my-ph       |    75.93 |    0.999 |  75.93 |  0.999 |         2802 |                 1000 |                64.3112 |
| pbsmt_5gram_main        | my-ph       |    75.9  |    1     |  75.9  |  1     |         2802 |                 1002 |                64.2398 |
| pbsmt_3gram_phrase5     | my-ph       |    75.89 |    0.999 |  75.89 |  0.999 |         2802 |                 1002 |                64.2398 |
| pbsmt_5gram_phrase13    | my-ph       |    75.72 |    1.001 |  75.72 |  1.001 |         2802 |                 1005 |                64.1328 |
| pbsmt_5gram_phrase7     | my-ph       |    75.75 |    1     |  75.75 |  1     |         2802 |                 1007 |                64.0614 |
| pbsmt_5gram_nbest200    | my-ph       |    75.69 |    1     |  75.69 |  1     |         2802 |                 1007 |                64.0614 |
| pbsmt_5gram_search10000 | my-ph       |    75.74 |    1     |  75.74 |  1     |         2802 |                 1009 |                63.99   |
| pbsmt_7gram_phrase9     | my-ph       |    75.73 |    0.999 |  75.73 |  0.999 |         2802 |                 1010 |                63.9543 |
| pbsmt_5gram_phrase11    | my-ph       |    75.57 |    1     |  75.57 |  1     |         2802 |                 1012 |                63.8829 |
| pbsmt_5gram_less_pruned | my-ph       |    75.86 |    1     |  75.86 |  1     |         2802 |                 1013 |                63.8473 |

### ph-my Direction

| experiment              | direction   |   bleu_c |   nist_c |   bleu |   nist |   test_items |   exact_match_errors |   exact_match_accuracy |
|:------------------------|:------------|---------:|---------:|-------:|-------:|-------------:|---------------------:|-----------------------:|
| pbsmt_5gram_less_pruned | ph-my       |    78.25 |    1.001 |  78.25 |  1.001 |         2802 |                  775 |                72.3412 |
| pbsmt_7gram_phrase9     | ph-my       |    78.26 |    1.001 |  78.26 |  1.001 |         2802 |                  777 |                72.2698 |
| pbsmt_5gram_phrase11    | ph-my       |    77.89 |    1.001 |  77.89 |  1.001 |         2802 |                  782 |                72.0914 |
| pbsmt_3gram_phrase7     | ph-my       |    77.98 |    1.001 |  77.98 |  1.001 |         2802 |                  784 |                72.02   |
| pbsmt_7gram_phrase11    | ph-my       |    78.06 |    1.001 |  78.06 |  1.001 |         2802 |                  785 |                71.9843 |
| pbsmt_5gram_nbest200    | ph-my       |    77.88 |    1.001 |  77.88 |  1.001 |         2802 |                  786 |                71.9486 |
| pbsmt_5gram_search10000 | ph-my       |    77.89 |    1.001 |  77.89 |  1.001 |         2802 |                  787 |                71.9129 |
| pbsmt_5gram_phrase7     | ph-my       |    77.94 |    1.001 |  77.94 |  1.001 |         2802 |                  788 |                71.8772 |
| pbsmt_5gram_main        | ph-my       |    77.91 |    1.001 |  77.91 |  1.001 |         2802 |                  789 |                71.8415 |
| pbsmt_3gram_phrase5     | ph-my       |    77.8  |    1.001 |  77.8  |  1.001 |         2802 |                  793 |                71.6988 |
| pbsmt_5gram_phrase13    | ph-my       |    76.31 |    1.001 |  76.31 |  1.001 |         2802 |                  824 |                70.5924 |



## Comparison

- **my-ph**: best system is `pbsmt_7gram_phrase11` with 998 exact-match errors. Compared with `pbsmt_5gram_main`, this changes the result by 4 errors and 0.1428 accuracy points.
- **ph-my**: best system is `pbsmt_5gram_less_pruned` with 775 exact-match errors. Compared with `pbsmt_5gram_main`, this changes the result by 14 errors and 0.4997 accuracy points.

The tuning results should be interpreted conservatively. Small differences of a few exact-match lines do not show that phrase length, n-gram order, pruning, n-best size, or stronger search is the real bottleneck.

## Error Analysis

| Root_Cause                                     |   Count |   Percentage |
|:-----------------------------------------------|--------:|-------------:|
| Context-dependent pronunciation variation      |     856 |        72.91 |
| Out of Vocabulary (OOV)                        |     232 |        19.76 |
| Missing pronunciation variant in training data |      85 |         7.24 |
| Rare over-generalization / segmentation error  |       1 |         0.09 |

The largest group is **context-dependent pronunciation variation** (856 cases, 72.91%). This is not simply a training-data mistake: it reflects the nature of Myanmar G2P, where some graphemes or words have different pronunciations depending on lexical item, neighboring syllables, and phonological context.

The remaining substantial errors are data coverage/preparation issues (OOV: 232, missing pronunciation variant: 85). These should be handled through data fixing, dictionary expansion, and verified pronunciation-variant additions.


### Examples of Missing Pronunciation Variants

| Source_Syllable   | Moses_Output   | True_Reference   | Root_Cause                                     |   Error_Count |
|:------------------|:---------------|:-----------------|:-----------------------------------------------|--------------:|
| ပျော်                 | pjo            | bjo              | Missing pronunciation variant in training data |             2 |
| ကုန်                | koun           | kou              | Missing pronunciation variant in training data |             2 |
| မြီး                 | mji:           | mhi:             | Missing pronunciation variant in training data |             2 |
| ဖွင့်                | hpwin.         | hpwin            | Missing pronunciation variant in training data |             1 |
| မြ                 | mja.           | mja              | Missing pronunciation variant in training data |             1 |
| မဲ့                 | me.            | me               | Missing pronunciation variant in training data |             1 |
| မောင်                | maun           | moun             | Missing pronunciation variant in training data |             1 |
| ဖွေ                 | hpwei          | bwei             | Missing pronunciation variant in training data |             1 |
| ဖည်                | hpe            | be               | Missing pronunciation variant in training data |             1 |
| မ                 | ma-            | a'               | Missing pronunciation variant in training data |             1 |
| ဘိ                 | bi.            | bein             | Missing pronunciation variant in training data |             1 |
| ပီ                 | pi             | pa-              | Missing pronunciation variant in training data |             1 |
| ပျက်                | pje'           | bje              | Missing pronunciation variant in training data |             1 |
| ပေး                 | pei:           | pjei:            | Missing pronunciation variant in training data |             1 |
| ပုံ                 | poun           | bou'             | Missing pronunciation variant in training data |             1 |
| ပန်း                | pan:           | pa-              | Missing pronunciation variant in training data |             1 |
| ပြူ                 | pju            | bju              | Missing pronunciation variant in training data |             1 |
| ပေါက်                | bau'           | pou'             | Missing pronunciation variant in training data |             1 |
| ပြိုင်                | pjain          | bjaun            | Missing pronunciation variant in training data |             1 |
| ကုမ်                | koun           | goun             | Missing pronunciation variant in training data |             1 |
| ပြည့်                | pjei.          | pji:             | Missing pronunciation variant in training data |             1 |
| ဝိ                 | wi.            | wei'             | Missing pronunciation variant in training data |             1 |
| အ                 | a-             | a'               | Missing pronunciation variant in training data |             1 |
| ဟီး                 | hi:            | he:              | Missing pronunciation variant in training data |             1 |
| ဟင်                | hin:           | hin              | Missing pronunciation variant in training data |             1 |
| သဲ                 | the:           | thei:            | Missing pronunciation variant in training data |             1 |
| အိပ်                | ei'            | hei'             | Missing pronunciation variant in training data |             1 |
| ကန်                | kan            | ka               | Missing pronunciation variant in training data |             1 |
| ရွား                 | jwa:           | jwa              | Missing pronunciation variant in training data |             1 |
| မ                 | ma.            | ma:              | Missing pronunciation variant in training data |             1 |

## Findings

1. **PBSMT tuning does not substantially improve the model.** The tuning runs change exact-match results only marginally.

2. **The largest error group is Myanmar G2P pronunciation variation.** The previously named ambiguity category is better understood as context-dependent pronunciation behavior in Myanmar. Some words or syllables naturally have different pronunciations depending on lexical and phonological context.

3. **Data coverage and preparation still matter.** OOV and missing pronunciation variants are actionable data problems. These can be improved by dictionary expansion, data correction, and adding verified pronunciation variants.

4. **Decoder search is not the main issue.** Stronger search does not materially improve the result, so further Moses tuning is unlikely to produce large gains.


## Conclusion

The confirmed 5-gram PBSMT baseline is strong, and additional PBSMT tuning gives only marginal improvements. The main bottleneck is linguistic and data-driven: Myanmar words can have context-dependent pronunciations, and the dataset still contains OOV or missing-variant cases.

The best next step is a hybrid improvement strategy: keep the original-template PBSMT system, add a small rule-based correction layer for high-frequency pronunciation variation, and improve the dataset with targeted augmentation and data fixes.

## Generated Artifacts

```text
pbsmt_experiment_results.csv
pbsmt_error_summary_report_labels.csv
pbsmt_error_leaderboard_report_labels.csv
```

## Sources

- Ye Kyaw Thu, `myG2P`: https://github.com/ye-kyaw-thu/myG2P
- Ye Kyaw Thu, `syl-Normalizer`: https://github.com/ye-kyaw-thu/syl-Normalizer
