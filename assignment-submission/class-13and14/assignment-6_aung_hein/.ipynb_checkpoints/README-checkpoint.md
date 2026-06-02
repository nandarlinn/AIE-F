# Myanmar G2P PBSMT Final Experiment Report

## Objective

This project evaluates Phrase-Based Statistical Machine Translation (PBSMT) for Myanmar grapheme-to-phoneme conversion. The final main model is the **5-gram PBSMT** configuration that follows the original successful setup. Additional 3-gram and 7-gram experiments are included for context-size comparison.

## Data Used

The experiment uses Sayar Ye Kyaw Thu's **myG2P** Myanmar/Burmese grapheme-to-phoneme dictionary data. The source side is Myanmar grapheme syllables (`.my`), and the target side is phoneme representation (`.ph`).

Final cleaned files:

```text
g2p-par/train_clean.my
g2p-par/train_clean.ph
g2p-par/dev_clean.my
g2p-par/dev_clean.ph
g2p-par/test_clean.my
g2p-par/test_clean.ph
```

## Normalization

The data was normalized using `syl-Normalizer` version `0.6` and the syllable dictionary:

```text
syl-Normalizer/ver_0.6/final_syl_dictionary_13Feb2024.sorted.txt
```

After normalization, test files were converted into Moses SGM format:

```text
g2p-par/sgm/test.my.src.sgm
g2p-par/sgm/test.my.ref.sgm
g2p-par/sgm/test.ph.src.sgm
g2p-par/sgm/test.ph.ref.sgm
```

## Final Configuration Decision

The previous generated configs forced monotonic decoding using `-distortion-limit 0`, which caused a large BLEU drop. The final clean setup removes this constraint and follows the original decoder behavior:

```text
[TUNING]
decoder-settings = ""

[EVALUATION]
decoder-settings = "-search-algorithm 1 -cube-pruning-pop-limit 5000 -s 5000"
```

## Experiment Matrix

Final experiment folder:

```text
pbsmt_final_clean_experiments/
```

Included variants:

```text
3g_phrase5_original_search
3g_phrase7_original_search
3g_phrase5_less_pruned
5g_phrase9_original_search_MAIN
5g_phrase9_less_pruned
5g_phrase11_original_search
7g_phrase9_original_search
7g_phrase11_original_search
7g_phrase9_less_pruned
```

## Results

Generated: 2026-06-02 00:25

| variant                         | direction   |   bleu_c |   nist_c |   bleu |   nist |   test_items |   exact_match_errors |   exact_match_accuracy |
|:--------------------------------|:------------|---------:|---------:|-------:|-------:|-------------:|---------------------:|-----------------------:|
| 5g_phrase9_original_search_MAIN | my-ph       |    36.02 |    0.659 |  36.02 |  0.659 |         2802 |                 1742 |                37.8301 |

## Comparison

Best model by direction:

- **my-ph**: `5g_phrase9_original_search_MAIN` with 1742 exact-match errors, BLEU-c=36.02

The final comparison should prioritize exact-match error count because G2P output should match the reference pronunciation sequence closely. BLEU/NIST are included for consistency with Moses EMS evaluation.

## Findings

- The original-like 5-gram config should be treated as the main baseline.
- Forced monotonic decoding was not part of the original successful experiment and should not be used in the final setup.
- 3-gram variants test lower context and sparsity robustness.
- 7-gram variants test whether longer phoneme context helps, but may be more sensitive to sparsity and Moses/KenLM order support.
- Error diagnostics should be run on the best `my-ph` model to separate ambiguity, OOV, missing-reference, and decoder errors.

## Conclusion

The final clean setup restores the original decoder behavior and uses the 5-gram model as the main experiment. 3-gram and 7-gram variants are retained only for comparison. The final model should be selected using exact-match errors first, then BLEU/NIST, then root-cause error analysis.

## Sources

- Ye Kyaw Thu, `myG2P`: https://github.com/ye-kyaw-thu/myG2P
- Ye Kyaw Thu, `syl-Normalizer`: https://github.com/ye-kyaw-thu/syl-Normalizer
