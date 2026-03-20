# Group 2 Assignment Submission

## Dataset

Our custom dataset comprises Facebook comments, synthetic data generated via generative models, and manually authored entries. It is categorized into the following six emotional classes:
- 0: Sadness
- 1: Joy
- 2: Love
- 3: Anger
- 4: Fear
- 5: Surprise

## Project Structure

```
group-2/
├── group2-hybrid-eliza.py          # single CLI entry: --mode train|eval|chat
├── data/
│   ├── raw_ungrouped/              # original team contributions (not merged)
│   ├── annotated_ungrouped/        # cleaned team contributions (not merged)
│   ├── merged/                     # merged spreadsheets before final export
│   ├── merged_preprocessed/        # final merged CSVs used for training
│   │   ├── data_before_downsampling.csv
│   │   └── data_after_downsampling.csv
│   └── stopwords.txt               # Burmese stopword list (see Sources)
├── checkpoints/                    # saved checkpoints and tokenizer/vocab artifacts
├── notebooks/                      # experimental analysis and EDA
├── scripts/
│   ├── train.py                    # model training execution
│   ├── eval.py                     # model evaluation/prediction
│   └── chat.py                     # interactive inference; reuses eval helpers
├── src/
│   ├── preprocessing.py            # Burmese text normalization, tokenization (see Sources), stopword removal
│   ├── rabbit.py                   # Zawgyi to Unicode conversion utilities (see Sources)
│   ├── vocab_builder.py            # vocab/token-id and label-id helpers
│   ├── prep_data.py                # shared preprocessing helpers for train/eval/chat
│   └── model.py                    # LSTM architecture and layer definitions
├── README.md                       # documentation
├── environment.yaml                # conda environment configuration
└── requirements.txt                # dependencies
```

## Sources:
- Burmese grammar: https://online.fliphtml5.com/rrlzh/mbir/#p=1
- Rabbit Zawgyi to Unicode Converter: https://github.com/Rabbit-Converter/Rabbit-Python 
- MMDT Tokenizer: https://github.com/Myanmar-Data-Tech/mmdt-tokenizer