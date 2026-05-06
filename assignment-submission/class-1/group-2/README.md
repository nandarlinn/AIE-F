# Group 2 Assignment Submission

## Dataset

Our custom dataset comprises Facebook comments, synthetic data generated via generative models, and manually authored entries. It is categorized into the following six emotional classes:
- 0: ဝမ်းနည်းမှု (sadness)
- 1: ပျော်ရွှင်မှု (joy)
- 2: ချစ်ခင်မှု (love)
- 3: ဒေါသ (anger)
- 4: ကြောက်ရွံ့မှု (fear)
- 5: အံ့အားသင့်မှု (surprise)

## Project Structure

```
group-2/
├── ...
├── group2-hybrid-eliza.py      # main CLI: --mode train | eval | chat
├── streamlit_app.py            # Streamlit entry for Cloud
├── img/
├── data/
│   ├── raw_ungrouped/          # original team files (not merged)
│   ├── annotated_ungrouped/    # cleaned/labeled team files (not merged)
│   ├── merged/                 # combined sheets
│   ├── merged_preprocessed/    # combined sheets before/after downsampling
│   └── stopwords.txt           # Burmese stopword list (see References)
├── checkpoints/                # saved `.pth` bundles
├── notebooks/                  # EDA and demos
├── scripts/
│   ├── train.py
│   ├── eval.py
│   ├── chat.py                 # shared chat logic; loads src/model.py + src/eliza.py
│   ├── streamlit_chatter.py    # Streamlit UI (subprocess from chat.py or streamlit_app.py)
│   └── custom_ui_chatter.py    # browser UI (subprocess from chat.py + UI from experiments/burmese_chat_ui.py)
├── experiments/                # early standalone hybrids, guides, logs, burmese_chat_ui.py
└── src/
    ├── preprocessing.py        # emotion pipeline: normalize, MMDT, stopwords, optional char n-grams
    ├── rabbit.py               # Zawgyi to Unicode
    ├── vocab_builder.py
    ├── prep_data.py            # shared tensors/loaders for train / eval / chat encoding
    ├── eliza_rules.py
    ├── eliza.py                # rule engine (separate tokenization from emotion model)
    ├── model.py                # BiLSTM + attention classifier
    └── plot.py                 # optional confusion matrix PNG (eval / train)
```

## Project Flow

The code is organized so a single wrapper script controls the high-level mode, while `src/` contains reusable building blocks.

![Project Mindmap](img/mindmap.png)

- `group2-hybrid-eliza.py` (primary entry point): the top-level CLI wrapper/dispatcher. It selects one of the modes and starts the corresponding script in `scripts/`.

- `scripts/train.py`: training orchestration. It uses `src/prep_data.py` to build train/validation tensors and class weights, uses `src/model.py` to define the model, and saves a checkpoint that includes model weights plus vocabulary/label mappings.

- `scripts/eval.py`: evaluation/prediction on a labeled dataset. It loads the saved checkpoint, uses `src/prep_data.py` to convert raw texts into padded token-id tensors using the same preprocessing pipeline, then runs `src/model.py` to produce predictions and confidence scores.

- `scripts/chat.py`: interactive inference. It loads the saved checkpoint and reuses the same preprocessing + model inference path as `scripts/eval.py` for user-entered text, then prints the predicted emotion and confidence.

- `src/prep_data.py`: shared data preparation pipeline used by both training and inference. It connects dataset reading, text preprocessing, token/id encoding, padding/truncation, train/val splitting (for training), and class-weight computation; it relies on `src/preprocessing.py` (TextProcessor) and `src/vocab_builder.py` (vocabulary + label mapping utilities).

- `src/preprocessing.py`: text preprocessing pipeline. It performs Zawgyi-to-Unicode normalization when needed, regex punctuation cleanup, tokenization with the MMDT tokenizer, and stopword removal; it relies on `src/rabbit.py` for Zawgyi conversion and on the Myanmar detection/tokenization libraries.

- `src/vocab_builder.py`: vocabulary and label mapping utilities. It builds the $word \rightarrow id$ vocabulary from training tokenized text and defines the fixed $label \leftrightarrow id$ order for the six emotion classes.

- `src/model.py`: model definition. It embeds token ids, runs a bidirectional LSTM, then pools sequence information (either attention pooling or final-state pooling) and applies a linear classifier to get logits.

- `src/rabbit.py`: Zawgyi-to-Unicode conversion rules used by the preprocessing pipeline.

In short: `group2-hybrid-eliza.py` starts the run; `scripts/*.py` control train/eval/chat; `src/*.py` implements the reusable preprocessing/model utilities.

## CLI Guide for `group2-hybrid-eliza.py`

Run from `group-2/`.

Modes: `--mode train`, `--mode eval`, `--mode chat`.

Important flags (not exhaustive):
- `--data_path`,
- `--checkpoint_path` (read/write for train; read for eval/chat),
- `--stopwords_path`,
- `--epochs`,
- `--batch_size`,
- `--lr`,
- `--embed_dim`,
- `--hidden_dim`,
- `--use_attention` / `--no-use_attention`,
- `--use_char_ngrams` / `--no-use_char_ngrams`,
- `--confusion_matrix_out`,
- `--chat_ui`, `--language`, `--custom_ui_host`, `--custom_ui_port`.

Defaults: training reads `./data/merged_preprocessed/data_after_downsampling.csv` and saves to `./checkpoints/bilstm_larger_params_after_downsampl.pth`. Train/eval also write a confusion matrix PNG under `./img/` unless disabled.

---

### Training Example

**1. Default**
```bash
python group2-hybrid-eliza.py --mode train
```

**2. Custom model shape & checkpoint name**
```bash
python group2-hybrid-eliza.py --mode train --embed_dim 512 --hidden_dim 256 --checkpoint_path ./checkpoints/bilstm_larger_params_after_downsampl.pth
```

**3. Further customizations**  
```bash
python group2-hybrid-eliza.py --mode train --data_path ./data/merged/Combined.csv --embed_dim 256 --hidden_dim 128 --checkpoint_path ./checkpoints/bilstm_smaller_params.pth
```
or
```bash
python group2-hybrid-eliza.py --mode train --data_path ./data/merged/Combined.csv --embed_dim 512 --hidden_dim 256 --checkpoint_path ./checkpoints/bilstm_larger_params.pth
```

---

### Evaluation Example

**1. Default**

```bash
python group2-hybrid-eliza.py --mode eval
```

**2. Match a specific training run**
```bash
python group2-hybrid-eliza.py --mode eval --data_path ./data/merged/Combined.csv --checkpoint_path ./checkpoints/bilstm_larger_params.pth
```

---

### Chat Example

**1. Streamlit UI**

```bash
python group2-hybrid-eliza.py --mode chat --chat_ui streamlit
```
or
```bash
streamlit run streamlit_app.py
```

**2. Custom UI**

```bash
python group2-hybrid-eliza.py --mode chat --chat_ui custom_ui
```

Optional: `--language en` or `--language mm`, `--custom_ui_host`, `--custom_ui_port`.

---

## Current Public Host

- Streamlit: https://group2-hybrid-eliza.streamlit.app/
- Render: https://group2-hybrid-eliza.onrender.com

---

## References

- Unicode Myanmar script blocks:
    - https://www.unicode.org/charts/PDF/U1000.pdf
    - https://www.unicode.org/charts/PDF/UAA60.pdf
- Burmese grammar: https://online.fliphtml5.com/rrlzh/mbir/#p=1
- Rabbit Zawgyi to Unicode Converter: https://github.com/Rabbit-Converter/Rabbit-Python 
- MMDT Tokenizer: https://github.com/Myanmar-Data-Tech/mmdt-tokenizer
