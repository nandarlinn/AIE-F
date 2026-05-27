# Step 3: Myanmar Tokenization

ဒီအဆင့်က `data_cleaning_process/cleaned/*.csv` ကို input အဖြစ်ယူပြီး tokenization output နှစ်မျိုး ထုတ်ဖို့ပါ။

- syllable-based: `sylbreak`
- word-based: `oppaWord`

CSV ထဲက text column တွေကိုပဲ tokenize လုပ်ပြီး `category` လို label column တွေကို မပြောင်းပါဘူး။

## Folder Structure

- `scripts/setup_tokenizers.sh` - `sylbreak` နဲ့ `oppaWord` ကို `tools/` ထဲ download လုပ်ရန်
- `scripts/tokenize_cleaned_csv.pl` - CSV-aware tokenizer runner
- `scripts/run_tokenization.sh` - syllable-based နဲ့ word-based နှစ်ခုလုံး run ရန်
- `syllable_based/` - syllable tokenized CSV output
- `word_based/` - word tokenized CSV output
- `reports/` - token count summary
- `tools/` - external tokenizer repositories

## Install Tokenizers

Local စက်ထဲမှာ tokenizer repo မရှိသေးရင်:

```bash
bash tokenization_process/scripts/setup_tokenizers.sh
```

Expected paths:

```text
tokenization_process/tools/sylbreak/python/sylbreak.py
tokenization_process/tools/oppaWord/oppa_word.py
tokenization_process/tools/oppaWord/data/myg2p_mypos.dict
```

မှတ်ချက်: `oppaWord` repo structure က version အလိုက် filename မတူရင် `--oppaword-py` နဲ့ path ကိုပြောင်းပေးနိုင်ပါတယ်။

## Run Tokenization

နှစ်ခုလုံး run:

```bash
bash tokenization_process/scripts/run_tokenization.sh
```

သီးသန့် syllable-based:

```bash
perl tokenization_process/scripts/tokenize_cleaned_csv.pl \
  --mode syllable \
  --output-dir tokenization_process/syllable_based
```

သီးသန့် word-based:

```bash
perl tokenization_process/scripts/tokenize_cleaned_csv.pl \
  --mode word \
  --output-dir tokenization_process/word_based
```

## Output

Syllable-based output:

```text
tokenization_process/syllable_based/four.csv
tokenization_process/syllable_based/one.csv
tokenization_process/syllable_based/three.csv
tokenization_process/syllable_based/two.csv
```

Word-based output:

```text
tokenization_process/word_based/four.csv
tokenization_process/word_based/one.csv
tokenization_process/word_based/three.csv
tokenization_process/word_based/two.csv
```

Reports:

```text
tokenization_process/reports/tokenization_syllable_summary.tsv
tokenization_process/reports/tokenization_word_summary.tsv
```

Current run summary:

```text
Syllable-based sylbreak
file       rows  text_columns  tokens
four.csv   42    text          3312
one.csv    636   Text-MM       20713
three.csv  44    text          1149
two.csv    6453  text          290666

Word-based oppaWord
file       rows  text_columns  tokens
four.csv   42    text          2185
one.csv    636   Text-MM       13005
three.csv  44    text          767
two.csv    6453  text          184002
```

## Process Explanation

1. Cleaned CSV ကိုဖတ်တယ်။
2. Header ရှိရင် `text`, `Text-MM`, `question`, `answer`, `content`, `body` စတဲ့ text column ကို detect လုပ်တယ်။
3. Text fields တွေကို file တစ်ခုချင်းစီ batch temporary file ထဲရေးတယ်။
4. Syllable mode မှာ `sylbreak.py` ကို run တယ်။
5. Word mode မှာ `oppa_word.py` ကို `myg2p_mypos.dict`, `my_not_num`, `Bi-MM fallback`, `bimm-boost 150` setting နဲ့ run တယ်။
6. Tokenizer output ထဲက line break တွေကို single-line CSV field ဖြစ်အောင် normalize လုပ်တယ်။
7. CSV structure ကိုပြန်ရေးပြီး label/category column ကိုမထိဘဲ output ထုတ်တယ်။
8. File တစ်ခုချင်းစီအတွက် row count နဲ့ token count summary ထုတ်တယ်။

## Delimiter

Default token delimiter က space ပါ။ Pipe delimiter လိုချင်ရင်:

```bash
perl tokenization_process/scripts/tokenize_cleaned_csv.pl \
  --mode syllable \
  --delimiter '|' \
  --output-dir tokenization_process/syllable_based_pipe
```

## Notes

- `sylbreak` က regular-expression based syllable segmentation tool ဖြစ်ပြီး Unicode Myanmar text အတွက်သုံးပါတယ်။
- `oppaWord` က Myanmar word segmentation အတွက်သုံးပါတယ်။
- `oppaWord` repo ကိုမရနိုင်တဲ့ environment မှာ `myWord` ကို fallback အဖြစ် path ပေးပြီး run နိုင်အောင် script ထဲမှာ `--myword-py` option ထည့်ထားပါတယ်။

References:

- `sylbreak`: https://github.com/ye-kyaw-thu/sylbreak
- `oppaWord`: https://github.com/ye-kyaw-thu/oppaWord
- `myWord` fallback: https://github.com/ye-kyaw-thu/myWord
