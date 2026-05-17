# Step 4: Train Base Language Model with KenLM

ဒီအဆင့်က Step 3 tokenized CSV output ကိုသုံးပြီး KenLM n-gram base language model train လုပ်ဖို့ပါ။

Input:

- `tokenization_process/syllable_based/*.csv`
- `tokenization_process/word_based/*.csv`

Output:

- plain text corpus
- ARPA language model
- KenLM binary language model
- training/build logs

## Folder Structure

- `scripts/extract_lm_corpus.pl` - tokenized CSV ထဲက text column ကို plain text corpus ထုတ်ရန်
- `scripts/train_kenlm.sh` - syllable LM နဲ့ word LM နှစ်ခုလုံး train/build လုပ်ရန်
- `scripts/query_lm.sh` - trained binary LM ကို query စမ်းရန်
- `corpus/` - KenLM train input text
- `models/arpa/` - `.arpa` language models
- `models/binary/` - KenLM binary models
- `reports/` - corpus/training/build summary logs
- `samples/` - sample queries ထားရန်

## KenLM Tools

Local system ထဲမှာ KenLM binaries ရှိနေပါတယ်:

```text
/home/phantom/mosesdecoder/bin/lmplz
/home/phantom/mosesdecoder/bin/build_binary
/home/phantom/mosesdecoder/bin/query
```

တခြား path သုံးချင်ရင်:

```bash
KENLM_BIN=/path/to/kenlm/build/bin bash language_model_process/scripts/train_kenlm.sh
```

## Train

Default က 5-gram LM ဖြစ်ပါတယ်။

```bash
bash language_model_process/scripts/train_kenlm.sh
```

3-gram စမ်းချင်ရင်:

```bash
ORDER=3 bash language_model_process/scripts/train_kenlm.sh
```

## Output Files

Syllable-based LM:

```text
language_model_process/corpus/syllable.txt
language_model_process/models/arpa/syllable_5gram.arpa
language_model_process/models/binary/syllable_5gram.binary
```

Word-based LM:

```text
language_model_process/corpus/word.txt
language_model_process/models/arpa/word_5gram.arpa
language_model_process/models/binary/word_5gram.binary
```

Summary:

```text
language_model_process/reports/lm_training_summary.tsv
language_model_process/reports/corpus_syllable_summary.tsv
language_model_process/reports/corpus_word_summary.tsv
language_model_process/reports/train_syllable_5gram.log
language_model_process/reports/train_word_5gram.log
```

Current 5-gram training summary:

```text
model     order  corpus_lines  arpa_size  binary_size
syllable  5      7175          36M        14M
word      5      7175          42M        13M
```

Corpus extraction summary:

```text
model     files  rows  kept  duplicates  too_short
syllable  4      7175  7175  0           0
word      4      7175  7175  0           0
```

## Query/Test

Word LM ကို query စမ်းရန်:

```bash
bash language_model_process/scripts/query_lm.sh word "မြန်မာ နိုင်ငံ တွင် မည် သည့် ကုမ္ပဏီ အမျိုးအစား များ အား မှတ်ပုံတင် ရ ပါ သနည်း"
```

Syllable LM ကို query စမ်းရန်:

```bash
bash language_model_process/scripts/query_lm.sh syllable "မြန် မာ နိုင် ငံ တွင် မည် သ ည့် ကုမ္ပ ဏီ အ မျိုး အ စား များ အား မှတ် ပုံ တင် ရ ပါ သ နည်း"
```

## Process Explanation

1. Tokenized CSV files ကိုဖတ်တယ်။
2. Header ရှိရင် `text`, `Text-MM`, `question`, `answer`, `content`, `body` column ကို text column အဖြစ် detect လုပ်တယ်။
3. Text column ကို line-per-sentence plain text corpus အဖြစ်ထုတ်တယ်။
4. Duplicate sentence ကို corpus ထဲမှာ တစ်ကြောင်းပဲထားတယ်။
5. `lmplz -o 5` နဲ့ ARPA LM train လုပ်တယ်။
6. `--discount_fallback` ထည့်ထားလို့ small/uneven corpus မှာ discount estimation မရတဲ့ n-gram order တွေကြောင့် train မပျက်ဘူး။
7. `build_binary` နဲ့ ARPA ကို faster query/load အတွက် binary LM ပြောင်းတယ်။
8. `query` command နဲ့ sentence score/perplexity စမ်းနိုင်တယ်။

## Notes

- Syllable-based LM က token ပိုများပြီး OOV နည်းနိုင်ပါတယ်။
- Word-based LM က semantic unit ပိုကောင်းပေမယ့် tokenizer quality နဲ့ dictionary coverage ပေါ်မူတည်ပါတယ်။
- Dataset သေးသေးမှာ 5-gram က sparse ဖြစ်နိုင်လို့ 3-gram/4-gram နဲ့လည်း နှိုင်းယှဥ်စမ်းသင့်ပါတယ်။
- Production LM အတွက် train/dev/test split, perplexity evaluation, pruning ကို ထပ်ထည့်သင့်ပါတယ်။
