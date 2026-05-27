# Step 5: Domain Test Set Preparation

ဒီ folder ထဲက domain မတူတဲ့ CSV 3 ဖိုင်ကို previous steps နဲ့ format တူအောင် prepare လုပ်ထားပါတယ်။

Input files:

- `FB Comment.csv`
- `Law .csv`
- `news1.csv`

လုပ်ထားတဲ့ pipeline:

1. Data cleaning
   - Unicode normalization
   - emoji/noise removal
   - punctuation removal
   - space normalization
   - non-Myanmar text ပါတဲ့ row reject
   - short sentence reject
   - duplicate reject
2. Tokenization
   - syllable-based: `sylbreak`
   - word-based: `oppaWord`
3. Same-size trimming
   - file တစ်ခုချင်းစီကို 200 tokens အတိဖြတ်ထုတ်
   - syllable output မှာ syllable-token 200
   - word output မှာ word-token 200

## Folder Structure

- `scripts/domain_prepare_200_tokens.pl` - cleaning + tokenization + 200-token trimming
- `scripts/run_domain_step5.sh` - one-command runner
- `cleaned/` - cleaned CSV files
- `syllable_based_200/` - sylbreak tokenized 200-token CSV files
- `word_based_200/` - oppaWord tokenized 200-token CSV files
- `reports/` - summary and rejected rows

## Run

```bash
bash DomainTest/scripts/run_domain_step5.sh
```

## Output

Cleaned:

```text
DomainTest/cleaned/FB Comment.csv
DomainTest/cleaned/Law .csv
DomainTest/cleaned/news1.csv
```

Syllable-based 200-token output:

```text
DomainTest/syllable_based_200/FB Comment.csv
DomainTest/syllable_based_200/Law .csv
DomainTest/syllable_based_200/news1.csv
```

Word-based 200-token output:

```text
DomainTest/word_based_200/FB Comment.csv
DomainTest/word_based_200/Law .csv
DomainTest/word_based_200/news1.csv
```

Reports:

```text
DomainTest/reports/domain_200_summary.tsv
DomainTest/reports/rejected_FB Comment.csv.tsv
DomainTest/reports/rejected_Law .csv.tsv
DomainTest/reports/rejected_news1.csv.tsv
```

Current run summary:

```text
file            rows_in  rows_cleaned  rejected_non_myanmar  rejected_short  duplicates  syllable_tokens  word_tokens  status
FB Comment.csv  19       15            0                     4               0           200              200          ok
Law .csv        19       19            0                     0               0           200              200          ok
news1.csv       19       19            0                     0               0           200              200          ok
```

Token count verification:

```text
DomainTest/syllable_based_200/FB Comment.csv  200
DomainTest/syllable_based_200/Law .csv        200
DomainTest/syllable_based_200/news1.csv       200
DomainTest/word_based_200/FB Comment.csv      200
DomainTest/word_based_200/Law .csv            200
DomainTest/word_based_200/news1.csv           200
```

## Notes

- Output CSV format က input နဲ့တူတူ `Text`/`text` header ကိုသိမ်းထားပါတယ်။
- Word output နဲ့ syllable output နှစ်ခုလုံး 200 tokens အတိဖြစ်အောင် row boundary အတွင်းက last row ကိုလိုသလောက် truncate လုပ်ထားပါတယ်။
- `sylbreak` နဲ့ `oppaWord` tool paths ကို script ထဲက option တွေနဲ့ပြောင်းနိုင်ပါတယ်။
- `FB Comment.csv` ထဲက space မပါဘဲရှည်နေတဲ့ comment 4 ကြောင်းဟာ previous cleaning rule အတိုင်း short/bad အဖြစ် reject ဖြစ်ထားပါတယ်။ အသေးစိတ်ကို `DomainTest/reports/rejected_FB Comment.csv.tsv` ထဲမှာကြည့်နိုင်ပါတယ်။
