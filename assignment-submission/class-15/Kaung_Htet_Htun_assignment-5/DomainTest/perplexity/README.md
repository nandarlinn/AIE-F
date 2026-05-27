# Step 6: Domain Perplexity

ဒီအဆင့်က Step 5 ထဲက `word_based_200` domain test files တွေကို Step 4 မှာ train ထားတဲ့ KenLM word 5-gram model နဲ့ perplexity တွက်ပါတယ်။

Input:

- `DomainTest/word_based_200/FB Comment.csv` -> Social
- `DomainTest/word_based_200/Law .csv` -> Law
- `DomainTest/word_based_200/news1.csv` -> News

Model:

```text
language_model_process/models/binary/word_5gram.binary
```

Run:

```bash
bash DomainTest/perplexity/scripts/run_step6.sh
```

Outputs:

```text
DomainTest/perplexity/reports/domain_ppl.tsv
DomainTest/perplexity/charts/domain_ppl_bar.png
```

Current result:

```text
News PPL = 1.3815
Social PPL = 1.1328
Law PPL = 1.5402
```

Detailed report:

```text
domain   sentences  query_tokens  oov  ppl
Social   13         213           0    1.1328
Law      10         210           1    1.5402
News     6          206           0    1.3815
```

Note: Step 5 files are 200 word tokens each. KenLM `query` token count is slightly higher because it includes sentence boundary tokens for each CSV row.

Interpretation:

- Lower PPL means the base LM finds that domain easier / more familiar.
- Higher PPL means the domain is harder / less matched to the base LM.
- Current hardest domain: `Law`.
