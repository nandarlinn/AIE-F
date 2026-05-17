# Assignment Report

## Project Title

Myanmar Text Cleaning, Tokenization, and Language Model Preparation

## Objective

The goal of this project was to prepare Myanmar text data for language-model analysis by:

1. cleaning raw CSV data
2. tokenizing the cleaned text in two ways
3. training a baseline language model
4. preparing a small domain test set
5. checking perplexity results for evaluation

## What I Did

### Step 1: Data Cleaning

I cleaned the raw files in `Dataset/` and produced cleaned CSVs in `data_cleaning_process/cleaned/`.

Main cleaning actions:

- removed Unicode noise and invisible characters
- removed emoji, URLs, emails, mentions, and hashtags
- normalized punctuation and spacing
- removed short or low-quality rows
- removed duplicate rows

### Step 2: Tokenization

I tokenized the cleaned data in two formats:

- syllable-based tokenization using `sylbreak`
- word-based tokenization using `oppaWord`

Outputs are stored in:

- `tokenization_process/syllable_based/`
- `tokenization_process/word_based/`

### Step 3: Language Model Training

I extracted plain-text corpora from the tokenized CSV files and trained KenLM 5-gram models.

Outputs are stored in:

- `language_model_process/corpus/`
- `language_model_process/models/arpa/`
- `language_model_process/models/binary/`

### Step 4: Domain Test Preparation

I prepared domain-specific CSV files in `DomainTest/` and limited each one to 200 tokens for fair comparison.

Outputs are stored in:

- `DomainTest/cleaned/`
- `DomainTest/syllable_based_200/`
- `DomainTest/word_based_200/`

### Step 5: Perplexity Evaluation

I checked domain perplexity results using the trained language model.

The report is already available in:

- `DomainTest/perplexity/reports/domain_ppl.tsv`

## Result Summary

### Cleaning Result

| file | rows_in | rows_kept |
| --- | ---: | ---: |
| four.csv | 102 | 42 |
| one.csv | 732 | 636 |
| three.csv | 102 | 44 |
| two.csv | 8116 | 6453 |

### Tokenization Result

| file | syllable tokens | word tokens |
| --- | ---: | ---: |
| four.csv | 3312 | 2185 |
| one.csv | 20713 | 13005 |
| three.csv | 1149 | 767 |
| two.csv | 290666 | 184002 |

### Language Model Result

I trained two 5-gram models:

- syllable-based LM
- word-based LM

Both models were built successfully and saved in binary and ARPA form.

### Domain Test Result

The domain test set was trimmed to 200 tokens for each file and prepared in both syllable and word formats.

## What I Should Do For Better Results

If I want a stronger result for the assignment, I should do these steps:

1. Split the data into train, development, and test sets instead of using only one cleaned pool.
2. Compare 3-gram, 4-gram, and 5-gram models to see which gives better perplexity.
3. Keep both syllable-based and word-based models, then compare which one performs better on the domain test set.
4. Review rejected rows from the cleaning step to make sure useful Myanmar text is not removed by mistake.
5. Add more domain data so the model is less sparse and more stable.
6. Use perplexity as the main evaluation metric and report the best model clearly.

## Suggested Workflow For Final Submission

If I want to finish the assignment cleanly, the best order is:

1. clean the raw CSV files
2. tokenize the cleaned text
3. train the language models
4. prepare the domain test set
5. compare perplexity results
6. write the final conclusion based on the best-performing model

## Short Conclusion

This project builds a complete Myanmar text pipeline from raw CSV data to cleaned text, tokenized text, language models, and domain perplexity evaluation.  
For a better assignment result, the next improvement should be proper train/dev/test splitting and a careful comparison between syllable-based and word-based models.

