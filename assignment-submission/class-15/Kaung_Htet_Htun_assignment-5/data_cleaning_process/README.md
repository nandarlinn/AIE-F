# Myanmar CSV Data Cleaning Process

ဒီ folder က `Dataset/*.csv` မူရင်းဖိုင်တွေကို မထိဘဲ cleaned CSV နဲ့ report တွေ ထုတ်ပေးဖို့ပါ။

## Folder Structure

- `scripts/inspect_dataset.sh` - CSV line count, encoding, duplicate, non-Myanmar token overview report ထုတ်ရန်
- `scripts/clean_myanmar_csv.pl` - Myanmar text cleaning pipeline
- `cleaned/` - conservative cleaned CSV output
- `reports/` - cleaning summary နဲ့ rejected rows

## Run

```bash
bash data_cleaning_process/scripts/inspect_dataset.sh
perl data_cleaning_process/scripts/clean_myanmar_csv.pl
```

Cleaned files:

```text
data_cleaning_process/cleaned/four.csv
data_cleaning_process/cleaned/one.csv
data_cleaning_process/cleaned/three.csv
data_cleaning_process/cleaned/two.csv
```

Summary:

```text
data_cleaning_process/reports/cleaning_summary.tsv
data_cleaning_process/reports/rejected_*.csv.tsv
```

Current conservative run summary:

```text
file       rows_in  rows_kept  non_myanmar  short_or_bad  duplicates  empty_after_clean
four.csv   102      42         48           10            2           0
one.csv    732      636        68           0             28          0
three.csv  102      44         50           3             5           0
two.csv    8116     6453       1517         19            122         5
```

## Cleaning Logic

Script က text column တွေကိုပဲ clean လုပ်ပါတယ်။ `category` လို label column တွေကို မထိပါဘူး။

လုပ်ဆောင်တဲ့အဆင့်တွေ:

1. Unicode normalization: `NFKC` ပြီး `NFC`
2. Invisible / zero-width noise ဖယ်ခြင်း
3. Emoji, pictograph, dingbat, variation selector ဖယ်ခြင်း
4. URL, email, hashtag, mention noise ဖယ်ခြင်း
5. punctuation နဲ့ symbol တွေကို space ပြောင်းခြင်း
6. non-Myanmar text ပါတဲ့ row တစ်ကြောင်းလုံး reject လုပ်ခြင်း
7. repeated character များကို default ၂ လုံးထိပဲထားခြင်း
8. multiple spaces ကို single space လုပ်ခြင်း
9. short/bad sentence ဖယ်ခြင်း
10. duplicate text row ဖယ်ခြင်း

## Conservative Defaults

Over-clean မဖြစ်အောင် default threshold တွေကို အရမ်းမတင်းကျပ်ထားပါဘူး။

- `--min-myanmar-chars 4`
- `--min-words 2`
- `--max-repeat 2`
- English/other non-Myanmar text ပါတဲ့ row တွေကို မဖျက်ပြင်ဘဲ row တစ်ခုလုံး reject လုပ်တယ်
- ASCII/Myanmar digits တွေကိုတော့ သိမ်းထားတယ်
- duplicate ကို file တစ်ခုချင်းစီအတွင်းမှာပဲ ဖယ်တယ်

## Useful Options

Non-Myanmar ပါတဲ့ row တွေကို မဖြုတ်ချင်ဘူးဆိုရင်:

```bash
perl data_cleaning_process/scripts/clean_myanmar_csv.pl --no-reject-non-myanmar
```

ဒါက row ကိုသိမ်းထားပြီး punctuation/emoji/space/repeated-character cleaning ပဲလုပ်ပါမယ်။

Short sentence filter ကိုပိုသက်သာချင်ရင်:

```bash
perl data_cleaning_process/scripts/clean_myanmar_csv.pl --min-words 1 --min-myanmar-chars 1
```

Duplicate မဖယ်ချင်ရင်:

```bash
perl data_cleaning_process/scripts/clean_myanmar_csv.pl --no-remove-duplicates
```

Repeated character ကို ၃ လုံးထိခွင့်ပြုချင်ရင်:

```bash
perl data_cleaning_process/scripts/clean_myanmar_csv.pl --max-repeat 3
```

## Ubuntu Command-Line Notes

Script run ပြီးနောက် reject ဖြစ်တဲ့ row တွေကို ကြည့်ပါ:

```bash
less data_cleaning_process/reports/rejected_two.csv.tsv
```

Non-Myanmar ပါလို့ reject ဖြစ်တဲ့ row တွေကိုပဲကြည့်ချင်ရင်:

```bash
grep '^non_myanmar' data_cleaning_process/reports/rejected_*.tsv
```

Cleaned output အရေအတွက်ကို စစ်ပါ:

```bash
wc -l Dataset/*.csv data_cleaning_process/cleaned/*.csv
```

Summary ကို column align နဲ့ကြည့်ပါ:

```bash
column -t -s $'\t' data_cleaning_process/reports/cleaning_summary.tsv
```
