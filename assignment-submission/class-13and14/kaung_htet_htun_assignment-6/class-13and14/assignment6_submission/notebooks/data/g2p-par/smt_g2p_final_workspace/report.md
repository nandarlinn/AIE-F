# Myanmar G2P SMT Experiment Report

## 1. Objective

ဒီ experiment ရဲ့ ရည်ရွယ်ချက်က Myanmar grapheme/syllable sequence (`my`) ကနေ phoneme/romanization sequence (`ph`) ကို Moses phrase-based SMT နည်းလမ်းနဲ့ train, tune, decode လုပ်ပြီး BLEU score တွက်ရန်ဖြစ်ပါတယ်။

Direction:

```text
-f my -e ph
```

Data split:

| Split | my lines | ph lines |
|---|---:|---:|
| train | 20,000 | 20,000 |
| dev | 2,000 | 2,000 |
| test | 2,802 | 2,802 |

## 2. Tools and Paths

အသုံးပြုထားသော toolkit:

| Tool | Path |
|---|---|
| Moses | `/home/phantom/mosesdecoder` |
| Moses decoder | `/home/phantom/mosesdecoder/bin/moses` |
| train-model.perl | `/home/phantom/mosesdecoder/scripts/training/train-model.perl` |
| mert-moses.pl | `/home/phantom/mosesdecoder/scripts/training/mert-moses.pl` |
| multi-bleu.perl | `/home/phantom/mosesdecoder/scripts/generic/multi-bleu.perl` |
| GIZA++ | `/home/phantom/giza-pp/GIZA++-v2/GIZA++` |
| mkcls | `/home/phantom/giza-pp/mkcls-v2/mkcls` |

## 3. Baseline PBSMT Training

ပထမဆုံး `assignment6_g2p_smt.ipynb` ထဲမှာ baseline PBSMT model ကို train လုပ်ခဲ့ပါတယ်။

လုပ်ဆောင်ခဲ့သော အဆင့်များ:

1. Source/target language သတ်မှတ်ခြင်း

```python
SRC = "my"
TGT = "ph"
```

2. Target side language model ဆောက်ခြင်း

```bash
lmplz -o 5 --text train.ph --arpa train.ph.arpa
build_binary train.ph.arpa train.ph.blm
```

3. Phrase-based SMT model train ခြင်း

```bash
train-model.perl \
  -root-dir work/my-ph \
  -corpus train \
  -f my -e ph \
  -alignment grow-diag-final-and \
  -reordering msd-bidirectional-fe \
  -lm 0:5:train.ph.blm:8
```

4. MERT tuning လုပ်ခြင်း

```bash
mert-moses.pl \
  dev.my dev.ph \
  moses \
  work/my-ph/model/moses.ini
```

5. Test set decode လုပ်ခြင်း

```bash
moses -f work/my-ph/mert/moses.ini < test.my > test.decoded.ph
```

6. BLEU score တွက်ခြင်း

```bash
multi-bleu.perl test.ph < test.decoded.ph
```

Baseline result:

```text
BLEU = 69.59, 85.2/72.6/64.5/58.8 (BP=1.000, ratio=1.000, hyp_len=8049, ref_len=8048)
```

## 4. BLEU Improvement Experiments

BLEU တက်နိုင်မလား စမ်းရန် `assignment6_g2p_smt_bleu_improve.ipynb` ထဲမှာ workspace အသစ်တစ်ခုသုံးပြီး experiments run ခဲ့ပါတယ်။

Improvement workspace:

```text
data/g2p-par/bleu_improve_workspace
```

အဓိကလုပ်ခဲ့တာတွေ:

1. Raw data ကို workspace အသစ်ထဲ copy လုပ်ခြင်း
2. Unicode/spacing normalization လုပ်ခြင်း
3. OOV tokens စစ်ခြင်း
4. LM order ပြောင်းစမ်းခြင်း
5. Phrase length ပြောင်းစမ်းခြင်း
6. Alignment method ပြောင်းစမ်းခြင်း
7. MERT random restarts တိုးခြင်း

OOV diagnostic:

```text
OOV count = 51
```

OOV များပါက decoded output ထဲမှာ unknown words ဖြစ်နိုင်ပြီး BLEU ကျစေနိုင်ပါတယ်။

## 5. Experiment Settings

### Experiment 1: 5-gram baseline rerun with stronger MERT

ပြင်ခဲ့သည့်နေရာ:

```text
lm_order = 5
max_phrase_length = 7
alignment = grow-diag-final-and
random_restarts = 30
maximum_iterations = 35
```

ဘာကြောင့်ပြင်လဲ:

MERT random restarts ကိုတိုးပြီး tuning weight ပိုကောင်းလာနိုင်မလား စမ်းရန်ဖြစ်ပါတယ်။

Result:

```text
BLEU = 69.59
```

### Experiment 2: Higher-order LM

မူလစမ်းချင်တာက 7-gram LM ဖြစ်ပါတယ်။ သို့သော် local KenLM build က max order 6 အထိပဲ support လုပ်သောကြောင့် 7-gram binary build မှာ error တက်ခဲ့ပါတယ်။

7-gram error:

```text
This model has order 7 but KenLM was compiled to support up to 6.
```

ထို့ကြောင့် support လုပ်နိုင်သော အမြင့်ဆုံး LM order ဖြစ်တဲ့ 6-gram ကို fallback အဖြစ် run ခဲ့ပါတယ်။

ပြင်ခဲ့သည့်နေရာ:

```text
lm_order = 6
max_phrase_length = 7
alignment = grow-diag-final-and
```

ဘာကြောင့်ပြင်လဲ:

G2P output ဖြစ်သော `ph` sequence တွေမှာ context အရှည်ပိုသုံးနိုင်ရင် phoneme sequence prediction ပိုကောင်းလာနိုင်သောကြောင့်ဖြစ်ပါတယ်။

Result:

```text
BLEU = 69.85, 85.4/73.1/64.9/58.8 (BP=1.000, ratio=1.000, hyp_len=8050, ref_len=8048)
```

### Experiment 3: Phrase Length 5

ပြင်ခဲ့သည့်နေရာ:

```text
max_phrase_length = 5
```

ဘာကြောင့်ပြင်လဲ:

G2P task မှာ grapheme/phoneme chunks တွေက တိုလေ့ရှိသောကြောင့် phrase length ကို 7 ကနေ 5 သို့လျှော့ပြီး phrase table ပိုတိကျလာနိုင်မလား စမ်းခဲ့ပါတယ်။

Result:

```text
BLEU = 69.59
```

### Experiment 4: Intersect Alignment

ပြင်ခဲ့သည့်နေရာ:

```text
alignment = intersect
```

ဘာကြောင့်ပြင်လဲ:

`grow-diag-final-and` က coverage များသော်လည်း noisy phrase pairs ပါနိုင်သောကြောင့် stricter alignment ဖြစ်တဲ့ `intersect` ကိုစမ်းခဲ့ပါတယ်။

Result:

```text
BLEU = 69.53
```

## 6. Results Summary

| Experiment | LM order | Max phrase length | Alignment | BLEU |
|---|---:|---:|---|---:|
| Baseline | 5 | 7 | grow-diag-final-and | 69.59 |
| norm_5g_p7_rr30 | 5 | 7 | grow-diag-final-and | 69.59 |
| norm_6g_p7_rr30 | 6 | 7 | grow-diag-final-and | 69.85 |
| norm_5g_p5_rr30 | 5 | 5 | grow-diag-final-and | 69.59 |
| norm_5g_p7_intersect_rr30 | 5 | 7 | intersect | 69.53 |

Best result:

```text
norm_6g_p7_rr30
BLEU = 69.85
```

Improvement:

```text
69.85 - 69.59 = +0.26 BLEU
```

## 7. Interpretation

BLEU တက်ခဲ့သော setting က 6-gram target language model ဖြစ်ပါတယ်။ ဆိုလိုတာက `ph` output sequence ရဲ့ context ကို 5-token အစား 6-token အထိကြည့်နိုင်တာက decoder အတွက် အနည်းငယ်အကျိုးရှိခဲ့ပါတယ်။

Phrase length 5 သည် baseline နှင့်တူသော score ရခဲ့ပြီး improvement မရှိပါ။ Intersect alignment သည် stricter ဖြစ်သော်လည်း BLEU အနည်းငယ်ကျသွားသောကြောင့် ဤ dataset အတွက် `grow-diag-final-and` ကပိုသင့်တော်ပါတယ်။

## 8. Final Clean Workspace

Final report နှင့် အရေးကြီးသော outputs များကို အောက်ပါ folder ထဲမှာ စုစည်းထားပါတယ်။

```text
data/g2p-par/smt_g2p_final_workspace
```

Folder structure:

```text
smt_g2p_final_workspace/
  report.md
  notebooks/
    assignment6_g2p_smt.ipynb
    assignment6_g2p_smt_bleu_improve.ipynb
  results/
    baseline/
      test.decoded.ph
      test.multi-bleu.my-ph.txt
    best_6gram/
      test.decoded.ph
      test.multi-bleu.txt
      moses.tuned.ini
      moses.untuned.ini
    experiments/
      summary.tsv
      oov_tokens.txt
```

## 9. Notes

`multi-bleu.perl` သည် internal comparison အတွက်အသုံးပြုနိုင်သော်လည်း publication-level standardized BLEU အတွက်တော့ tokenizer/detokenizer consistency ကိုသေချာစစ်သင့်ပါတယ်။

နောက်ထပ် BLEU တက်စေရန် အထိရောက်ဆုံးလုပ်နိုင်သောအချက်များ:

1. OOV 51 tokens ကို train data ထဲဖြည့်ခြင်း
2. my/ph side spelling and segmentation consistency စစ်ခြင်း
3. 6-gram LM ကိုထားပြီး data cleaning ထပ်လုပ်ခြင်း
4. Additional parallel G2P pairs ထည့်ခြင်း

