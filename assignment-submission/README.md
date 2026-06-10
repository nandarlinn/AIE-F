# Assignments

## Assignment No.1  
### (@Class-1)
Build a small emotional dataset in the Myanmar language, design several regular expression rules for Myanmar words, and experiment with the provided demo code (*Hybrid-ELIZA.py*).  

The main purpose of this assignment is to understand traditional rule-based chatbots and how they can be combined with deep learning approaches such as bi-LSTM. This is a group project aimed at developing experience in group discussion and brainstorming.  

## Assignment No.2  
### (@Class-4)  

Watch the lecture:  
"Four Ways of Thinking: Statistical, Interactive, Chaotic and Complex" by David Sumpter:  
[https://www.youtube.com/watch?v=PPCfDe8TfJQ](https://www.youtube.com/watch?v=PPCfDe8TfJQ)   

Then, write a one-page report on your understanding of the lecture.  
The main purpose of this assignment is to introduce high-level concepts of applying mathematics in different ways.  

## Assignment No.3
### (@Class-6)  

Python code will be provided for collecting Myanmar handwriting syllable data, including stroke information. Based on the collected data, develop a Myanmar syllable handwriting recognition system.  

The total number of classes (syllables) exceeds 4,300, making this a challenging task.  

The main purpose of this individual assignment is to provide hands-on experience with the full pipeline: data collection, model building, experimentation, and evaluation and to understand the effort involved in each stage.  

## Assignment No.4
### (@Class-10)  

For Assignment 4, I would like you to brainstorm with your group members to identify a real-world problem near you, or another interesting problem that can be solved using Ripple-Down Rules (RDR). Then, prepare simulation data (e.g., in CSV or Excel format) and build an RDR tree or set of rules using the demo Python code [scrdr_interactive.py](https://github.com/ye-kyaw-thu/SCRDR_tutorial/blob/main/inter/scrdr_interactive.py).  

Each group should submit the dataset, the RDR model, and a report covering two different scenarios or real-world problems.  

## Assignment No.5
### (@Class-15)

**Domain Adaptation with Language Model**  

**Theme:** LM Evaluation + Domain Analysis
**Objective:** Investigate how well a general-domain Myanmar LM performs on specific domains, and explore adaptation strategies.

### Tasks

1. Online မှာရှိတဲ့ မြန်မာစာ corpus တွေကို ကိုယ့်စက်ထဲကို download လုပ်ယူပါ။ များနိုင်သမျှ များများစုပါ။ ဒိုမိန်းလည်း မျိုးစုံ ပါရင် ပိုကောင်းတယ်။ ရည်ရွယ်တာက general domain ဖြစ်စေချင်တာ။ ဥပမာ ALT Treebank Corpus ထဲက မြန်မာစာဒေတာ၊ myPOS မြန်မာစာဒေတာ၊ တခြား GitHub/HuggingFace က မြန်မာစာဒေတာ
2. Data cleaning (ပုဒ်ထီး၊ ပုဒ်မတွေ၊ တခြား မလိုတဲ့ သင်္ကေတတွေကို ရှင်းတာမျိုးတော့ အနည်းဆုံး လုပ်ပါ)
3. Tokenization (e.g. syllable segmentation with sylbreak tool or Word tokenization with oppaWord)
4. Train Base LM (e.g. KenLM or LSTM)
5. Prepare some example domains to explore (collect 300 tokens each):
- News articles
- Social media (Facebook posts)
- Legal documents
- Medical/Health text
- Literary/Novel text
- Religious text (Buddhist scriptures)
  အနည်းဆုံး မတူတဲ့ ဒိုမိန်း ၃ ခုအတွက် testset အသစ် သုံးခုကို ပြင်ဆင်ပါ။ ဥပမာ ဂါထာတွေ၊ ဥပဒေနဲ့ ပတ်သက်တဲ့ စာကြောင်းတွေ၊ Facebook က comment တွေ။ အရေးကြီးတာက fair compare လုပ်လို့ ရအောင် token အရေအတွက်ကို တူအောင်ထားပါ။ ပြီးတော့ token type ကလည်း base LM နဲ့ တူရပါမယ်။ တကယ်လို့ ကိုယ့် base LM က syllable ဖြတ်ထားပြီး မော်ဒယ်ဆောက်ခဲ့ရင် test လုပ်မယ့် ဒိုမိန်းမတူတဲ့ ဒေတာတွေကိုလည်း syllable ဖြတ်ထားရပါမယ်။ ပြီးတော့ testset တစ်ခုစီကို syllable token 200 စီ ညီအောင် ညှိပါ။ စာကြောင်းရေလည်း တူအောင်ထားပါ။ ဥပမာ တစ်ကြောင်းမှာ syllable ၂၀ လုံးစီပါတဲ့ စာကြောင်းမျိုး။

6. Compute PPL of base LM on each domain (ကိုယ်ရွေးထားတဲ့ မတူတဲ့ ဒိုမိန်း သုံးမျိုး tes data နဲ့ PPL တိုင်းတာကြည့်ပါ)
Visualize with bar chart: Which domains are "hardest"?
Brainstorming on Domain Adaptation Strategies (ဥပမာ Domain-specific vocabulary expansion)

PPL က နည်းရင် ရလဒ်က ပိုကောင်းတာမို့ ကိုယ်ရွေးထားတဲ့ မတူတဲ့ ဒိုမိန်း သုံးခု test data နဲ့ စမ်းထားတဲ့ PPL ကို လျော့အောင် ကြိုးစားကြည့်ပါ။
ပြီးတဲ့ သူက သုံးထားတဲ့ ဒေတာနဲ့ python code ဒါမှမဟုတ်ရင်လည်း Jupyter Notebook နဲ့ report တင်ပါ။ zip လုပ်ပြီးတင်ရင် ပိုကောင်းလိမ့်မယ်။ ဆရာ AIEF GitHub Repository မှာ class-15 ဆိုပြီး ဖိုလ်ဒါဆောက်ပေးထားလိုက်မယ်။ ဒါမှမဟုတ်ရင်လည်း GoogleDrive နဲ့ ဆရာ့ကို ရှဲပေးကြပါ။ ဒီ exercise က လွယ်ပါတယ်။ ဖြစ်နိုင်ရင် တပတ်အတွင်းမှာ ပြီးအောင် လုပ်ကြပါ။

## Assignment No.6
### (@Class-13 and 14)

Automatic Speech Recognition (ASR), Text to Speech (TTS) အလုပ်တွေအတွက် အရေးကြီးတဲ့ grapheme2phoneme ပြောင်းတဲ့ အပိုင်းကို အတန်းထဲမှာ သင်ထားတဲ့ Statistical Machine Translation ကို သုံးပြီး ရလဒ်ကို ကောင်းအောင် ကြိုးစားကြည့်ပါ။  

### Data Information

Data က myG2P corpus ကနေ ယူထားတာပါ။  
Link က [https://github.com/ye-kyaw-thu/myG2P](https://github.com/ye-kyaw-thu/myG2P)  

Original corpus format က အောက်ပါအတိုင်း ရှိတယ်။ အဲဒီကနေ field 3 နဲ့ 4 ကို ဆွဲထုတ်ယူလိုက်ပြီး parallel corpus ဆောက်ပေးထားတယ်။  

```
$ head ./myg2p.ver2.0.txt
1       ...ဖြစ်စေ...ဖြစ်စေ      ... ဖြစ် စေ ... ဖြစ် စေ ... hpji' sei ... hpji' sei     ... pʰjɪʔ sè ... pʰjɪʔ sè
2       ...ရိုး...စဉ်   ... ရိုး ... စဉ်        ... jou: ... sin        ... jó ... sɪ̀ɴ
3       ...ရိုး...စဉ်   ... ရိုး ... စဉ်        ... jou: ... zin        ... jó ... zɪ̀ɴ
4       ...လို...ငြား   ... လို ... ငြား        ... lou ... nja:        ... lò ... ɲá
5       ကကတစ်   က က တစ် ka. ga- di'     ka̰ gə dɪʔ
6       ကကတိုး  က က တိုး        ka. ga- dou:    ka̰ gə dó
7       ကကုသန်  က ကု သန်        ka. ku. than    ka̰ kṵ θàɴ
8       ကကုသန်  က ကု သန်        kau' ka- than   kaʊʔ kə θàɴ
9       ကကူရံ   က ကူ ရံ ka. ku jan      ka̰ kù jàɴ
10      ကကြိုး  က ကြိုး ka. gyou:       ka̰ dʑó
```

### Source  

Machine Translation လုပ်ဖို့အတွက် Source နဲ့ target ဒေတာတွေက အတွဲလိုက် shuffle လုပ်ထားပြီးသားပါ။   

```
$ wc *.my
  2000   5687  59222 dev.my
  2802   8047  83959 test.my
 20000  57336 594183 train.my
 24802  71070 737364 total
```

```
$ head -n 5 *.my
==> dev.my <==
ရပ် မှု ရွာ မှု
ပြီး ဆေး
မြင်း စား ဂျုံ
အ ငံ့ ပေး
အ နေ အ ထား

==> test.my <==
တက် တက် ပြောင်
ကပ် ပိ
ရှုံ့ မဲ့
ညှဉ်း ပန်း
မွမ်း မံ

==> train.my <==
ပြာ သာဒ် ဆောင်
ကိုယ် ပိုင် စာ ကြည့် တိုက်
သိန် ဓော ဆား
စား သောက် ဆိုင်
ကြိုး စင်
```

### Target  

```
ye@lst-hpc3090:/mnt/disk1/ye/data/zg-un/myg2p/uni/src-tgt/g2p-par$ wc *.ph
  2000   5688  25849 dev.ph
  2802   8048  36532 test.ph
 20000  57346 260356 train.ph
 24802  71082 322737 total
ye@lst-hpc3090:/mnt/disk1/ye/data/zg-un/myg2p/uni/src-tgt/g2p-par$
```

```
$ head -n 5 *.ph
==> dev.ph <==
ja' mhu. jwa mhu.
pi: zei:
mjin: za: gyoun
a- ngan pei:
a- nei a- hta:

==> test.ph <==
te' te' pjaun
ka' pi.
shoun. me.
njhin: ban:
mun: man

==> train.ph <==
pja' tha' hsaun
kou bain sa kyi. dai'
thein: do: hsa:
sa: thau' hsain
kyou: zin
```

အခု ဆရာပြင်ပေးထားတဲ့ ဒေတာကို သုံးပြီး Phrase-based SMT ကို ကိုယ့်စက်ထဲမှာ run ကြည့်ပါ။ ထွက်လာတဲ့ baseline ရလဒ်ကိုမှ ပိုကောင်းအောင် စဉ်းစားပြီး BLEU score ကို တင်ကြည့်ပါ။  
အနည်းဆုံးတော့ SMT ရဲ့ baseline ကိုတော့ ကိုယ့်ဖာသာကိုယ် ရအောင်ထုတ်ကြည့်ပါ။  
Assignment တင်တဲ့အခါမှာ perl script တွေ ပြင်တာ၊ test data ကို SGM format ပြောင်းတာ၊ configuration file ကို my-ph, ph-my အတွက် ထုတ်တာ၊ SMT training/tuning/evaluation စတာတွေ လုပ်ထားတဲ့ Jupyter notebook ကို တင်ကြပါ။  

SMT tutorial နဲ့ ဒေတာက အောက်ပါ လင့်ကနေ ရယူနိုင်ပါတယ်။  
[https://github.com/ye-kyaw-thu/AIE-F/tree/main/slide-code/class-13and14](https://github.com/ye-kyaw-thu/AIE-F/tree/main/slide-code/class-13and14)  

Assignment-6 အတွက် moses SMT framework version 2 ကို source ကနေ installation လုပ်ဖို့ ခက်တဲ့သူတွေက binary ဖိုင်ကို ကိုယ့်စက်ထဲကို download လုပ်ပြီး tgz (i.e. zip) ဖိုင်ကို ဖြေလိုက်ပြီး ခေါ်သုံးရင် ရပါပြီ။ ဆရာ လုပ်ပြခဲ့သလို /home/ye/tool/ အောက်ထဲမှာ ဖြေလိုက်သလိုမျိုးပေါ့။  ဆရာ သုံးပြခဲ့တာက ubuntu-17.04.tgz ပါ။  

တကယ်ကတော့ Windows အတွက်ရော Mac OS အတွက်ရော binary ဖိုင်တွေက ရှိပါတယ်။ ဆရာ့အတွေ့အကြုံအရ Linux OS မှာ ပိုလုပ်ရတာ လွယ်ပါလိမ့်မယ်။  
Download link က အောက်ပါအတိုင်းပါ။  

[https://www.statmt.org/moses-release/RELEASE-4.0/binaries/](https://www.statmt.org/moses-release/RELEASE-4.0/binaries/)  

## Assignment No.7  

1. Prepare Marian NMT framework
2. Run Grapheme-to-Phoneme Translation with Sequence-to-Sequence Model
3. Run Grapheme-to-Phoneme Translation with Transformer Model
4. Try to increase baseline results of Seq2Seq and Transformer models

Refer following Jupyter Notebooks:  

**Sequence to Sequence NMT with Marian:** [https://github.com/ye-kyaw-thu/AIE-F/blob/main/slide-code/class-22/NMT-notebooks/Seq2Seq-NMT-marian-ph2gp.ipynb](https://github.com/ye-kyaw-thu/AIE-F/blob/main/slide-code/class-22/NMT-notebooks/Seq2Seq-NMT-marian-ph2gp.ipynb)  

**Transformer NMT with Marian:** [https://github.com/ye-kyaw-thu/AIE-F/blob/main/slide-code/class-22/NMT-notebooks/Transformer-NMT-marian-ph2gp.ipynb](https://github.com/ye-kyaw-thu/AIE-F/blob/main/slide-code/class-22/NMT-notebooks/Transformer-NMT-marian-ph2gp.ipynb)      

လေ့လာရလွယ်ကူအောင် Marian နဲ့ run ပြထားတဲ့ Jupyter notebook နှစ်ခုကို PDF ဖိုင်အနေနဲ့လည်း ပြောင်းပေးထားပါတယ်။   

**Sequence to Sequence NMT with OpenNMT:** [https://github.com/ye-kyaw-thu/AIE-F/blob/main/slide-code/class-22/NMT-notebooks/pdf/Seq2Seq-NMT-marian-ph2gp.pdf](https://github.com/ye-kyaw-thu/AIE-F/blob/main/slide-code/class-22/NMT-notebooks/pdf/Seq2Seq-NMT-marian-ph2gp.pdf)  
**Transformer NMT with Marian:** [https://github.com/ye-kyaw-thu/AIE-F/blob/main/slide-code/class-22/NMT-notebooks/pdf/Transformer-NMT-marian-ph2gp.pdf](https://github.com/ye-kyaw-thu/AIE-F/blob/main/slide-code/class-22/NMT-notebooks/pdf/Transformer-NMT-marian-ph2gp.pdf)  

## Assignment No.8 (MSL Recognition Tutorial)

### Jupyter Notebooks

Jupyter Notebook တွေကို ကြည့်ပြီး လေ့လာပါ။   

1. [MSL-Video-Recognition-Tutorial.ipynb](https://github.com/ye-kyaw-thu/AIE-F/blob/main/slide-code/class-25/msl_recognition/MSL-Video-Recognition-Tutorial.ipynb)
   ဒီ notebook က အတန်းထဲမှာ ဆရာ အစအဆုံး ၂နာရီကျော် အချိန်ယူပြီး run ပြခဲ့တဲ့ notebook ပါ။   
2. [Single-Video-Inference.ipynb](https://github.com/ye-kyaw-thu/AIE-F/blob/main/slide-code/class-25/msl_recognition/Single-Video-Inference.ipynb)
   ဒီ notebook ကတော့ ဆောက်ထားပြီးသား မော်ဒယ် သုံးမျိုးကို သုံးပြီး ဗီဒီယိုဖိုင် တစ်ဖိုင်ချင်းစီရော၊ ကိုယ် test လုပ်ချင်တဲ့ ဗီဒီယိုဖိုင်တွေကို ဖိုလ်ဒါတစ်ခုအောက်မှာ စုသိမ်းထားပြီး batch အလိုက် recognition လုပ်တာကိုကော ဒီမို လုပ်ပြထားတဲ့ notebook ပါ။
3. [Python_Codes_for_MSL_Recognition.ipynb](https://github.com/ye-kyaw-thu/AIE-F/blob/main/slide-code/class-25/msl_recognition/Python_Codes_for_MSL_Recognition.ipynb)
   ဒီ notebook ကတော့ python code အကုန်လုံးနီးပါးကို တစုတစည်းတည်း notebook ပေါ်ကို တင်ပေးထားတာပါ။
4. [Export-to-ONNX-and-Infer-Experiment.ipynb](https://github.com/ye-kyaw-thu/AIE-F/blob/main/slide-code/class-25/msl_recognition/Export-to-ONNX-and-Infer-Experiment.ipynb)
   ဒီ နံပါတ် ၄ notebook ကတော့ Linux ဆာဗာပေါ်မှာ ဆောက်ထားတဲ့ မော်ဒယ်တွေကို Windows OS ပေါ်မှာ ခေါ်သုံးနိုင်အောင်လို့ ONNX export လုပ်ပေးတဲ့ code နဲ့ infer လုပ်တဲ့ code တွေကိုရေးနေတဲ့ အချိန်မှာ debugging log လုပ်ခဲ့တာတွေကို မှတ်တမ်းတင်ထားတာပါ။ infer_onnx.py က ဆရာတို့ debugging လုပ်ဖို့ ကျန်နေပါသေးတယ်။ အရင်ရေးထားတဲ့ code တွေကို Claude ကိုသုံးပြီး ပြန် update/debugg လုပ်တဲ့အခါမှာ ဘယ်လိုမျိုး အမှားတွေ ရှိတယ်ဆိုတဲ့ လက်တွေ့အင်ဂျင်နီယာအပိုင်းကိုလည်း မြင်အောင်လို့ပါ။

### Scripts

scripts ဖိုလ်ဒါအောက်မှာက experiment လုပ်တဲ့အခါမှာ တစ်ခါထက်မက run ကြရတာမို့ အဆင်ပြေအောင် ပြင်ဆင်ထားတဲ့ shell script တွေပါ။ ဒေတာကို ပြင်ဆင်ပြီး၊ နံပါတ် အစီအစဉ်လိုက် အဆင့်ဆင့် run တာကို ဆရာ အတန်းထဲမှာ ပြပြီးသားပါ။  

### Src

src ဖိုလ်ဒါအောက်မှာကတော့ Python code တွေကို သိမ်းထားတာပါ။ လက်ရှိဗားရှင်းတွေကို တင်ပေးထားလိုက်ပါတယ်။ infer_onnx.py က ဆရာ့ Windows OS မှာ run လို့ အဆင်မပြေသေးပါဘူး။ debug လုပ်နေတုန်းပါ။ အဲဒီအပိုင်းကို debug လုပ်နိုင်တဲ့သူတွေကလည်း လုပ်ကြည့်ကြပါ။  

### Config

config ဖိုလ်ဒါအောက်မှာက experiment အတွက် သုံးခဲ့တဲ့ configuration ဖိုင်ကို သိမ်းထားပါတယ်။  

### tools

tools ဖိုလ်ဒါထဲက ဖိုင်တွေက ဆရာ အတန်းထဲမှာ run မပြခဲ့ပါဘူး။ Server အသစ်ပေါ်ကို code တွေရွှေ့လာပြီး အရင်ရေးထားတဲ့ ဗားရှင်းတွေကို ပြန်ပြင်တဲ့အခါမှာ mediapipe library import path ကို စစ်ဆေးဖို့ ရေးခဲ့တာပါ။ lookup_video_label.py ကတော့ data preparation လုပ်တဲ့အခါမှာ လေဘယ်နဲ့ ဗီဒီယိုနဲ့ ကိုက်ရဲ့လား ဆိုတဲ့ အပိုင်းနဲ့ ဆိုင်တဲ့ ကုဒ်ပါ။  

### Requirements

requirements.txt ဖိုင်ကတော့ လိုအပ်တဲ့ python library တွေကို installation လုပ်တဲ့အခါမှာ အသုံးဝင်ပါလိမ့်မယ်။  

### Data

data ဖိုလ်ဒါအောက်မှာ videos ဆိုတဲ့ ဖိုလ်ဒါအသစ်ဆောက်ပြီး MSL4Emergency ဗီဒီယိုဖိုင်တွေကို ဒေါင်းလုဒ်လုပ်ထားပါ။ annotations.txt ဖိုင်ကိုတော့ ဆရာ ပြင်ပေးထားပါတယ်။  

### Assignment No.8 Information

- Video recognition ကို ကိုယ့်စက်ထဲမှာ ဖြစ်ဖြစ်၊ Colab မှာဖြစ်ဖြစ် run ကြည့်ပါ။ 
- အရင်ဆုံး requirements.txt ဖိုင်ကို အခြေခံပြီး လိုအပ်တဲ့ Python environment ကို ဆောက်ပါ။
- ဆရာ တင်ပေးထားတဲ့ Python နဲ့ shell script တွေကို src/ နဲ့ scripts/ ဖိုလ်ဒါအောက်မှာ သိမ်းပါ။
- data/ ဖိုလ်ဒါ ကိုပြင်ဆင်ပြီး [MSL4Emergency](https://github.com/ye-kyaw-thu/MSL4Emergency) repository ကနေ ဗီဒီယိုဖိုင်တွေကို download လုပ်ယူပြီး ပြင်ဆင်ပါ။
- config/ ဖိုလ်ဒါအောက်မှာ config.yaml ဖိုင်ကိုလည်း ပြင်ဆင်ထားဖို့ လိုအပ်ပါလိမ့်မယ်။
- ပြီးရင်တော့ script/ ဖိုလ်ဒါအောက်ထဲက shell script တွေကို နံပါတ်စဉ်အတိုင်း အဆင့်ဆင့် run သွားရင် အဆင်ပြေပါလိမ့်မယ်။
- ဒီ Assignment မှာ အဓိက အပြောင်းအလဲ လုပ်ကြည့်စေချင်တာက annotations.txt ဖိုင်ထဲက ကော်လံနီပါတ် (၂) MSL gloss ကို သုံးပြီး video recognition လုပ်ကြည့်စေချင်တာပါ။ ရလဒ်တွေနဲ့တကွ run ထားတဲ့ Jupyter Notebook ကို ဆရာ့ဆီကို အီးမေးလ်နဲ့ ပို့ပေးတာ ဖြစ်ဖြစ်၊ ဒါမှမဟုတ် ဆရာ ပြင်ဆင်ပေးထားတဲ့ GitHub Repository [assignment-submission/class-25/](https://github.com/ye-kyaw-thu/AIE-F/tree/main/assignment-submission/class-25) အောက်မှာ Pull request လုပ်ထားပေးကြပါ။    
- လေ့လာရတဲ့အခါမှာ လွယ်ကူအောင်လို့ Jupyter notebook တွေကို PDF ဖိုင်အဖြစ်လည်း ပြောင်းပေးထားပါတယ်။  
- ဖြစ်နိုင်ရင် တပတ်အတွင်းမှာ ပြီးအောင် ကြိုးစား လုပ်ကြပါ။  




