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

6.Compute PPL of base LM on each domain (ကိုယ်ရွေးထားတဲ့ မတူတဲ့ ဒိုမိန်း သုံးမျိုး tes data နဲ့ PPL တိုင်းတာကြည့်ပါ)
Visualize with bar chart: Which domains are "hardest"?
Brainstorming on Domain Adaptation Strategies (ဥပမာ Domain-specific vocabulary expansion)

PPL က နည်းရင် ရလဒ်က ပိုကောင်းတာမို့ ကိုယ်ရွေးထားတဲ့ မတူတဲ့ ဒိုမိန်း သုံးခု test data နဲ့ စမ်းထားတဲ့ PPL ကို လျော့အောင် ကြိုးစားကြည့်ပါ။
ပြီးတဲ့ သူက သုံးထားတဲ့ ဒေတာနဲ့ python code ဒါမှမဟုတ်ရင်လည်း Jupyter Notebook နဲ့ report တင်ပါ။ zip လုပ်ပြီးတင်ရင် ပိုကောင်းလိမ့်မယ်။ ဆရာ AIEF GitHub Repository မှာ Assignment-5 အတွက် ဖိုလ်ဒါဆောက်ပေးထားလိုက်မယ်။ ဒါမှမဟုတ်ရင်လည်း GoogleDrive နဲ့ ဆရာ့ကို ရှဲပေးကြပါ။ ဒီ exercise က လွယ်ပါတယ်။ ဖြစ်နိုင်ရင် တပတ်အတွင်းမှာ ပြီးအောင် လုပ်ကြပါ။

