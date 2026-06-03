#!/usr/bin/env bash
cd /media/kyalkalay/Data/AI-Class/assignment-6_aung_hein/pbsmt_original_clean_experiments/pbsmt_5gram_less_pruned/ph-my/tuning/tmp.1
/home/kyalkalay/tools/mosesdecoder/bin/extractor --sctype BLEU --scconfig case:true  --scfile run9.scores.dat --ffile run9.features.dat -r /media/kyalkalay/Data/AI-Class/assignment-6_aung_hein/g2p-par/dev_clean.my -n run9.best100.out.gz
