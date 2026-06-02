#!/usr/bin/env bash
cd /media/kyalkalay/Data/AI-Class/assignment-6_aung_hein/pbsmt_original_clean_experiments/pbsmt_3gram_phrase7/ph-my/tuning/tmp.1
/home/kyalkalay/tools/mosesdecoder/bin/extractor --sctype BLEU --scconfig case:true  --scfile run12.scores.dat --ffile run12.features.dat -r /media/kyalkalay/Data/AI-Class/assignment-6_aung_hein/g2p-par/dev_clean.my -n run12.best100.out.gz
