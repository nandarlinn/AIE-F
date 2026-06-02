#!/usr/bin/env bash
cd /media/kyalkalay/Data/AI-Class/assignment-6_aung_hein/pbsmt_original_clean_experiments/pbsmt_3gram_phrase5/ph-my/tuning/tmp.1
/home/kyalkalay/tools/mosesdecoder/bin/extractor --sctype BLEU --scconfig case:true  --scfile run4.scores.dat --ffile run4.features.dat -r /media/kyalkalay/Data/AI-Class/assignment-6_aung_hein/g2p-par/dev_clean.my -n run4.best100.out.gz
