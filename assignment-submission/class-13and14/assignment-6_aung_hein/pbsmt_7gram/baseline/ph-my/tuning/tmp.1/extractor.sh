#!/usr/bin/env bash
cd /media/kyalkalay/Data/AI-Class/assignment-6_aung_hein/pbsmt_7gram/baseline/ph-my/tuning/tmp.1
/home/kyalkalay/tools/mosesdecoder/bin/extractor --sctype BLEU --scconfig case:true  --scfile run6.scores.dat --ffile run6.features.dat -r /media/kyalkalay/Data/AI-Class/assignment-6_aung_hein/g2p-par/dev_clean.my -n run6.best100.out.gz
