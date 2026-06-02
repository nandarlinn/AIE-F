#!/usr/bin/env bash
cd /media/kyalkalay/Data/AI-Class/assignment-6_aung_hein/pbsmt-big-normalize/baseline/ph-my/tuning/tmp.1
/home/kyalkalay/tools/mosesdecoder/bin/extractor --sctype BLEU --scconfig case:true  --scfile run5.scores.dat --ffile run5.features.dat -r /media/kyalkalay/Data/AI-Class/assignment-6_aung_hein/g2p-par/dev_clean.my -n run5.best100.out.gz
