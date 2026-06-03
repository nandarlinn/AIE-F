#!/usr/bin/env bash
cd /mnt/data/personal_projects/NandarLinn-SMT-Assignment/pbsmt-normalize/baseline/ph-my/tuning/tmp.1
/home/elio/tool/mosesbin/ubuntu-17.04/moses/bin/extractor --sctype BLEU --scconfig case:true  --scfile run5.scores.dat --ffile run5.features.dat -r /mnt/data/personal_projects/NandarLinn-SMT-Assignment/clean-data/dev.my -n run5.best100.out.gz
