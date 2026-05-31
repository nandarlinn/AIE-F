#!/usr/bin/env bash
cd /mnt/data/personal_projects/NandarLinn-SMT-Assignment/pbsmt/baseline/my-ph/tuning/tmp.1
/home/elio/tool/mosesbin/ubuntu-17.04/moses/bin/extractor --sctype BLEU --scconfig case:true  --scfile run4.scores.dat --ffile run4.features.dat -r /mnt/data/personal_projects/NandarLinn-SMT-Assignment/clean-data/dev.ph -n run4.best100.out.gz
