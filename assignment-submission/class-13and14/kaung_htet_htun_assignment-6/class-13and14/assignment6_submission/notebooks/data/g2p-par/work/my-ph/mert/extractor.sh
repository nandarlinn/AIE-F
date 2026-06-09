#!/usr/bin/env bash
cd /home/phantom/Desktop/Git/AI_Assignment/assignment-6/class-13and14/assignment6_submission/notebooks/data/g2p-par/work/my-ph/mert
/home/phantom/mosesdecoder/bin/extractor --sctype BLEU --scconfig case:true  --scfile run3.scores.dat --ffile run3.features.dat -r /home/phantom/Desktop/Git/AI_Assignment/assignment-6/class-13and14/assignment6_submission/notebooks/data/g2p-par/dev.ph -n run3.best100.out.gz
