#!/usr/bin/env bash
cd /home/phantom/Desktop/Git/AI_Assignment/assignment-6/class-13and14/data/g2p-par/bleu_improve_workspace/experiments/norm_5g_p7_intersect_rr30/mert
/home/phantom/mosesdecoder/bin/extractor --sctype BLEU --scconfig case:true  --scfile run4.scores.dat --ffile run4.features.dat -r /home/phantom/Desktop/Git/AI_Assignment/assignment-6/class-13and14/data/g2p-par/bleu_improve_workspace/normalized/dev.ph -n run4.best100.out.gz
