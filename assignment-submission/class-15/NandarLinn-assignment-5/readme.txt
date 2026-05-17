python preprocess_corpus.py

python reformat_test_sets.py

# 3 gram
python train_evaluate_lm.py \
  --order 3 \
  --lmplz /home/elio/Downloads/assignment5/kenlm/build/bin/lmplz \
  --build-binary /home/elio/Downloads/assignment5/kenlm/build/bin/build_binary

# 5 gram
python train_evaluate_lm.py \
  --order 5 \
  --lmplz /home/elio/Downloads/assignment5/kenlm/build/bin/lmplz \
  --build-binary /home/elio/Downloads/assignment5/kenlm/build/bin/build_binary

# Realtime interactive test
python interactive_test.py
