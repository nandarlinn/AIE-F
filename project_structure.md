# Assignment-5-ACN — Project Structure

This document lists the top-level files and folders in this project and short descriptions so contributors can quickly understand where things live.

**Top level**
- README_Assignment5-ACN.md: Project README and high-level instructions.
- requirements.txt: Python dependencies for training/evaluation/plotting.

**data/**
- data/mypos_v3.word.clean: (example) cleaned dataset used for baseline experiments.
- data/raw_samples/: raw input samples split by domain (news, legal, medical).
- data/prepared/: processed training/test files used by scripts (e.g., train_news_adapt.txt).

**script/**
- 01_train_lstm.py: Training script for LSTM language models.
- 02_evaluate_lstm.py: Evaluation script for trained models.
- 03_domain_adaptation.py: Domain adaptation workflow/driver.
- 04_plot_domain_ppl.py: Plots domain perplexity / comparison charts.
- 05_plot_results_auto.py: Automated plotting utilities.

**results/**
- results/: model outputs and evaluation artifacts organized by experiment name.
  - lstm_news/, lstm_news_adapted/, lstm_medical/, lstm_medical_adapted/, lstm_legal/, lstm_legal_adapted/, lstm_mypos_baseline/
  - Typical files: best_model.pt, vocab.json, history.json, eval_results.json

