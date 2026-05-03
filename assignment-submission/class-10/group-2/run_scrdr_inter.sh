#!/bin/bash

# Building RDR Rules
python scrdr_interactive.py \
  --input ./data/cleaned/movies_small_v1.csv \
  --target rating_class \
  --exclude title genres original_language vote_average \
  --tree ./rules/inter_movies_rules.json \
  --mode build \
  | tee ./logs/rdr_inter_movies.log

# Testing with RDR Model
python scrdr_interactive.py \
  --input ./data/cleaned/movies_small_v1.csv \
  --target rating_class \
  --exclude title genres original_language vote_average \
  --tree ./rules/inter_movies_rules.json \
  --mode test