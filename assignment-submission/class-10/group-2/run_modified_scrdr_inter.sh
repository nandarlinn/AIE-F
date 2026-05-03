#!/bin/bash

set -euo pipefail

# Building RDR Rules
python modified_scrdr_interactive.py \
  --input ./data/cleaned/movies_small_v1_recommend.csv \
  --target recommend \
  --title_col title \
  --in_set Recommend \
  --exclude budget popularity revenue vote_count rating_class \
  --genre_col genres \
  --tree ./rules/modified_inter_movies_rules.json \
  --mode build | tee ./logs/modified_rdr_inter_movies.log

# Testing with RDR Model
python modified_scrdr_interactive.py \
  --input ./data/cleaned/movies_small_v1_recommend.csv \
  --target recommend \
  --title_col title \
  --in_set Recommend \
  --exclude budget popularity revenue vote_count rating_class \
  --genre_col genres \
  --tree ./rules/modified_inter_movies_rules.json \
  --mode test
