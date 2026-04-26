#!/bin/bash

time python ./scrdr_learner.py --input ./data/cleaned/movies_small_v1.csv --target rating_class \
--exclude title --plot ./img/rdr_learner_movies_cf.png --output ./rules/learner_movies_rules.json | tee ./logs/rdr_learner_movies.log