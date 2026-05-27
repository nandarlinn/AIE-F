# Group 2 Assignment Submission: Movie Classification and Recommendation with RDR

## Overview

In this project, we explore movie classification and recommendation using three different Ripple-Down Rules (RDR) systems:

1. **Standard SCRDR for Multi-class Rating**: The classic `scrdr_interactive.py` builds an RDR for predicting the `rating_class` (e.g., Poor/Average/Good/Excellent) of a movie based on numeric features. The system interactively prompts the user to add rules when predictions are incorrect, gradually refining its classification tree. The goal is enhanced accuracy in multi-class classification of movie quality.

2. **SCRDR Learner Automation**: `scrdr_learner.py` implements an automated RDR construction, using all input data and greedily picking best splits based on error reduction (rather than stepwise user prompts). This approach is designed to scale rule induction to larger datasets and to benchmark the hand-built interactive RDRs on the same problem.

3. **Movie Recommendation-oriented SCRDR**: `modified_scrdr_interactive.py` is tailored to a binary/coarse-label recommendation task (e.g., Recommend/Not). Unlike standard multi-class RDRs, it uses genre subset matching rules (`has` operator) and focuses on producing a list of titles recommended by learned rules. The system outputs:
    - (a) recommendation/classification accuracy, and
    - (b) an explicit list of movies whose final predicted status matches the "in_set" of recommendation labels.

## Dataset

The [original dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) is cleaned to prepare it as input for RDR.

## Project Structure

```
group-2/
├── ...
├── data/
│   ├── raw/                        # original data files
│   └── cleaned/                    # cleaned/labeled data file
├── img/
├── logs/
├── notebooks/
├── rules/                          # rules from SCRDR
├── run_modified_scrdr_inter.sh
├── run_scrdr_inter.sh              # originally Sayar's # modified for the project
├── run_scrdr_learner.sh            # originally Sayar's # modified for the project
├── modified_scrdr_interactive.py   # modified `scrdr_interactive` for the project
├── scrdr_interactive.py            # originally Sayar's
└── scrdr_learner.py                # originally Sayar's
```
