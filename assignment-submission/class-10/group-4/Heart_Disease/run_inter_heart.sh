#!/bin/bash

## Building RDR Rules
python3 scrdr_interactive.py --input ./data/heart.csv --target target \
--tree heart_demo.json --mode build | tee running_inter_heart_demo.log

## Testing with RDR Model
python3 scrdr_interactive.py --input ./data/heart.csv --target target \
--tree heart_demo.json --mode test
