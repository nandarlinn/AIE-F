#!/bin/bash

## Building RDR Rules
python3 scrdr_interactive.py --input ./data/adult_small.csv --target income \
--tree adult_demo.json --mode build | tee running_inter_adult_demo.log

## Testing with the FULL dataset
python3 scrdr_interactive.py --input ./data/adult.csv --target income \
--tree adult_demo.json --mode test