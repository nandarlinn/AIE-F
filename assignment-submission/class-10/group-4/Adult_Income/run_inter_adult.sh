#!/bin/bash

## Building RDR Rules
python scrdr_interactive.py --input ./data/adult.csv --target income \
--tree adult_demo.json --mode build | tee running_inter_adult_demo.log

## Testing with RDR Model
python scrdr_interactive.py --input ./data/adult.csv --target income \
--tree adult_demo.json --mode test