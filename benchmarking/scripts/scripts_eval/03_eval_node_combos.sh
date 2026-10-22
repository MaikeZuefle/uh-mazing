#!/bin/bash

INPUT=generated_output/analysis_node_combos
REFERENCES=../translation_data/uh-mazing-with-node-combos.csv
OUTPUT=results/node_combos/analysis_node_combos.csv


cd "$(dirname "$0")/../.." || exit 1

mkdir -p "$(dirname "$OUTPUT")"

python src/eval/eval_node_combos.py \
    --input "$INPUT" \
    --references "$REFERENCES" \
    --output "$OUTPUT"


