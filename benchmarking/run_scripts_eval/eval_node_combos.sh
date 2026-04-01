#!/bin/bash

INPUT=generated_output
REFERENCES=translation_data/uh-mazing-with-node-combos.csv
OUTPUT=results/node_combos/analysis_node_combos.csv


cd "$(dirname "$0")/.." || exit 1

python eval/eval_node_combos.py \
    --input "$INPUT" \
    --references "$REFERENCES" \
    --output "$OUTPUT"

OUTPUT_TABLE="${OUTPUT%.csv}_table.csv"

python eval/transform_scores_for_table.py \
    --input "$OUTPUT" \
    --output "$OUTPUT_TABLE" \
    --metric comet
