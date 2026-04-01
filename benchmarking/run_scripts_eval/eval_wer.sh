#!/bin/bash

cd "$(dirname "$0")/.." || exit 1

INPUT=generated_output/transcript
REFERENCES=translation_data/references.csv
OUTPUT=results/wer/wer_results.csv

python eval/eval_wer.py \
    --input "$INPUT" \
    --references "$REFERENCES" \
    --output "$OUTPUT"
