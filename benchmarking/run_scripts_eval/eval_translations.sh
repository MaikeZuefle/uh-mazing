#!/bin/bash

INPUT=generated_output # folder with model subfolders
REFERENCES=../translation_data/references.csv
OUTPUT=results/translations/evaluation_results.csv

cd "$(dirname "$0")/.." || exit 1

python eval/eval_translations.py \
    --input "$INPUT" \
    --references "$REFERENCES" \
    --output "$OUTPUT"

OUTPUT_COMET_TABLE="${OUTPUT%.csv}_comet_table.csv"
OUTPUT_CHRF_TABLE="${OUTPUT%.csv}_chrf_table.csv"

python eval/transform_scores_for_table.py \
    --input "$OUTPUT" \
    --output "$OUTPUT_COMET_TABLE" \
    --metric comet

python eval/transform_scores_for_table.py \
    --input "$OUTPUT" \
    --output "$OUTPUT_CHRF_TABLE" \
    --metric chrf
