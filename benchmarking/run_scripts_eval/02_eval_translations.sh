#!/bin/bash

INPUT=generated_output # folder with model subfolders
REFERENCES=../translation_data/references.csv
OUTPUT=results/translations/evaluation_results.csv
PER_ITEM_OUTPUT=results/translations/evaluation_results_per_item.csv  # per-item chrF/COMET, for human-eval correlation
ONLY_CHRF=false  # set to true to skip COMET and only (re)compute ChrF scores

cd "$(dirname "$0")/.." || exit 1

CHRF_FLAG=""
[ "$ONLY_CHRF" = "true" ] && CHRF_FLAG="--only-chrf"

python eval/eval_translations.py \
    --input "$INPUT" \
    --references "$REFERENCES" \
    --output "$OUTPUT" \
    --per-item-output "$PER_ITEM_OUTPUT" \
    $CHRF_FLAG

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
