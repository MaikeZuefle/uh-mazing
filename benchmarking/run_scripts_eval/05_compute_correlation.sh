#!/bin/bash

TRANSLATIONS_PER_ITEM_CSV=results/translations/evaluation_results_per_item.csv
HUMAN_JSON=../humeval_translation_quality/annotations_human_scoring.json

cd "$(dirname "$0")/.." || exit 1

# The script itself lives in humeval_translation_quality/scripts/ (not under benchmarking/)
# since that's where the human-annotation JSON lives; paths above are relative to benchmarking/
# and get re-resolved relative to that script's directory below.
python ../humeval_translation_quality/scripts/05-compute_correlation.py \
    --llm-judge-csv results/llm_judge/llm_judge_per_item.csv \
    --llm-judge-label llm_judge \
    --llm-judge-ref-csv results/llm_judge_with_references/llm_judge_per_item.csv \
    --llm-judge-ref-label llm_judge_ref \
    --translations-per-item-csv "$TRANSLATIONS_PER_ITEM_CSV" \
    --human-json "$HUMAN_JSON" \
    --output results/correlation/correlation_results.csv
