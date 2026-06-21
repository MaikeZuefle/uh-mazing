#!/bin/bash

COMET=results/translations/evaluation_results_comet_table.csv
CHRF=results/translations/evaluation_results_chrf_table.csv
ASR=results/ASR_results/asr_scores.csv
OUTPUT_DIR=results/latex_tables

cd "$(dirname "$0")/.." || exit 1

python eval/create_latex_tables.py \
    --comet "$COMET" \
    --chrf "$CHRF" \
    --asr "$ASR" \
    --output-dir "$OUTPUT_DIR"
