#!/bin/bash

COMET=results/translations/evaluation_results_comet_table.csv
CHRF=results/translations/evaluation_results_chrf_table.csv
ASR=results/ASR_results/asr_scores.csv
LLM_JUDGE_STYLE=results/llm_judge/llm_judge_style_table.csv
LLM_JUDGE_MEANING=results/llm_judge/llm_judge_meaning_table.csv
OUTPUT_DIR=results/latex_tables

cd "$(dirname "$0")/../.." || exit 1

python src/eval/create_latex_tables.py \
    --comet "$COMET" \
    --chrf "$CHRF" \
    --asr "$ASR" \
    --llm-judge-style "$LLM_JUDGE_STYLE" \
    --llm-judge-meaning "$LLM_JUDGE_MEANING" \
    --output-dir "$OUTPUT_DIR"
