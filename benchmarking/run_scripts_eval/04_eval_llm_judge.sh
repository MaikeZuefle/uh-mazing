#!/bin/bash

INPUT=results/llm_judge_scores_without_references.csv
OUTPUT_DIR=results/llm_judge

cd "$(dirname "$0")/.." || exit 1

python eval/eval_llm_judge.py \
    --input "$INPUT" \
    --output-dir "$OUTPUT_DIR"
