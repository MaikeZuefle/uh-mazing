#!/bin/bash

cd "$(dirname "$0")/.." || exit 1

python eval/eval_llm_judge.py \
    --input results/llm_judge_scores_without_references.csv \
    --output-dir results/llm_judge

python eval/eval_llm_judge.py \
    --input results/llm_judge_scores_with_references.csv \
    --output-dir results/llm_judge_with_references
