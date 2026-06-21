#!/bin/bash

SCATTER_INPUT=results/llm_judge/llm_judge_summary.csv
SCATTER_OUTPUT=results/llm_judge/style_vs_meaning_scatter.pdf

NODE_COMBOS_INPUT=results/node_combos/analysis_node_combos_en-x_structured.csv
NODE_COMBOS_OUTPUT=results/node_combos/node_combos_delta.pdf

cd "$(dirname "$0")/.." || exit 1

python eval/plot_llm_judge_scatter.py \
    --input "$SCATTER_INPUT" \
    --output "$SCATTER_OUTPUT"

python eval/plot_node_combos.py \
    --input "$NODE_COMBOS_INPUT" \
    --output "$NODE_COMBOS_OUTPUT"
