#!/bin/bash
# Usage: bash scripts/02_analyse_llm_emotions.sh

cd "$(dirname "$0")/.." || exit 1

python src/analyse_llm_emotions.py
