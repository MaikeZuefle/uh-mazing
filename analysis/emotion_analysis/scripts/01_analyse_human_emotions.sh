#!/bin/bash
# Usage: bash scripts/01_analyse_human_emotions.sh

cd "$(dirname "$0")/.." || exit 1

python src/analyse_human_emotions.py
