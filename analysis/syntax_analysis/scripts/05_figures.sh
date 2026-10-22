#!/bin/bash
# Usage: bash scripts/05_figures.sh

cd "$(dirname "$0")/.." || exit 1

python src/plot_comparison.py
