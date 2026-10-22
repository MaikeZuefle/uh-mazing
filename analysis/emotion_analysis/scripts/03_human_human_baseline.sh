#!/bin/bash
# Usage: bash scripts/03_human_human_baseline.sh

cd "$(dirname "$0")/.." || exit 1

python src/compute_human_human_baseline.py
