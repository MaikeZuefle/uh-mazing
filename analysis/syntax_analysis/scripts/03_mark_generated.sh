#!/bin/bash
# Usage: bash scripts/03_mark_generated.sh [--limit N] [--workers N]

cd "$(dirname "$0")/.." || exit 1

python src/mark_disfluencies.py --mode references "$@"
python src/evaluate_marking.py
python src/mark_disfluencies.py --mode systems "$@"
