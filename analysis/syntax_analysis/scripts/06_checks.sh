#!/bin/bash
# Usage: bash scripts/06_checks.sh

cd "$(dirname "$0")/.." || exit 1

python src/check_context_effects.py
