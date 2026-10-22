#!/bin/bash
# Usage: bash scripts/02_run_analysis.sh
# Run after 01_parse_dependency.sh has produced dependency_results.csv and
# fluent_parse_results.csv in data/reference_parses. This script never touches
# Stanza, so re-running it after tweaking analyse_dependencies.py is fast.

cd "$(dirname "$0")/.." || exit 1

python src/analyse_dependencies.py \
    --dep_csv data/reference_parses/dependency_results.csv \
    --fluent_csv data/reference_parses/fluent_parse_results.csv \
    --output output/references/analysis_results \
    "$@"
