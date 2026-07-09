#!/bin/bash
# Usage: bash 02_run_analysis.sh
# Run after 01_parse_dependency.sh has produced dependency_results.csv and
# fluent_parse_results.csv in this directory. This script never touches
# Stanza, so re-running it after tweaking analyse_dependencies.py is fast.

cd "$(dirname "$0")" || exit 1

DEP_CSV=dependency_results.csv
FLUENT_CSV=fluent_parse_results.csv
OUTPUT=analysis_results

python analyse_dependencies.py \
    --dep_csv "$DEP_CSV" \
    --fluent_csv "$FLUENT_CSV" \
    --output "$OUTPUT" \
    "$@"
