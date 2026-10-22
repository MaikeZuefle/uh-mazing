#!/bin/bash
# Usage: bash scripts/01_parse_dependency.sh [--langs en de zh ...] [--stanza_dir DIR]

cd "$(dirname "$0")/.." || exit 1

python src/dependency_parsing.py \
    --input ../../translation_data/references.csv \
    --output data/reference_parses \
    "$@"
