#!/bin/bash
# Usage: bash parse_dependency.sh [--langs en de zh ...]
# Requires stanza in the comet conda env: pip install stanza

cd "$(dirname "$0")" || exit 1

INPUT=../references.csv
OUTPUT=.
STANZA_DIR=/home/jhu/mzufle1/uh-mazing/stanza

python dependency_parsing.py \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --stanza_dir "$STANZA_DIR" \
    "$@"
