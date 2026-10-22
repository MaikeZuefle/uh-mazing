#!/bin/bash
# Usage (from the repo root): bash myjob.sh -f analysis/syntax_analysis/scripts/04_analyse_generated_gpu.sh -n stanza -t 04:00:00

eval "$(conda shell.bash hook)"
conda activate comet
bash "$(dirname "$0")/04_analyse_generated.sh" "$@"
