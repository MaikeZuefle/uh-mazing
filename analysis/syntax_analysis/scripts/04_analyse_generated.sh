#!/bin/bash
# Usage: bash scripts/04_analyse_generated.sh [--stanza_dir DIR]

cd "$(dirname "$0")/.." || exit 1

for csv in data/marked/*.csv; do
    system=$(basename "$csv" .csv)
    [[ $system == marked_* || $system == excluded_* ]] && continue
    out=output/generated_translations/parsed/$system
    echo "== $system"
    if [[ -d $out/per_lang && $csv -nt $out/per_lang ]]; then
        rm -r "$out/per_lang"
    fi
    python src/dependency_parsing.py --input "$csv" --output "$out" "$@"
    python src/analyse_dependencies.py --dep_csv "$out/dependency_results.csv" \
        --fluent_csv "$out/fluent_parse_results.csv" --output "$out/analysis_results"
done

python src/compare_to_references.py
