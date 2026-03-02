import os
import pandas as pd
import re
from comet import download_model, load_from_checkpoint
from utils import load_xml_hypotheses, clean_disfluency_text, extract_language_from_filename, extract_prompt_type_from_filename
from utils import REQUIRED_DISFLUENCY_COLUMNS
import csv
from collections import defaultdict

MODEL_TYPES = ["analysis_node_combos"]


def load_comet_model(model_name="Unbabel/wmt22-cometkiwi-da"):
    model_path = download_model(model_name)
    return load_from_checkpoint(model_path).to("cuda")


def prepare_comet_data(csv_path, xml_path):

    def read_disfluency_type_transcript(csv_path):
        sources = []

        with open(csv_path, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)

            missing_cols = [
                col for col in REQUIRED_DISFLUENCY_COLUMNS
                if col not in reader.fieldnames
            ]

            if missing_cols:
                raise KeyError(
                    f"Missing required columns in CSV: {missing_cols}"
                )

            for row in reader:
                sample_id = row["ID"]

                variants = {
                    col: row[col].strip()
                    for col in REQUIRED_DISFLUENCY_COLUMNS
                    if col != "ID"
                }
                for variant_name, variant in variants.items():
                    text_clean = clean_disfluency_text(variant)
                    sources.append((f"{sample_id}##{variant_name}", text_clean))
        return sources

    sources = read_disfluency_type_transcript(csv_path)
    hyps_dict = load_xml_hypotheses(xml_path)

    qe_data = {}
    for source_id, source in sources:
        if source_id not in hyps_dict:
            raise ValueError(f"Source ID '{source_id}' not found in XML hypotheses")

        translation_text = hyps_dict[source_id]
        qe_data[source_id] = {"src": source, "mt": translation_text}
    return qe_data


def compute_comet_scores(model, qe_dict):
    """
    Returns a dict mapping disfluency_type -> mean COMET score across all samples.
    Keys in qe_dict are like 'sw2027_A_65##EN_fluent'.
    """
    grouped = defaultdict(list)
    for key, val in qe_dict.items():
        sample_id, disfluency_type = key.split("##")
        grouped[disfluency_type].append({"src": val["src"], "mt": val["mt"]})

    scores_by_type = {}
    for disfluency_type, data in grouped.items():
        result = model.predict(data, batch_size=8, gpus=1)
        mean_score = sum(result.scores) / len(result.scores)
        scores_by_type[disfluency_type] = mean_score

    return scores_by_type


def save_pivot_tables(output_csv):
    final_df = pd.read_csv(output_csv)

    DISFLUENCY_ORDER = [
        "EN_fluent",
        "EN_disfluent",
        "edited_nodes_only",
        "intj_only",
        "prn_only",
        "edited_intj",
        "intj_prn",
        "edited_prn",
        "edited_prn_intj",
    ]

    DISFLUENCY_LABELS = {
        "EN_fluent": "Fluent",
        "EN_disfluent": "Disfluent",
        "edited_nodes_only": "Edited Nodes Only",
        "intj_only": "Intj Only",
        "prn_only": "Prn Only",
        "edited_intj": "Edited Intj",
        "intj_prn": "intj PRN",
        "edited_prn": "edited prn",
        "edited_prn_intj": "edited prn intj",
    }

    MODEL_ORDER = ["llama", "tower"]
    LANG_ORDER = ["zh", "es", "hi", "fr", "de", "it", "cs"]

    # Full pivot table
    print("\nPivot table (model + disfluency_type vs target_lang):")
    pivot = final_df.pivot_table(
        index=["model", "disfluency_type"],
        columns="target_lang",
        values="comet_score",
        aggfunc="mean"
    )
    print(pivot)

    pivot_structured = (pivot * 100).round(2)

    # Add missing language columns as NaN
    for lang in LANG_ORDER:
        if lang not in pivot_structured.columns:
            pivot_structured[lang] = float("nan")

    pivot_structured = pivot_structured[LANG_ORDER]

    # Sort by model first, then disfluency type
    available_disfluency = [d for d in DISFLUENCY_ORDER if d in pivot_structured.index.get_level_values("disfluency_type")]

    if missing_disfluency := [d for d in DISFLUENCY_ORDER if d not in pivot_structured.index.get_level_values("disfluency_type")]:
        print(f"Warning: disfluency types not yet in results: {missing_disfluency}")

    model_rank = {m: i for i, m in enumerate(MODEL_ORDER)}
    disfl_rank = {d: i for i, d in enumerate(available_disfluency)}

    sort_keys = [
        (model_rank.get(m, 99), disfl_rank.get(d, 99))
        for m, d in zip(
            pivot_structured.index.get_level_values("model"),
            pivot_structured.index.get_level_values("disfluency_type")
        )
    ]
    sorted_indices = sorted(range(len(sort_keys)), key=lambda i: sort_keys[i])
    pivot_structured = pivot_structured.iloc[sorted_indices]

    # Rename disfluency types to display labels
    pivot_structured.index = pd.MultiIndex.from_arrays([
        pivot_structured.index.get_level_values("model"),
        pivot_structured.index.get_level_values("disfluency_type").map(DISFLUENCY_LABELS)
    ], names=["model", "disfluency_type"])

    output_csv_structured = output_csv.replace(".csv", "_en-x_structured.csv")
    pivot_structured.to_csv(output_csv_structured)
    print(f"\nSaved structured pivot to {output_csv_structured}")

def evaluate_all_translations(base_dir, csv_path, output_csv):

    model = load_comet_model("Unbabel/wmt22-cometkiwi-da")

    if os.path.exists(output_csv):
        print(f"Output file {output_csv} already exists. Appending new results...")
        existing_df = pd.read_csv(output_csv)
        evaluated_keys = set(
            existing_df.apply(
                lambda row: f"{row['model_type']}_{row['model']}_{row['filename']}_{row['disfluency_type']}",
                axis=1
            )
        )
    else:
        evaluated_keys = set()

    results_count = 0

    for model_type in MODEL_TYPES:
        model_type_path = os.path.join(base_dir, model_type)

        if not os.path.exists(model_type_path):
            raise FileNotFoundError(f"Model type directory not found: {model_type_path}")

        print(f"\nProcessing model_type: {model_type}")

        for model_name in os.listdir(model_type_path):
            model_path = os.path.join(model_type_path, model_name)

            if not os.path.isdir(model_path):
                continue

            print(f"  Processing model: {model_name}")

            for filename in os.listdir(model_path):
                if not filename.endswith(".xml"):
                    continue

                source_lang, target_lang = extract_language_from_filename(filename)
                if source_lang is None or target_lang is None:
                    raise ValueError(f"Could not extract language pair from {filename}")

                prompt_type = extract_prompt_type_from_filename(filename)
                if prompt_type is None:
                    raise ValueError(f"Could not extract prompt type from {filename}")

                xml_path = os.path.join(model_path, filename)

                qe_dict = prepare_comet_data(csv_path, xml_path)

                # Determine which disfluency types exist in this file
                disfluency_types_in_file = set(key.split("##")[1] for key in qe_dict)
                keys_for_file = {
                    f"{model_type}_{model_name}_{filename}_{dt}"
                    for dt in disfluency_types_in_file
                }

                if keys_for_file.issubset(evaluated_keys):
                    print(f"    Skipping {filename} (all disfluency types already evaluated)")
                    continue

                print(f"    Evaluating {filename} (lang: {source_lang}→{target_lang}, prompt: {prompt_type})")
                scores_by_type = compute_comet_scores(model, qe_dict)

                for disfluency_type, comet_score in scores_by_type.items():
                    file_key = f"{model_type}_{model_name}_{filename}_{disfluency_type}"

                    if file_key in evaluated_keys:
                        print(f"      Skipping {disfluency_type} (already evaluated)")
                        continue

                    new_result = pd.DataFrame([{
                        "model_type": model_type,
                        "model": model_name,
                        "source_lang": source_lang,
                        "target_lang": target_lang,
                        "lang_pair": f"{source_lang}-{target_lang}",
                        "prompt_type": prompt_type,
                        "disfluency_type": disfluency_type,
                        "comet_score": comet_score,
                        "filename": filename
                    }])

                    if not os.path.exists(output_csv):
                        new_result.to_csv(output_csv, mode='w', header=True, index=False)
                    else:
                        new_result.to_csv(output_csv, mode='a', header=False, index=False)

                    evaluated_keys.add(file_key)
                    results_count += 1
                    print(f"      {disfluency_type} COMET score: {comet_score:.4f}")

    if results_count > 0 or os.path.exists(output_csv):
        print(f"\n{'='*60}")
        print(f"Results saved to {output_csv}")
        print(f"New evaluations in this run: {results_count}")
        final_df = pd.read_csv(output_csv)
        print(f"Total evaluations in file: {len(final_df)}")
        print(f"{'='*60}")

        save_pivot_tables(output_csv)
    else:
        print("\nNo results to save.")


if __name__ == "__main__":
    evaluate_all_translations(
        base_dir="generated_output",
        csv_path="translation_data/uh-mazing-with-node-combos.csv",
        output_csv="eval/results_non_commercial/analysis_node_combos.csv"
    )