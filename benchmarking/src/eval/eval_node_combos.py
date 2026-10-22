import os
import pandas as pd
import csv
from collections import defaultdict
from utils import load_xml_hypotheses, clean_disfluency_text
from utils import load_comet_model, append_result, parse_xml_filename
from utils import REQUIRED_DISFLUENCY_COLUMNS, COMMERCIAL_LANG_CODE_MAP

MODEL_TYPES = ["commercial_models", "llama", "tower"]


def parse_csv_filename(filename):
    """
    Parse node-combos_{model}_{prompt_type}_{disfluency_label}[-noextra].csv
    Each file corresponds to exactly one disfluency type used as translation source.
    The "-noextra" suffix is optional -- older/regenerated commercial-model files omit it.
    Returns (model, prompt_type, col_name, out_name) where:
      col_name: actual column in the CSV (e.g. 'text_fluent', 'edited_intj')
      out_name: canonical name for output (e.g. 'EN_fluent', 'edited_intj')
    """
    if not filename.startswith("node-combos_") or not filename.endswith(".csv"):
        raise ValueError(f"Unexpected node-combos filename: {filename}")
    inner = filename[len("node-combos_"):-len(".csv")]
    parts = inner.split("_")
    if len(parts) < 3:
        raise ValueError(f"Cannot parse node-combos filename (too few parts): {filename}")
    model = parts[0]
    prompt_type = parts[1]
    disfl_with_suffix = "_".join(parts[2:])
    disfl_label = disfl_with_suffix[:-len("-noextra")] if disfl_with_suffix.endswith("-noextra") else disfl_with_suffix

    # Map short filename labels to (CSV column name, canonical output name)
    label_map = {
        "fluent": ("text_fluent", "EN_fluent"),
        "disfluent": ("text_disfluent", "EN_disfluent"),
    }
    col_name, out_name = label_map.get(disfl_label, (disfl_label, disfl_label))
    return model, prompt_type, col_name, out_name


def prepare_comet_data_csv(csv_path, disfluency_type):
    """
    Returns (lang_qe_data, missing_entries).
    lang_qe_data: dict target_lang -> list of {"src": ..., "mt": ...}
    missing_entries: list of (col, lang, src_text) for rows with empty translations.
    Source text is taken from the disfluency_type column; translations from {LANG}_{model} columns.
    """
    lang_qe_data = defaultdict(list)
    missing_entries = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        lang_columns = {
            col: COMMERCIAL_LANG_CODE_MAP[col.split("_", 1)[0]]
            for col in reader.fieldnames
            if col.split("_", 1)[0] in COMMERCIAL_LANG_CODE_MAP
        }
        if not lang_columns:
            raise ValueError(f"No translation columns found in {csv_path}")
        for row in reader:

            src_text = clean_disfluency_text(row[disfluency_type].strip())


            for col, lang in lang_columns.items():
                mt_text = row[col].strip()
                if not mt_text:
                    missing_entries.append((col, lang, f"{row['file']} {row['speaker']} {row['turn']}"))
                    continue
                lang_qe_data[lang].append({"src": src_text, "mt": mt_text})
    return lang_qe_data, missing_entries


def prepare_comet_data_xml(csv_path, xml_path):
    sources = []
    with open(csv_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        missing_cols = [col for col in REQUIRED_DISFLUENCY_COLUMNS if col not in reader.fieldnames]
        if missing_cols:
            raise KeyError(f"Missing required columns in CSV: {missing_cols}")
        for row in reader:
            sample_id = row["ID"]
            for col in REQUIRED_DISFLUENCY_COLUMNS:
                if col == "ID":
                    continue
                text_clean = clean_disfluency_text(row[col].strip())
                sources.append((f"{sample_id}##{col}", text_clean))

    hyps_dict = load_xml_hypotheses(xml_path)
    qe_data = {}
    for source_id, source in sources:
        if source_id not in hyps_dict:
            raise ValueError(f"Source ID '{source_id}' not found in XML hypotheses")
        qe_data[source_id] = {"src": source, "mt": hyps_dict[source_id]}
    return qe_data


def compute_comet_scores(model, qe_dict):
    """
    Returns a dict mapping disfluency_type -> mean COMET score across all samples.
    Keys in qe_dict are like 'sw2027_A_65##EN_fluent'.
    """
    grouped = defaultdict(list)
    for key, val in qe_dict.items():
        _, disfluency_type = key.split("##")
        grouped[disfluency_type].append({"src": val["src"], "mt": val["mt"]})

    return {
        disfluency_type: sum(model.predict(data, batch_size=8, gpus=1).scores) / len(data)
        for disfluency_type, data in grouped.items()
    }


def make_eval_key(model_type, model, filename, disfluency_type, target_lang):
    return f"{model_type}_{model}_{filename}_{disfluency_type}_{target_lang}"


def save_pivot_tables(output_csv):
    DISFLUENCY_ORDER = [
        "EN_fluent", "EN_disfluent", "edited_nodes_only", "intj_only", "prn_only",
        "edited_intj", "intj_prn", "edited_prn", "edited_prn_intj",
    ]
    DISFLUENCY_LABELS = {
        "EN_fluent": "Fluent",
        "EN_disfluent": "Disfluent",
        "edited_nodes_only": "Edited Nodes Only",
        "intj_only": "Intj Only",
        "prn_only": "Prn Only",
        "edited_intj": "Edited Intj",
        "intj_prn": "Intj PRN",
        "edited_prn": "Edited PRN",
        "edited_prn_intj": "Edited PRN Intj",
    }
    MODEL_ORDER = ["llama", "tower", "gemini-2.5-flash", "gpt52"]
    LANG_ORDER = ["zh", "es", "hi", "fr", "de", "it", "cs"]

    final_df = pd.read_csv(output_csv)

    print("\nPivot table (model + disfluency_type vs target_lang):")
    pivot = final_df.pivot_table(
        index=["model", "disfluency_type"],
        columns="target_lang",
        values="comet_score",
        aggfunc="mean"
    )
    print(pivot)

    pivot_structured = (pivot * 100).round(2)
    for lang in LANG_ORDER:
        if lang not in pivot_structured.columns:
            pivot_structured[lang] = float("nan")
    pivot_structured = pivot_structured[LANG_ORDER]

    available_disfluency = [d for d in DISFLUENCY_ORDER if d in pivot_structured.index.get_level_values("disfluency_type")]
    missing_disfluency = [d for d in DISFLUENCY_ORDER if d not in pivot_structured.index.get_level_values("disfluency_type")]
    if missing_disfluency:
        print(f"Warning: disfluency types not yet in results: {missing_disfluency}")

    model_rank = {m: i for i, m in enumerate(MODEL_ORDER)}
    disfl_rank = {d: i for i, d in enumerate(available_disfluency)}
    sort_keys = [
        (model_rank.get(m, 99), disfl_rank.get(d, 99))
        for m, d in zip(
            pivot_structured.index.get_level_values("model"),
            pivot_structured.index.get_level_values("disfluency_type"),
        )
    ]
    pivot_structured = pivot_structured.iloc[sorted(range(len(sort_keys)), key=lambda i: sort_keys[i])]

    pivot_structured.index = pd.MultiIndex.from_arrays([
        pivot_structured.index.get_level_values("model"),
        pivot_structured.index.get_level_values("disfluency_type").map(DISFLUENCY_LABELS),
    ], names=["model", "disfluency_type"])

    output_csv_structured = output_csv.replace(".csv", "_en-x_structured.csv")
    pivot_structured.to_csv(output_csv_structured)
    print(f"\nSaved structured pivot to {output_csv_structured}")


def evaluate_all_translations(base_dir, csv_path, output_csv):
    model = load_comet_model()

    if os.path.exists(output_csv):
        print(f"Output file {output_csv} already exists. Appending new results...")
        existing_df = pd.read_csv(output_csv)
        evaluated_keys = set(
            existing_df.apply(
                lambda row: make_eval_key(row["model_type"], row["model"], row["filename"], row["disfluency_type"], row["target_lang"]),
                axis=1,
            )
        )
    else:
        evaluated_keys = set()

    results_count = 0
    all_missing = []  # list of (filename, col, lang, src_text)

    for model_type in MODEL_TYPES:
        model_type_path = os.path.join(base_dir, model_type)
        if not os.path.exists(model_type_path):
            raise FileNotFoundError(f"Model type directory not found: {model_type_path}")

        print(f"\nProcessing model_type: {model_type}")

        for filename in sorted(os.listdir(model_type_path)):
            filepath = os.path.join(model_type_path, filename)
            if not os.path.isfile(filepath):
                raise NotADirectoryError(f"Expected file but found directory: {filepath}")

            if filename.endswith(".xml"):
                is_commercial = False
            elif filename.endswith(".csv"):
                is_commercial = True
            else:
                continue  # skip .log and other non-translation files

            if is_commercial:
                commercial_model, prompt_type, col_disfluency, out_disfluency = parse_csv_filename(filename)
                print(f"  Evaluating {filename} (model: {commercial_model}, prompt: {prompt_type}, disfl: {out_disfluency})")

                lang_qe_data, missing = prepare_comet_data_csv(filepath, col_disfluency)
                all_missing.extend((filename, col, lang, src) for col, lang, src in missing)

                if all(make_eval_key(model_type, commercial_model, filename, out_disfluency, lang) in evaluated_keys for lang in lang_qe_data):
                    print(f"    Skipping {filename} (all languages already evaluated)")
                    continue

                for target_lang, qe_list in lang_qe_data.items():
                    file_key = make_eval_key(model_type, commercial_model, filename, out_disfluency, target_lang)
                    if file_key in evaluated_keys:
                        print(f"    Skipping {out_disfluency}/{target_lang} (already evaluated)")
                        continue
                    comet_score = sum(model.predict(qe_list, batch_size=8, gpus=1).scores) / len(qe_list)
                    append_result(output_csv, {
                        "model_type": model_type,
                        "model": commercial_model,
                        "source_lang": "en",
                        "target_lang": target_lang,
                        "lang_pair": f"en-{target_lang}",
                        "prompt_type": prompt_type,
                        "disfluency_type": out_disfluency,
                        "comet_score": comet_score,
                        "filename": filename,
                    })
                    evaluated_keys.add(file_key)
                    results_count += 1
                    print(f"    {out_disfluency}/{target_lang} COMET score: {comet_score:.4f}")

            else:
                model_name = model_type  # e.g. "llama" or "tower"
                source_lang, target_lang, prompt_type = parse_xml_filename(filename)
                qe_dict = prepare_comet_data_xml(csv_path, filepath)
                disfluency_types_in_file = set(key.split("##")[1] for key in qe_dict)

                if all(make_eval_key(model_type, model_name, filename, dt, target_lang) in evaluated_keys for dt in disfluency_types_in_file):
                    print(f"  Skipping {filename} (all disfluency types already evaluated)")
                    continue

                print(f"  Evaluating {filename} (lang: {source_lang}→{target_lang}, prompt: {prompt_type})")
                scores_by_type = compute_comet_scores(model, qe_dict)

                for disfluency_type, comet_score in scores_by_type.items():
                    file_key = make_eval_key(model_type, model_name, filename, disfluency_type, target_lang)
                    if file_key in evaluated_keys:
                        print(f"    Skipping {disfluency_type} (already evaluated)")
                        continue
                    append_result(output_csv, {
                        "model_type": model_type,
                        "model": model_name,
                        "source_lang": source_lang,
                        "target_lang": target_lang,
                        "lang_pair": f"{source_lang}-{target_lang}",
                        "prompt_type": prompt_type,
                        "disfluency_type": disfluency_type,
                        "comet_score": comet_score,
                        "filename": filename,
                    })
                    evaluated_keys.add(file_key)
                    results_count += 1
                    print(f"    {disfluency_type} COMET score: {comet_score:.4f}")

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

    print("Missing entries")
    for row in all_missing:
        print(row)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate analysis node combos translations with COMET QE.")
    parser.add_argument("--input", required=True, help="Base directory containing generated output (e.g. generated_output)")
    parser.add_argument("--references", required=True, help="Path to node-combos source CSV (e.g. translation_data/uh-mazing-with-node-combos.csv)")
    parser.add_argument("--output", required=True, help="Path to output CSV (e.g. eval/results_non_commercial/analysis_node_combos.csv)")
    args = parser.parse_args()
    evaluate_all_translations(
        base_dir=args.input,
        csv_path=args.references,
        output_csv=args.output,
    )
