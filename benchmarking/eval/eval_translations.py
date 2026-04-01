import os
import pandas as pd
from sacrebleu.metrics import CHRF
from utils import load_csv_references, load_xml_hypotheses, clean_disfluency_text
from utils import load_comet_model, append_result, parse_xml_filename
from utils import (
    LANG_COLUMN_MAP, MODEL_TYPES,
    COMMERCIAL_LANG_CODES, COMMERCIAL_LANG_CODE_MAP, COMMERCIAL_REF_LANG_MAP,
)


# ---------------------------------------------------------------------------
# Non-commercial (XML) helpers
# ---------------------------------------------------------------------------

def prepare_data_xml(csv_path, xml_path, source_lang, target_lang):
    """
    Returns (sources, hypotheses, references) for COMET QE and CHRF evaluation.
    sources: disfluent source text (for COMET QE)
    hypotheses: model translations (from XML)
    references: reference translations (for CHRF)
    """
    if target_lang not in LANG_COLUMN_MAP:
        raise ValueError(f"Target language '{target_lang}' not found in LANG_COLUMN_MAP")

    refs_df = load_csv_references(csv_path)
    hyps_dict = load_xml_hypotheses(xml_path)

    ref_column = LANG_COLUMN_MAP[target_lang]
    if ref_column not in refs_df.columns:
        breakpoint()
        raise ValueError(f"Reference column '{ref_column}' not found in CSV")

    src_column = f"{source_lang.upper()}_disfluent"
    if src_column not in refs_df.columns:
        raise ValueError(f"Source column '{src_column}' not found in CSV")

    sources, hypotheses, references = [], [], []
    for _, row in refs_df.iterrows():
        sample_id = row["sample_id"]
        if sample_id not in hyps_dict:
            raise ValueError(f"Sample ID '{sample_id}' not found in XML hypotheses")

        raw_src = row[src_column]
        if pd.isna(raw_src) or str(raw_src).strip() == "":
            raise ValueError(f"Empty source for sample '{sample_id}'")

        raw_ref = row[ref_column]
        if pd.isna(raw_ref) or str(raw_ref).strip() == "":
            raise ValueError(f"Empty reference for sample '{sample_id}'")

        sources.append(clean_disfluency_text(str(raw_src)))
        hypotheses.append(hyps_dict[sample_id])
        references.append(clean_disfluency_text(str(raw_ref)))

    return sources, hypotheses, references


# ---------------------------------------------------------------------------
# Commercial (CSV) helpers
# ---------------------------------------------------------------------------

def parse_commercial_filename(filename):
    """
    Parse commercial CSV filename.
    Format: uh-mazing_{model}_{modality}_{prompt_output}.csv
    Returns (model, modality, prompt_output).
    """
    if not filename.startswith("uh-mazing_") or not filename.endswith(".csv"):
        raise ValueError(f"Unexpected commercial filename format: {filename}")
    inner = filename[len("uh-mazing_"):-len(".csv")]
    parts = inner.split("_")
    if len(parts) < 3:
        raise ValueError(f"Cannot parse commercial filename (too few parts): {filename}")
    model = parts[0]
    modality = parts[1]
    prompt_output = "_".join(parts[2:])
    return model, modality, prompt_output


def prepare_data_csv(df, csv_path, model, lang_code):
    """
    Prepare (sources, hypotheses, references) for a single language from a commercial CSV.
    df: already-loaded DataFrame (pass in to avoid re-reading per lang).
    lang_code: uppercase, e.g. 'ZH', 'ES'.
    references is None when no reference column exists (e.g. SW, LG).
    """
    source_column = "EN_disfluent"
    if source_column not in df.columns:
        raise ValueError(f"Source column '{source_column}' not found in {csv_path}")

    hyp_column = f"{lang_code}_{model}"
    if hyp_column not in df.columns:
        raise ValueError(f"Hypothesis column '{hyp_column}' not found in {csv_path}")

    ref_lang_code = COMMERCIAL_REF_LANG_MAP.get(lang_code, lang_code)
    ref_column = f"{ref_lang_code}_disfluent"
    has_reference = ref_column in df.columns

    sources, hypotheses, references = [], [], []
    for _, row in df.iterrows():
        raw_src = row[source_column]
        if pd.isna(raw_src) or str(raw_src).strip() == "":
            raise ValueError(f"Empty source in {csv_path}")
        sources.append(clean_disfluency_text(str(raw_src)))

        raw_hyp = row[hyp_column]
        if pd.isna(raw_hyp):
            raw_hyp = ""
            print(f"Empty hypothesis for lang {lang_code} in {csv_path}")
            # raise ValueError(f"Empty hypothesis for lang {lang_code} in {csv_path}")
        hypotheses.append(str(raw_hyp))

        if has_reference:
            raw_ref = row[ref_column]
            if pd.isna(raw_ref) or str(raw_ref).strip() == "":
                raise ValueError(f"Empty reference for lang {lang_code} in {csv_path}")
            references.append(clean_disfluency_text(str(raw_ref)))

    return sources, hypotheses, references if has_reference else None


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_chrf(hypotheses, references):
    return CHRF(word_order=2).corpus_score(hypotheses, [references]).score


def compute_comet_qe(model, sources, hypotheses):
    data = [{"src": src, "mt": mt} for src, mt in zip(sources, hypotheses)]
    return model.predict(data, batch_size=8, gpus=1).system_score


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def make_eval_key(model_type, model, filename, target_lang):
    return f"{model_type}_{model}_{filename}_{target_lang}"


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate_all_translations(base_dir, csv_path, output_csv):
    comet_model = load_comet_model()

    if os.path.exists(output_csv):
        print(f"Output file {output_csv} already exists. Appending new results...")
        existing_df = pd.read_csv(output_csv)
        evaluated_keys = set(
            existing_df.apply(
                lambda row: make_eval_key(row["model_type"], row["model"], row["filename"], row["target_lang"]),
                axis=1,
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

        for entry in os.listdir(model_type_path):
            entry_path = os.path.join(model_type_path, entry)

            if os.path.isdir(entry_path):
                # Non-commercial: entry is a model subdirectory containing XML files
                model_name = entry
                print(f"  Processing model: {model_name}")

                for filename in os.listdir(entry_path):
                    if filename.endswith(".log"):
                        continue
                    elif filename.endswith(".xml"):
                        source_lang, target_lang, prompt_type = parse_xml_filename(filename)
                        if target_lang == "lg":
                            continue # excluded from experiments

                        eval_key = make_eval_key(model_type, model_name, filename, target_lang)
                        if eval_key in evaluated_keys:
                            print(f"    Skipping {filename} (already evaluated)")
                            continue

                        print(f"    Evaluating {filename} (lang: {source_lang}→{target_lang}, prompt: {prompt_type})")

                        sources, hypotheses, references = prepare_data_xml(
                            csv_path, os.path.join(entry_path, filename), source_lang, target_lang
                        )
                        chrf_score = compute_chrf(hypotheses, references)
                        comet_score = compute_comet_qe(comet_model, sources, hypotheses)

                        append_result(output_csv, {
                            "model_type": model_type,
                            "model": model_name,
                            "modality": "",
                            "source_lang": source_lang,
                            "target_lang": target_lang,
                            "lang_pair": f"{source_lang}-{target_lang}",
                            "prompt_type": prompt_type,
                            "chrf_score": chrf_score,
                            "comet_score": comet_score,
                            "filename": filename,
                        })
                        evaluated_keys.add(eval_key)
                        results_count += 1
                        print(f"      CHRF++ score: {chrf_score:.4f}, COMET QE score: {comet_score:.4f}")

                    else: 
                        raise ValueError(f"Unexpected file type in {entry_path}: {filename}")
            else:
                if entry_path.endswith(".csv"):
                    # Commercial: entry is a flat CSV file (uh-mazing_*.csv)
                    filename = entry
                    commercial_model, modality, prompt_output = parse_commercial_filename(filename)
                    print(f"  Processing commercial file: {filename} (model={commercial_model}, modality={modality})")
                    commercial_df = pd.read_csv(entry_path)

                    for lang_code in COMMERCIAL_LANG_CODES:
                        target_lang = COMMERCIAL_LANG_CODE_MAP.get(lang_code, lang_code.lower())
                        eval_key = make_eval_key(model_type, commercial_model, filename, target_lang)
                        if eval_key in evaluated_keys:
                            print(f"    Skipping {lang_code} in {filename} (already evaluated)")
                            continue

                        sources, hypotheses, references = prepare_data_csv(commercial_df, entry_path, commercial_model, lang_code)

                        chrf_score = None
                        if references is not None:
                            chrf_score = compute_chrf(hypotheses, references)

                        comet_score = compute_comet_qe(comet_model, sources, hypotheses)

                        chrf_str = f"{chrf_score:.4f}" if chrf_score is not None else "N/A"
                        print(f"    {lang_code}: CHRF++ = {chrf_str}, COMET QE = {comet_score:.4f}")

                        append_result(output_csv, {
                            "model_type": model_type,
                            "model": commercial_model,
                            "modality": modality,
                            "source_lang": "en",
                            "target_lang": target_lang,
                            "lang_pair": f"en-{target_lang}",
                            "prompt_type": prompt_output,
                            "chrf_score": chrf_score,
                            "comet_score": comet_score,
                            "filename": filename,
                        })
                        evaluated_keys.add(eval_key)
                        results_count += 1
                else: 
                    raise ValueError(f"Unexpected file type in {entry_path}: {filename}")


    if results_count > 0 or os.path.exists(output_csv):
        print(f"\n{'='*60}")
        print(f"Results saved to {output_csv}")
        print(f"New evaluations in this run: {results_count}")
        final_df = pd.read_csv(output_csv)
        print(f"Total evaluations in file: {len(final_df)}")
        print(f"{'='*60}")
        print("\nCHRF++ summary by model_type:")
        print(final_df.groupby("model_type")["chrf_score"].agg(["mean", "std", "count"]))
        print("\nCHRF++ summary by language pair:")
        print(final_df.groupby("lang_pair")["chrf_score"].agg(["mean", "std", "count"]))
        print("\nCOMET QE summary by model_type:")
        print(final_df.groupby("model_type")["comet_score"].agg(["mean", "std", "count"]))
        print("\nCOMET QE summary by language pair:")
        print(final_df.groupby("lang_pair")["comet_score"].agg(["mean", "std", "count"]))
    else:
        print("\nNo results to save.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate translations with CHRF++ and COMET QE.")
    parser.add_argument("--input", required=True, help="Base directory containing generated output (e.g. generated_output)")
    parser.add_argument("--references", required=True, help="Path to references CSV (e.g. translation_data/references.csv)")
    parser.add_argument("--output", required=True, help="Path to output CSV (e.g. eval/results_non_commercial/evaluation_results.csv)")
    args = parser.parse_args()
    evaluate_all_translations(
        base_dir=args.input,
        csv_path=args.references,
        output_csv=args.output,
    )
