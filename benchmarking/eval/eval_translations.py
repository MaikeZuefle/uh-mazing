import os
import jieba
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
    Returns (sources, hypotheses, references, sample_ids) for COMET QE and CHRF evaluation.
    sources: disfluent source text (for COMET QE)
    hypotheses: model translations (from XML)
    references: reference translations (for CHRF)
    sample_ids: per-row identifiers, parallel to the above
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

    sources, hypotheses, references, sample_ids = [], [], [], []
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
        sample_ids.append(sample_id)

    return sources, hypotheses, references, sample_ids


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


def prepare_data_csv(df, csv_path, model, lang_code, refs_df):
    """
    Prepare (sources, hypotheses, references, sample_ids) for a single language from a
    commercial CSV.
    df: already-loaded commercial DataFrame.
    refs_df: references DataFrame loaded from references.csv (the authoritative source),
        already carrying a 'sample_id' column via load_csv_references.
    lang_code: uppercase, e.g. 'ZH', 'ES'.
    references is None when no reference column exists (e.g. LG).
    """
    source_column = "EN_disfluent"
    if source_column not in df.columns:
        raise ValueError(f"Source column '{source_column}' not found in {csv_path}")

    hyp_column = f"{lang_code}_{model}"
    if hyp_column not in df.columns:
        raise ValueError(f"Hypothesis column '{hyp_column}' not found in {csv_path}")

    ref_lang_code = COMMERCIAL_REF_LANG_MAP.get(lang_code, lang_code)
    ref_column = f"{ref_lang_code}_disfluent"
    has_reference = ref_column in refs_df.columns

    if has_reference:
        merged = df[["ID", source_column, hyp_column]].merge(
            refs_df[["ID", ref_column, "sample_id"]], on="ID", how="left"
        )
    else:
        merged = df[["ID", source_column, hyp_column]].merge(
            refs_df[["ID", "sample_id"]], on="ID", how="left"
        )

    sources, hypotheses, references, sample_ids = [], [], [], []
    for _, row in merged.iterrows():
        raw_src = row[source_column]
        if pd.isna(raw_src) or str(raw_src).strip() == "":
            raise ValueError(f"Empty source in {csv_path}")
        sources.append(clean_disfluency_text(str(raw_src)))

        raw_hyp = row[hyp_column]
        if pd.isna(raw_hyp):
            raw_hyp = ""
            print(f"Empty hypothesis for lang {lang_code} in {csv_path}")
        hypotheses.append(str(raw_hyp))
        sample_ids.append(row["sample_id"])

        if has_reference:
            raw_ref = row[ref_column]
            if pd.isna(raw_ref) or str(raw_ref).strip() == "":
                raise ValueError(f"Empty reference for lang {lang_code} in {csv_path}")
            references.append(clean_disfluency_text(str(raw_ref)))

    return sources, hypotheses, (references if has_reference else None), sample_ids


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _maybe_segment_chinese(texts, target_lang):
    """chrF++ (word_order=2) adds word-level n-grams on top of character n-grams, and
    sacrebleu splits "words" on whitespace. Chinese has no whitespace between words, so
    without segmentation the word-n-gram component never matches. Word-segment with jieba
    first so chrF++ behaves the same way for Chinese as for whitespace-delimited languages.
    """
    if target_lang != "zh":
        return texts
    return [" ".join(jieba.cut(text)) for text in texts]


def compute_chrf(hypotheses, references, target_lang):
    """Corpus-level chrF (not decomposable from per-sentence scores)."""
    hypotheses = _maybe_segment_chinese(hypotheses, target_lang)
    references = _maybe_segment_chinese(references, target_lang)
    return CHRF(word_order=2).corpus_score(hypotheses, [references]).score


def compute_chrf_per_item(hypotheses, references, target_lang):
    hypotheses = _maybe_segment_chinese(hypotheses, target_lang)
    references = _maybe_segment_chinese(references, target_lang)
    chrf = CHRF(word_order=2)
    return [chrf.sentence_score(hyp, [ref]).score for hyp, ref in zip(hypotheses, references)]


def compute_comet_qe(model, sources, hypotheses):
    """Returns (system_score, per_item_scores). COMET computes per-segment scores internally
    regardless; we just keep both instead of discarding the per-segment ones."""
    data = [{"src": src, "mt": mt} for src, mt in zip(sources, hypotheses)]
    prediction = model.predict(data, batch_size=8, gpus=1)
    return prediction.system_score, prediction.scores


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def make_eval_key(model_type, model, filename, target_lang):
    return f"{model_type}_{model}_{filename}_{target_lang}"


def append_per_item_results(per_item_csv, rows):
    if not rows:
        return
    out_dir = os.path.dirname(per_item_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    new_df = pd.DataFrame(rows)
    if os.path.exists(per_item_csv):
        new_df.to_csv(per_item_csv, mode="a", header=False, index=False)
    else:
        new_df.to_csv(per_item_csv, mode="w", header=True, index=False)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate_all_translations(base_dir, csv_path, output_csv, only_chrf=False, per_item_csv=None):
    comet_model = None if only_chrf else load_comet_model()
    # load_csv_references (not a plain pd.read_csv) so refs_df carries the 'sample_id' column
    # that prepare_data_csv needs for per-item output.
    refs_df = load_csv_references(csv_path)

    # A combo only counts as already done if it also has per-item rows saved, since those
    # were never computed/saved by earlier runs of this script.
    per_item_keys = set()
    existing_per_item_df = None
    if per_item_csv and os.path.exists(per_item_csv):
        existing_per_item_df = pd.read_csv(per_item_csv)
        per_item_keys = set(
            existing_per_item_df.apply(
                lambda row: make_eval_key(row["model_type"], row["model"], row["filename"], row["target_lang"]),
                axis=1,
            )
        )

    if os.path.exists(output_csv):
        existing_df = pd.read_csv(output_csv)
        all_keys = set(
            existing_df.apply(
                lambda row: make_eval_key(row["model_type"], row["model"], row["filename"], row["target_lang"]),
                axis=1,
            )
        )
        if only_chrf:
            # Skip rows that already have a chrf score and update in-place instead of appending,
            # except target_lang == "zh": chrF++ now word-segments Chinese first (see
            # _maybe_segment_chinese), so any existing zh score predates that fix and is stale.
            evaluated_keys = set(
                existing_df[
                    existing_df["chrf_score"].notna()
                    & (existing_df["chrf_score"] != "")
                    & (existing_df["target_lang"] != "zh")
                ].apply(
                    lambda row: make_eval_key(row["model_type"], row["model"], row["filename"], row["target_lang"]),
                    axis=1,
                )
            )
            key_to_idx = {
                make_eval_key(row["model_type"], row["model"], row["filename"], row["target_lang"]): idx
                for idx, row in existing_df.iterrows()
            }
            chrf_updates = {}
            per_item_chrf_updates = {}
        else:
            print(f"Output file {output_csv} already exists. Appending new results...")
            evaluated_keys = all_keys
    else:
        evaluated_keys = set()
        if only_chrf:
            key_to_idx = {}
            chrf_updates = {}
            per_item_chrf_updates = {}

    if per_item_csv and not only_chrf:
        evaluated_keys = evaluated_keys & per_item_keys

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

                        sources, hypotheses, references, sample_ids = prepare_data_xml(
                            csv_path, os.path.join(entry_path, filename), source_lang, target_lang
                        )
                        chrf_score = compute_chrf(hypotheses, references, target_lang)

                        if only_chrf:
                            print(f"      CHRF++ score: {chrf_score:.4f}")
                            if eval_key in key_to_idx:
                                # Update chrf score in existing row
                                chrf_updates[eval_key] = chrf_score
                                if per_item_csv:
                                    chrf_per_item = compute_chrf_per_item(hypotheses, references, target_lang)
                                    for sample_id, item_chrf in zip(sample_ids, chrf_per_item):
                                        per_item_chrf_updates[(eval_key, sample_id)] = item_chrf
                            else:
                                # New entry not yet in CSV: append it directly
                                append_result(output_csv, {
                                    "model_type": model_type,
                                    "model": model_name,
                                    "modality": "",
                                    "source_lang": source_lang,
                                    "target_lang": target_lang,
                                    "lang_pair": f"{source_lang}-{target_lang}",
                                    "prompt_type": prompt_type,
                                    "chrf_score": chrf_score,
                                    "comet_score": None,
                                    "filename": filename,
                                })
                                if per_item_csv:
                                    chrf_per_item = compute_chrf_per_item(hypotheses, references, target_lang)
                                    append_per_item_results(per_item_csv, [
                                        {
                                            "model_type": model_type,
                                            "model": model_name,
                                            "target_lang": target_lang,
                                            "prompt_type": prompt_type,
                                            "filename": filename,
                                            "sample_id": sample_id,
                                            "chrf_score": item_chrf,
                                            "comet_score": None,
                                        }
                                        for sample_id, item_chrf in zip(sample_ids, chrf_per_item)
                                    ])
                        else:
                            comet_score, comet_per_item = compute_comet_qe(comet_model, sources, hypotheses)
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
                            print(f"      CHRF++ score: {chrf_score:.4f}, COMET QE score: {comet_score:.4f}")

                            if per_item_csv:
                                chrf_per_item = compute_chrf_per_item(hypotheses, references, target_lang)
                                append_per_item_results(per_item_csv, [
                                    {
                                        "model_type": model_type,
                                        "model": model_name,
                                        "target_lang": target_lang,
                                        "prompt_type": prompt_type,
                                        "filename": filename,
                                        "sample_id": sample_id,
                                        "chrf_score": item_chrf,
                                        "comet_score": item_comet,
                                    }
                                    for sample_id, item_chrf, item_comet in zip(
                                        sample_ids, chrf_per_item, comet_per_item
                                    )
                                ])
                        evaluated_keys.add(eval_key)
                        results_count += 1

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

                        sources, hypotheses, references, sample_ids = prepare_data_csv(
                            commercial_df, entry_path, commercial_model, lang_code, refs_df
                        )

                        chrf_score = None
                        if references is not None:
                            chrf_score = compute_chrf(hypotheses, references, target_lang)

                        if only_chrf:
                            chrf_str = f"{chrf_score:.4f}" if chrf_score is not None else "N/A"
                            print(f"    {lang_code}: CHRF++ = {chrf_str}")
                            chrf_updates[eval_key] = chrf_score
                            if per_item_csv and references is not None:
                                chrf_per_item = compute_chrf_per_item(hypotheses, references, target_lang)
                                for sample_id, item_chrf in zip(sample_ids, chrf_per_item):
                                    per_item_chrf_updates[(eval_key, sample_id)] = item_chrf
                        else:
                            comet_score, comet_per_item = compute_comet_qe(comet_model, sources, hypotheses)
                            chrf_str = f"{chrf_score:.4f}" if chrf_score is not None else "N/A"
                            print(f"    {lang_code}: CHRF++ = {chrf_str}, COMET QE = {comet_score:.4f}")

                            if per_item_csv:
                                chrf_per_item = (
                                    compute_chrf_per_item(hypotheses, references, target_lang)
                                    if references is not None
                                    else [None] * len(hypotheses)
                                )
                                append_per_item_results(per_item_csv, [
                                    {
                                        "model_type": model_type,
                                        "model": commercial_model,
                                        "target_lang": target_lang,
                                        "prompt_type": prompt_output,
                                        "filename": filename,
                                        "sample_id": sample_id,
                                        "chrf_score": item_chrf,
                                        "comet_score": item_comet,
                                    }
                                    for sample_id, item_chrf, item_comet in zip(
                                        sample_ids, chrf_per_item, comet_per_item
                                    )
                                ])

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


    if only_chrf and chrf_updates:
        for key, score in chrf_updates.items():
            if key in key_to_idx:
                existing_df.at[key_to_idx[key], "chrf_score"] = score
        existing_df.to_csv(output_csv, index=False)
        print(f"\nUpdated {len(chrf_updates)} chrf_score values in {output_csv}")

    if only_chrf and per_item_csv and per_item_chrf_updates:
        if existing_per_item_df is None:
            print(f"Warning: {per_item_csv} does not exist yet, cannot apply per-item chrf updates.")
        else:
            row_eval_keys = existing_per_item_df.apply(
                lambda row: make_eval_key(row["model_type"], row["model"], row["filename"], row["target_lang"]),
                axis=1,
            )
            updated_count = 0
            for idx, (eval_key, sample_id) in enumerate(zip(row_eval_keys, existing_per_item_df["sample_id"])):
                update = per_item_chrf_updates.get((eval_key, sample_id))
                if update is not None:
                    existing_per_item_df.at[idx, "chrf_score"] = update
                    updated_count += 1
            existing_per_item_df.to_csv(per_item_csv, index=False)
            print(f"Updated {updated_count} per-item chrf_score values in {per_item_csv}")

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
    parser.add_argument("--only-chrf", action="store_true", help="Only compute/update ChrF scores (skip COMET, update in-place)")
    parser.add_argument(
        "--per-item-output", default=None,
        help="Optional path to also save per-item chrF/COMET scores (ignored with --only-chrf)",
    )
    args = parser.parse_args()
    evaluate_all_translations(
        base_dir=args.input,
        csv_path=args.references,
        output_csv=args.output,
        only_chrf=args.only_chrf,
        per_item_csv=args.per_item_output,
    )
