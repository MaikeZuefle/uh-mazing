import os
import pandas as pd
import re
from sacrebleu.metrics import CHRF
from utils import load_xml_hypotheses, clean_disfluency_text, extract_language_from_filename, extract_prompt_type_from_filename, extract_decode_strategy_from_filename
from utils import LANG_COLUMN_MAP, MODEL_TYPES


def init_chrf():
    return CHRF(word_order=2)  # word_order=2 enables CHRF++

def compute_chrf_scores(references_csv, xml_path, source_lang, target_lang):

    refs_df = pd.read_csv(references_csv)
    hyps_dict = load_xml_hypotheses(xml_path)
    
    if target_lang not in LANG_COLUMN_MAP:
        print(f"Warning: Target language {target_lang} not found in reference columns")
        return None
    
    ref_column = LANG_COLUMN_MAP[target_lang]
    
    if ref_column not in refs_df.columns:
        print(f"Warning: Column {ref_column} not found in references CSV")
        return None
    
    references = []
    hypotheses = []
    
    for _, row in refs_df.iterrows():
        sample_id = row["ID"]
        sample_id = sample_id.split("sw")[-1]
        sample_id = f"sw0{sample_id}"

        ref_text = clean_disfluency_text(row[ref_column])

        if sample_id not in hyps_dict:
            continue
        hyp_text = hyps_dict[sample_id]
        
        # Skip if reference is NaN or empty
        if pd.isna(ref_text) or ref_text.strip() == "":
            continue
        
        references.append(ref_text)
        hypotheses.append(hyp_text)
    
    if not references:
        return None
    
    chrf = init_chrf()
    return chrf.corpus_score(hypotheses, [references]).score




def evaluate_all_translations(base_dir="generated_output", references_csv="references.csv", output_csv="chrf_evaluation_results.csv"):
    
    # Check if output file exists to avoid re-evaluating
    if os.path.exists(output_csv):
        print(f"Output file {output_csv} already exists. Appending new results...")
        existing_df = pd.read_csv(output_csv)
        # Create set of already evaluated files for quick lookup
        evaluated_files = set(
            existing_df.apply(lambda row: f"{row['model_type']}_{row['model']}_{row['filename']}", axis=1)
        )
    else:
        existing_df = pd.DataFrame()
        evaluated_files = set()
    
    results_count = 0
    
    for model_type in MODEL_TYPES:
        model_type_path = os.path.join(base_dir, model_type)
        
        if not os.path.exists(model_type_path):
            print(f"Skipping {model_type} (directory not found)")
            continue
        
        print(f"\nProcessing model_type: {model_type}")
        
        # Iterate through model folders (second level)
        for model_name in os.listdir(model_type_path):
            model_path = os.path.join(model_type_path, model_name)
            
            if not os.path.isdir(model_path):
                continue
            
            print(f"  Processing model: {model_name}")
            
            # Iterate through XML files in model folder
            for filename in os.listdir(model_path):
                if not filename.endswith(".xml"):
                    continue
                
                # Check if already evaluated
                file_key = f"{model_type}_{model_name}_{filename}"
                if file_key in evaluated_files:
                    print(f"    Skipping {filename} (already evaluated)")
                    continue
                
                # Extract language pair from filename
                source_lang, target_lang = extract_language_from_filename(filename)
                if source_lang is None or target_lang is None:
                    print(f"    Warning: Could not extract language pair from {filename}")
                    continue
                
                # Extract prompt type from filename
                prompt_type = extract_prompt_type_from_filename(filename)
                if prompt_type is None:
                    print(f"    Warning: Could not extract prompt type from {filename}")
                    continue

                decode_strategy = extract_decode_strategy_from_filename(filename)
                xml_path = os.path.join(model_path, filename)

                print(f"    Evaluating {filename} (lang: {source_lang}→{target_lang}, prompt: {prompt_type}, decode: {decode_strategy})")

                chrf_score = compute_chrf_scores(references_csv, xml_path, source_lang, target_lang)

                if chrf_score is not None:
                    # Create new result row
                    new_result = pd.DataFrame([{
                        "model_type": model_type,
                        "model": model_name,
                        "source_lang": source_lang,
                        "target_lang": target_lang,
                        "lang_pair": f"{source_lang}-{target_lang}",
                        "prompt_type": prompt_type,
                        "decode_strategy": decode_strategy,
                        "chrf_score": chrf_score,
                        "filename": filename
                    }])
                    
                    # Append to CSV immediately
                    if not os.path.exists(output_csv):
                        # First write - include header
                        new_result.to_csv(output_csv, mode='w', header=True, index=False)
                    else:
                        # Append without header
                        new_result.to_csv(output_csv, mode='a', header=False, index=False)
                    
                    results_count += 1
                    print(f"      CHRF++ score: {chrf_score:.4f} (saved to {output_csv})")
                    
                    # Add to evaluated set to avoid re-evaluation in this run
                    evaluated_files.add(file_key)
    
    # Print final summary
    if results_count > 0 or os.path.exists(output_csv):
        print(f"\n{'='*60}")
        print(f"Results saved to {output_csv}")
        print(f"New evaluations in this run: {results_count}")
        
        # Load final results and print summary
        final_df = pd.read_csv(output_csv)
        print(f"Total evaluations in file: {len(final_df)}")
        print(f"{'='*60}")
        
        # Print summary statistics
        print("\nSummary by model_type:")
        summary = final_df.groupby("model_type")["chrf_score"].agg(["mean", "std", "count"])
        print(summary)
        
        print("\nSummary by language pair:")
        summary_lang = final_df.groupby("lang_pair")["chrf_score"].agg(["mean", "std", "count"])
        print(summary_lang)
    else:
        print("\nNo results to save.")


if __name__ == "__main__":
    evaluate_all_translations(
        base_dir="generated_output",
        references_csv="translation_data/references.csv",
        output_csv="eval/results_non_commercial/chrf_evaluation_results.csv"
    )