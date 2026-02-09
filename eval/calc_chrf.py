import os
import pandas as pd
import re
from sacrebleu.metrics import CHRF
from utils import load_xml_hypotheses, clean_disfluency_text

# ----------------------------
# Initialize CHRF++ metric
# ----------------------------
def init_chrf():
    """Initialize CHRF++ metric."""
    # CHRF++ uses word n-grams in addition to character n-grams
    return CHRF(word_order=2)  # word_order=2 enables CHRF++

# ----------------------------
# Compute CHRF++ scores
# ----------------------------
def compute_chrf_scores(references_csv, xml_path, lang):
    """
    Compute CHRF++ scores.
    
    Args:
        references_csv: Path to CSV with reference translations
        xml_path: Path to XML with translations (hypotheses)
        lang: Target language code (e.g., 'de', 'fr', 'ar')
    
    Returns:
        CHRF++ score (float) or None if no valid samples
    """
    # Load references
    refs_df = pd.read_csv(references_csv)
    
    # Load hypotheses
    hyps_dict = load_xml_hypotheses(xml_path)
    
    # Language code mapping to column names
    lang_column_map = {
        'ar': 'AR_disfluent',
        'cs': 'CS_disfluent',
        'de': 'DE_disfluent',
        'es': 'ES_disfluent',
        'fr': 'FR_disfluent',
        'hi': 'HI_disfluent',
        'it': 'IT_disfluent',
        'zh': 'ZH_disfluent',
        'sw': 'SW_disfluent',
        'lg': 'LG_disfluent'
    }

    if lang not in lang_column_map:
        print(f"Warning: Language {lang} not found in reference columns")
        return None
    
    ref_column = lang_column_map[lang]
    
    if ref_column not in refs_df.columns:
        print(f"Warning: Column {ref_column} not found in references CSV")
        return None
    
    # Prepare references and hypotheses
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
    
    # Compute CHRF++
    chrf = init_chrf()
    score = chrf.corpus_score(hypotheses, [references])
    
    return score.score

# ----------------------------
# Extract language from filename
# ----------------------------
def extract_language_from_filename(filename):
    """
    Extract target language code from filename.
    Expected format: *_en_LANG_*_output.xml
    Example: text_en_de_standard_llama_output.xml -> de
    """
    match = re.search(r'_en_([a-z]{2})_', filename)
    if match:
        return match.group(1)
    return None

# ----------------------------
# Extract prompt type from filename
# ----------------------------
def extract_prompt_type_from_filename(filename):
    """
    Extract prompt type from filename.
    Expected format: *_PROMPT_TYPE_*_output.xml
    Example: text_en_de_standard_llama_output.xml -> standard
    Example: text_en_de_disfluency_aware_llama_output.xml -> disfluency_aware
    """
    if "disfluency_aware" in filename:
        return "disfluency_aware"
    elif "standard" in filename:
        return "standard"
    return None

# ----------------------------
# Main evaluation loop
# ----------------------------
def evaluate_all_translations(base_dir="generated_output", references_csv="references.csv", output_csv="chrf_evaluation_results.csv"):
    """
    Evaluate all translation XML files in the directory structure using CHRF++.
    Writes results incrementally to CSV after each evaluation.
    
    Args:
        base_dir: Base directory containing model_type folders
        references_csv: Path to CSV with reference translations
        output_csv: Output CSV file path
    """
    print("Using CHRF++ metric...")
    
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
    
    # Iterate through model_type folders (first level)
    model_types = ["canary_cascaded", "end2end", "gold_cascaded", "speechllm", "whisper_cascaded"]
    
    for model_type in model_types:
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
                
                # Extract language from filename
                lang = extract_language_from_filename(filename)
                if lang is None:
                    print(f"    Warning: Could not extract language from {filename}")
                    continue
                
                # Extract prompt type from filename
                prompt_type = extract_prompt_type_from_filename(filename)
                if prompt_type is None:
                    print(f"    Warning: Could not extract prompt type from {filename}")
                    continue
                
                xml_path = os.path.join(model_path, filename)
                
                print(f"    Evaluating {filename} (lang: {lang}, prompt: {prompt_type})")

                chrf_score = compute_chrf_scores(references_csv, xml_path, lang)
                
                if chrf_score is not None:
                    # Create new result row
                    new_result = pd.DataFrame([{
                        "model_type": model_type,
                        "model": model_name,
                        "lang": lang,
                        "prompt_type": prompt_type,
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
    else:
        print("\nNo results to save.")

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    evaluate_all_translations(
        base_dir="generated_output",
        references_csv="eval/references.csv",
        output_csv="chrf_evaluation_results.csv"
    )