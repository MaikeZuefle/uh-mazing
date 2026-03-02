import os
import pandas as pd
import re
from comet import download_model, load_from_checkpoint
from utils import load_csv_references, load_xml_hypotheses, clean_disfluency_text

# ----------------------------
# Load COMET-KIWI model
# ----------------------------
def load_comet_model(model_name="Unbabel/wmt22-cometkiwi-da"):
    """Load COMET-KIWI quality estimation model."""
    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path).to("cuda")

    return model

# ----------------------------
# Compute COMET-KIWI scores
# ----------------------------
def compute_comet_scores(csv_path, xml_path, model, source_lang, target_lang):
    """
    Compute COMET-KIWI quality estimation scores.
    
    Args:
        csv_path: Path to CSV with source texts (disfluent)
        xml_path: Path to XML with translations (hypotheses)
        model: Loaded COMET model
        source_lang: Source language code (e.g., 'en', 'de')
        target_lang: Target language code (e.g., 'de', 'en')
    
    Returns:
        System score (float) or None if no valid samples
    """
    # Load data
    refs_df = load_csv_references(csv_path)
    hyps_dict = load_xml_hypotheses(xml_path)
    
    # Prepare data for COMET
    sources = []
    translations = []
    

    for _, row in refs_df.iterrows():

        sample_id = row["sample_id"]
        if sample_id not in hyps_dict:
            continue
        
        source_text = clean_disfluency_text(row[f"{source_lang.upper()}_disfluent"])
        translation_text = hyps_dict[sample_id]
        sources.append(source_text)
        translations.append(translation_text)
    
    if not sources:
        return None
    
    # Prepare input for COMET-KIWI

    data = [{"src": src, "mt": mt} for src, mt in zip(sources, translations)]

    # Compute scores
    model_output = model.predict(data, batch_size=8, gpus=1)
    
    return model_output.system_score

# ----------------------------
# Extract language pair from filename
# ----------------------------
def extract_language_from_filename(filename):
    """
    Extract source and target language codes from filename.
    Expected formats:
    - en→x: *_en_LANG_*_output.xml (e.g., text_en_de_standard_llama_output.xml)
    - x→en: *_LANG_en_*_output.xml (e.g., text_de_en_standard_llama_output.xml)
    
    Returns:
        tuple: (source_lang, target_lang) or (None, None) if not found
    """
    # Try en→x pattern first
    match = re.search(r'_en_([a-z]{2})_', filename)
    if match:
        return ("en", match.group(1))
    
    # Try x→en pattern
    match = re.search(r'_([a-z]{2})_en_', filename)
    if match:
        return (match.group(1), "en")
    
    return (None, None)

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
def evaluate_all_translations(base_dir="generated_output", csv_path="utils_processing_data/translation-dataset-with-timestamps.csv", output_csv="comet_evaluation_results.csv"):
    """
    Evaluate all translation XML files in the directory structure.
    Writes results incrementally to CSV after each evaluation.
    
    Args:
        base_dir: Base directory containing model_type folders
        csv_path: Path to CSV with source texts
        output_csv: Output CSV file path
    """
    # Load COMET model once
    print("Loading COMET-KIWI model...")
    model = load_comet_model("Unbabel/wmt22-cometkiwi-da")
    
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
                
                xml_path = os.path.join(model_path, filename)
                
                print(f"    Evaluating {filename} (lang: {source_lang}→{target_lang}, prompt: {prompt_type})")
                
                comet_score = compute_comet_scores(csv_path, xml_path, model, source_lang, target_lang)
                
                if comet_score is not None:
                    # Create new result row
                    new_result = pd.DataFrame([{
                        "model_type": model_type,
                        "model": model_name,
                        "source_lang": source_lang,
                        "target_lang": target_lang,
                        "lang_pair": f"{source_lang}-{target_lang}",
                        "prompt_type": prompt_type,
                        "comet_score": comet_score,
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
                    print(f"      COMET score: {comet_score:.4f} (saved to {output_csv})")
                    
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
        summary = final_df.groupby("model_type")["comet_score"].agg(["mean", "std", "count"])
        print(summary)
        
        print("\nSummary by language pair:")
        summary_lang = final_df.groupby("lang_pair")["comet_score"].agg(["mean", "std", "count"])
        print(summary_lang)
    else:
        print("\nNo results to save.")

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    evaluate_all_translations(
        base_dir="benchmarking/generated_output",
        csv_path="translation_data/references.csv",
        output_csv="comet_evaluation_results.csv"
    )   