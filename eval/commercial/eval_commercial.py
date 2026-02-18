import os
import pandas as pd
import re
from sacrebleu.metrics import CHRF
from comet import download_model, load_from_checkpoint
from jiwer import Compose, RemoveMultipleSpaces, Strip

# ----------------------------
# Text cleaning
# ----------------------------
normalizer = Compose([
    RemoveMultipleSpaces(),
    Strip()
])

def clean_disfluency_text(text: str) -> str:
    """
    Remove underscore-based disfluency markers.
    Example: _uh_ -> uh
    """
    if pd.isna(text):
        return ""
    return normalizer(str(text).replace("_", "").replace("ـ", ""))

# ----------------------------
# Initialize metrics
# ----------------------------
def init_chrf():
    """Initialize CHRF++ metric."""
    return CHRF(word_order=2)  # word_order=2 enables CHRF++

def load_comet_model(model_name="Unbabel/wmt22-cometkiwi-da"):
    """Load COMET-KIWI quality estimation model."""
    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)
    return model

# ----------------------------
# Compute CHRF++ scores
# ----------------------------
def compute_chrf_score(csv_path, target_lang_code):
    """
    Compute CHRF++ score for a given CSV file and target language.
    
    Args:
        csv_path: Path to CSV with translations
        target_lang_code: Target language code (e.g., 'ZH', 'ES', 'HI', etc.)
    
    Returns:
        CHRF++ score (float) or None if no valid samples
    """
    df = pd.read_csv(csv_path)
    
    # Column names
    ref_column = f"{target_lang_code}_disfluent"
    hyp_column = f"{target_lang_code}_gemini-2.5-flash" if "gemini" in csv_path else f"{target_lang_code}_gpt52"
    
    # Check if columns exist
    if ref_column not in df.columns:
        print(f"  Warning: Reference column {ref_column} not found")
        return None
    if hyp_column not in df.columns:
        print(f"  Warning: Hypothesis column {hyp_column} not found")
        return None
    
    # Prepare references and hypotheses
    references = []
    hypotheses = []
    
    for _, row in df.iterrows():
        ref_text = clean_disfluency_text(row[ref_column])
        hyp_text = str(row[hyp_column]) if not pd.isna(row[hyp_column]) else ""
        
        # Skip if reference is empty
        if ref_text.strip() == "":
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
# Compute COMET scores
# ----------------------------
def compute_comet_score(csv_path, target_lang_code, model):
    """
    Compute COMET-KIWI score for a given CSV file and target language.
    
    Args:
        csv_path: Path to CSV with translations
        target_lang_code: Target language code (e.g., 'ZH', 'ES', 'HI', etc.)
        model: Loaded COMET model
    
    Returns:
        COMET score (float) or None if no valid samples
    """
    df = pd.read_csv(csv_path)
    
    # Column names
    source_column = "EN_disfluent"
    hyp_column = f"{target_lang_code}_gemini-2.5-flash" if "gemini" in csv_path else f"{target_lang_code}_gpt52"
    
    # Check if columns exist
    if source_column not in df.columns:
        print(f"  Warning: Source column {source_column} not found")
        return None
    if hyp_column not in df.columns:
        print(f"  Warning: Hypothesis column {hyp_column} not found")
        return None
    
    # Prepare data for COMET
    sources = []
    translations = []
    
    for _, row in df.iterrows():
        src_text = clean_disfluency_text(row[source_column])
        mt_text = str(row[hyp_column]) if not pd.isna(row[hyp_column]) else ""
        
        # Skip if source is empty
        if src_text.strip() == "":
            continue
        
        sources.append(src_text)
        translations.append(mt_text)
    
    if not sources:
        return None
    
    # Prepare input for COMET-KIWI
    data = [{"src": src, "mt": mt} for src, mt in zip(sources, translations)]
    
    # Compute scores
    model_output = model.predict(data, batch_size=8, gpus=1)
    
    return model_output.system_score

# ----------------------------
# Parse filename
# ----------------------------
def parse_filename(filename):
    """
    Parse filename to extract metadata.
    Format: uh-mazing_{model}_{modality}_{prompt_type}[_output_type].csv
    
    Examples:
    - uh-mazing_gemini-2.5-flash_audio_disfluent-noextra.csv
    - uh-mazing_gpt52_text_standard_fluent.csv
    
    Returns:
        dict with keys: model, modality, prompt_type, output_type (or full_prompt_output)
    """
    parts = filename.replace("uh-mazing_", "").replace(".csv", "").split("_")
    
    if len(parts) < 3:
        return None
    
    model = parts[0]
    modality = parts[1]
    
    # The rest could be prompt_type + output_type or just a combined string
    remaining = "_".join(parts[2:])
    
    return {
        "model": model,
        "modality": modality,
        "prompt_output": remaining
    }

# ----------------------------
# Main evaluation function
# ----------------------------
def evaluate_all_files(files_dir=".", output_dir="evaluation_results"):
    """
    Evaluate all CSV files in the directory using COMET and CHRF++.
    
    Args:
        files_dir: Directory containing the CSV files
        output_dir: Directory to save evaluation results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Language codes for the models
    lang_codes = ["ZH", "ES", "HI", "FR", "DE", "IT", "SW", "CZ", "AR", "LG"]
    
    # Find all CSV files
    csv_files = [f for f in os.listdir(files_dir) if f.startswith("uh-mazing_") and f.endswith(".csv")]
    
    if not csv_files:
        print("No CSV files found!")
        return
    
    print(f"Found {len(csv_files)} CSV files to evaluate")
    print("="*60)
    
    # Load COMET model once
    print("\nLoading COMET-KIWI model...")
    comet_model = load_comet_model()
    print("COMET model loaded!")
    
    # Store all results
    all_chrf_results = []
    all_comet_results = []
    
    # Evaluate each file
    for csv_file in sorted(csv_files):
        csv_path = os.path.join(files_dir, csv_file)
        
        print(f"\n{'='*60}")
        print(f"Evaluating: {csv_file}")
        print(f"{'='*60}")
        
        # Parse filename
        metadata = parse_filename(csv_file)
        if metadata is None:
            print(f"  Skipping {csv_file} (cannot parse filename)")
            continue
        
        # Evaluate each language
        for lang_code in lang_codes:
            print(f"\n  Language: {lang_code}")
            
            # Compute CHRF++
            chrf_score = compute_chrf_score(csv_path, lang_code)
            if chrf_score is not None:
                print(f"    CHRF++: {chrf_score:.4f}")
                all_chrf_results.append({
                    "filename": csv_file,
                    "model": metadata["model"],
                    "modality": metadata["modality"],
                    "prompt_output": metadata["prompt_output"],
                    "target_lang": lang_code,
                    "lang_pair": f"EN-{lang_code}",
                    "chrf_score": chrf_score
                })
            
            # Compute COMET
            comet_score = compute_comet_score(csv_path, lang_code, comet_model)
            if comet_score is not None:
                print(f"    COMET:  {comet_score:.4f}")
                all_comet_results.append({
                    "filename": csv_file,
                    "model": metadata["model"],
                    "modality": metadata["modality"],
                    "prompt_output": metadata["prompt_output"],
                    "target_lang": lang_code,
                    "lang_pair": f"EN-{lang_code}",
                    "comet_score": comet_score
                })
    
    # Save results
    if all_chrf_results:
        chrf_df = pd.DataFrame(all_chrf_results)
        chrf_output = os.path.join(output_dir, "chrf_evaluation_results.csv")
        chrf_df.to_csv(chrf_output, index=False)
        print(f"\n{'='*60}")
        print(f"CHRF++ results saved to: {chrf_output}")
        print(f"Total evaluations: {len(chrf_df)}")
        
        # Print summary
        print("\nCHRF++ Summary by model:")
        print(chrf_df.groupby("model")["chrf_score"].agg(["mean", "std", "count"]))
        
        print("\nCHRF++ Summary by modality:")
        print(chrf_df.groupby("modality")["chrf_score"].agg(["mean", "std", "count"]))
    
    if all_comet_results:
        comet_df = pd.DataFrame(all_comet_results)
        comet_output = os.path.join(output_dir, "comet_evaluation_results.csv")
        comet_df.to_csv(comet_output, index=False)
        print(f"\n{'='*60}")
        print(f"COMET results saved to: {comet_output}")
        print(f"Total evaluations: {len(comet_df)}")
        
        # Print summary
        print("\nCOMET Summary by model:")
        print(comet_df.groupby("model")["comet_score"].agg(["mean", "std", "count"]))
        
        print("\nCOMET Summary by modality:")
        print(comet_df.groupby("modality")["comet_score"].agg(["mean", "std", "count"]))
    
    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"{'='*60}")

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # Evaluate all files in current directory
    evaluate_all_files(
        files_dir=".",
        output_dir="evaluation_results"
    )