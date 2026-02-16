import pandas as pd

# Define the order of languages
LANGUAGES = ["zh", "es", "hi", "fr", "de", "it", "sw", "cs", "ar", "lg"]

# Define model order
MODELS = [
    "gold-gemini",
    "gold-chatgpt",
    "audio-gemini",
    "audio-chatgpt",
    "audio-noextra-gemini",
    "audio-noextra-chatgpt"
]

# Define prompt types
PROMPT_TYPES = ["standard", "disfluent"]

def normalize_score(score, score_name):
    """Normalize score values based on metric type."""
    if len(score) > 0:
        if "comet" in score_name:
            score = round(score[0]*100, 2)
        else:
            score = round(score[0], 2)
    else:
        score = None
    return score

def map_to_model_name(filename, modality, prompt_output):
    """
    Map filename components to standardized model name.
    
    Args:
        filename: Original filename
        modality: 'audio' or 'text'
        prompt_output: Combined prompt and output type (e.g., 'disfluent-noextra', 'standard_fluent')
    
    Returns:
        Tuple of (model_name, prompt_type)
    """
    # Determine base model
    if "gemini" in filename:
        base_model = "gemini"
    elif "gpt52" in filename or "gpt" in filename:
        base_model = "chatgpt"
    else:
        base_model = "unknown"
    
    # Determine model type and prompt type
    if modality == "text":
        # Text modality = gold cascaded
        model_name = f"gold-{base_model}"
        # Extract prompt type from prompt_output
        if "standard" in prompt_output:
            prompt_type = "standard"
        elif "disfluent" in prompt_output:
            prompt_type = "disfluent"
        else:
            prompt_type = "unknown"
    
    elif modality == "audio":
        # Audio modality
        if "noextra" in prompt_output:
            model_name = f"audio-noextra-{base_model}"
        else:
            model_name = f"audio-{base_model}"
        
        # Extract prompt type from prompt_output
        if "standard" in prompt_output:
            prompt_type = "standard"
        elif "disfluent" in prompt_output:
            prompt_type = "disfluent"
        else:
            prompt_type = "unknown"
    else:
        model_name = "unknown"
        prompt_type = "unknown"
    
    return model_name, prompt_type

def transform_results(input_csv, output_csv):
    """
    Transform evaluation results into a structured format for spreadsheet.
    
    Args:
        input_csv: Path to input CSV with evaluation results
        output_csv: Path to output CSV in structured format
    """
    # Load the data
    df = pd.read_csv(input_csv)
    
    # Determine score column name
    if "comet" in input_csv:
        score_name = "comet_score"
    elif "chrf" in input_csv:
        score_name = "chrf_score"
    else:
        # Try to detect from columns
        if "comet_score" in df.columns:
            score_name = "comet_score"
        elif "chrf_score" in df.columns:
            score_name = "chrf_score"
        else:
            raise ValueError("Could not determine score column name")
    
    # Add standardized model name and prompt type columns
    df[["model_name", "prompt_type"]] = df.apply(
        lambda row: pd.Series(map_to_model_name(row["filename"], row["modality"], row["prompt_output"])),
        axis=1
    )
    
    # Convert target_lang to lowercase for consistency
    df["lang"] = df["target_lang"].str.lower()
    
    # Create pivot table structure
    rows = []
    
    for model in MODELS:
        # Filter data for this model
        model_df = df[df["model_name"] == model]
        
        if len(model_df) == 0:
            continue
        
        # Create rows for each prompt type
        for prompt_type in PROMPT_TYPES:
            row = {"model": model, "prompt_type": prompt_type}
            
            # Add score for each language
            for lang in LANGUAGES:
                score = model_df[
                    (model_df["lang"] == lang) & 
                    (model_df["prompt_type"] == prompt_type)
                ][score_name].values
                
                row[lang] = normalize_score(score, score_name)
            
            rows.append(row)
    
    # Create DataFrame
    result_df = pd.DataFrame(rows)
    
    # Reorder columns
    columns = ["model", "prompt_type"] + LANGUAGES
    result_df = result_df[columns]
    
    # Save to CSV
    result_df.to_csv(output_csv, index=False)
    
    print(f"Transformed results saved to {output_csv}")
    print(f"\nPreview:")
    print(result_df.to_string())
    
    return result_df

if __name__ == "__main__":
    import os
    
    # Transform COMET results
    if os.path.exists("evaluation_results/comet_evaluation_results.csv"):
        print("="*60)
        print("Transforming COMET results...")
        print("="*60)
        transform_results(
            input_csv="evaluation_results/comet_evaluation_results.csv",
            output_csv="evaluation_results/comet_results_structured.csv"
        )
    
    print("\n")
    
    # Transform CHRF results
    if os.path.exists("evaluation_results/chrf_evaluation_results.csv"):
        print("="*60)
        print("Transforming CHRF++ results...")
        print("="*60)
        transform_results(
            input_csv="evaluation_results/chrf_evaluation_results.csv",
            output_csv="evaluation_results/chrf_results_structured.csv"
        )