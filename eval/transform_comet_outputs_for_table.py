import pandas as pd

# Define the order of languages and models
LANGUAGES = ["zh", "es", "hi", "fr", "de", "it", "sw", "cs", "ar", "lg"]
MODELS = [
    "canary-llama",
    "canary-tower", 
    "whisper-llama",
    "whisper-tower",
    "gold-transcript-llama",
    "gold-transcript-tower",
    "canary",
    "whisper",
    "phi",
    "qwen"
]

def normalize_score(score):
    if len(score) > 0:
        score = round(score[0]*100, 2)
    else:
        score = None
    return score
                

def transform_results(input_csv, output_csv):
    """
    Transform COMET results into a structured format for spreadsheet.
    
    Args:
        input_csv: Path to input CSV with evaluation results
        output_csv: Path to output CSV in structured format
    """
    # Load the data
    df = pd.read_csv(input_csv)
    
    # Create a mapping from (model_type, model) to standardized model name
    model_mapping = {
        ("canary_cascaded", "llama"): "canary-llama",
        ("canary_cascaded", "tower"): "canary-tower",
        ("whisper_cascaded", "llama"): "whisper-llama",
        ("whisper_cascaded", "tower"): "whisper-tower",
        ("gold_cascaded", "llama"): "gold-transcript-llama",
        ("gold_cascaded", "tower"): "gold-transcript-tower",
        ("end2end", "canary"): "canary",
        ("end2end", "whisper"): "whisper",
        ("speechllm", "phi"): "phi",
        ("speechllm", "qwen"): "qwen",
    }
    
    # Add standardized model name column
    df["model_name"] = df.apply(
        lambda row: model_mapping.get((row["model_type"], row["model"]), f"{row['model_type']}-{row['model']}"),
        axis=1
    )
    
    # Create pivot table structure
    rows = []
    
    for model in MODELS:
        # Filter data for this model
        model_df = df[df["model_name"] == model]
        
        # Check if this model has prompt types (all except canary and whisper)
        has_prompt_types = model not in ["canary", "whisper"]
        
        if has_prompt_types:
            # Create two rows: one for standard, one for disfluency_aware
            for prompt_type in ["standard", "disfluency_aware"]:
                row = {"model": model, "prompt_type": prompt_type}
                
                # Add score for each language
                for lang in LANGUAGES:
                    score = model_df[
                        (model_df["lang"] == lang) & 
                        (model_df["prompt_type"] == prompt_type)
                    ]["comet_score"].values
                    
                    row[lang] = normalize_score(score)
                
                rows.append(row)
        else:
            # Single row for models without prompt types
            row = {"model": model, "prompt_type": "N/A"}
            
            # Add score for each language
            for lang in LANGUAGES:
                score = model_df[model_df["lang"] == lang]["comet_score"].values               
                row[lang] = row[lang] = normalize_score(score)
            
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
    # Transform the results
    transform_results(
        input_csv="comet_evaluation_results.csv",
        output_csv="comet_results_structured.csv"
    )