import pandas as pd

# Define the order of languages and models
LANGUAGES_EN_X = ["zh", "es", "hi", "fr", "de", "it", "sw", "cs", "ar", "lg"]
LANGUAGES_X_EN = ["zh", "es", "hi", "fr", "de", "it", "sw", "cs", "ar", "lg"]

MODELS_EN_X = [
    "canary-llama",
    "canary-tower",
    "whisper-llama",
    "whisper-tower",
    "gold-transcript-llama",
    "gold-transcript-tower",
    "owsm",
    "canary",
    "phi",
    "qwen",
    "gold-text-gemini",
    "gold-text-chatgpt",
    "asr-text-gemini",
    "asr-text-chatgpt",
    "audio-gemini",
    "audio-chatgpt",
    "audio-noextra-gemini",
    "audio-noextra-chatgpt",
]

MODELS_X_EN = [
    "gold-transcript-llama",
    "gold-transcript-tower",
    "gold-transcript-chatgpt",
    "gold-transcript-gemini",
]

def normalize_score(score, score_name):
    if len(score) > 0:
        if "comet" in score_name:
            score = round(score[0]*100, 2)
        else:
            score = round(score[0], 2)
    else:
        score = None
    return score

def transform_results(input_csv, output_csv, metric=None):
    """
    Transform evaluation results into a structured format for spreadsheet.
    Handles both en→x and x→en translations.

    Args:
        input_csv: Path to input CSV with evaluation results
        output_csv: Path to output CSV in structured format
        metric: "comet" or "chrf" to override auto-detection
    """
    # Load the data
    df = pd.read_csv(input_csv)

    # Determine score column name
    if metric == "comet":
        score_name = "comet_score"
    elif metric == "chrf":
        score_name = "chrf_score"
    elif "comet" in input_csv:
        score_name = "comet_score"
    elif "chrf" in input_csv:
        score_name = "chrf_score"
    elif "comet_score" in df.columns:
        score_name = "comet_score"
    elif "chrf_score" in df.columns:
        score_name = "chrf_score"
    else:
        raise ValueError("Could not determine score column name")
    
    # Create a mapping from (model_type, model) to standardized model name
    model_mapping = {
        ("canary_cascaded", "llama"): "canary-llama",
        ("canary_cascaded", "tower"): "canary-tower",
        ("whisper_cascaded", "llama"): "whisper-llama",
        ("whisper_cascaded", "tower"): "whisper-tower",
        ("gold_cascaded", "llama"): "gold-transcript-llama",
        ("gold_cascaded", "tower"): "gold-transcript-tower",
        ("gold_cascaded", "chatgpt"): "gold-transcript-chatgpt",
        ("gold_cascaded", "gemini"): "gold-transcript-gemini",
        ("end2end", "canary"): "canary",
        ("end2end", "owsm"): "owsm",
        ("speechllm", "phi_multimodal"): "phi",
        ("speechllm", "qwen_omni"): "qwen",
    }
    
    COMMERCIAL_SHORT = {"gemini-2.5-flash": "gemini", "gpt52": "chatgpt"}

    def get_model_name(row):
        if row["model_type"] == "commercial_models":
            short = COMMERCIAL_SHORT.get(row["model"], row["model"])
            pt = row["prompt_type"]
            if row["modality"] == "text":
                return f"gold-text-{short}" if pt.endswith("_fluent") else f"asr-text-{short}"
            else:  # audio
                return f"audio-noextra-{short}" if "noextra" in pt else f"audio-{short}"
        return model_mapping.get((row["model_type"], row["model"]), f"{row['model_type']}-{row['model']}")

    def normalize_prompt_type(row):
        if row["model_type"] == "commercial_models":
            return "standard" if row["prompt_type"].startswith("standard") else "disfluency_aware"
        return row["prompt_type"]

    df["model_name"] = df.apply(get_model_name, axis=1)
    df["prompt_type"] = df.apply(normalize_prompt_type, axis=1)
    
    # Separate en→x and x→en translations
    # Check if we have the new format with source_lang and target_lang columns
    if "source_lang" in df.columns and "target_lang" in df.columns:
        df_en_x = df[df["source_lang"] == "en"].copy()
        df_x_en = df[df["target_lang"] == "en"].copy()
        # For en→x, use target_lang as "lang"
        df_en_x["lang"] = df_en_x["target_lang"]
        # For x→en, use source_lang as "lang"
        df_x_en["lang"] = df_x_en["source_lang"]
    else:
        # Old format - assume all are en→x
        df_en_x = df.copy()
        df_x_en = pd.DataFrame()
    
    # Create pivot table structure
    rows = []
    
    # ==========================================
    # SECTION 1: en→x translations
    # ==========================================
    rows.append({"model": "=== EN → X TRANSLATIONS ===", "prompt_type": "", **{lang: "" for lang in LANGUAGES_EN_X}})
    
    for model in MODELS_EN_X:
        # Filter data for this model
        model_df = df_en_x[df_en_x["model_name"] == model]
        
        if len(model_df) == 0:
            continue
        
        has_prompt_types = model not in ["canary", "owsm"]
        
        if has_prompt_types:
            # Create two rows: one for standard, one for disfluency_aware
            for prompt_type in ["standard", "disfluency_aware"]:
                row = {"model": model, "prompt_type": prompt_type}
                
                # Add score for each language
                for lang in LANGUAGES_EN_X:
                    score = model_df[
                        (model_df["lang"] == lang) & 
                        (model_df["prompt_type"] == prompt_type)
                    ][score_name].values
                    
                    row[lang] = normalize_score(score, score_name)
                
                rows.append(row)
        else:
            # Single row for models without prompt types
            row = {"model": model, "prompt_type": "N/A"}
            
            # Add score for each language
            for lang in LANGUAGES_EN_X:
                score = model_df[model_df["lang"] == lang][score_name].values
                row[lang] = normalize_score(score, score_name)
            
            rows.append(row)
    
    # ==========================================
    # SECTION 2: x→en translations
    # ==========================================
    if len(df_x_en) > 0:
        # Add separator row
        rows.append({"model": "", "prompt_type": "", **{lang: "" for lang in LANGUAGES_X_EN}})
        rows.append({"model": "=== X → EN TRANSLATIONS ===", "prompt_type": "", **{lang: "" for lang in LANGUAGES_X_EN}})
        
        for model in MODELS_X_EN:
            # Filter data for this model
            model_df = df_x_en[df_x_en["model_name"] == model]
            
            if len(model_df) == 0:
                continue
            
            # Models without prompt type variants (single row each)
            has_prompt_types = model not in [
                "canary", "owsm", "phi", "qwen",
                "gemini-2.5-flash-audio", "gemini-2.5-flash-text",
                "gpt52-audio", "gpt52-text",
            ]
            
            if has_prompt_types:
                # Create two rows: one for standard, one for disfluency_aware
                for prompt_type in ["standard", "disfluency_aware"]:
                    row = {"model": model, "prompt_type": prompt_type}
                    
                    # Add score for each language
                    for lang in LANGUAGES_X_EN:
                        score = model_df[
                            (model_df["lang"] == lang) & 
                            (model_df["prompt_type"] == prompt_type)
                        ][score_name].values
                        
                        row[lang] = normalize_score(score, score_name)
                    
                    rows.append(row)
            else:
                # Single row for models without prompt types
                row = {"model": model, "prompt_type": "N/A"}
                
                # Add score for each language
                for lang in LANGUAGES_X_EN:
                    score = model_df[model_df["lang"] == lang][score_name].values
                    row[lang] = normalize_score(score, score_name)
                
                rows.append(row)
    
    # Create DataFrame
    result_df = pd.DataFrame(rows)
    
    # Reorder columns
    columns = ["model", "prompt_type"] + LANGUAGES_EN_X
    result_df = result_df[columns]
    
    # Save to CSV
    result_df.to_csv(output_csv, index=False)
    
    print(f"Transformed results saved to {output_csv}")
    print(f"\nPreview:")
    print(result_df.to_string())
    
    return result_df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Transform evaluation results into structured table format.")
    parser.add_argument("--input", required=True, help="Path to input CSV with evaluation results")
    parser.add_argument("--output", required=True, help="Path to output structured CSV")
    parser.add_argument("--metric", choices=["comet", "chrf"], default=None,
                        help="Metric to use (overrides auto-detection from filename/columns)")
    args = parser.parse_args()
    transform_results(input_csv=args.input, output_csv=args.output, metric=args.metric)