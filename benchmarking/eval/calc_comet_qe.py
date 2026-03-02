import os
import pandas as pd
import xml.etree.ElementTree as ET
from comet import download_model, load_from_checkpoint
from utils import load_csv_references, load_xml_hypotheses, clean_disfluency_text, MODEL_TYPES, extract_language_from_filename, extract_prompt_type_from_filename

def load_comet_model(model_name="Unbabel/wmt22-cometkiwi-da"):
    model_path = download_model(model_name)
    return load_from_checkpoint(model_path).to("cuda")

def prepare_comet_data(csv_path, xml_path, source_lang):

    src_df = load_csv_references(csv_path)
    hyps_dict = load_xml_hypotheses(xml_path)
    
    sources = []; translations = []
    
    for _, row in src_df.iterrows():
        sample_id = row["sample_id"]
        if sample_id not in hyps_dict:
            continue
        
        source_text = clean_disfluency_text(row[f"{source_lang.upper()}_disfluent"])
        translation_text = hyps_dict[sample_id]
        sources.append(source_text)
        translations.append(translation_text)
    
    return sources, translations

def compute_comet_scores(model, sources, translations):
    data = [{"src": src, "mt": mt} for src, mt in zip(sources, translations)]
    scores =  model.predict(data, batch_size=8, gpus=1).system_score
    return scores
    

def evaluate_all_translations(base_dir="generated_output", csv_path="utils_processing_data/translation-dataset-with-timestamps.csv", output_csv="comet_evaluation_results.csv"):

    model = load_comet_model("Unbabel/wmt22-cometkiwi-da")
    
    # Check if output file exists to avoid re-evaluating
    if os.path.exists(output_csv):
        print(f"Output file {output_csv} already exists. Appending new results...")
        existing_df = pd.read_csv(output_csv)
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
        
        for model_name in os.listdir(model_type_path):
            model_path = os.path.join(model_type_path, model_name)
            if not os.path.isdir(model_path):
                continue
            print(f"  Processing model: {model_name}")

            for filename in os.listdir(model_path):
                if not filename.endswith(".xml"):
                    continue
                
                # Check if already evaluated
                file_key = f"{model_type}_{model_name}_{filename}"
                if file_key in evaluated_files:
                    print(f"    Skipping {filename} (already evaluated)")
                    continue
                
                source_lang, target_lang = extract_language_from_filename(filename)
                prompt_type = extract_prompt_type_from_filename(filename)
                xml_path = os.path.join(model_path, filename)
                print(f"    Evaluating {filename} (lang: {source_lang}→{target_lang}, prompt: {prompt_type})")
                

                sources, translations = prepare_comet_data(csv_path, xml_path, source_lang)
                comet_score = compute_comet_scores(model, sources, translations)
                
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
                

                if not os.path.exists(output_csv):
                    new_result.to_csv(output_csv, mode='w', header=True, index=False)
                else:
                    new_result.to_csv(output_csv, mode='a', header=False, index=False)
                
                results_count += 1
                evaluated_files.add(file_key)
                print(f"      COMET score: {comet_score:.4f} (saved to {output_csv})")
                
                
    
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


if __name__ == "__main__":
    evaluate_all_translations(
        base_dir="generated_output",
        csv_path="translation_data/references.csv",
        output_csv="eval/results_non_commercial/comet_evaluation_results.csv"
    )   