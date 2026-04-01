import os
import argparse
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, RemoveMultipleSpaces, Strip, ExpandCommonEnglishContractions
from utils import load_csv_references, load_xml_hypotheses, clean_disfluency_text, append_result

normalizer = Compose([
    ToLowerCase(),
    ExpandCommonEnglishContractions(),
    RemovePunctuation(),
    RemoveMultipleSpaces(),
    Strip()
])


def compute_wers(csv_path, xml_path):
    refs_df = load_csv_references(csv_path)
    hyps_dict = load_xml_hypotheses(xml_path)

    fluent_wers = []
    disfluent_wers = []

    for _, row in refs_df.iterrows():
        sample_id = row["sample_id"]
        if sample_id not in hyps_dict:
            raise ValueError(f"Sample ID '{sample_id}' not found in XML hypotheses")

        hyp_text = normalizer(hyps_dict[sample_id])
        fluent_wers.append(wer(normalizer(row["EN_fluent"]), hyp_text))
        disfluent_wers.append(wer(normalizer(clean_disfluency_text(row["EN_disfluent"])), hyp_text))

    return {
        "fluent_wer": sum(fluent_wers) / len(fluent_wers) if fluent_wers else None,
        "disfluent_wer": sum(disfluent_wers) / len(disfluent_wers) if disfluent_wers else None,
        "num_samples": len(fluent_wers),
    }


def evaluate_all_wer(base_dir, csv_path, output_csv):
    for model_name in os.listdir(base_dir):
        model_path = os.path.join(base_dir, model_name)
        if not os.path.isdir(model_path):
            raise NotADirectoryError(f"Expected directory but found file: {model_path}")

        print(f"Processing model: {model_name}")

        for filename in os.listdir(model_path):
            if not filename.endswith(".xml"):
                raise ValueError(f"Unexpected file type in {model_path}: {filename}")

            xml_path = os.path.join(model_path, filename)
            print(f"  Evaluating {filename}")

            results = compute_wers(csv_path, xml_path)
            append_result(output_csv, {
                "model": model_name,
                "filename": filename,
                "fluent_wer": results["fluent_wer"],
                "disfluent_wer": results["disfluent_wer"],
                "num_samples": results["num_samples"],
            })
            print(f"    fluent WER: {results['fluent_wer']:.4f}, disfluent WER: {results['disfluent_wer']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ASR transcriptions with WER.")
    parser.add_argument("--input", required=True, help="Directory containing transcript model subdirs (e.g. generated_output/transcript)")
    parser.add_argument("--references", required=True, help="Path to references CSV")
    parser.add_argument("--output", required=True, help="Path to output CSV (e.g. results/wer_results.csv)")
    args = parser.parse_args()
    evaluate_all_wer(
        base_dir=args.input,
        csv_path=args.references,
        output_csv=args.output,
    )
