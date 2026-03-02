import pandas as pd
import xml.etree.ElementTree as ET
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, RemoveMultipleSpaces, Strip, ExpandCommonEnglishContractions
from utils import load_csv_references, load_xml_hypotheses, clean_disfluency_text

# ----------------------------
# Text normalization
# ----------------------------
normalizer = Compose([
    ToLowerCase(),
    ExpandCommonEnglishContractions(),
    RemovePunctuation(),
    RemoveMultipleSpaces(),
    Strip()
])



# ----------------------------
# Compute WERs
# ----------------------------
def compute_wers(csv_path, xml_path):
    refs_df = load_csv_references(csv_path)  # CSV is reference
    hyps_dict = load_xml_hypotheses(xml_path)  # XML is hypothesis

    fluent_wers = []
    disfluent_wers = []

    for _, row in refs_df.iterrows():
        sample_id = row["sample_id"]

        if sample_id not in hyps_dict:
            # skip missing samples
            continue

        hyp_text = normalizer(hyps_dict[sample_id])

        fluent_ref = normalizer(row["text_fluent"])
        disfluent_ref = normalizer(clean_disfluency_text(row["text_disfluent"]))

        # WER(reference, hypothesis)
        fluent_wers.append(wer(fluent_ref, hyp_text))
        disfluent_wers.append(wer(disfluent_ref, hyp_text))

    return {
        "fluent_WER": sum(fluent_wers) / len(fluent_wers) if fluent_wers else None,
        "disfluent_WER": sum(disfluent_wers) / len(disfluent_wers) if disfluent_wers else None,
        "num_samples": len(fluent_wers)
    }

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    results = compute_wers(
        csv_path="translation_data/translation-dataset-with-timestamps.csv",
        xml_path="benchmarking/generated_output/transcript/canary/audio_en_en_standard_canary_output.xml"
    )
    # generated_output/transcript/whisper/audio_en_en_standard_whisper_output.xml
    # generated_output/transcript/canary/audio_en_en_standard_canary_output.xml

    print("Samples evaluated:", results["num_samples"])
    print("WER (ref: text_fluent):", results["fluent_WER"])
    print("WER (ref: text_disfluent):", results["disfluent_WER"])