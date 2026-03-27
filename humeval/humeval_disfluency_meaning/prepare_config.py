import csv
import json
import os
import re

CSV_PATH = "/net/scratch/hscra/plgrid/plgzuefle/disfluent/translation_data/references.csv"
AUDIO_DIR = "assets/translation_dataset_audio_preprocessed"
AUDIO_SRC_DIR = "/net/scratch/hscra/plgrid/plgzuefle/disfluent/translation_data/translation_dataset_audio_preprocessed"
TEMPLATE_PATH = "disfl_meaning_config.json"
OUTPUT_PATH = "disfl_meaning_config_filled.json"
NUM_ANNOTATORS = 9


def csv_id_to_audio_filename(csv_id):
    # sw2005_B_2 -> sw02005_B_2.wav
    parts = csv_id.split("_", 1)  # ["sw2005", "B_2"]
    prefix = parts[0]             # "sw2005"
    rest = parts[1]               # "B_2"
    number = prefix[2:]           # "2005"
    padded = "sw0" + number       # "sw02005"
    return f"{padded}_{rest}.wav"


def clean_text(text):
    text = text.replace('_', '')
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\s+,', ',', text)
    text = re.sub(r'\s+\.', '.', text)
    text = re.sub(r"\s+'", "'", text)
    return text


def make_item(src, item_id):
    return {
        "src": src,
        "tgt": {"a": "", "b": "", "c": "", "d": "", "e": ""},
        "item_id": item_id,
    }


def main():
    with open(TEMPLATE_PATH) as f:
        config = json.load(f)

    audio_files = set(os.listdir(AUDIO_SRC_DIR))

    # Collect all samples
    samples = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            csv_id = row["ID"].strip()
            fluent = clean_text(row["EN_fluent"])
            disfluent = clean_text(row["EN_disfluent"])

            audio_filename = csv_id_to_audio_filename(csv_id)
            if audio_filename not in audio_files:
                print(f"WARNING: audio file not found for {csv_id} ({audio_filename})")
                continue

            audio_src = f'<audio controls src="{AUDIO_DIR}/{audio_filename}"></audio><br>'
            samples.append((csv_id, audio_src, disfluent, fluent))

    # Build per-annotator item lists using rotation:
    # sample i -> audio to annotator i%9, disfluent to (i+3)%9, fluent to (i+6)%9
    annotator_items = [[] for _ in range(NUM_ANNOTATORS)]

    for i, (csv_id, audio_src, disfluent, fluent) in enumerate(samples):
        item_id = i + 1
        annotator_items[i % NUM_ANNOTATORS].append(
            make_item(audio_src, f"{item_id}_audio")
        )
        annotator_items[(i + 3) % NUM_ANNOTATORS].append(
            make_item(disfluent, f"{item_id}_disfluent_transcript")
        )
        annotator_items[(i + 6) % NUM_ANNOTATORS].append(
            make_item(fluent, f"{item_id}_fluent_transcript")
        )

    # task-based format: data[annotator][document][item]
    # each item is its own document (single-item document)
    data = [[[item] for item in items] for items in annotator_items]

    config["data"] = data

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"Loaded {len(samples)} samples")
    for i, items in enumerate(annotator_items):
        modality_counts = {"audio": 0, "disfluent": 0, "fluent": 0}
        for item in items:
            iid = item["item_id"]
            if iid.endswith("_audio"):
                modality_counts["audio"] += 1
            elif iid.endswith("_disfluent_transcript"):
                modality_counts["disfluent"] += 1
            elif iid.endswith("_fluent_transcript"):
                modality_counts["fluent"] += 1
        print(f"  Annotator {i}: {len(items)} items — {modality_counts}")
    print(f"Written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
