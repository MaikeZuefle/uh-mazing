import csv
import json
import os
import re

CSV_PATH = "../translation_data/references.csv"
AUDIO_DIR = "assets/translation_dataset_audio_preprocessed"
AUDIO_SRC_DIR = "../translation_data/translation_dataset_audio_preprocessed"
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

    # Fixed example item shown first to all annotators (taken from template)
    example_item = config["data"][0][0]

    # Build per-annotator item lists (mixed modalities):
    # Samples split into 3 groups; modality assignments rotate across annotator groups:
    #   Annotators 0-2: group0=audio,  group1=disfluent, group2=fluent
    #   Annotators 3-5: group0=disfluent, group1=fluent, group2=audio
    #   Annotators 6-8: group0=fluent,  group1=audio,  group2=disfluent
    # Each sample seen by 9 people total (3 per modality); each annotator sees all 3 modalities.
    n = len(samples)
    # sample group: 0 if i < n//3, 1 if i < 2*n//3, else 2
    modality_matrix = [
        ["audio",     "disfluent", "fluent"],    # annotator group 0 (0,1,2)
        ["disfluent", "fluent",    "audio"],      # annotator group 1 (3,4,5)
        ["fluent",    "audio",     "disfluent"],  # annotator group 2 (6,7,8)
    ]

    annotator_items = [[example_item] for _ in range(NUM_ANNOTATORS)]

    for i, (csv_id, audio_src, disfluent, fluent) in enumerate(samples):
        item_id = i + 1
        sample_group = 0 if i < n // 3 else (1 if i < 2 * n // 3 else 2)
        src_map = {"audio": audio_src, "disfluent": disfluent, "fluent": fluent}
        suffix_map = {"audio": "audio", "disfluent": "disfluent_transcript", "fluent": "fluent_transcript"}

        for ag in range(3):  # annotator group
            modality = modality_matrix[ag][sample_group]
            src = src_map[modality]
            suffix = suffix_map[modality]
            for a in range(ag * 3, ag * 3 + 3):
                annotator_items[a].append(make_item(src, f"{item_id}_{suffix}"))

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
            if iid == "example":
                pass
            elif iid.endswith("_audio"):
                modality_counts["audio"] += 1
            elif iid.endswith("_disfluent_transcript"):
                modality_counts["disfluent"] += 1
            elif iid.endswith("_fluent_transcript"):
                modality_counts["fluent"] += 1
        print(f"  Annotator {i}: {len(items)} items — {modality_counts}")
    print(f"Written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
