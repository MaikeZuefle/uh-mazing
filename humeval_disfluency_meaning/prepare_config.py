import csv
import json
import os
import re

CSV_PATH = "../translation_data/references.csv"
AUDIO_DIR = "assets/translation_dataset_audio_preprocessed"
AUDIO_SRC_DIR = "../translation_data/translation_dataset_audio_preprocessed"
TEMPLATE_PATH = "disfl_meaning_config.json"
OUTPUT_PATH = "disfl_meaning_config_filled.json"
NUM_ANNOTATORS = 18


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
    if NUM_ANNOTATORS % 9 != 0:
        raise ValueError(f"NUM_ANNOTATORS must be a multiple of 9, got {NUM_ANNOTATORS}")

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

    # Design: NUM_ANNOTATORS // 9 replica groups, each of 9 annotators.
    # Each replica handles a separate chunk of samples using a 3x3 mixed modality design:
    #   Annotator sub-groups 0,1,2 (3 each) × sample sub-groups 0,1,2 → Latin square modality rotation.
    # Result: each sample seen by 3 annotators per modality (9 total), each annotator sees all 3 modalities.
    num_replicas = NUM_ANNOTATORS // 9
    modality_matrix = [
        ["audio",     "disfluent", "fluent"],
        ["disfluent", "fluent",    "audio"],
        ["fluent",    "audio",     "disfluent"],
    ]
    suffix_map = {"audio": "audio", "disfluent": "disfluent_transcript", "fluent": "fluent_transcript"}

    annotator_items = [[example_item] for _ in range(NUM_ANNOTATORS)]

    for r in range(num_replicas):
        # Samples assigned to this replica
        start = r * len(samples) // num_replicas
        end = (r + 1) * len(samples) // num_replicas if r < num_replicas - 1 else len(samples)
        replica_samples = samples[start:end]
        n_r = len(replica_samples)
        base_a = r * 9  # first annotator index in this replica

        for i, (csv_id, audio_src, disfluent, fluent) in enumerate(replica_samples):
            item_id = start + i + 1  # global 1-based item id
            sample_group = 0 if i < n_r // 3 else (1 if i < 2 * n_r // 3 else 2)
            src_map = {"audio": audio_src, "disfluent": disfluent, "fluent": fluent}

            for ag in range(3):  # annotator sub-group within replica
                modality = modality_matrix[ag][sample_group]
                src = src_map[modality]
                suffix = suffix_map[modality]
                for a in range(base_a + ag * 3, base_a + ag * 3 + 3):
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
