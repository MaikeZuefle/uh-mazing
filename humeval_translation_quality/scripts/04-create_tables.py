import json
import os
import collections

LANGUAGES = ["zh", "es", "hi", "fr", "de", "it", "sw", "cs", "ar"]
SYSTEMS = ["gold_cascaded/tower", "end2end/canary", "speechllm/phi_multimodal", "end2end/gemini-2.5-flash"]

INPUT_JSON = "../annotations_human_scoring.json"
OUTPUT_DIR = "../../benchmarking/results/human_judge"


def load_ratings(input_json):
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # rows[(system, lang)] = list of (style, meaning) ratings
    rows = collections.defaultdict(list)
    for campaign, sessions in data.items():
        lang = campaign.split("_")[-1]
        for session in sessions:
            # form/tutorial entries have item lists of len != 1; "__RESET__" entries have no item at all
            if session.get("item") is None or len(session["item"]) != 1:
                continue
            annotation = session["annotation"][0]
            if not isinstance(annotation, dict):
                continue
            for system, model_kv in annotation.items():
                sliders = model_kv.get("sliders", {})
                style = sliders.get("Style preservation")
                meaning = sliders.get("Meaning preservation")
                if style is None or meaning is None:
                    continue
                rows[(system, lang)].append((style, meaning))
    return rows


def write_table(rows, metric_index, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("system," + ",".join(LANGUAGES) + "\n")
        for system in SYSTEMS:
            scores = []
            for lang in LANGUAGES:
                ratings = rows.get((system, lang), [])
                if not ratings:
                    scores.append("")
                else:
                    values = [r[metric_index] if metric_index is not None else (r[0] + r[1]) / 2 for r in ratings]
                    scores.append(f"{sum(values) / len(values):.2f}")
            f.write(f"{system}," + ",".join(scores) + "\n")


def create_tables(input_json=INPUT_JSON, output_dir=OUTPUT_DIR):
    rows = load_ratings(input_json)

    os.makedirs(output_dir, exist_ok=True)
    write_table(rows, 0, os.path.join(output_dir, "human_judge_style_table.csv"))
    write_table(rows, 1, os.path.join(output_dir, "human_judge_meaning_table.csv"))
    write_table(rows, None, os.path.join(output_dir, "human_judge_overall_table.csv"))

    print(f"Saved style/meaning/overall tables to {output_dir}")


if __name__ == "__main__":
    create_tables()
