import json
from xml.etree import ElementTree
import collections
import os
import random

os.makedirs("humeval/data/campaigns", exist_ok=True)
with open("humeval/data/data_welcome.json", "r") as f:
    data_welcome = json.load(f)
langs = ["ar", "cs", "de", "es", "fr", "hi", "it", "zh"]
models = ["speechllm/phi_multimodal", "end2end/canary", "gold_cascaded/tower"]

data_gold = {}
with open("humeval/data/audio_en_en_standard_gold_output.xml", "r") as f:
    tree = ElementTree.parse(f)
    tasks = tree.findall("task")
    for task in tasks:
        samples = task.findall("sample")
        for sample in samples:
            data_gold[sample.get("id")] = sample.text

for lang in langs:
    data = collections.defaultdict(dict)
    _models = list(models)
    if lang in {"ar", "zh", "hi"}:
        _models.remove("end2end/canary")
    for model in _models:
        model_sub = model.split("/")[1]
        for fname in [
            f"humeval/data/generated_output_non_commercial/{model}/text_en_{lang}_standard_{model_sub}_output.xml",
            f"humeval/data/generated_output_non_commercial/{model}/audio_en_{lang}_standard_{model_sub}_output.xml",
        ]:
            if os.path.exists(fname):
                break
        else:
            raise Exception(f"{fname} not found")
        tree = ElementTree.parse(fname)
        tasks = tree.findall("task")
        for task in tasks:
            samples = task.findall("sample")
            for sample in samples:
                data[sample.get("id")][model] = sample.text

    data_out = []
    for item_id, item in data.items():
        median_len = len(sorted(item.values(), key=len)[len(item) // 2])
        data_out.append(
            [
                {
                    "src": f'<audio controls src="assets/preprocessed_audio_switchboard/{item_id}"></audio><br>{data_gold[item_id]}',
                    "tgt": {model: item[model][: 2 * median_len] for model in _models},
                    "item_id": item_id,
                }
            ]
        )

    random.Random(0).shuffle(data_out)
    # chunk to size 20
    data_out = [data_out[i : i + 20] for i in range(0, len(data_out), 20)]

    campaign = {
        "info": {
            "assignment": "task-based",
            "protocol": "MQM",
            "mqm_categories": {
                "Disfluency": [
                    "Deletion",
                    "Extra disfluency",
                    "Type mismatch",
                    "Other",
                ],
                "General": [
                    "Incorrect language",
                    "Missing content",
                    "Looping",
                    "Extra content",
                ],
                "Other": [],
            },
            "sliders": [
                {"name": "Style preservation", "min": 0, "max": 100, "step": 1},
                {"name": "Meaning preservation", "min": 0, "max": 100, "step": 1},
            ],
            "instructions": "TODO",
        },
        "campaign_id": f"uhmazing_v0_{lang}.json",
        # "data_welcome": data_welcome,
        "data": data_out,
    }
    # assets can be attached to only one
    if lang == "cs":
        campaign["assets"] = {
            "source": "humeval/data/preprocessed_audio_switchboard",
            "destination": "assets/preprocessed_audio_switchboard",
        }
    with open(f"humeval/data/campaigns/{lang}.json", "w") as f:
        json.dump(campaign, f, ensure_ascii=False, indent=2)
