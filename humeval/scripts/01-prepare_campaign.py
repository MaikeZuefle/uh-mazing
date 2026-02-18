import json
from xml.etree import ElementTree
import collections
import os
import random

os.makedirs("humeval/data/campaigns", exist_ok=True)
langs = ["ar", "cs", "de", "es", "fr", "hi", "it", "zh"]
lang2name = {
    "ar": "Arabic",
    "cs": "Czech",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "it": "Italian",
    "zh": "Chinese",
}
models = [
    "speechllm/phi_multimodal",
    "end2end/canary",
    "gold_cascaded/tower",
    "end2end/gemini-2.5-flash",
]

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
        if lang == "ar" and not all(model in item for model in _models):
            print("Skipping", item_id)
            continue
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

    with open("humeval/data/data_welcome.json", "r") as f:
        data_welcome = json.loads(f.read().replace("LANG2", lang2name[lang]))

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
            "instructions": """
We have a speech audio and its translation. The speech audio contains disfluencies, such as "uh," "um," repeated words, self-corrections, etc.
Your task is to evaluate both:
<ul>
<li>How well the disfluencies are preserved in the translation</li>
<li>How natural these disfluencies sound in the target language</li>
</ul>
Specifically, you will:
<ul>
<li>Use sliders (0-100) to rate preservation of Style and Meaning</li>
<li>If you notice an error, please annotate one of the following error categories:

For general errors that affect text and translation quality, you can select:

<ul>
<li>General > Incorrect language: Wrong language used</li>
<li>General > Missing content: Part of the source content was not translated</li>
<li>General > Extra content: Content was added that is not in the source</li>
<li>General > Looping: Unintended repetition in the translation</li>
</ul>
<ul>
For errors that directly relate to disfluencies, you can select:
<ul>
<li>Disfluency > Deletion: A source disfluency is missing in the translation</li>
<li>Disfluency > Extra: A new disfluency is added in the translation, absent, from the source</li>
<li>Disfluency > Type mismatch: Disfluency type is changed (e.g., repetition → filler)</li>
</ul>
</ul>
Lastly, you will rate the overall quality of the translation in two aspects on a 0-100 scale, using sliders:
<ul>
<li>Style Preservation: How well are disfluencies preserved (presence, type, placement, spoken feel)?</li>
<li>Meaning Preservation: How accurately and completely is the source meaning preserved?</li>
<li>Rate each from 0% (poor) to 100% (excellent).</li>
</ul>
Thanks for your help!
""",
        },
        "campaign_id": f"uhmazing_v0_{lang}.json",
        "data_welcome": data_welcome,
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
