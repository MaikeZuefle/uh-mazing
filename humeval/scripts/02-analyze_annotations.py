# %%

import json
import collections
import statistics
import scipy.stats as st

with open("../data/annotations.json", "r") as f:
    data = json.load(f)["uhmazing_v1_cs"]

data_agg = collections.defaultdict(lambda: collections.defaultdict(list))
error_spans = []
for item in data:
    if "form" in item["item"][0]:
        # form
        print(item["annotation"])
    elif len(item["item"]) == 2:
        # tutorial
        pass
    elif len(item["item"]) == 1:
        # actual annotation
        # print(item["annotation"][0])
        for model, model_kv in item["annotation"][0].items():
            error_spans += model_kv["error_spans"]
            for slider_name, value in model_kv["sliders"].items():
                data_agg[model][slider_name].append(value)
    else:
        raise Exception("Unknown item format")


def confidence_interval(ys, confidence=0.95):
    intervals = st.t.interval(
        confidence,
        len(ys) - 1,
        loc=statistics.mean(ys),
        scale=statistics.stdev(ys) / len(ys) ** 0.5,
    )
    return (intervals[1] - intervals[0]) / 2


for model in data_agg:
    for slider_name in data_agg[model]:
        vals = data_agg[model][slider_name]
        print(
            f"{model:<30} {slider_name:>20} {statistics.mean(vals):.2f} ±{confidence_interval(vals):.2f}"
        )

for error_span in error_spans:
    print(error_span["severity"], error_span["category"])
