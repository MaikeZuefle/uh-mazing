# %%

import json
import collections

with open("annotations.json", "r") as f:
    data = json.load(f)["uh-mazing_disfluency_meaning"]

# %%
import statistics

data_category_agg = collections.defaultdict(list)
data_user_agg = collections.defaultdict(list)

for line in data:
    category = line["item"][0]["item_id"].split("_", 1)[-1]
    if category == "example":
        continue

    for annotation in line["annotation"][0].values():
        emotion, intensity = (
            annotation["textfield"].lower(),
            annotation["sliders"]["Intensity"],
        )
        if emotion.strip() == "" or intensity == 0:
            continue
        data_category_agg[category].append((emotion, intensity, category))
        data_user_agg[line["user_id"]].append((emotion, intensity, category))

for category, data_local in data_category_agg.items():
    print(f"{category:>20} {statistics.mean([x[1] for x in data_local]):.2f}")

# %%
import matplotlib.pyplot as plt

data_peri_agg = collections.defaultdict(list)

for user, data_local in data_user_agg.items():
    for x_i, (emotion, intensity, category) in enumerate(data_local):
        data_peri_agg[x_i].append(intensity)

plt.plot([statistics.mean(x) for x in data_peri_agg.values()])
