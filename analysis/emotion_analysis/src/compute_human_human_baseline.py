"""
Human-human leave-one-out baseline for per-item emotion intensity.

The analyse_llm_emotions.py computes human-to-LLM Pearson r = 0.35 on per-item mean
intensity (averaged across all labels for that item). This script computes the
human-human ceiling on the same metric so the 0.35 can be interpreted.

Each of the 160 (utterance, condition) items was annotated by ~3 human annotators.
For each annotator, we compute:
  - their per-item mean intensity (mean across their up to 5 labels for that item)
  - the LOO consensus: mean intensity of all OTHER annotators on that same item
  - Pearson r between those two series (only items the annotator actually rated)
We report the mean Pearson r across all annotators.
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from pathlib import Path

ROOT       = Path(__file__).resolve().parents[3]
HUMAN_JSON = ROOT / 'humeval_disfluency_meaning' / 'annotations.json'
SLOTS      = ['a', 'b', 'c', 'd', 'e']

# ── Load human annotations (transcript conditions only, matching the LLM study) ──
raw = json.loads(HUMAN_JSON.read_text())
rows = []
for entry in raw['uh-mazing_disfluency_meaning']:
    ann = entry['annotation'][0] if isinstance(entry['annotation'], list) else None
    if not isinstance(ann, dict) or 'a' not in ann:
        continue
    iid = entry.get('item', [{}])[0].get('item_id', '')
    if iid == 'example' or '_' not in iid:
        continue
    cond = iid.split('_', 1)[1]
    if cond == 'audio':
        continue
    base = iid.split('_', 1)[0]
    uid  = entry.get('user_id', '')
    for s in SLOTS:
        lab = (ann[s].get('textfield') or '').strip().lower()
        if not lab:
            continue
        rows.append({
            'base_item': base,
            'condition': cond,
            'user_id':   uid,
            'intensity': ann[s]['sliders']['Intensity'],
        })

df = pd.DataFrame(rows)
print(f"{len(df):,} rows  "
      f"{df['user_id'].nunique()} annotators  "
      f"{df.groupby(['base_item','condition']).ngroups} (item, cond) pairs\n")

# Per-annotator per-item mean intensity.
ann_item = (df.groupby(['user_id', 'base_item', 'condition'])['intensity']
              .mean()
              .rename('intensity')
              .reset_index())

# Overall per-item mean across all annotators (used to build the LOO consensus).
item_mean = (df.groupby(['base_item', 'condition'])['intensity']
               .agg(['mean', 'sum', 'count'])
               .reset_index()
               .rename(columns={'mean': 'all_mean', 'sum': 'all_sum', 'count': 'all_count'}))

merged = ann_item.merge(item_mean, on=['base_item', 'condition'])
# LOO consensus: subtract this annotator's contribution, re-average.
merged['loo_sum']   = merged['all_sum'] - merged['intensity']
merged['loo_count'] = merged['all_count'] - 1
merged['loo_mean']  = merged['loo_sum'] / merged['loo_count'].clip(lower=1)
# Discard items where the annotator was the only rater (no consensus possible).
merged = merged[merged['loo_count'] >= 1]

# Per-annotator Pearson r between their ratings and the LOO consensus.
print("Leave-one-out human-human Pearson r per annotator:\n")
annotator_rs = []
for uid, grp in merged.groupby('user_id'):
    grp = grp.dropna(subset=['intensity', 'loo_mean'])
    if len(grp) < 3:
        continue
    r, p = pearsonr(grp['intensity'], grp['loo_mean'])
    annotator_rs.append(r)
    print(f"  {uid:<35}  n={len(grp):3d}  r={r:+.3f}  p={p:.4f}")

mean_r = float(np.mean(annotator_rs))
print(f"\nMean human-human LOO Pearson r = {mean_r:.3f}  "
      f"(across {len(annotator_rs)} annotators)\n")
print("Compare to human-to-LLM Pearson r from analyse_llm_emotions.py:")
print("  mean human↔seed r = 0.345  (range 0.319–0.389)")
