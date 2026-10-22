"""Analyse the human emotion annotations across the Audio, Disfluent and Fluent conditions (Section 2 of the paper)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]



# Emotion-label analysis


# §1. Setup: load annotations, embeddings, and NRC lexicon


import json, os, re, time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from nrclex import NRCLex
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import nltk
from dotenv import load_dotenv

import warnings; warnings.filterwarnings('ignore')

# WordNet is only downloaded on the first run; silent on subsequent runs.
try:    nltk.data.find('corpora/wordnet')
except LookupError: nltk.download('wordnet', quiet=True)

ALPHA  = 0.05
SLOTS  = ['a','b','c','d','e']
CONDS  = ['audio','disfluent_transcript','fluent_transcript']
CSHORT = {'audio':'Audio','disfluent_transcript':'Disfluent','fluent_transcript':'Fluent'}

sns.set_theme(style='whitegrid', context='notebook')


# Flatten the Pearmut export to one row per (annotator, item, slot).
# item_id looks like '12_audio' or '12_disfluent_transcript'; base_item = '12'.
with open(ROOT / 'humeval_disfluency_meaning' / 'annotations.json') as f:
    raw = json.load(f)

rows = []
for entry in raw['uh-mazing_disfluency_meaning']:
    annotation = entry['annotation'][0] if isinstance(entry['annotation'], list) else None
    if not isinstance(annotation, dict) or 'a' not in annotation:
        continue
    item_id = entry.get('item', [{}])[0].get('item_id', '')
    if item_id == 'example':
        continue
    condition = item_id.split('_', 1)[1] if '_' in item_id else 'unknown'
    base_item = item_id.split('_', 1)[0] if '_' in item_id else item_id
    for slot in SLOTS:
        s = annotation[slot]
        label = (s.get('textfield') or '').strip().lower()
        if not label:
            continue
        rows.append({
            'user_id':   entry['user_id'],
            'item_id':   item_id,
            'base_item': base_item,
            'condition': condition,
            'label':     label,
            'intensity': s['sliders']['Intensity'],
        })

df = pd.DataFrame(rows)
unique_labels = sorted(df['label'].unique())
print(f'{len(df):,} rows · {len(unique_labels)} unique labels · '
      f'{df["user_id"].nunique()} annotators · {df["base_item"].nunique()} base utterances')


# Classify each label by cosine distance to positive/neutral/negative anchor words.
# Cache to parquet so this only hits OpenAI once (~$0.01 for 565 strings).
from openai import OpenAI
_client = None


def get_client():
    global _client
    if _client is None:
        load_dotenv(ROOT / '.env')
        _client = OpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            organization=os.getenv('OPENAI_ORG_KEY'),
            project=os.getenv('OPENAI_PROJECT_KEY'),
        )
    return _client
EMBED_MODEL = 'text-embedding-3-small'
ANCHORS     = ['positive', 'neutral', 'negative']
CACHE_PATH  = (HERE / '../data/human_label_embeddings.parquet')

def embed_one(text):
    for attempt in range(4):
        try:
            return get_client().embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding
        except Exception:
            if attempt == 3: raise
            time.sleep(2 ** attempt)

cache = pd.read_parquet(CACHE_PATH) if CACHE_PATH.exists() else pd.DataFrame(columns=['text','embedding'])
to_embed = [t for t in unique_labels + ANCHORS if t not in set(cache['text'])]
if to_embed:
    print(f'embedding {len(to_embed)} new strings via {EMBED_MODEL} ...')
    cache = pd.concat([cache, pd.DataFrame([{'text':t,'embedding':embed_one(t)} for t in to_embed])],
                      ignore_index=True)
    cache.to_parquet(CACHE_PATH)

emb_lookup = {r['text']: np.array(r['embedding'], dtype=np.float32) for _, r in cache.iterrows()}
print(f'{len(emb_lookup)} embeddings loaded (dim={len(next(iter(emb_lookup.values())))})')


# For each label, embed_valence = whichever anchor is closest in cosine space.
def cos(a, b): return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

anchor_vecs = {a: emb_lookup[a] for a in ANCHORS}
label_to_valence = {
    lab: max(anchor_vecs, key=lambda a: cos(emb_lookup[lab], anchor_vecs[a]))
    for lab in unique_labels
}
df['embed_valence'] = df['label'].map(label_to_valence)
print('label-count by valence:', df['embed_valence'].value_counts().to_dict())


# Implements §1.4 steps 1–7.
NRC_LEX      = NRCLex().__lexicon__                                   # {word: [emotion, ...]}
NRC_EMOTIONS = ['anger','anticipation','disgust','fear','joy','sadness','surprise','trust']

# Build stem→emotions and lemma→emotions indexes ONCE from the NRC lexicon so the
# fallbacks in label_to_nrc() are O(1) per token.
stemmer    = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()

def _canonical_emos(word):
    """Return the subset of NRC_EMOTIONS associated with `word` (empty list if none)."""
    return [e for e in NRC_LEX.get(word, []) if e in NRC_EMOTIONS]

# Precompute the two fallback maps. Each takes the *union* over lexicon entries that
# map to the same stem/lemma, so morphological variants converge.
NRC_STEM_LEX  = {}    # stem -> set(emotions)
NRC_LEMMA_LEX = {}    # lemma -> set(emotions)
for word, emos in NRC_LEX.items():
    canon = [e for e in emos if e in NRC_EMOTIONS]
    if not canon:
        continue
    NRC_STEM_LEX .setdefault(stemmer.stem(word),          set()).update(canon)
    NRC_LEMMA_LEX.setdefault(lemmatizer.lemmatize(word),  set()).update(canon)

print(f'NRC lexicon: {len(NRC_LEX):,} words   '
      f'stem index: {len(NRC_STEM_LEX):,} stems   '
      f'lemma index: {len(NRC_LEMMA_LEX):,} lemmas')

def label_to_nrc(label):
    """
    Map a free-form emotion label to a set of NRC emotions using cascading fallbacks.
      • Step 1 — tokenize on lowercase alphabetic runs.
      • Step 2 — direct surface lookup in NRC_LEX.
      • Step 3 — if miss, Snowball-stem the token and hit NRC_STEM_LEX.
      • Step 4 — if still miss, WordNet-lemmatize the token and hit NRC_LEMMA_LEX.
      • Step 5 — keep only the 8 canonical emotions.
      • Step 6 — union across tokens in a multi-word label.
    Returns a sorted list so downstream code is deterministic.
    """
    emos = set()
    for tok in re.findall(r'[a-z]+', label.lower()):                          # step 1
        hit = _canonical_emos(tok)                                            # step 2
        if not hit:
            hit = NRC_STEM_LEX.get(stemmer.stem(tok), set())                  # step 3
        if not hit:
            hit = NRC_LEMMA_LEX.get(lemmatizer.lemmatize(tok), set())         # step 4
        emos.update(hit)                                                      # steps 5 + 6
    return sorted(emos)

# Precompute the mapping once per unique label.
label_to_emotions = {lab: label_to_nrc(lab) for lab in unique_labels}

# Diagnostic: coverage lift from stem+lemma fallbacks vs. surface-only.
def _surface_only(label):
    emos = set()
    for tok in re.findall(r'[a-z]+', label.lower()):
        emos.update(_canonical_emos(tok))
    return sorted(emos)

n_surface = sum(1 for lab in unique_labels if _surface_only(lab))
n_all     = sum(1 for lab in unique_labels if label_to_emotions[lab])
print(f'\ncoverage — surface only     : {n_surface}/{len(unique_labels)} ({n_surface/len(unique_labels):.0%})')
print(f'coverage — + stem + lemma  : {n_all}/{len(unique_labels)} ({n_all/len(unique_labels):.0%})   '
      f'(+{n_all - n_surface} labels recovered)')

# Step 7 — explode df into nrc_df: one row per (annotator × item × slot × NRC emotion).
nrc_rows = [
    {'user_id':   r['user_id'],
     'item_id':   r['item_id'],
     'base_item': r['base_item'],
     'condition': r['condition'],
     'intensity': r['intensity'],
     'nrc':       emo}
    for _, r in df.iterrows()
    for emo in label_to_emotions[r['label']]
]
nrc_df = pd.DataFrame(nrc_rows)
print(f'\n{len(df):,} df rows → {len(nrc_df):,} nrc_df rows')

# Example mappings — trace which fallback fired per token so you can eyeball the algorithm.
def _explain(label):
    parts = []
    for tok in re.findall(r'[a-z]+', label.lower()):
        if _canonical_emos(tok):
            parts.append(f'{tok}=surface')
        elif NRC_STEM_LEX.get(stemmer.stem(tok)):
            parts.append(f'{tok}→stem({stemmer.stem(tok)})')
        elif NRC_LEMMA_LEX.get(lemmatizer.lemmatize(tok)):
            parts.append(f'{tok}→lemma({lemmatizer.lemmatize(tok)})')
        else:
            parts.append(f'{tok}=miss')
    return ', '.join(parts)

example_labels = ['frustrated', 'mild disappointment', 'hesitancy', 'aggravated', 'contemplation']
print('\nExample label → NRC mappings (with fallback trace):')
for lab in example_labels:
    emos_str = str(label_to_nrc(lab))
    print(f'  {lab!r:<24} → {emos_str:<32}   [{_explain(lab)}]')


# §1.4 How free-form labels map to NRC emotions


# §2. Baseline sentiment findings


# Reusable helpers implementing §2's three tests.
# Each accepts a subset of df so §3 can call them on LOAO/LOSO slices.

def _one_vs_rest_chi2(sub, target):
    """
    Chi-square that asks: does the target embed_valence class appear more in some
    conditions than others?
      1. Count target vs. not-target rows per condition → 2×3 contingency table.
      2. Compute chi² p-value with scipy.stats.chi2_contingency.
      3. Effect size: Cramér's V = sqrt(chi²  /  (n × (min(rows,cols)-1))).
    Returns (V, p) or (nan, nan) if the target has too few rows to test.
    """
    in_v  = (sub['embed_valence'] == target).groupby(sub['condition']).sum().reindex(CONDS, fill_value=0)
    tot   = sub['condition'].value_counts().reindex(CONDS, fill_value=0)
    not_v = tot - in_v
    if in_v.sum() < 5 or not_v.sum() < 5:
        return (np.nan, np.nan)
    chi2, p, _, _ = stats.chi2_contingency(pd.DataFrame([in_v, not_v]))
    n = in_v.sum() + not_v.sum()
    V = float(np.sqrt(chi2 / (n * (min(2, 3) - 1))))
    return (V, p)

def _neg_intensity_kw(sub):
    """
    Kruskal-Wallis on intensity across conditions, restricted to negative-valence rows.
      1. Split negative-valence rows into three groups by condition.
      2. stats.kruskal returns H statistic + p-value (non-parametric ANOVA analog).
      3. Effect size: ε² = (H - k + 1) / (n - k), clamped to [0, 1].
    """
    neg    = sub[sub['embed_valence'] == 'negative']
    groups = [neg[neg['condition'] == c]['intensity'].values for c in CONDS]
    if min(len(g) for g in groups) < 3:
        return (np.nan, np.nan)
    H, p   = stats.kruskal(*groups)
    n_t, k = len(neg), 3
    eps    = max(0.0, (H - k + 1) / (n_t - k))
    return (eps, p)

def sentiment_tests(sub):
    """Run all three sentiment tests on `sub`. Returns {name: (effect_size, p_value)}."""
    return {
        'neutral_share':      _one_vs_rest_chi2(sub, 'neutral'),
        'negative_share':     _one_vs_rest_chi2(sub, 'negative'),
        'negative_intensity': _neg_intensity_kw(sub),
    }

# ── Baseline: run on the full data ─────────────────────────────────────────
baseline = sentiment_tests(df)
print('Baseline sentiment findings (full data):')
for name, (es, p) in baseline.items():
    print(f'  {name:<20}  effect={es:.3f}   p={p:.4f}')

# Descriptive means we quote directly in the paper text.
print('\nEmbed-valence share by condition (in-text numbers):')
shares = pd.crosstab(df['embed_valence'], df['condition'], normalize='columns').loc[ANCHORS, CONDS]
print((shares * 100).round(1).rename(columns=CSHORT).to_string())

print('\nNegative-intensity means by condition:')
neg = df[df['embed_valence'] == 'negative']
print(neg.groupby('condition')['intensity'].mean().reindex(CONDS).rename(index=CSHORT).round(2).to_string())


# §3. Robustness — leave-one-annotator-out & leave-one-sample-out


def _loo_sweep(unit_col):
    """
    Leave-one-{unit}-out resampling.
      1. For each unique value in `unit_col` (e.g. one annotator or one base_item),
         drop every row with that value from df.
      2. Run the §2 sentiment tests on the remaining subset.
      3. Collect (effect_size, p_value) tuples across all iterations.
    Returns {finding_name: [(es, p), ...]}.
    """
    out = {name: [] for name in baseline}
    for value in df[unit_col].unique():
        sub = df[df[unit_col] != value]                      # step 1
        for name, (es, p) in sentiment_tests(sub).items():   # step 2
            out[name].append((es, p))                        # step 3
    return out

def _summarize(records, label):
    """
    Collapse per-iteration records into a one-row summary per finding:
      • baseline effect size (from full-data run)
      • min / median / max effect size across iterations
      • % of iterations with p < ALPHA
    """
    return pd.DataFrame([{
        'finding':          name,
        'baseline_es':      baseline[name][0],
        f'{label}_es_min':  min(es for es, _ in series if not np.isnan(es)),
        f'{label}_es_med':  float(np.median([es for es, _ in series if not np.isnan(es)])),
        f'{label}_es_max':  max(es for es, _ in series if not np.isnan(es)),
        f'{label}_pct_sig': float(np.mean([p < ALPHA for _, p in series if not np.isnan(p)])),
    } for name, series in records.items()])

# Run both sweeps and report.
loao_df = _summarize(_loo_sweep('user_id'),   'LOAO')     # 18 iterations
loso_df = _summarize(_loo_sweep('base_item'), 'LOSO')     # 80 iterations

print('Leave-one-annotator-out (18 iterations):')
print(loao_df.round(3).to_string(index=False))
print('\nLeave-one-sample-out (80 iterations):')
print(loso_df.round(3).to_string(index=False))


# §4. NRC intensity across conditions + direction consistency


# Table (§4 step 1): mean intensity per (NRC emotion × condition), sorted by Δ.
#   1. Group nrc_df by (nrc, condition) and mean the intensity column.
#   2. Unstack conditions to columns so we get an 8×3 table.
#   3. Reindex to canonical CONDS order and rename to short labels for display.
#   4. Add derived columns: delta_FminA (the direction-consistency effect) + per-emotion n.
#   5. Sort descending by delta_FminA so the strongest movers sit at the top.
nrc_int = (nrc_df.groupby(['nrc','condition'])['intensity']
                 .mean().unstack('condition')
                 .reindex(columns=CONDS)
                 .rename(columns=CSHORT))
nrc_int['delta_FminA'] = nrc_int['Fluent'] - nrc_int['Audio']
nrc_int['n']           = nrc_df['nrc'].value_counts()
nrc_int = nrc_int.sort_values('delta_FminA', ascending=False)
print(nrc_int.round(2).to_string())


# Direction-consistency figure. Each row shows one NRC emotion's intensity shift
# vs. the Audio baseline.
#   bar length   = Δ_{Fluent−Audio}     (color-coded by magnitude)
#   vertical tick = Δ_{Disfluent−Audio}  (black)
# Positive → that transcript condition (fluent or disfluent) was rated MORE
# INTENSE than audio.
# Tick inside bar → monotonic trajectory (Audio ≤ Disfluent ≤ Fluent).
# Tick past bar tip or past 0 → non-monotonic (see surprise, disgust).
fig, ax = plt.subplots(figsize=(10, 6.5))

emos     = nrc_int.index.tolist()
delta_DA = (nrc_int['Disfluent'] - nrc_int['Audio']).values    # Disfluent − Audio
delta_FA = (nrc_int['Fluent']    - nrc_int['Audio']).values    # Fluent − Audio

LARGE_TH, MID_TH = 0.30, 0.15
# Paper-standard palette (per Maike's comment): dark blue / red / orange,
# matching the other figures in the manuscript.
LARGE_COLOR, MID_COLOR, SMALL_COLOR = '#b2182b', '#e08214', '#1f4788'

def _color(d):
    if d > LARGE_TH: return LARGE_COLOR
    if d > MID_TH:   return MID_COLOR
    return SMALL_COLOR

y = np.arange(len(emos))
for yi, dDA, dFA in zip(y, delta_DA, delta_FA):
    ax.barh(yi, dFA, color=_color(dFA), edgecolor='#333', linewidth=0.5, height=0.7)
    ax.plot([dDA, dDA], [yi - 0.32, yi + 0.32], color='#000', linewidth=2.8)
    # Place the label outside the bar's tip: to the right for positive Δ,
    # to the left for negative Δ (so it doesn't overlap the bar).
    if dFA >= 0:
        ax.text(dFA + 0.01, yi, f'{dFA:+.2f}', va='center', ha='left',
                fontsize=17, fontweight='bold')
    else:
        ax.text(dFA - 0.01, yi, f'{dFA:+.2f}', va='center', ha='right',
                fontsize=17, fontweight='bold')

ax.axvline(0, color='#333', linewidth=1)
ax.set_yticks(y)
ax.set_yticklabels(emos, fontsize=19)
ax.set_xlabel(r'intensity shift vs. audio  '
              r'(positive = fluent (bar) or disfluent (tick) rated more intensely than audio)',
              fontsize=15)
ax.set_xlim(-0.15, 0.4)
ax.invert_yaxis()
ax.tick_params(axis='x', labelsize=15)

from matplotlib.lines import Line2D
BAR_LABEL   = r'$\Delta_{\mathrm{Fluent}-\mathrm{Audio}}$'
TICK_LABEL  = r'$\Delta_{\mathrm{Disfluent}-\mathrm{Audio}}$'
legend = [
    Line2D([0], [0], color=LARGE_COLOR, lw=12,  label=f'{BAR_LABEL} > {LARGE_TH:.2f}   (large)'),
    Line2D([0], [0], color=MID_COLOR,   lw=12,  label=f'{BAR_LABEL} > {MID_TH:.2f}   (medium)'),
    Line2D([0], [0], color=SMALL_COLOR, lw=12,  label=f'{BAR_LABEL} $\\leq$ {MID_TH:.2f}   (small)'),
    Line2D([0], [0], color='#000',      lw=2.8, label=f'tick position: {TICK_LABEL}'),
]
ax.legend(handles=legend, loc='lower right', fontsize=15, frameon=True,
          title='bar length = ' + BAR_LABEL, title_fontsize=15)
sns.despine(left=True, bottom=False)

plt.tight_layout()
plt.savefig(HERE / '../output/nrc_direction_consistency.pdf', bbox_inches='tight')
plt.savefig(HERE / '../output/nrc_direction_consistency.png', dpi=150, bbox_inches='tight')
print('saved nrc_direction_consistency.{pdf,png}')


# Small-multiples barplot: one panel per NRC emotion, mean intensity by condition
# with seaborn's 95% bootstrap CI. Appendix figure — shows all three condition means
# and CI widths so the reader can see how much overlap remains.
#
# Bars use pastel per-condition colors; error bars are drawn in grey so the CIs
# read as neutral uncertainty markers rather than competing for attention.
PASTEL = {'Audio':'#a6cee3', 'Disfluent':'#fdbf6f', 'Fluent':'#b2df8a'}

fig, axes = plt.subplots(2, 4, figsize=(15, 8), sharey=True)
for ax, emo in zip(axes.flat, nrc_int.index):
    sub = nrc_df[nrc_df['nrc'] == emo].copy()
    sub['cond_lbl'] = sub['condition'].map(CSHORT)

    sns.barplot(
        data=sub, x='cond_lbl', y='intensity',
        order=['Audio','Disfluent','Fluent'],
        palette=PASTEL, hue='cond_lbl', legend=False,
        errorbar=('ci', 95), n_boot=1000, seed=0,
        edgecolor='#333', linewidth=0.6,
        err_kws={'color': '#555', 'linewidth': 1.5},   # grey CI whiskers
        capsize=0.18,
        ax=ax,
    )

    n = int(nrc_int.loc[emo, 'n'])
    d = nrc_int.loc[emo, 'delta_FminA']
    # Per-panel subplot label: which emotion + its total Δ (not a figure title).
    ax.set_title(f'{emo}  (n={n}, Δ={d:+.2f})', fontsize=14)
    ax.set_xlabel(''); ax.set_ylabel('')
    ax.set_ylim(2.0, 3.1)
    ax.tick_params(axis='x', labelsize=13); ax.tick_params(axis='y', labelsize=12)

axes[0,0].set_ylabel('intensity (1–5)', fontsize=14)
axes[1,0].set_ylabel('intensity (1–5)', fontsize=14)
# Overall figure title lives in the markdown cell above this one.
plt.tight_layout()
plt.savefig(HERE / '../output/nrc_intensity_by_condition.pdf', bbox_inches='tight')
plt.savefig(HERE / '../output/nrc_intensity_by_condition.png', dpi=150, bbox_inches='tight')
print('saved nrc_intensity_by_condition.{pdf,png}')


# §5. Inter-annotator agreement on dominant valence


# For each item: count each embed_valence, find the dominant, report the share matching it.
agree = []
for item_id, group in df.groupby('item_id'):
    if group['user_id'].nunique() < 2:
        continue
    dominant, dom_n = Counter(group['embed_valence']).most_common(1)[0]
    agree.append({
        'item_id':   item_id,
        'condition': group['condition'].iloc[0],
        'agree':     dom_n / len(group),
        'dominant':  dominant,
    })
iaa = pd.DataFrame(agree)

print(f'items with ≥ 2 annotators: {len(iaa)}')
print(f'mean agreement (overall): {iaa["agree"].mean():.1%}\n')

per_cond = (iaa.groupby('condition')['agree']
               .agg(['mean','count']).reindex(CONDS).rename(index=CSHORT))
per_cond['mean'] = per_cond['mean'].map(lambda v: f'{v:.1%}')
print(per_cond.to_string())


# Significance test: are the differences in IAA agreement across conditions reliable?
# The same 80 utterances appear in all three conditions, so observations are paired.
# We use the Wilcoxon signed-rank test (non-parametric, paired analog of a t-test).

_iaa = iaa.copy()
_iaa['base_item'] = _iaa['item_id'].str.split('_', n=1).str[0]
iaa_wide = _iaa.pivot(index='base_item', columns='condition', values='agree').dropna()
print(f"Utterances matched across all 3 conditions: {len(iaa_wide)}\n")

comparisons = [
    ('fluent_transcript', 'audio',               'Fluent > Audio',      'greater'),
    ('fluent_transcript', 'disfluent_transcript', 'Fluent > Disfluent',  'greater'),
    ('audio',             'disfluent_transcript', 'Audio vs Disfluent',  'two-sided'),
]
for cond_a, cond_b, label, alt in comparisons:
    a, b = iaa_wide[cond_a], iaa_wide[cond_b]
    W, p = stats.wilcoxon(a, b, alternative=alt)
    mean_diff = (a - b).mean()
    sig = '**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.')
    print(f"{label:<25}  mean diff={mean_diff:+.3f}  W={W:.0f}  p={p:.4f}  {sig}")


# §6. Qualitative examples — largest cross-condition swings


# Implements §6 steps 1–3.

# Step 1: fraction of labels classified 'negative' per (base_item, condition), + row count.
#   We add is_neg as a 0/1 column so the mean gives us the share directly.
neg_share = (df.assign(is_neg=(df['embed_valence']=='negative').astype(int))
               .groupby(['base_item','condition'])
               .agg(is_neg=('is_neg','mean'),      # negative-share
                    n     =('is_neg','size'))      # row count
               .reset_index())

# Pivot to a wide (base_item × condition) table so we can compute the max-min gap.
wide = neg_share.pivot(index='base_item', columns='condition', values='is_neg')
n_w  = neg_share.pivot(index='base_item', columns='condition', values='n').fillna(0)

# Step 2: keep only utterances with ≥ 3 rows in every condition. Otherwise the share
# is estimated from too few labels to trust.
valid = wide.dropna().index.intersection((n_w >= 3).all(axis=1)[lambda x: x].index)
wide  = wide.loc[valid]

# Step 3: rank by cross-condition max-min gap. The 8 utterances that shift most sit at the top.
wide['max_gap'] = wide.max(axis=1) - wide.min(axis=1)

TOP = 8
print(f'Top {TOP} utterances by cross-condition negative-share gap:\n')
for base in wide.sort_values('max_gap', ascending=False).head(TOP).index:
    r = wide.loc[base]
    print(f'── utterance {base}    '
          f'Audio {r["audio"]:.0%}  Disfluent {r["disfluent_transcript"]:.0%}  '
          f'Fluent {r["fluent_transcript"]:.0%}    gap={r["max_gap"]:.0%}')
    # For each condition, print per-annotator label sets so the reader can eyeball the shift.
    for cond in CONDS:
        sub = df[(df['base_item'] == base) & (df['condition'] == cond)]
        for uid, group in sub.groupby('user_id'):
            labs = ', '.join(f"{r['label']}[{r['embed_valence'][0]}]" for _, r in group.iterrows())
            print(f'   {CSHORT[cond]:<10} {uid[:20]:<20} → {labs}')
    print()


# §6a. Appendix table: source text per condition for the top-swing utterances


# Pull each base_item's source text per condition from the JSON.
# For transcript conditions, item['src'] is the English text. For audio it's the
# <audio> tag which we strip. We only need Disfluent + Fluent for the table.
def clean_src(s):
    s = re.sub(r'<audio[^>]*></audio>\s*(<br>)?', '', str(s or ''))
    return re.sub(r'\s+', ' ', s).strip()

src_by_item = {}
for entry in raw['uh-mazing_disfluency_meaning']:
    item = entry.get('item', [{}])[0]
    iid  = item.get('item_id', '')
    if iid == 'example' or '_' not in iid:
        continue
    base = iid.split('_', 1)[0]
    cond = iid.split('_', 1)[1]
    txt  = clean_src(item.get('src', ''))
    if txt:
        src_by_item.setdefault(base, {}).setdefault(cond, txt)

# Top-N most-swung utterances by cross-condition negative-share gap.
TOP = 5
top_bases = wide.sort_values('max_gap', ascending=False).head(TOP).index

def esc(s):
    """Minimal LaTeX escape for source text and emotion labels."""
    return (s.replace('\\', r'\textbackslash{}')
             .replace('&', r'\&').replace('%', r'\%')
             .replace('_', r'\_').replace('#', r'\#')
             .replace('$', r'\$').replace('{', r'\{').replace('}', r'\}'))

# For each utterance, rank emotion labels by how much their appearance shifted
# between Disfluent and Fluent. shift = count(Fluent) − count(Disfluent).
# Positive shift → label appeared more in Fluent (added by disfluency removal).
# Negative shift → label appeared more in Disfluent (lost by disfluency removal).
def label_shift(base_item, top_k=5):
    sub    = df[(df['base_item'] == base_item) & (df['condition'].isin(['disfluent_transcript','fluent_transcript']))]
    agg    = sub.groupby(['label','condition']).agg(n=('intensity','size'), mean_int=('intensity','mean')).reset_index()
    pivot_n = agg.pivot(index='label', columns='condition', values='n').fillna(0)
    pivot_i = agg.pivot(index='label', columns='condition', values='mean_int')
    pivot_n['shift'] = pivot_n.get('fluent_transcript', 0) - pivot_n.get('disfluent_transcript', 0)
    ranked  = pivot_n.reindex(pivot_n['shift'].abs().sort_values(ascending=False).index)
    out = []
    for lab in ranked.head(top_k).index:
        d_n = int(pivot_n.loc[lab].get('disfluent_transcript', 0))
        f_n = int(pivot_n.loc[lab].get('fluent_transcript',   0))
        d_i = pivot_i.loc[lab].get('disfluent_transcript', np.nan)
        f_i = pivot_i.loc[lab].get('fluent_transcript',   np.nan)
        out.append({'label': lab, 'D_n': d_n, 'F_n': f_n, 'D_int': d_i, 'F_int': f_i,
                    'shift': int(f_n - d_n)})
    return out

rows_out = []
for base in top_bases:
    r = wide.loc[base]
    rows_out.append({
        'base':       base,
        'audio_pct':  int(round(r['audio'] * 100)),
        'disfl_pct':  int(round(r['disfluent_transcript'] * 100)),
        'fluent_pct': int(round(r['fluent_transcript'] * 100)),
        'disfluent':  src_by_item.get(base, {}).get('disfluent_transcript', '(text not found)'),
        'fluent':     src_by_item.get(base, {}).get('fluent_transcript',    '(text not found)'),
        'shifts':     label_shift(base, top_k=5),
    })

# ── Human-readable preview ─────────────────────────────────────────────────
print('Top-swing utterances — source text + top-shifted emotion labels:\n')
for r in rows_out:
    print(f'── utterance {r["base"]}    '
          f'neg-share A/D/F = {r["audio_pct"]}%/{r["disfl_pct"]}%/{r["fluent_pct"]}%')
    print(f'   Disfluent: {r["disfluent"]}')
    print(f'   Fluent:    {r["fluent"]}')
    print(f'   {"label":<22} {"D (n×μ)":>10}   {"F (n×μ)":>10}   {"shift":>6}')
    for s in r['shifts']:
        d = f'{s["D_n"]}×{s["D_int"]:.1f}' if s['D_n'] else '—'
        f_ = f'{s["F_n"]}×{s["F_int"]:.1f}' if s['F_n'] else '—'
        arrow = '↑' if s['shift'] > 0 else ('↓' if s['shift'] < 0 else '·')
        print(f'   {s["label"]:<22} {d:>10}   {f_:>10}   {arrow}{abs(s["shift"]):>2}')
    print()

# ── LaTeX: source-text table (unchanged) ───────────────────────────────────
print('─' * 60)
print('LaTeX for source-text table (tab:qual-examples):\n')
print(r'\begin{table*}[t]')
print(r'\centering\small')
print(r'\begin{tabular}{clp{5.8cm}p{5.8cm}}')
print(r'\toprule')
print(r'utt. & \makecell{neg-share\\A/D/F} & Disfluent transcript & Fluent transcript \\')
print(r'\midrule')
for r in rows_out:
    share = f'{r["audio_pct"]}/{r["disfl_pct"]}/{r["fluent_pct"]}\\%'
    print(f'{r["base"]} & \\footnotesize {share} & '
          f'\\emph{{{esc(r["disfluent"])}}} & \\emph{{{esc(r["fluent"])}}} \\\\')
    print(r'\addlinespace')
print(r'\bottomrule')
print(r'\end{tabular}')
print(r'\caption{Utterances with the largest cross-condition swings in negative-label share '
      r'(Audio\,/\,Disfluent\,/\,Fluent). Removing disfluencies replaces contemplative labels '
      r'with hostile or distressed ones.}')
print(r'\label{tab:qual-examples}')
print(r'\end{table*}')

# ── LaTeX: emotion-shift table ─────────────────────────────────────────────
print('\n' + '─' * 60)
print('LaTeX for emotion-shift table (tab:qual-shifts):\n')
print(r'\begin{table*}[t]')
print(r'\centering\small')
print(r'\begin{tabular}{clllc}')
print(r'\toprule')
print(r'utt. & label & \makecell{Disfluent\\(n × mean int.)} & \makecell{Fluent\\(n × mean int.)} & shift \\')
print(r'\midrule')
for r in rows_out:
    for i, s in enumerate(r['shifts']):
        base_col = str(r['base']) if i == 0 else ''
        d = f'{s["D_n"]} × {s["D_int"]:.1f}' if s['D_n'] else '—'
        f_ = f'{s["F_n"]} × {s["F_int"]:.1f}' if s['F_n'] else '—'
        arrow = r'$\uparrow$' if s['shift'] > 0 else (r'$\downarrow$' if s['shift'] < 0 else '')
        print(f'{base_col} & \\emph{{{esc(s["label"])}}} & {d} & {f_} & {arrow}{abs(s["shift"])} \\\\')
    print(r'\midrule')
print(r'\bottomrule')
print(r'\end{tabular}')
print(r'\caption{Top-5 emotion labels by cross-condition appearance shift for each swing '
      r'utterance in Table~\ref{tab:qual-examples}. $\uparrow$ = added by disfluency removal '
      r'(appears more in Fluent). $\downarrow$ = lost by disfluency removal '
      r'(appears more in Disfluent). Intensity is annotator-reported mean on the $[1, 5]$ scale.}')
print(r'\label{tab:qual-shifts}')
print(r'\end{table*}')


# §7. Embedding validation of the NRC mapping


# §7 step 1: embed the 8 NRC emotion words (add to the on-disk cache if any are missing).
missing = [e for e in NRC_EMOTIONS if e not in emb_lookup]
if missing:
    print(f'embedding {len(missing)} NRC emotion words: {missing}')
    new_rows = [{'text': e, 'embedding': embed_one(e)} for e in missing]
    cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
    cache.to_parquet(CACHE_PATH)
    for r in new_rows:
        emb_lookup[r['text']] = np.array(r['embedding'], dtype=np.float32)

nrc_vecs = np.stack([emb_lookup[e] for e in NRC_EMOTIONS])            # (8, dim)
nrc_norm = nrc_vecs / np.linalg.norm(nrc_vecs, axis=1, keepdims=True)  # unit vectors

# §7 step 2: for each label, cosine sim to all 8 NRC emotions → take the max.
# Vectorized: stack all label vectors, normalize, matmul with the 8×dim NRC matrix,
# then take argmax/max along axis 1.
labels_arr    = np.array(unique_labels)
label_vecs    = np.stack([emb_lookup[lab] for lab in labels_arr])
label_norm    = label_vecs / np.linalg.norm(label_vecs, axis=1, keepdims=True)
sim_matrix    = label_norm @ nrc_norm.T                                # (n_labels, 8)
max_sim       = sim_matrix.max(axis=1)
nearest_emo   = np.array(NRC_EMOTIONS)[sim_matrix.argmax(axis=1)]

# §7 step 3: bucket each label by its §1.4 match status.
def _bucket(label):
    if _surface_only(label):          return 'surface'
    if label_to_emotions[label]:      return 'stem+lemma'    # fell to the fallback
    return 'miss'

emb_val = pd.DataFrame({
    'label':       labels_arr,
    'max_cos_sim': max_sim,
    'nearest_emo': nearest_emo,
    'bucket':      [_bucket(l) for l in labels_arr],
})

# Sanity: report bucket counts + summary stats per bucket.
print('Labels per bucket + max-cosine-sim summary:')
print(emb_val.groupby('bucket', observed=True)['max_cos_sim']
             .agg(['count','mean','std','min','median','max'])
             .round(3)
             .reindex(['surface','stem+lemma','miss'])
             .to_string())

# A few concrete examples per bucket (highest and lowest sim in each).
print('\nHighest and lowest max-sim example per bucket:')
for bkt in ['surface','stem+lemma','miss']:
    sub = emb_val[emb_val['bucket']==bkt].sort_values('max_cos_sim')
    if len(sub) == 0: continue
    lo, hi = sub.iloc[0], sub.iloc[-1]
    print(f'  {bkt:<12} lowest  {lo["label"]!r:<32} (sim={lo["max_cos_sim"]:.2f} to {lo["nearest_emo"]!r})')
    print(f'  {bkt:<12} highest {hi["label"]!r:<32} (sim={hi["max_cos_sim"]:.2f} to {hi["nearest_emo"]!r})')


# Overlaid histograms of max_cos_sim per bucket. Shared bin edges so bars are directly comparable.
BUCKET_COLORS = {'surface':'#fbb4ae', 'stem+lemma':'#fed9a6', 'miss':'#b3cde3'}
BUCKET_ORDER  = ['surface','stem+lemma','miss']

fig, ax = plt.subplots(figsize=(10, 5.5))
lo, hi = emb_val['max_cos_sim'].min(), emb_val['max_cos_sim'].max()
bins   = np.linspace(np.floor(lo * 20) / 20, np.ceil(hi * 20) / 20, 26)   # ~0.02-wide bins

for bkt in BUCKET_ORDER:
    sub = emb_val[emb_val['bucket'] == bkt]['max_cos_sim']
    ax.hist(sub, bins=bins, alpha=0.6, label=f'{bkt}  (n={len(sub)})',
            color=BUCKET_COLORS[bkt], edgecolor='#333', linewidth=0.4)
    # Mark each bucket's median with a dashed vertical line so shifts are readable
    ax.axvline(sub.median(), color=BUCKET_COLORS[bkt], linestyle='--', linewidth=1.5,
               alpha=0.9)

ax.set_xlabel('max cosine similarity to the 8 NRC emotion words', fontsize=13)
ax.set_ylabel('label count', fontsize=13)
ax.tick_params(axis='both', labelsize=12)
ax.legend(fontsize=12, frameon=True, loc='upper right')     # legend top-right per feedback
sns.despine()

plt.tight_layout()
plt.savefig(HERE / '../output/nrc_bucket_maxsim_hist.pdf', bbox_inches='tight')
plt.savefig(HERE / '../output/nrc_bucket_maxsim_hist.png', dpi=150, bbox_inches='tight')
print('saved nrc_bucket_maxsim_hist.{pdf,png}')
