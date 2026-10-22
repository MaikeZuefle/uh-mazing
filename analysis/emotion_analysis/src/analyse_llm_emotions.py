"""Compare the human emotion annotations with seven GPT annotation seeds."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]



# LLM emotion annotator — 7-seed variability sweep


import json, re
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from nrclex import NRCLex
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import nltk
try: nltk.data.find('corpora/wordnet')
except LookupError: nltk.download('wordnet', quiet=True)

SEEDS = [1, 2, 3, 4, 5, 6, 7]
sns.set_theme(style='whitegrid', context='notebook')

HUMAN_JSON = ROOT / 'humeval_disfluency_meaning' / 'annotations.json'


# Load all 7 seed CSVs into a single long-format df.
llm_dfs = []
for s in SEEDS:
    p = (HERE / f'../data/scores_seed{s}.csv')
    if not p.exists():
        print(f'  missing {p} — run run_seeds.py first'); continue
    df = pd.read_csv(p)
    df['seed'] = s
    df['base_item'] = df['base_item'].astype(str)
    llm_dfs.append(df)
llm = pd.concat(llm_dfs, ignore_index=True)
print(f'{len(llm):,} LLM rows across {len(SEEDS)} seeds')
print(llm.groupby('seed').size().to_string())


# Load human annotations, restricted to the two text conditions (matches LLM restriction).
SLOTS = ['a','b','c','d','e']
raw = json.loads(HUMAN_JSON.read_text())
hrows = []
for entry in raw['uh-mazing_disfluency_meaning']:
    ann = entry['annotation'][0] if isinstance(entry['annotation'], list) else None
    if not isinstance(ann, dict) or 'a' not in ann: continue
    iid = entry.get('item', [{}])[0].get('item_id', '')
    if iid == 'example' or '_' not in iid: continue
    cond = iid.split('_', 1)[1]
    if cond == 'audio': continue
    base = iid.split('_', 1)[0]
    for s in SLOTS:
        lab = (ann[s].get('textfield') or '').strip().lower()
        if not lab: continue
        hrows.append({'base_item':base, 'condition':cond, 'emotion':lab,
                      'intensity': ann[s]['sliders']['Intensity']})
human = pd.DataFrame(hrows)
human['base_item'] = human['base_item'].astype(str)
print(f'{len(human):,} human rows')


# §2. NRC mapping


NRC_LEX = NRCLex().__lexicon__
NRC_EMOTIONS = ['anger','anticipation','disgust','fear','joy','sadness','surprise','trust']
stemmer, lemmatizer = SnowballStemmer('english'), WordNetLemmatizer()
def _canon(w): return [e for e in NRC_LEX.get(w, []) if e in NRC_EMOTIONS]
STEM, LEMMA = {}, {}
for w, emos in NRC_LEX.items():
    canon = [e for e in emos if e in NRC_EMOTIONS]
    if not canon: continue
    STEM .setdefault(stemmer.stem(w), set()).update(canon)
    LEMMA.setdefault(lemmatizer.lemmatize(w), set()).update(canon)
def label_to_nrc(label):
    out = set()
    for tok in re.findall(r'[a-z]+', label.lower()):
        hit = _canon(tok)
        if not hit: hit = STEM .get(stemmer.stem(tok), set())
        if not hit: hit = LEMMA.get(lemmatizer.lemmatize(tok), set())
        out.update(hit)
    return sorted(out)

def expand_nrc(df, seed_col=False):
    all_labs = df['emotion'].unique()
    l2n = {lab: label_to_nrc(lab) for lab in all_labs}
    rows = []
    for _, r in df.iterrows():
        for emo in l2n[r['emotion']]:
            row = {'base_item':r['base_item'], 'condition':r['condition'],
                   'intensity':r['intensity'], 'nrc':emo}
            if seed_col: row['seed'] = r['seed']
            rows.append(row)
    return pd.DataFrame(rows)

nrc_l = expand_nrc(llm, seed_col=True)
nrc_h = expand_nrc(human)
print(f'NRC rows — human: {len(nrc_h):,}   LLM (7 seeds): {len(nrc_l):,}')


# §3. Direction consistency per seed


def per_nrc_delta(nrc_df):
    m = (nrc_df.groupby(['nrc','condition'])['intensity'].mean().unstack('condition')
               .rename(columns={'disfluent_transcript':'D','fluent_transcript':'F'}))
    return (m['F'] - m['D']).rename('ΔF-D')

h_dir = per_nrc_delta(nrc_h).rename('human')
seed_dirs = {f'seed{s}': per_nrc_delta(nrc_l[nrc_l['seed']==s]) for s in SEEDS}
table = pd.concat([h_dir] + [seed_dirs[f'seed{s}'].rename(f'seed{s}') for s in SEEDS], axis=1)

# Add summary columns: LLM mean, LLM std, seeds agreeing with human sign.
seed_cols = [f'seed{s}' for s in SEEDS]
table['llm_mean'] = table[seed_cols].mean(axis=1)
table['llm_std']  = table[seed_cols].std(axis=1)
table['n_agree']  = (np.sign(table[seed_cols]).eq(np.sign(table['human']), axis=0)).sum(axis=1)
table = table.reindex(table['human'].sort_values(ascending=False).index)

print('Full 7-seed direction table (sorted by human ΔF-D):\n')
print(table.round(3).to_string())

print('\nSummary — per-seed positive-Δ count:')
for col in ['human'] + seed_cols:
    print(f'  {col:<7}: {int((table[col] > 0).sum())}/8 positive')


# Grouped bars: for each NRC emotion, one bar per annotator (human + 7 seeds).
long = table[['human'] + seed_cols].reset_index().melt(id_vars='nrc', var_name='annotator', value_name='delta')

# Palette: human in blue, seeds in a graded warm colormap so they're visually a group.
seed_palette = sns.color_palette('YlOrRd', n_colors=len(SEEDS) + 2)[1:-1]  # trim endpoints
palette = {'human': '#274b7d'}
palette.update({f'seed{s}': tuple(c) for s, c in zip(SEEDS, seed_palette)})

fig, ax = plt.subplots(figsize=(14, 7))
sns.barplot(data=long, x='nrc', y='delta', hue='annotator',
            order=table.index.tolist(),
            hue_order=['human'] + seed_cols,
            palette=palette, edgecolor='#333', linewidth=0.5, ax=ax)
ax.axhline(0, color='#333', linewidth=1)
ax.set_xlabel('NRC emotion', fontsize=19)
ax.set_ylabel(r'$\Delta_{\mathrm{Fluent}-\mathrm{Disfluent}}$ intensity', fontsize=19)
ax.tick_params(axis='x', labelsize=17)
ax.tick_params(axis='y', labelsize=15)
ax.legend(title='annotator', fontsize=13, title_fontsize=14, ncols=2, loc='best')
sns.despine()
plt.tight_layout()
plt.savefig(HERE / '../output/seed_direction_comparison.pdf', bbox_inches='tight')
plt.savefig(HERE / '../output/seed_direction_comparison.png', dpi=150, bbox_inches='tight')
print('saved seed_direction_comparison.{pdf,png}')


# §4. Per-item correlation matrix — humans + 7 LLM seeds


def per_item_mean(df):
    return df.groupby(['base_item','condition'])['intensity'].mean()

series = {'human': per_item_mean(human)}
for s in SEEDS:
    series[f'seed{s}'] = per_item_mean(llm[llm['seed']==s])
wide = pd.concat(series, axis=1).dropna()
print(f'n = {len(wide)} paired items across all annotators\n')

corr_pearson  = wide.corr(method='pearson')
print('Pearson correlation matrix:')
print(corr_pearson.round(3).to_string())

print('\nSummary — between-seed pairwise means:')
seed_pairs = [(a, b) for i, a in enumerate(seed_cols) for b in seed_cols[i+1:]]
seed_seed_rs = [corr_pearson.loc[a, b] for a, b in seed_pairs]
print(f'  mean seed↔seed Pearson r  = {np.mean(seed_seed_rs):+.3f}   (min {np.min(seed_seed_rs):+.3f}, max {np.max(seed_seed_rs):+.3f})')

human_seed_rs = [corr_pearson.loc['human', s] for s in seed_cols]
print(f'  mean human↔seed Pearson r = {np.mean(human_seed_rs):+.3f}   (min {np.min(human_seed_rs):+.3f}, max {np.max(human_seed_rs):+.3f})')


fig, ax = plt.subplots(figsize=(9, 7.8))
sns.heatmap(corr_pearson.round(2), annot=True, fmt='.2f',
            cmap='Blues', vmin=0, vmax=1, cbar_kws={'label':'Pearson r'},
            linewidths=0.5, linecolor='white',
            annot_kws={'fontsize':16}, ax=ax)
ax.tick_params(axis='both', labelsize=14)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=12)
cbar.set_label('Pearson r', fontsize=14)
plt.tight_layout()
plt.savefig(HERE / '../output/seed_correlation_heatmap.pdf', bbox_inches='tight')
plt.savefig(HERE / '../output/seed_correlation_heatmap.png', dpi=150, bbox_inches='tight')
print('saved seed_correlation_heatmap.{pdf,png}')


# §4b. Per-item NRC-emotion *presence* correlation


def presence_vector(nrc_df):
    """Build a 160 × 8 → flat 1280-length 0/1 presence vector per annotator."""
    piv = (nrc_df.assign(present=1)
                 .pivot_table(index=['base_item','condition'], columns='nrc',
                              values='present', aggfunc='max', fill_value=0)
                 .reindex(columns=NRC_EMOTIONS, fill_value=0))
    return piv.sort_index()

vecs = {'human': presence_vector(nrc_h)}
for s in SEEDS:
    vecs[f'seed{s}'] = presence_vector(nrc_l[nrc_l['seed']==s])

# Align on the shared (item, condition) index and flatten.
shared_idx = vecs['human'].index
for k in vecs: vecs[k] = vecs[k].reindex(shared_idx, fill_value=0)
flat = {k: v.values.flatten() for k, v in vecs.items()}
print(f'presence vectors: {len(shared_idx)} (item, cond) × 8 NRC = {len(next(iter(flat.values())))} bits per annotator\n')

# Per-annotator base rate (share of 1s in the vector) — needed to interpret phi.
print('Base rate (fraction of NRC bits set) per annotator:')
for k, v in flat.items():
    print(f'  {k:<7}: {v.mean():.3f}')

# Pearson correlation matrix = phi coefficient on binary data.
presence_wide = pd.DataFrame(flat)
corr_presence = presence_wide.corr(method='pearson')
print('\nPresence correlation matrix (Pearson r on binary presence vectors = phi):')
print(corr_presence.round(3).to_string())

# Summary — mean between-seed and mean human↔seed.
seed_pairs = [(a, b) for i, a in enumerate([f'seed{s}' for s in SEEDS]) for b in [f'seed{t}' for t in SEEDS[i+1:]]]
ss = [corr_presence.loc[a, b] for a, b in seed_pairs]
hs = [corr_presence.loc['human', f'seed{s}'] for s in SEEDS]
print(f'\nMean seed↔seed  phi = {np.mean(ss):+.3f}  (min {np.min(ss):+.3f}, max {np.max(ss):+.3f})')
print(f'Mean human↔seed phi = {np.mean(hs):+.3f}  (min {np.min(hs):+.3f}, max {np.max(hs):+.3f})')

# Per-NRC breakdown — for each NRC emotion, correlate human presence vs each seed.
print('\nPer-NRC emotion — human↔seed phi (averaged across 7 seeds):')
per_nrc = {}
for emo_i, emo in enumerate(NRC_EMOTIONS):
    h_col = vecs['human'][emo].values
    llm_rs = []
    for s in SEEDS:
        s_col = vecs[f'seed{s}'][emo].values
        if h_col.std() == 0 or s_col.std() == 0:
            llm_rs.append(np.nan)
        else:
            llm_rs.append(np.corrcoef(h_col, s_col)[0,1])
    per_nrc[emo] = {'human_share': h_col.mean(),
                     'seed_mean_share': np.mean([vecs[f'seed{s}'][emo].values.mean() for s in SEEDS]),
                     'phi_mean': np.nanmean(llm_rs),
                     'phi_min':  np.nanmin(llm_rs),
                     'phi_max':  np.nanmax(llm_rs)}
per_nrc_df = pd.DataFrame(per_nrc).T[['human_share','seed_mean_share','phi_mean','phi_min','phi_max']]
per_nrc_df = per_nrc_df.sort_values('phi_mean', ascending=False).round(3)
print(per_nrc_df.to_string())


# §4c. Fair comparison — is the LLM really more selective than a human?


# Human load — need to reconstruct annotator counts per item to make a fair comparison.
# Load the raw JSON again to grab user_id (dropped from the human df earlier).
raw = json.loads(HUMAN_JSON.read_text())
h_full_rows = []
for entry in raw['uh-mazing_disfluency_meaning']:
    ann = entry['annotation'][0] if isinstance(entry['annotation'], list) else None
    if not isinstance(ann, dict) or 'a' not in ann: continue
    iid = entry.get('item', [{}])[0].get('item_id', '')
    if iid == 'example' or '_' not in iid: continue
    cond = iid.split('_', 1)[1]
    if cond == 'audio': continue
    base = iid.split('_', 1)[0]
    uid = entry.get('user_id', '')
    for s in ['a','b','c','d','e']:
        lab = (ann[s].get('textfield') or '').strip().lower()
        if not lab: continue
        h_full_rows.append({'base_item':str(base), 'condition':cond, 'user_id':uid, 'emotion':lab})
hf = pd.DataFrame(h_full_rows)

# Sanity check: how many humans per (item, condition)?
per_item_h = hf.groupby(['base_item','condition'])['user_id'].nunique()
print('Human annotators per (item, condition):')
print(f'  mean = {per_item_h.mean():.2f}   min={per_item_h.min()}   max={per_item_h.max()}\n')

# ---- Per-annotator NRC coverage ----
# For each (item, condition, annotator), count distinct NRC emotions their labels mapped to.
labs_seen_h  = {lab: label_to_nrc(lab) for lab in hf['emotion'].unique()}
labs_seen_l  = {lab: label_to_nrc(lab) for lab in llm['emotion'].unique()}

def nrc_set(labs, mapper):
    out = set()
    for lab in labs:
        out.update(mapper[lab])
    return out

h_per = (hf.groupby(['base_item','condition','user_id'])['emotion']
           .apply(lambda s: len(nrc_set(s, labs_seen_h)))
           .rename('n_nrc'))
l_per = (llm.groupby(['base_item','condition','seed'])['emotion']
            .apply(lambda s: len(nrc_set(s, labs_seen_l)))
            .rename('n_nrc'))

print(f'Distinct NRC emotions per (item, cond, SINGLE annotator):')
print(f'  human: mean={h_per.mean():.2f}   sd={h_per.std():.2f}   median={h_per.median():.0f}')
print(f'  llm  : mean={l_per.mean():.2f}   sd={l_per.std():.2f}   median={l_per.median():.0f}   '
      f'(across all 7 seeds combined as separate observations)')

# Distinct labels (before mapping) per annotator per item — pure vocabulary count.
h_labs = hf.groupby(['base_item','condition','user_id'])['emotion'].nunique()
l_labs = llm.groupby(['base_item','condition','seed'])['emotion'].nunique()
print(f'\nDistinct raw labels per (item, cond, SINGLE annotator):')
print(f'  human: mean={h_labs.mean():.2f}   sd={h_labs.std():.2f}   median={h_labs.median():.0f}')
print(f'  llm  : mean={l_labs.mean():.2f}   sd={l_labs.std():.2f}   median={l_labs.median():.0f}')

# ---- Pooled coverage curve ----
# For each (item, condition), sample N annotators (with replacement or fixed) and take
# the union of NRC emotions they mentioned. Plot mean union size vs N.
rng = np.random.default_rng(0)
def pooled_coverage(df, id_col, pool_sizes, n_boot=200):
    grouped = df.groupby(['base_item','condition'])
    curves = {n: [] for n in pool_sizes}
    for (base, cond), g in grouped:
        annots_avail = g[id_col].unique()
        if len(annots_avail) == 0: continue
        for n in pool_sizes:
            for _ in range(n_boot):
                take = rng.choice(annots_avail, size=min(n, len(annots_avail)), replace=False)
                labs = g[g[id_col].isin(take)]['emotion'].unique()
                mapper = labs_seen_h if id_col == 'user_id' else labs_seen_l
                curves[n].append(len(nrc_set(labs, mapper)))
    return {n: (np.mean(vs), np.std(vs)) for n, vs in curves.items()}

# Human has typically 3 annotators; LLM has 7 seeds.
h_curve = pooled_coverage(hf,  'user_id', [1, 2, 3])
l_curve = pooled_coverage(llm, 'seed',    [1, 2, 3, 4, 5, 6, 7])

print(f'\nPooled coverage: mean distinct NRC emotions per (item, cond) as pool size grows.')
print(f'{"pool":<6} {"human":<18} {"llm":<18}')
for n in [1, 2, 3]:
    h = f'{h_curve[n][0]:.2f} ± {h_curve[n][1]:.2f}'
    l = f'{l_curve[n][0]:.2f} ± {l_curve[n][1]:.2f}'
    print(f'{n:<6} {h:<18} {l:<18}')
for n in [4, 5, 6, 7]:
    l = f'{l_curve[n][0]:.2f} ± {l_curve[n][1]:.2f}'
    print(f'{n:<6} {"—":<18} {l:<18}')

# ---- NRC base rate at matched pool size = 1 annotator ----
print('\nFraction of the 8 NRC emotions mentioned at least once per (item, cond), single annotator:')
h_base_single = h_per.mean() / 8
l_base_single = l_per.mean() / 8
print(f'  human: {h_base_single:.3f}   llm: {l_base_single:.3f}   ratio: {h_base_single/l_base_single:.2f}')


# Cool viz: the coverage-saturation curves for humans vs LLM seeds.
h_sizes = [1, 2, 3]
l_sizes = [1, 2, 3, 4, 5, 6, 7]

h_means = np.array([h_curve[n][0] for n in h_sizes])
h_stds  = np.array([h_curve[n][1] for n in h_sizes])
l_means = np.array([l_curve[n][0] for n in l_sizes])
l_stds  = np.array([l_curve[n][1] for n in l_sizes])

fig, ax = plt.subplots(figsize=(11, 7))

# Shaded confidence bands.
ax.fill_between(l_sizes, l_means - l_stds, l_means + l_stds,
                color='#fed9a6', alpha=0.35, zorder=1)
ax.fill_between(h_sizes, h_means - h_stds, h_means + h_stds,
                color='#a6cee3', alpha=0.35, zorder=1)

# Lines with markers.
ax.plot(l_sizes, l_means, marker='o', markersize=14,
        color='#b25a1a', linewidth=3, zorder=3, label='LLM seeds (7 total)')
ax.plot(h_sizes, h_means, marker='s', markersize=14,
        color='#274b7d', linewidth=3, zorder=3, label='Humans (3 total)')

# Reference line showing where 3 pooled humans sit.
h_ceiling = h_means[-1]
ax.axhline(h_ceiling, color='#274b7d', linewidth=1.6, linestyle=':', alpha=0.7)
ax.text(7.15, h_ceiling, f'3 humans = {h_ceiling:.2f}', va='center',
        color='#274b7d', fontsize=16, fontweight='bold')

llm_final = l_means[-1]
ax.text(7.15, llm_final, f'7 seeds = {llm_final:.2f}', va='center',
        color='#b25a1a', fontsize=16, fontweight='bold')

# The gap that isn't closing.
ax.annotate('', xy=(7.05, h_ceiling), xytext=(7.05, llm_final),
            arrowprops=dict(arrowstyle='<->', color='#555', linewidth=2))
ax.text(6.85, (h_ceiling + llm_final) / 2, f'Δ = {h_ceiling - llm_final:.2f}',
        color='#555', fontsize=16, fontweight='bold', ha='right', va='center')

ax.set_xlim(0.6, 8.6)
ax.set_ylim(3.0, 7.5)
ax.set_xticks(range(1, 8))
ax.set_xlabel('pool size (annotators pooled per item)', fontsize=19)
ax.set_ylabel('mean distinct NRC emotions per (item, condition)', fontsize=19)
ax.tick_params(axis='both', labelsize=17)
ax.legend(loc='upper left', fontsize=17, frameon=True)
sns.despine()

plt.tight_layout()
plt.savefig(HERE / '../output/coverage_saturation.pdf', bbox_inches='tight')
plt.savefig(HERE / '../output/coverage_saturation.png', dpi=150, bbox_inches='tight')
print('saved coverage_saturation.{pdf,png}')


# §4d. Embedding-based coverage — beyond the 8 NRC categories


# Embed every unique free-form label from either annotator population.
CACHE = (HERE / '../data/llm_label_embeddings.parquet')
all_labels = sorted(set(hf['emotion']) | set(llm['emotion']))
print(f'{len(all_labels)} unique free-form labels total to embed')

# Reuse the same OpenAI client used in run_seeds.py — cheap, ~$0.001 for all labels.
import os
from openai import OpenAI
from dotenv import load_dotenv
_client = None


def get_client():
    global _client
    if _client is None:
        load_dotenv(ROOT / '.env')
        _client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'),
                         organization=os.getenv('OPENAI_ORG_KEY'),
                         project=os.getenv('OPENAI_PROJECT_KEY'))
    return _client


MODEL_EMB = 'text-embedding-3-small'

cache = pd.read_parquet(CACHE) if CACHE.exists() else pd.DataFrame(columns=['text','embedding'])
to_embed = [l for l in all_labels if l and l not in set(cache['text'])]
if to_embed:
    print(f'  embedding {len(to_embed)} new labels via {MODEL_EMB}')
    import time
    new_rows = []
    for lab in to_embed:
        for attempt in range(3):
            try:
                v = get_client().embeddings.create(model=MODEL_EMB, input=lab).data[0].embedding
                new_rows.append({'text':lab, 'embedding':v}); break
            except Exception:
                if attempt == 2: raise
                time.sleep(2**attempt)
    cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
    cache.to_parquet(CACHE)
emb = {r['text']: np.array(r['embedding'], dtype=np.float32) for _, r in cache.iterrows()}
# L2-normalize once so cosine is a simple dot product.
emb_norm = {t: v / np.linalg.norm(v) for t, v in emb.items()}
print(f'{len(emb)} embeddings loaded (dim={len(next(iter(emb.values())))})')

def mean_pairwise_cos_dist(labels):
    """Mean pairwise cosine distance among the unique labels in `labels`."""
    uniq = [l for l in set(labels) if l in emb_norm]
    if len(uniq) < 2: return np.nan
    V = np.stack([emb_norm[l] for l in uniq])
    sims = V @ V.T                                            # (k, k) cosine sims
    dists = 1.0 - sims
    iu = np.triu_indices_from(dists, k=1)                    # upper triangle only
    return dists[iu].mean()

# ── Per (item, cond, single annotator) — how spread are one annotator's labels? ──
h_disp_ann = (hf.groupby(['base_item','condition','user_id'])['emotion']
                .apply(list).apply(mean_pairwise_cos_dist).dropna())
l_disp_ann = (llm.groupby(['base_item','condition','seed'])['emotion']
                 .apply(list).apply(mean_pairwise_cos_dist).dropna())
print(f'\nMean pairwise cosine distance among labels of a SINGLE annotator (per item):')
print(f'  human: mean={h_disp_ann.mean():.3f}   sd={h_disp_ann.std():.3f}   n={len(h_disp_ann)}')
print(f'  llm  : mean={l_disp_ann.mean():.3f}   sd={l_disp_ann.std():.3f}   n={len(l_disp_ann)}   '
      f'(each seed treated as a single annotator)')

# ── Pooled coverage curve — mirror the NRC-coverage analysis in continuous space. ──
rng2 = np.random.default_rng(0)
def pooled_dispersion(df, id_col, pool_sizes, n_boot=200):
    grouped = df.groupby(['base_item','condition'])
    curves = {n: [] for n in pool_sizes}
    for (base, cond), g in grouped:
        avail = g[id_col].unique()
        for n in pool_sizes:
            if len(avail) == 0: continue
            for _ in range(n_boot):
                take = rng2.choice(avail, size=min(n, len(avail)), replace=False)
                labs = g[g[id_col].isin(take)]['emotion'].tolist()
                d = mean_pairwise_cos_dist(labs)
                if not np.isnan(d): curves[n].append(d)
    return {n: (np.mean(vs), np.std(vs)) for n, vs in curves.items()}

h_disp_curve = pooled_dispersion(hf,  'user_id', [1, 2, 3])
l_disp_curve = pooled_dispersion(llm, 'seed',    [1, 2, 3, 4, 5, 6, 7])

print(f'\nPooled embedding dispersion (mean pairwise cosine distance) as pool size grows:')
print(f'{"pool":<6} {"human":<20} {"llm":<20}')
for n in [1, 2, 3]:
    h = f'{h_disp_curve[n][0]:.3f} ± {h_disp_curve[n][1]:.3f}'
    l = f'{l_disp_curve[n][0]:.3f} ± {l_disp_curve[n][1]:.3f}'
    print(f'{n:<6} {h:<20} {l:<20}')
for n in [4, 5, 6, 7]:
    l = f'{l_disp_curve[n][0]:.3f} ± {l_disp_curve[n][1]:.3f}'
    print(f'{n:<6} {"—":<20} {l:<20}')


# Cool viz for embedding dispersion — mirror the NRC coverage plot.
h_sizes_e = [1, 2, 3]
l_sizes_e = [1, 2, 3, 4, 5, 6, 7]

h_means_e = np.array([h_disp_curve[n][0] for n in h_sizes_e])
h_stds_e  = np.array([h_disp_curve[n][1] for n in h_sizes_e])
l_means_e = np.array([l_disp_curve[n][0] for n in l_sizes_e])
l_stds_e  = np.array([l_disp_curve[n][1] for n in l_sizes_e])

fig, ax = plt.subplots(figsize=(11, 7))

ax.fill_between(l_sizes_e, l_means_e - l_stds_e, l_means_e + l_stds_e,
                color='#fed9a6', alpha=0.35, zorder=1)
ax.fill_between(h_sizes_e, h_means_e - h_stds_e, h_means_e + h_stds_e,
                color='#a6cee3', alpha=0.35, zorder=1)

ax.plot(l_sizes_e, l_means_e, marker='o', markersize=14,
        color='#b25a1a', linewidth=3, zorder=3, label='LLM seeds (7 total)')
ax.plot(h_sizes_e, h_means_e, marker='s', markersize=14,
        color='#274b7d', linewidth=3, zorder=3, label='Humans (3 total)')

h_ceiling_e = h_means_e[-1]
llm_final_e = l_means_e[-1]
ax.axhline(h_ceiling_e, color='#274b7d', linewidth=1.6, linestyle=':', alpha=0.7)
ax.text(7.15, h_ceiling_e, f'3 humans = {h_ceiling_e:.3f}',
        va='center', color='#274b7d', fontsize=16, fontweight='bold')
ax.text(7.15, llm_final_e, f'7 seeds = {llm_final_e:.3f}',
        va='center', color='#b25a1a', fontsize=16, fontweight='bold')

ax.annotate('', xy=(7.05, h_ceiling_e), xytext=(7.05, llm_final_e),
            arrowprops=dict(arrowstyle='<->', color='#555', linewidth=2))
ax.text(6.85, (h_ceiling_e + llm_final_e) / 2, f'Δ = {h_ceiling_e - llm_final_e:+.3f}',
        color='#555', fontsize=16, fontweight='bold', ha='right', va='center')

ax.set_xlim(0.6, 8.6)
ax.set_xticks(range(1, 8))
ax.set_xlabel('pool size (annotators pooled per item)', fontsize=19)
ax.set_ylabel('mean pairwise cosine distance among labels', fontsize=19)
ax.tick_params(axis='both', labelsize=17)
ax.legend(loc='upper left', fontsize=17, frameon=True)
sns.despine()

plt.tight_layout()
plt.savefig(HERE / '../output/embedding_dispersion_saturation.pdf', bbox_inches='tight')
plt.savefig(HERE / '../output/embedding_dispersion_saturation.png', dpi=150, bbox_inches='tight')
print('saved embedding_dispersion_saturation.{pdf,png}')


# Combined figure: NRC-category coverage (top) + embedding dispersion (bottom).
# Compact 2x1 stack with big fonts for single-column layout.
h_sizes = [1, 2, 3]
l_sizes = [1, 2, 3, 4, 5, 6, 7]

fig, (axT, axB) = plt.subplots(2, 1, figsize=(9, 8.5))

def _draw(ax, h_means, h_stds, l_means, l_stds, ylabel, delta_fmt, show_legend=False):
    ax.fill_between(l_sizes, l_means - l_stds, l_means + l_stds,
                    color='#fed9a6', alpha=0.35, zorder=1)
    ax.fill_between(h_sizes, h_means - h_stds, h_means + h_stds,
                    color='#a6cee3', alpha=0.35, zorder=1)
    ax.plot(l_sizes, l_means, marker='o', markersize=14,
            color='#b25a1a', linewidth=3, zorder=3, label='LLM seeds')
    ax.plot(h_sizes, h_means, marker='s', markersize=14,
            color='#274b7d', linewidth=3, zorder=3, label='humans')
    h_ceil, l_end = h_means[-1], l_means[-1]
    ax.axhline(h_ceil, color='#274b7d', linewidth=1.5, linestyle=':', alpha=0.7)
    ax.annotate('', xy=(7.05, h_ceil), xytext=(7.05, l_end),
                arrowprops=dict(arrowstyle='<->', color='#555', linewidth=2))
    ax.text(6.85, (h_ceil + l_end) / 2, delta_fmt.format(h_ceil - l_end),
            color='#555', fontsize=19, fontweight='bold', ha='right', va='center')
    ax.text(7.15, h_ceil, f'3 humans\n= {h_ceil:.2f}',
            va='center', color='#274b7d', fontsize=17, fontweight='bold')
    ax.text(7.15, l_end, f'7 seeds\n= {l_end:.2f}',
            va='center', color='#b25a1a', fontsize=17, fontweight='bold')
    ax.set_xlim(0.6, 8.7)
    ax.set_xticks(range(1, 8))
    ax.set_xlabel('pool size (annotators pooled per item)', fontsize=19)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.tick_params(axis='both', labelsize=17)
    if show_legend:
        ax.legend(loc='lower right', fontsize=17, frameon=True)

# TOP — NRC-category coverage
h_means_c = np.array([h_curve[n][0] for n in h_sizes])
h_stds_c  = np.array([h_curve[n][1] for n in h_sizes])
l_means_c = np.array([l_curve[n][0] for n in l_sizes])
l_stds_c  = np.array([l_curve[n][1] for n in l_sizes])
_draw(axT, h_means_c, h_stds_c, l_means_c, l_stds_c,
      'NRC emotions per item',
      'Δ = {:+.2f}', show_legend=True)
axT.set_title('(a) NRC-category coverage', fontsize=20, pad=8)

# BOTTOM — embedding dispersion
h_means_e = np.array([h_disp_curve[n][0] for n in h_sizes])
h_stds_e  = np.array([h_disp_curve[n][1] for n in h_sizes])
l_means_e = np.array([l_disp_curve[n][0] for n in l_sizes])
l_stds_e  = np.array([l_disp_curve[n][1] for n in l_sizes])
_draw(axB, h_means_e, h_stds_e, l_means_e, l_stds_e,
      'pairwise cosine distance',
      'Δ = {:+.3f}')
axB.set_title('(b) Embedding-space dispersion', fontsize=20, pad=8)

for ax in (axT, axB):
    sns.despine(ax=ax)

plt.tight_layout()
plt.savefig(HERE / '../output/coverage_combined.pdf', bbox_inches='tight')
plt.savefig(HERE / '../output/coverage_combined.png', dpi=150, bbox_inches='tight')
print('saved coverage_combined.{pdf,png}')


# §5. Vocabulary stability across seeds


TOP = 15
vocab_by_seed = {f'seed{s}': llm[llm['seed']==s]['emotion'].value_counts().head(TOP)
                 for s in SEEDS}
cmp_v = pd.concat(vocab_by_seed, axis=1).fillna(0).astype(int)
print(f'Top {TOP} emotion labels per seed:')
print(cmp_v.to_string())

# Pairwise Jaccard on top-N sets.
sets = {k: set(v.index) for k, v in vocab_by_seed.items()}
jaccards = []
for i, a in enumerate(SEEDS):
    for b in SEEDS[i+1:]:
        A, B = sets[f'seed{a}'], sets[f'seed{b}']
        jaccards.append(len(A & B) / len(A | B))
print(f'\nMean pairwise Jaccard of top-{TOP} label sets: {np.mean(jaccards):.3f}  '
      f'(min {min(jaccards):.3f}, max {max(jaccards):.3f})')


# §6. Interpretation


# Pull the numbers to interpret.
n_all_agree     = int((table['n_agree'] == len(SEEDS)).sum())
n_majority      = int((table['n_agree'] >= 5).sum())
mean_ss         = float(np.mean(seed_seed_rs))
mean_hs         = float(np.mean(human_seed_rs))
mean_jaccard    = float(np.mean(jaccards))

# Key negative emotions the paper's headline finding hinges on.
key = ['anger', 'fear', 'sadness']
print('§3-4 headline numbers:')
print(f'   • NRC emotions where ALL 7 seeds match the human sign:   {n_all_agree}/8')
print(f'   • NRC emotions where ≥5/7 seeds match the human sign:    {n_majority}/8')
print(f'   • Between-seed  Pearson r (per-item mean intensity):     {mean_ss:+.3f}')
print(f'   • Human↔seed    Pearson r (per-item mean intensity):     {mean_hs:+.3f}')
print(f'   • Top-15 label Jaccard across seeds:                     {mean_jaccard:.3f}')

print('\nPaper-relevant negative emotions — per-seed sign vs. human:')
for emo in key:
    if emo not in table.index: continue
    row = table.loc[emo]
    signs = ['+' if row[f'seed{s}'] > 0 else '−' for s in SEEDS]
    print(f'   {emo:<8}   human ΔF-D = {row["human"]:+.2f};   '
          f'seed signs = [{" ".join(signs)}];   '
          f'seed mean {row["llm_mean"]:+.3f} (sd {row["llm_std"]:.3f}, {int(row["n_agree"])}/7 agree)')

print('\n──────────────────────────────────────────')
print('READING:')
print('──────────────────────────────────────────\n')

def _tier(r, hi=0.7, mid=0.4):
    return 'high' if r >= hi else 'moderate' if r >= mid else 'low'

print(f'1. LLM self-consistency is {_tier(mean_ss)} '
      f'(between-seed r ≈ {mean_ss:.2f}, top-{15} label Jaccard ≈ {mean_jaccard:.2f}).')
print(f'   The LLM is essentially the same reader run to run: same vocabulary,')
print(f'   similar per-item intensities.\n')

print(f'2. LLM↔human correlation is {_tier(mean_hs)} '
      f'and much lower than seed↔seed (r ≈ {mean_hs:.2f} vs {mean_ss:.2f}).')
print(f'   The LLM has a stable internal reading that is only modestly aligned')
print(f'   with what human annotators produce. This gap is not seed noise —')
print(f'   it persists across all 7 seeds at the same level.\n')

if n_all_agree >= 6:
    dir_line = 'The direction pattern is robust across seeds.'
elif n_majority >= 6:
    dir_line = 'The direction pattern is majority-robust across seeds but has notable exceptions.'
else:
    dir_line = 'The direction pattern is NOT robust across seeds.'
print(f'3. {dir_line}')
print(f'   Only {n_all_agree}/8 NRC emotions have all 7 seeds agreeing with the human sign;')
print(f'   {n_majority}/8 have a majority (≥5/7) of seeds agreeing.')
print(f'   For the paper-relevant negative emotions specifically:')
for emo in key:
    if emo not in table.index: continue
    row = table.loc[emo]
    print(f'     - {emo}: {int(row["n_agree"])}/7 seeds agree with the human sign')
print()

print('4. TAKEAWAY for the paper.')
print('   Framing the LLM as *convergent evidence* for the human ΔF-D story is')
print('   fragile: the LLM produces a stable but different reading. Safer framing:')
print('   "The LLM annotator produces highly self-consistent readings but agrees')
print('   only modestly with humans (r≈{:.2f}), suggesting the emotional interpretations'.format(mean_hs))
print('   humans read into disfluent speech are not fully recovered by text-only')
print('   LLM prompting.\"')
