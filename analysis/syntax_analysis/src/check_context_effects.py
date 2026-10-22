"""Check how much subject pronouns and coordinating conjunctions explain the difference to the references."""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compare_to_references import (counts_by_utt, load_excluded, load_spans, pct_predicates_with_pronoun_subject,
                                   tvd_with_ci)
from paper_style import LANG_DISPLAY, LANGS, LLM_JUDGE_SYSTEM_ORDER, MIN_UTTERANCES, SYSTEM_ID, SYSTEM_MARKER, SYSTEM_MARKER_SIZE
from plot_comparison import SYSTEM_COLOR, add_legend, style_axes, system_handles

HERE = os.path.dirname(os.path.abspath(__file__))
VARIANTS = {False: "All words", True: "Subject pronouns skipped"}


def compute(reference_dir, systems_dir, excluded_csv, n_boot):
    rng = np.random.default_rng(0)
    refs = {skip: load_spans(reference_dir, skip) for skip in VARIANTS}
    rows = []
    for system, system_id in SYSTEM_ID.items():
        excluded = load_excluded(excluded_csv, system)
        parsed = {skip: load_spans(os.path.join(systems_dir, system), skip, excluded) for skip in VARIANTS}
        sys_utts, sys_fluent = parsed[False][1], parsed[False][2]
        for lang in LANGS:
            utts = sorted(sys_utts.get(lang, []))
            if len(utts) < MIN_UTTERANCES:
                continue
            row = {"system_id": system_id, "lang": lang,
                   "pct_predicates_pron_subj_ref": pct_predicates_with_pronoun_subject(refs[False][2], lang, utts),
                   "pct_predicates_pron_subj_sys": pct_predicates_with_pronoun_subject(sys_fluent, lang, utts)}
            for skip in VARIANTS:
                s, r = parsed[skip][0], refs[skip][0]
                s = s[(s["lang"] == lang) & s["utt_id"].isin(utts)]
                r = r[(r["lang"] == lang) & r["utt_id"].isin(utts)]
                for side in ["before", "after"]:
                    col = f"{side}_deprel"
                    tvd = tvd_with_ci(counts_by_utt(s, col), counts_by_utt(r, col), utts, n_boot, rng)[1]
                    row[f"tvd_{side}_{'skipped' if skip else 'all'}"] = tvd
            rows.append(row)
    return pd.DataFrame(rows)


def plot(df, out_path):
    plt.rcParams["font.family"] = "DejaVu Sans"
    groups = [("Subject can be omitted", [l for l in LANGS if l not in ("de", "fr")]), ("Subject required", ["de", "fr"])]
    positions, x = {}, 0.0
    for _, langs in groups:
        for lang in langs:
            positions[lang] = x
            x += 1
        x += 0.6
    offsets = dict(zip(LLM_JUDGE_SYSTEM_ORDER, [-0.21, -0.07, 0.07, 0.21]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, skip in zip(axes, VARIANTS):
        style_axes(ax)
        for _, row in df.iterrows():
            ax.scatter(positions[row["lang"]] + offsets[row["system_id"]],
                       row[f"tvd_after_{'skipped' if skip else 'all'}"], marker=SYSTEM_MARKER[row["system_id"]],
                       s=0.6 * SYSTEM_MARKER_SIZE[row["system_id"]], color=SYSTEM_COLOR, edgecolors="white",
                       linewidths=0.4, zorder=3)
        for name, langs in groups:
            ax.text(np.mean([positions[l] for l in langs]), 1.02, name, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=13, color="#555555")
        ax.set_xticks(list(positions.values()))
        ax.set_xticklabels([LANG_DISPLAY[l] for l in positions], fontsize=14)
        ax.set_title(f"Word after the disfluency: {VARIANTS[skip].lower()}", fontsize=15, pad=26)
        ax.set_ylim(0, None)
    axes[0].set_ylabel("% of contexts differing from references", fontsize=14)
    add_legend(fig, system_handles(SYSTEM_COLOR))
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def conjunction_checks(reference_dir, systems_dir, excluded_csv):
    """Conjunctions per 100 words overall, and conjunction neighbours of disfluencies split by whether a system kept all disfluencies of an utterance."""
    ref_spans, _, ref_fluent = load_spans(reference_dir)
    overall, selection = [], []
    for system, system_id in SYSTEM_ID.items():
        spans, utts, fluent = load_spans(os.path.join(systems_dir, system), excluded=load_excluded(excluded_csv, system))
        for lang in LANGS:
            utts_lang = utts.get(lang, set())
            if len(utts_lang) < MIN_UTTERANCES:
                continue
            for who, f in [("reference", ref_fluent), ("system", fluent)]:
                words = f[(f["lang"] == lang) & f["utt_id"].isin(utts_lang) & (f["upos"] != "PUNCT")]
                overall.append({"system_id": system_id, "lang": lang, "who": who,
                                "cc_per_100_words": 100 * (words["deprel"] == "cc").mean()})
            s = spans[(spans["lang"] == lang) & spans["utt_id"].isin(utts_lang)]
            r = ref_spans[(ref_spans["lang"] == lang) & ref_spans["utt_id"].isin(utts_lang)]
            kept = s.groupby("utt_id").size()
            for utt, n_ref in r.groupby("utt_id").size().items():
                group = "all kept" if kept.get(utt, 0) >= n_ref else "some deleted"
                for who, df in [("reference", r[r["utt_id"] == utt]), ("system", s[s["utt_id"] == utt])]:
                    for side in ["before", "after"]:
                        selection.append({"system_id": system_id, "lang": lang, "utt_id": utt, "group": group,
                                          "who": who, "side": side, "n_spans": len(df),
                                          "n_cc": int((df[f"{side}_deprel"] == "cc").sum())})
    return pd.DataFrame(overall), pd.DataFrame(selection)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference_dir", default=os.path.join(HERE, "../data/reference_parses"))
    parser.add_argument("--systems_dir", default=os.path.join(HERE, "../output/generated_translations/parsed"))
    parser.add_argument("--output", default=os.path.join(HERE, "../output/generated_translations/checks"))
    parser.add_argument("--excluded_csv", default=os.path.join(HERE, "../data/marked/excluded_outputs.csv"))
    parser.add_argument("--n_boot", type=int, default=200)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    df = compute(args.reference_dir, args.systems_dir, args.excluded_csv, args.n_boot)
    df.to_csv(os.path.join(args.output, "pronoun_effect.csv"), index=False)
    plot(df, os.path.join(args.output, "pronoun_effect.png"))
    print(df.groupby("lang")[[c for c in df.columns if c.startswith(("tvd_", "pct_"))]].mean().round(1).to_string())

    overall, selection = conjunction_checks(args.reference_dir, args.systems_dir, args.excluded_csv)
    overall.to_csv(os.path.join(args.output, "conjunction_overall.csv"), index=False)
    selection.to_csv(os.path.join(args.output, "conjunction_selection.csv"), index=False)
    print("\nConjunctions per 100 words:")
    print(overall.pivot_table(index="lang", columns="who", values="cc_per_100_words").round(2).to_string())
    pooled = selection.groupby(["group", "side", "who"])[["n_cc", "n_spans"]].sum()
    print("\n% of disfluencies with a conjunction as neighbour:")
    print((100 * pooled["n_cc"] / pooled["n_spans"]).unstack("who").round(1).to_string())


if __name__ == "__main__":
    main()
