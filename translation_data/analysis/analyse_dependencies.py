#!/usr/bin/env python3
"""
Analysis of where disfluencies occur syntactically, built entirely from the
raw per-token CSVs dependency_parsing.py produces (dependency_results.csv,
fluent_parse_results.csv). This script never touches Stanza -- every
interpretive choice (what counts as a "span", how to find a "neighbor",
whether to skip punctuation, how to pair EN with a target language) lives
here, in plain pandas, so changing any of it is a re-run of THIS script
only, not a re-parse.

Three questions, each answered with two independent signals (the direct
disfluent-token parse, and the parser-quality-independent fluent-only parse)
so neither is treated as ground truth on its own:

1. Role: what syntactic role do disfluencies play (subject, verb modifier,
   discourse marker, ...)? A disfluency is often more than one word (e.g. a
   multi-word false start), and UD has no "this whole span is one modifier"
   label -- every word gets its own head. So for each contiguous disfluent
   SPAN we find its "anchor": the word(s) inside the span whose head points
   outside it, i.e. the point where the span attaches to the rest of the
   sentence. A span is "coherent" if it has exactly one anchor; spans with
   zero or multiple anchors mean the parser fragmented the span, which is
   itself a reliability signal given disfluent text is harder to parse.
   As a second, independent signal, we also look at the nearest fluent
   CONTENT word (punctuation skipped -- every punctuation mark is UD-tagged
   deprel=punct/upos=PUNCT regardless of context, so it carries no signal)
   immediately left/right of the span, from the fluent-only parse.

2. Cross-language: how does the role distribution compare across languages?
   Reported as per-language proportions (not raw counts, since totals
   differ), for both the anchor deprel/upos and the left/right neighbor
   deprel/upos.

3. English -> target shift: for each target language, we take every
   coherent English span and every coherent target-language span in the
   same utterance and cross them to build a role-transition table (e.g.
   nsubj -> advmod). This is a co-occurrence table, not a strict token
   alignment (translators marked disfluencies independently per language,
   so span counts differ and there's no word alignment to exploit) -- an
   utterance with 2 EN spans and 3 target spans contributes 6 pairs. This
   uses far more of the data than requiring exactly one span per side, and
   avoids selecting only for utterances with a single, simple filler word.

Output files (written to --output dir):
  reliability/coherence_by_language.csv     - fraction of spans with exactly
                                               one anchor, per language
  reliability/span_length_distribution.csv  - span length counts per language
  role_by_language/anchor_{deprel,upos}_by_language.{csv,png}
  role_by_language/{left,right}_deprel_by_language.{csv,png}
  position_by_language/position_bucket_by_language.{csv,png} - initial/
                                               medial/final proportions
  position_by_language/relative_position_summary.csv - mean/median/std of
                                               relative position (0=start,
                                               1=end), per language
  position_by_language/relative_position_boxplot.png
  en_to_target/transition_{deprel,upos}_all.csv - stacked EN->X role pairs
  en_to_target/coverage_summary.csv         - how many utterances/pairs feed
                                               the transition tables
  en_to_target/span_count_shift.csv         - does translation add/drop
                                               disfluencies, per language
  en_to_target/heatmaps/en_to_{lang}_deprel.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def disfluent_spans_from_tokens(g):
    """g: token rows for one (utt_id, lang), any order. Returns list of (start, end) token_idx pairs."""
    idx_flag = (
        g[["token_idx", "is_disfluent"]]
        .drop_duplicates("token_idx")
        .sort_values("token_idx")
    )
    token_idxs = idx_flag["token_idx"].tolist()
    flags = idx_flag["is_disfluent"].tolist()
    spans = []
    i, n = 0, len(token_idxs)
    while i < n:
        if not flags[i]:
            i += 1
            continue
        start = i
        while i < n and flags[i]:
            i += 1
        spans.append((token_idxs[start], token_idxs[i - 1]))
    return spans


def position_bucket(relative_center):
    if relative_center < 1 / 3:
        return "initial"
    if relative_center < 2 / 3:
        return "medial"
    return "final"


def build_spans(dep_df):
    """
    One row per contiguous disfluent span: utt_id, lang, span_start, span_end,
    span_len, span_text, num_anchors, anchor_deprel, anchor_upos, coherent,
    plus linear-position fields (relative_center in [0, 1], position_bucket
    in {initial, medial, final}) -- literal "where in the sentence", distinct
    from the syntactic-role fields above.
    "Anchor" = word(s) in the span whose head lies outside the span (ROOT counts
    as outside); a span is coherent iff it has exactly one anchor.
    """
    rows = []
    for (utt_id, lang), g in dep_df.groupby(["utt_id", "lang"]):
        g = g.sort_values("token_idx")
        utt_len = g["token_idx"].max() + 1
        for start, end in disfluent_spans_from_tokens(g):
            span_rows = g[(g["token_idx"] >= start) & (g["token_idx"] <= end)]
            anchors = span_rows[
                (span_rows["head_token_idx"] < start) | (span_rows["head_token_idx"] > end)
            ]
            relative_center = ((start + end) / 2) / (utt_len - 1) if utt_len > 1 else 0.5
            rows.append({
                "utt_id": utt_id,
                "lang": lang,
                "span_start": start,
                "span_end": end,
                "span_len": end - start + 1,
                "span_text": " ".join(span_rows.sort_values("token_idx")["token"]),
                "utt_len": utt_len,
                "relative_center": relative_center,
                "position_bucket": position_bucket(relative_center),
                "num_anchors": len(anchors),
                "anchor_deprel": anchors.iloc[0]["deprel"] if len(anchors) >= 1 else None,
                "anchor_upos": anchors.iloc[0]["upos"] if len(anchors) >= 1 else None,
                "coherent": len(anchors) == 1,
            })
    return pd.DataFrame(rows)


def attach_fluent_neighbors(spans_df, fluent_df):
    """
    For each span, find the nearest fluent CONTENT word (upos != PUNCT) to its
    left and right, using fluent_df's orig_token_idx (which lives in the same
    index space as span_start/span_end). Adds left_/right_ token/upos/deprel
    columns to spans_df.
    """
    content = fluent_df[fluent_df["upos"] != "PUNCT"].sort_values(
        ["utt_id", "lang", "orig_token_idx"]
    )
    grouped = {key: g for key, g in content.groupby(["utt_id", "lang"])}

    left_cols = {"left_token": [], "left_upos": [], "left_deprel": []}
    right_cols = {"right_token": [], "right_upos": [], "right_deprel": []}
    for _, span in spans_df.iterrows():
        g = grouped.get((span["utt_id"], span["lang"]))
        before = g[g["orig_token_idx"] < span["span_start"]] if g is not None else None
        after = g[g["orig_token_idx"] > span["span_end"]] if g is not None else None

        if before is not None and not before.empty:
            r = before.iloc[-1]
            left_cols["left_token"].append(r["token"])
            left_cols["left_upos"].append(r["upos"])
            left_cols["left_deprel"].append(r["deprel"])
        else:
            left_cols["left_token"].append(None)
            left_cols["left_upos"].append(None)
            left_cols["left_deprel"].append(None)

        if after is not None and not after.empty:
            r = after.iloc[0]
            right_cols["right_token"].append(r["token"])
            right_cols["right_upos"].append(r["upos"])
            right_cols["right_deprel"].append(r["deprel"])
        else:
            right_cols["right_token"].append(None)
            right_cols["right_upos"].append(None)
            right_cols["right_deprel"].append(None)

    spans_df = spans_df.copy()
    for col, vals in {**left_cols, **right_cols}.items():
        spans_df[col] = vals
    return spans_df


def proportions(df, value_col):
    counts = df.groupby(["lang", value_col]).size().reset_index(name="count")
    totals = df.groupby("lang").size().rename("total")
    counts = counts.merge(totals, on="lang")
    counts["proportion"] = counts["count"] / counts["total"]
    return counts.sort_values(["lang", "proportion"], ascending=[True, False])


def save_heatmap(counts, value_col, out_png, title):
    pivot = counts.pivot_table(index="lang", columns=value_col, values="proportion", fill_value=0)
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(pivot.columns)), max(3, 0.4 * len(pivot.index))))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="proportion")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def save_transition_heatmap(trans_df, en_col, x_col, out_png, title):
    """Row-normalized heatmap: given the EN role (rows), distribution over target roles (columns)."""
    pivot = trans_df.pivot_table(index=en_col, columns=x_col, values="count", fill_value=0, aggfunc="sum")
    if pivot.empty:
        return
    row_sums = pivot.sum(axis=1)
    pivot = pivot.div(row_sums, axis=0)
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(pivot.columns)), max(3, 0.4 * len(pivot.index))))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("target role")
    ax.set_ylabel("EN role")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="P(target role | EN role)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def role_by_language(spans_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    coherent = spans_df[spans_df["coherent"]]

    deprel_counts = proportions(coherent, "anchor_deprel")
    deprel_counts.to_csv(os.path.join(output_dir, "anchor_deprel_by_language.csv"), index=False)
    save_heatmap(deprel_counts, "anchor_deprel", os.path.join(output_dir, "anchor_deprel_by_language.png"),
                 "Disfluency anchor deprel, by language (coherent spans only)")

    upos_counts = proportions(coherent, "anchor_upos")
    upos_counts.to_csv(os.path.join(output_dir, "anchor_upos_by_language.csv"), index=False)
    save_heatmap(upos_counts, "anchor_upos", os.path.join(output_dir, "anchor_upos_by_language.png"),
                 "Disfluency anchor upos, by language (coherent spans only)")

    left_counts = proportions(spans_df, "left_deprel")
    left_counts.to_csv(os.path.join(output_dir, "left_deprel_by_language.csv"), index=False)
    save_heatmap(left_counts, "left_deprel", os.path.join(output_dir, "left_deprel_by_language.png"),
                 "Deprel of nearest fluent CONTENT word to the LEFT of a disfluency, by language")

    right_counts = proportions(spans_df, "right_deprel")
    right_counts.to_csv(os.path.join(output_dir, "right_deprel_by_language.csv"), index=False)
    save_heatmap(right_counts, "right_deprel", os.path.join(output_dir, "right_deprel_by_language.png"),
                 "Deprel of nearest fluent CONTENT word to the RIGHT of a disfluency, by language")


def reliability(spans_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    coherence = spans_df.groupby("lang")["coherent"].mean().reset_index(name="coherence_rate")
    coherence.to_csv(os.path.join(output_dir, "coherence_by_language.csv"), index=False)

    span_len = spans_df.groupby(["lang", "span_len"]).size().reset_index(name="count")
    span_len.to_csv(os.path.join(output_dir, "span_length_distribution.csv"), index=False)


def position_by_language(spans_df, output_dir):
    """
    Literal linear position of disfluencies within the utterance (0=start,
    1=end) -- distinct from syntactic role. Uses ALL spans, not just coherent
    ones, since position doesn't depend on how cleanly the span attaches.
    """
    os.makedirs(output_dir, exist_ok=True)

    bucket_counts = proportions(spans_df, "position_bucket")
    bucket_counts.to_csv(os.path.join(output_dir, "position_bucket_by_language.csv"), index=False)
    save_heatmap(bucket_counts, "position_bucket", os.path.join(output_dir, "position_bucket_by_language.png"),
                 "Disfluency position within the utterance, by language")

    summary = spans_df.groupby("lang")["relative_center"].describe()[["mean", "50%", "std"]]
    summary.columns = ["mean", "median", "std"]
    summary.reset_index().to_csv(os.path.join(output_dir, "relative_position_summary.csv"), index=False)

    langs = sorted(spans_df["lang"].unique())
    data = [spans_df[spans_df["lang"] == lang]["relative_center"].values for lang in langs]
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(langs)), 4))
    ax.boxplot(data, labels=langs, showmeans=True)
    ax.set_ylabel("relative position in utterance (0=start, 1=end)")
    ax.set_title("Disfluency position within the utterance, by language")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "relative_position_boxplot.png"), dpi=150)
    plt.close(fig)


def en_to_target(spans_df, output_dir, source_lang="en"):
    heatmap_dir = os.path.join(output_dir, "heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)

    en_spans = spans_df[(spans_df["lang"] == source_lang) & (spans_df["coherent"])]
    n_utts_total = spans_df["utt_id"].nunique()

    transition_deprel_all, transition_upos_all, coverage_rows, shift_rows = [], [], [], []
    for lang in sorted(spans_df["lang"].unique()):
        if lang == source_lang:
            continue
        x_spans = spans_df[(spans_df["lang"] == lang) & (spans_df["coherent"])]

        # Cross-product of all coherent EN spans x all coherent target spans in the
        # same utterance -- a co-occurrence table, not a strict 1:1 alignment.
        pairs = en_spans[["utt_id", "anchor_deprel", "anchor_upos"]].merge(
            x_spans[["utt_id", "anchor_deprel", "anchor_upos"]],
            on="utt_id", suffixes=("_en", "_x"),
        )

        deprel_trans = pairs.groupby(["anchor_deprel_en", "anchor_deprel_x"]).size().reset_index(name="count")
        deprel_trans.insert(0, "target_lang", lang)
        transition_deprel_all.append(deprel_trans)

        upos_trans = pairs.groupby(["anchor_upos_en", "anchor_upos_x"]).size().reset_index(name="count")
        upos_trans.insert(0, "target_lang", lang)
        transition_upos_all.append(upos_trans)

        save_transition_heatmap(
            deprel_trans, "anchor_deprel_en", "anchor_deprel_x",
            os.path.join(heatmap_dir, f"en_to_{lang}_deprel.png"),
            f"EN -> {lang}: disfluency anchor deprel co-occurrence (n={len(pairs)} pairs)",
        )

        n_en_utts = en_spans["utt_id"].nunique()
        n_x_utts = x_spans["utt_id"].nunique()
        n_both = len(set(en_spans["utt_id"]) & set(x_spans["utt_id"]))
        coverage_rows.append({
            "target_lang": lang,
            "n_utts_total": n_utts_total,
            "n_utts_with_en_coherent_span": n_en_utts,
            "n_utts_with_target_coherent_span": n_x_utts,
            "n_utts_with_both": n_both,
            "n_role_pairs": len(pairs),
        })

        en_counts = spans_df[spans_df["lang"] == source_lang].groupby("utt_id").size().rename("n_en")
        x_counts = spans_df[spans_df["lang"] == lang].groupby("utt_id").size().rename("n_x")
        span_counts = pd.concat([en_counts, x_counts], axis=1).fillna(0).reset_index()
        span_counts["target_lang"] = lang
        span_counts["diff"] = span_counts["n_x"] - span_counts["n_en"]
        shift_rows.append(span_counts)

    pd.concat(transition_deprel_all, ignore_index=True).to_csv(
        os.path.join(output_dir, "transition_deprel_all.csv"), index=False
    )
    pd.concat(transition_upos_all, ignore_index=True).to_csv(
        os.path.join(output_dir, "transition_upos_all.csv"), index=False
    )
    pd.DataFrame(coverage_rows).to_csv(os.path.join(output_dir, "coverage_summary.csv"), index=False)
    pd.concat(shift_rows, ignore_index=True).to_csv(
        os.path.join(output_dir, "span_count_shift.csv"), index=False
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dep_csv", default="dependency_results.csv")
    parser.add_argument("--fluent_csv", default="fluent_parse_results.csv")
    parser.add_argument("--output", default="analysis_results")
    parser.add_argument("--source_lang", default="en", help="Reference language for the shift analysis")
    args = parser.parse_args()

    dep_df = pd.read_csv(args.dep_csv)
    fluent_df = pd.read_csv(args.fluent_csv)

    print("Building disfluent spans from token-level parse...")
    spans_df = build_spans(dep_df)
    print(f"  {len(spans_df)} spans across {spans_df['lang'].nunique()} languages "
          f"({spans_df['coherent'].mean():.1%} coherent)")

    print("Attaching nearest fluent content-word neighbors...")
    spans_df = attach_fluent_neighbors(spans_df, fluent_df)

    reliability(spans_df, os.path.join(args.output, "reliability"))
    role_by_language(spans_df, os.path.join(args.output, "role_by_language"))
    position_by_language(spans_df, os.path.join(args.output, "position_by_language"))
    en_to_target(spans_df, os.path.join(args.output, "en_to_target"), source_lang=args.source_lang)

    print(f"Done. Results in {args.output}/")


if __name__ == "__main__":
    main()
