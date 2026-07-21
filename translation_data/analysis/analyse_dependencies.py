#!/usr/bin/env python3
"""
Analysis of where disfluencies occur, built entirely from the raw per-token
CSVs dependency_parsing.py produces (dependency_results.csv,
fluent_parse_results.csv). This script never touches Stanza -- every
interpretive choice (what counts as a "span", how to find a "neighbor",
whether to skip punctuation) lives here, in plain pandas, so changing any of
it is a re-run of THIS script only, not a re-parse.

The core signal, reliable regardless of disfluency length and deliberately
excluding anything that requires trusting Stanza's parse of the disfluent
words themselves: "position" here means the nearest fluent CONTENT word
(punctuation skipped -- every punctuation mark is UD-tagged deprel=punct/
upos=PUNCT regardless of context, so it carries no signal) immediately
before/after each disfluent span, from the fluent-only parse. This never
asks Stanza to parse the disfluent words, so it's equally trustworthy for a
1-word filler or a 10-word false start.

Explicitly NOT reported: the disfluency's own deprel/upos (what we called
its "anchor" role), and literal linear position (initial/medial/final) --
both were tried and dropped: the anchor role because the overwhelming
majority of disfluencies longer than one word don't resolve to one clean
attachment point (coherence drops from 100% at span_len=1 to ~50% at
span_len=2 and keeps falling), and literal position because it wasn't
informative for the question being asked here.

Output files (written to --output dir):
  context_by_language/{before,after}_deprel_by_language.{csv,png}
  context_by_language/before_after_deprel_side_by_side.{png,pdf}
  context_by_language/before_after_deprel_diverging.{png,pdf} - red=after, blue=before
  basic_statistics/basic_statistics.csv     - avg utterance/disfluency length
                                               (words), utterance/span counts,
                                               per language (Chinese uses
                                               jieba word-segmentation, same
                                               as benchmarking/eval's chrF,
                                               since dependency_results.csv's
                                               tokens are per-CHARACTER for zh)
"""
import argparse
import os

import jieba
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


def build_spans(dep_df):
    """One row per contiguous disfluent span: utt_id, lang, span_start, span_end, span_len, span_text."""
    rows = []
    for (utt_id, lang), g in dep_df.groupby(["utt_id", "lang"]):
        g = g.sort_values("token_idx")
        for start, end in disfluent_spans_from_tokens(g):
            span_rows = g[(g["token_idx"] >= start) & (g["token_idx"] <= end)]
            rows.append({
                "utt_id": utt_id,
                "lang": lang,
                "span_start": start,
                "span_end": end,
                "span_len": end - start + 1,
                "span_text": " ".join(span_rows.sort_values("token_idx")["token"]),
            })
    return pd.DataFrame(rows)


def attach_fluent_neighbors(spans_df, fluent_df):
    """
    For each span, find the nearest fluent CONTENT word (upos != PUNCT)
    immediately before and after it, using fluent_df's orig_token_idx (which
    lives in the same index space as span_start/span_end). Adds
    before_/after_ token/upos/deprel columns to spans_df.
    """
    content = fluent_df[fluent_df["upos"] != "PUNCT"].sort_values(
        ["utt_id", "lang", "orig_token_idx"]
    )
    grouped = {key: g for key, g in content.groupby(["utt_id", "lang"])}

    before_cols = {"before_token": [], "before_upos": [], "before_deprel": []}
    after_cols = {"after_token": [], "after_upos": [], "after_deprel": []}
    for _, span in spans_df.iterrows():
        g = grouped.get((span["utt_id"], span["lang"]))
        before = g[g["orig_token_idx"] < span["span_start"]] if g is not None else None
        after = g[g["orig_token_idx"] > span["span_end"]] if g is not None else None

        if before is not None and not before.empty:
            r = before.iloc[-1]
            before_cols["before_token"].append(r["token"])
            before_cols["before_upos"].append(r["upos"])
            before_cols["before_deprel"].append(r["deprel"])
        else:
            before_cols["before_token"].append(None)
            before_cols["before_upos"].append(None)
            before_cols["before_deprel"].append(None)

        if after is not None and not after.empty:
            r = after.iloc[0]
            after_cols["after_token"].append(r["token"])
            after_cols["after_upos"].append(r["upos"])
            after_cols["after_deprel"].append(r["deprel"])
        else:
            after_cols["after_token"].append(None)
            after_cols["after_upos"].append(None)
            after_cols["after_deprel"].append(None)

    spans_df = spans_df.copy()
    for col, vals in {**before_cols, **after_cols}.items():
        spans_df[col] = vals
    return spans_df


def proportions(df, value_col):
    counts = df.groupby(["lang", value_col]).size().reset_index(name="count")
    totals = df.groupby("lang").size().rename("total")
    counts = counts.merge(totals, on="lang")
    counts["proportion"] = counts["count"] / counts["total"]
    return counts.sort_values(["lang", "proportion"], ascending=[True, False])


def top_categories(counts, value_col, top_n):
    return list(counts.groupby(value_col)["count"].sum().sort_values(ascending=False).head(top_n).index)


def abbrev_labels(categories):
    """UD deprel codes as abbreviation-style labels (e.g. nsubj -> Nsubj.) for axis
    ticks. Full names belong in the caption/figure glossary, not inline -- expanding
    them here (e.g. "nominal subject") would reintroduce the label-width/overlap
    issues already fixed for this figure."""
    return [str(c).capitalize() + "." for c in categories]


def lang_labels(langs):
    """Language codes uppercased for axis ticks (e.g. en -> EN)."""
    return [str(l).upper() for l in langs]


def save_heatmap(counts, value_col, out_png, title, categories=None):
    if categories is not None:
        counts = counts[counts[value_col].isin(categories)]
    pivot = counts.pivot_table(index="lang", columns=value_col, values="proportion", fill_value=0)
    if categories is not None:
        pivot = pivot.reindex(columns=categories)
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(pivot.columns)), max(3, 0.4 * len(pivot.index))))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(abbrev_labels(pivot.columns), rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(lang_labels(pivot.index))
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="proportion")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def savefig_both(fig, out_png):
    """Save both a .png (for quick viewing) and a .pdf (vector, for the paper) alongside it."""
    fig.savefig(out_png, dpi=150, bbox_inches="tight", pad_inches=0.3)
    fig.savefig(os.path.splitext(out_png)[0] + ".pdf", bbox_inches="tight", pad_inches=0.3)


def save_side_by_side_heatmap(before_counts, after_counts, categories, out_png, suptitle):
    before_pivot = before_counts[before_counts["before_deprel"].isin(categories)].pivot_table(
        index="lang", columns="before_deprel", values="proportion", fill_value=0
    ).reindex(columns=categories)
    after_pivot = after_counts[after_counts["after_deprel"].isin(categories)].pivot_table(
        index="lang", columns="after_deprel", values="proportion", fill_value=0
    ).reindex(columns=categories)

    vmax = max(before_pivot.values.max(), after_pivot.values.max())
    fig, axes = plt.subplots(1, 2, figsize=(2.2 * len(categories) + 4, 1.3 * len(before_pivot.index) + 4))
    for ax, pivot, label in [(axes[0], before_pivot, "BEFORE"), (axes[1], after_pivot, "AFTER")]:
        im = ax.imshow(pivot.values, cmap="viridis", aspect="auto", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(abbrev_labels(pivot.columns), rotation=45, ha="right", fontsize=32)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(lang_labels(pivot.index), fontsize=32)
        ax.set_title(label, fontsize=36)
    cbar = fig.colorbar(im, ax=axes, orientation="horizontal", pad=0.18, fraction=0.05)
    cbar.set_label("proportion", fontsize=26)
    cbar.ax.tick_params(labelsize=24)
    fig.suptitle(suptitle, fontsize=28)
    savefig_both(fig, out_png)
    plt.close(fig)


def save_diverging_heatmap(before_counts, after_counts, categories, out_png):
    """Single heatmap of after_proportion - before_proportion: positive (red) = more
    common AFTER the disfluency, negative (blue) = more common BEFORE it."""
    b = before_counts[before_counts["before_deprel"].isin(categories)][["lang", "before_deprel", "proportion"]]
    b = b.rename(columns={"before_deprel": "deprel", "proportion": "before"})
    a = after_counts[after_counts["after_deprel"].isin(categories)][["lang", "after_deprel", "proportion"]]
    a = a.rename(columns={"after_deprel": "deprel", "proportion": "after"})
    merged = b.merge(a, on=["lang", "deprel"], how="outer").fillna(0)
    merged["diff"] = merged["after"] - merged["before"]

    pivot = merged.pivot_table(index="lang", columns="deprel", values="diff", fill_value=0).reindex(columns=categories)
    if pivot.empty:
        return
    bound = max(abs(pivot.values.min()), abs(pivot.values.max()), 1e-9)
    fig, ax = plt.subplots(figsize=(1.1 * len(pivot.columns) + 2.5, 0.75 * len(pivot.index) + 3.5))
    im = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto", vmin=-bound, vmax=bound)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(abbrev_labels(pivot.columns), rotation=45, ha="right", fontsize=40)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(lang_labels(pivot.index), fontsize=40)
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.3, fraction=0.05, aspect=40)
    cbar.ax.tick_params(labelsize=32)
    cbar.ax.text(0.0, -3.2, "before disfluency ←", transform=cbar.ax.transAxes, ha="left", va="top", fontsize=32)
    cbar.ax.text(1.0, -3.2, "→ after disfluency", transform=cbar.ax.transAxes, ha="right", va="top", fontsize=32)
    fig.tight_layout()
    savefig_both(fig, out_png)
    plt.close(fig)


def context_by_language(spans_df, output_dir, top_n=10):
    """
    Deprel/upos of the nearest fluent CONTENT word immediately before/after
    each span -- reliable for any span length, since it never parses the
    disfluent words.
    """
    os.makedirs(output_dir, exist_ok=True)

    before_counts = proportions(spans_df, "before_deprel")
    before_counts.to_csv(os.path.join(output_dir, "before_deprel_by_language.csv"), index=False)

    after_counts = proportions(spans_df, "after_deprel")
    after_counts.to_csv(os.path.join(output_dir, "after_deprel_by_language.csv"), index=False)

    # Shared top-N categories (by combined before+after frequency) so all three figures
    # below are directly comparable and not cluttered by the long tail of rare deprels.
    combined_counts = pd.concat([
        before_counts.rename(columns={"before_deprel": "deprel"})[["deprel", "count"]],
        after_counts.rename(columns={"after_deprel": "deprel"})[["deprel", "count"]],
    ])
    categories = top_categories(combined_counts, "deprel", top_n)

    save_heatmap(before_counts, "before_deprel", os.path.join(output_dir, "before_deprel_by_language.png"),
                 "Deprel of nearest fluent CONTENT word BEFORE a disfluency, by language",
                 categories=categories)
    save_heatmap(after_counts, "after_deprel", os.path.join(output_dir, "after_deprel_by_language.png"),
                 "Deprel of nearest fluent CONTENT word AFTER a disfluency, by language",
                 categories=categories)
    save_side_by_side_heatmap(
        before_counts, after_counts, categories,
        os.path.join(output_dir, "before_after_deprel_side_by_side.png"),
        "Deprel of nearest fluent CONTENT word, BEFORE vs AFTER a disfluency",
    )
    save_diverging_heatmap(
        before_counts, after_counts, categories,
        os.path.join(output_dir, "before_after_deprel_diverging.png"),
    )


def basic_statistics(dep_df, spans_df, output_dir):
    """
    Per language: average utterance length (words), average disfluency length
    (words per span), number of utterances, and number of disfluent spans.

    dependency_results.csv's token_idx is per-CHARACTER for Chinese (its
    references.csv column has space-separated characters, not words), so its
    raw token counts aren't comparable to the whitespace-delimited word
    counts used for every other language. Recomputed for zh using jieba
    word-segmentation, the same tokenizer benchmarking/eval/eval_translations.py
    uses for zh chrF scoring, from the characters already in dependency_results.csv
    (no need to re-read references.csv).
    """
    os.makedirs(output_dir, exist_ok=True)

    utt_len = dep_df.groupby(["utt_id", "lang"])["token_idx"].max().add(1).reset_index(name="utt_len")
    avg_utt_len = utt_len.groupby("lang")["utt_len"].mean().rename("avg_utterance_length_words")
    n_utts = utt_len.groupby("lang").size().rename("n_utterances")
    avg_span_len = spans_df.groupby("lang")["span_len"].mean().rename("avg_disfluency_length_words")
    n_spans = spans_df.groupby("lang").size().rename("n_disfluent_spans")

    stats = pd.concat([avg_utt_len, n_utts, avg_span_len, n_spans], axis=1).reset_index()

    if "zh" in stats["lang"].values:
        zh_tokens = (
            dep_df[dep_df["lang"] == "zh"]
            .drop_duplicates(["utt_id", "token_idx"])
            .sort_values(["utt_id", "token_idx"])
        )
        zh_utt_text = zh_tokens.groupby("utt_id")["token"].apply("".join)
        zh_avg_utt_words = zh_utt_text.apply(lambda t: len(list(jieba.cut(t)))).mean()

        zh_span_text = spans_df.loc[spans_df["lang"] == "zh", "span_text"]
        zh_avg_span_words = zh_span_text.apply(lambda t: len(list(jieba.cut(t.replace(" ", ""))))).mean()

        stats.loc[stats["lang"] == "zh", "avg_utterance_length_words"] = zh_avg_utt_words
        stats.loc[stats["lang"] == "zh", "avg_disfluency_length_words"] = zh_avg_span_words

    stats.to_csv(os.path.join(output_dir, "basic_statistics.csv"), index=False)
    write_basic_statistics_latex(stats, os.path.join(output_dir, "basic_statistics.tex"))
    return stats


def write_basic_statistics_latex(stats, out_tex):
    """Booktabs table of basic_statistics.csv, one row per language (n_utterances
    omitted: it's the same 80 for every language, so it belongs in prose, not a column)."""
    stats = stats.sort_values("lang")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        " & ".join(r"\textbf{" + h + "}" for h in [
            "Lang", "Avg.\\ utt.\\ length", "Avg.\\ disfl.\\ length", "\\# disfluencies",
        ]) + r" \\",
        r"\midrule",
    ]
    for _, row in stats.iterrows():
        lines.append(
            f"{lang_labels([row['lang']])[0]} & "
            f"{row['avg_utterance_length_words']:.1f} & "
            f"{row['avg_disfluency_length_words']:.1f} & "
            f"{int(row['n_disfluent_spans'])} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Average utterance length and disfluency length (in words), and total "
        r"number of disfluencies (contiguous disfluency-marked spans -- a single filler "
        r"word and a 5-word false start each count as one), per language. Chinese lengths "
        r"use jieba word-segmentation (see text) rather than raw token counts.}",
        r"\label{tab:basic-statistics}",
        r"\end{table}",
    ]
    with open(out_tex, "w") as f:
        f.write("\n".join(lines) + "\n")


def context_shift(spans_df, output_dir, source_lang="en"):
    """
    Is the before/after neighbor relationship PRESERVED on translation? E.g.
    if a disfluency sits right after the subject (before_deprel=nsubj) in
    English, does it still sit right after the subject in the target
    language, for the same utterance? Cross every source-language span
    against every target-language span in the same utterance (weighted so
    each utterance contributes equal total weight, same reasoning as
    elsewhere in this script) and report the match rate, both overall and
    broken down by the source language's before_deprel/after_deprel category
    so a specific question like "was it after subj, is it still after subj"
    is directly answerable from by_category.csv.
    """
    os.makedirs(output_dir, exist_ok=True)
    src = spans_df[spans_df["lang"] == source_lang][["utt_id", "before_deprel", "after_deprel"]]

    overall_rows, before_category_rows, after_category_rows = [], [], []
    for lang in sorted(spans_df["lang"].unique()):
        if lang == source_lang:
            continue
        tgt = spans_df[spans_df["lang"] == lang][["utt_id", "before_deprel", "after_deprel"]]
        pairs = src.merge(tgt, on="utt_id", suffixes=("_src", "_tgt"))
        if pairs.empty:
            continue
        pairs["weight"] = 1.0 / pairs.groupby("utt_id")["utt_id"].transform("size")
        pairs["before_match"] = pairs["before_deprel_src"] == pairs["before_deprel_tgt"]
        pairs["after_match"] = pairs["after_deprel_src"] == pairs["after_deprel_tgt"]

        def weighted_pct(df, col):
            return 100 * (df["weight"] * df[col]).sum() / df["weight"].sum()

        overall_rows.append({
            "target_lang": lang,
            "pct_before_preserved": weighted_pct(pairs, "before_match"),
            "pct_after_preserved": weighted_pct(pairs, "after_match"),
            "n_utts_with_both": pairs["utt_id"].nunique(),
        })

        for category, g in pairs.groupby("before_deprel_src"):
            before_category_rows.append({
                "target_lang": lang,
                "before_deprel_in_source": category,
                "pct_before_preserved": weighted_pct(g, "before_match"),
                "n_pairs": len(g),
            })

        for category, g in pairs.groupby("after_deprel_src"):
            after_category_rows.append({
                "target_lang": lang,
                "after_deprel_in_source": category,
                "pct_after_preserved": weighted_pct(g, "after_match"),
                "n_pairs": len(g),
            })

    pd.DataFrame(overall_rows).to_csv(os.path.join(output_dir, "context_shift_overall.csv"), index=False)

    before_df = pd.DataFrame(before_category_rows).sort_values(["target_lang", "before_deprel_in_source"])
    before_df.to_csv(os.path.join(output_dir, "context_shift_by_category.csv"), index=False)

    after_df = pd.DataFrame(after_category_rows).sort_values(["target_lang", "after_deprel_in_source"])
    after_df.to_csv(os.path.join(output_dir, "context_shift_by_category_after.csv"), index=False)

    write_context_shift_latex(
        before_df, after_df, os.path.join(output_dir, "context_shift_by_category.tex"),
        source_lang=source_lang,
    )


def write_context_shift_latex(before_df, after_df, out_tex, top_n=10, source_lang="en"):
    """
    Single booktabs table with two row-blocks (preceding word / following word),
    each showing its own top_n categories (by total n_pairs, same cutoff logic
    as the figures); columns are target languages. Cells are % of spans where
    that before/after-deprel relationship is preserved on translation, with the
    highest value in each row bolded. Missing category/language combinations (a
    deprel that never co-occurred with a source-language span in that language
    pair) show "--".
    """
    def block_pivot(category_df, category_col, value_col):
        totals = category_df.groupby(category_col)["n_pairs"].sum()
        categories = totals.sort_values(ascending=False).head(top_n).index.tolist()
        pivot = category_df[category_df[category_col].isin(categories)].pivot_table(
            index=category_col, columns="target_lang", values=value_col
        ).reindex(index=categories)
        return categories, pivot

    before_categories, before_pivot = block_pivot(before_df, "before_deprel_in_source", "pct_before_preserved")
    after_categories, after_pivot = block_pivot(after_df, "after_deprel_in_source", "pct_after_preserved")

    langs = sorted(set(before_pivot.columns) | set(after_pivot.columns))
    before_pivot = before_pivot.reindex(columns=langs)
    after_pivot = after_pivot.reindex(columns=langs)

    source_name = {"en": "English"}.get(source_lang, source_lang.capitalize())
    header_cells = [f"{source_name} word"] + lang_labels(langs)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\begin{tabular}{l" + "r" * len(langs) + "}",
        r"\toprule",
        " & ".join(r"\textbf{" + h + "}" for h in header_cells) + r" \\",
    ]

    has_missing = False

    def add_block(title, categories, pivot):
        nonlocal has_missing
        lines.append(r"\midrule")
        lines.append(r"\multicolumn{" + str(len(langs) + 1) + r"}{l}{\textit{" + title + r"}} \\")
        lines.append(r"\midrule")
        for category in categories:
            row = pivot.loc[category]
            valid = row.dropna()
            best = valid.idxmax() if not valid.empty else None
            row_vals = []
            for lang in langs:
                v = row[lang]
                if pd.isna(v):
                    has_missing = True
                    row_vals.append("--")
                elif lang == best:
                    row_vals.append(r"\textbf{" + f"{v:.1f}" + "}")
                else:
                    row_vals.append(f"{v:.1f}")
            lines.append(abbrev_labels([category])[0] + " & " + " & ".join(row_vals) + r" \\")

    add_block("Preceding word", before_categories, before_pivot)
    add_block("Following word", after_categories, after_pivot)

    caption = (
        f"Percentage of disfluencies whose nearest preceding/following fluent word's "
        f"dependency relation in {source_lang.upper()} is preserved after translation, "
        "by target language and source relation type. The highest value in each row is bolded."
    )
    if has_missing:
        caption += " `--' marks combinations with no co-occurring spans."
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{" + caption + "}",
        r"\label{tab:context-shift-by-category}",
        r"\end{table}",
    ]
    with open(out_tex, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dep_csv", default="dependency_results.csv")
    parser.add_argument("--fluent_csv", default="fluent_parse_results.csv")
    parser.add_argument("--output", default="analysis_results")
    parser.add_argument("--source_lang", default="en", help="Reference language for the context-shift comparison")
    args = parser.parse_args()

    dep_df = pd.read_csv(args.dep_csv)
    fluent_df = pd.read_csv(args.fluent_csv)

    print("Building disfluent spans from token-level parse...")
    spans_df = build_spans(dep_df)
    print(f"  {len(spans_df)} spans across {spans_df['lang'].nunique()} languages")

    print("Attaching nearest fluent content-word neighbors...")
    spans_df = attach_fluent_neighbors(spans_df, fluent_df)

    context_by_language(spans_df, os.path.join(args.output, "context_by_language"))
    basic_statistics(dep_df, spans_df, os.path.join(args.output, "basic_statistics"))
    context_shift(spans_df, os.path.join(args.output, "context_shift"), source_lang=args.source_lang)

    print(f"Done. Results in {args.output}/")


if __name__ == "__main__":
    main()
