import statistics
import pandas as pd
import matplotlib.pyplot as plt

LANG_COLS = ["zh", "es", "hi", "fr", "de", "it", "cs"]

MODEL_ORDER = ["llama", "tower", "gemini-2.5-flash", "gpt52"]

# Same combinatorial order as the appendix table: singles, pairs, then Disfluent. The
# synthetic triple-combo (Edited+INTJ+PRN) is omitted: once the source-text cleaning bug was
# fixed, it scores within noise of the natural Disfluent transcript, so Disfluent alone
# represents that point in the combination space (and is drawn with all three colors, since
# it is itself a mix of all three types).
# Fluent is the baseline every bar is measured against, so it isn't plotted as its own bar;
# a dashed line at 0 marks it instead.
CONDITION_ORDER = [
    "Intj Only", "Prn Only", "Edited Nodes Only",
    "Intj PRN", "Edited Intj", "Edited PRN",
    "Disfluent",
]
CONDITION_DISPLAY = {
    "Intj Only": "INTJ", "Prn Only": "PRN", "Edited Nodes Only": "EDITED",
    "Intj PRN": "INTJ+PRN", "Edited Intj": "Ed.+INTJ", "Edited PRN": "Ed.+PRN",
    "Disfluent": "Disfluent",
}
# Each combination bar is split into colored segments for the base types it contains, in
# this fixed order, so e.g. "EDITED+INTJ" always shows red then blue, left to right.
BASE_TYPE_COLOR = {"EDITED": "#C53A32", "INTJ": "tab:blue", "PRN": "tab:orange"}
CONDITION_BASE_TYPES = {
    "Intj Only": ["INTJ"], "Prn Only": ["PRN"], "Edited Nodes Only": ["EDITED"],
    "Intj PRN": ["INTJ", "PRN"], "Edited Intj": ["EDITED", "INTJ"], "Edited PRN": ["EDITED", "PRN"],
    "Disfluent": ["EDITED", "INTJ", "PRN"],
}

BASE_TYPE_GLOSS = {
    "INTJ": "Filled Pause",
    "PRN": "Discourse Marker",
    "EDITED": "False Start",
}


def compute_deltas(df):
    """Returns {condition: [delta, ...]} pooling all (model, language) pairs, where delta
    is that condition's score minus the SAME model's Fluent score for the SAME language."""
    deltas = {c: [] for c in CONDITION_ORDER}
    for model in MODEL_ORDER:
        model_rows = df[df["model"] == model].set_index("disfluency_type")
        fluent_row = model_rows.loc["Fluent"]
        for condition in CONDITION_ORDER:
            if condition not in model_rows.index:
                continue
            row = model_rows.loc[condition]
            for lang in LANG_COLS:
                deltas[condition].append(row[lang] - fluent_row[lang])
    return deltas


def draw_segmented_bar(ax, x, height, condition, width=0.65):
    types = CONDITION_BASE_TYPES[condition]
    seg_width = width / len(types)
    for i, t in enumerate(types):
        seg_x = x - width / 2 + seg_width * (i + 0.5)
        ax.bar([seg_x], [height], color=BASE_TYPE_COLOR[t], width=seg_width)


def plot_node_combos(input_csv, output_path):
    df = pd.read_csv(input_csv)
    deltas = compute_deltas(df)

    conditions = [c for c in CONDITION_ORDER if deltas[c]]
    means = [statistics.mean(deltas[c]) for c in conditions]
    stdevs = [statistics.stdev(deltas[c]) for c in conditions]
    labels = [CONDITION_DISPLAY[c] for c in conditions]

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)

    for x, (condition, mean) in enumerate(zip(conditions, means)):
        draw_segmented_bar(ax, x, mean, condition)
    ax.errorbar(range(len(conditions)), means, yerr=stdevs, fmt="none",
                ecolor="black", elinewidth=1.2, capsize=4, zorder=3)

    ylim = max(abs(m) + s for m, s in zip(means, stdevs)) * 1.4
    ax.set_ylim(-ylim, ylim)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=22)
    ax.set_ylabel("$\\Delta$ COMET from Fluent", fontsize=22)
    ax.tick_params(axis="y", labelsize=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=BASE_TYPE_COLOR[t], label=f"{t} ({BASE_TYPE_GLOSS[t]})")
        for t in ["EDITED", "INTJ", "PRN"]
    ]
    legend = ax.legend(handles=legend_handles, loc="upper center", ncol=3, fontsize=18,
                        handlelength=1.0, handleheight=1.0,
                        columnspacing=0.8, handletextpad=0.4)
    legend.get_frame().set_facecolor("#e8e8e8")
    legend.get_frame().set_edgecolor("none")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved node-combos plot to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bar chart of COMET delta from fluent, by disfluency node combination.")
    parser.add_argument("--input", default="results/node_combos/analysis_node_combos_en-x_structured.csv")
    parser.add_argument("--output", default="results/node_combos/node_combos_delta.pdf")
    args = parser.parse_args()
    plot_node_combos(args.input, args.output)
