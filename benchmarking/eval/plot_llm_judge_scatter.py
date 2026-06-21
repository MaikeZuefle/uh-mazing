import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SYSTEM_DISPLAY = {
    "end2end/canary": "Canary",
    "end2end/gemini-2.5-flash": "Gemini",
    "speechllm/phi_multimodal": "Phi-4",
    "gold_cascaded/tower": "Tower",
}
SYSTEM_MARKER = {
    "gold_cascaded/tower": "o",
    "end2end/canary": "s",
    "speechllm/phi_multimodal": "^",
    "end2end/gemini-2.5-flash": "D",
}
# 'o'/'^' render visually smaller than 's'/'D' at the same matplotlib `s` value, so they get
# a size boost to look equally sized on the page.
SYSTEM_MARKER_SIZE = {
    "gold_cascaded/tower": 230,
    "end2end/canary": 190,
    "speechllm/phi_multimodal": 250,
    "end2end/gemini-2.5-flash": 190,
}
LANG_DISPLAY = {
    "ar": "AR", "cs": "CS", "de": "DE", "es": "ES",
    "fr": "FR", "hi": "HI", "it": "IT", "zh": "ZH",
}

# Languages are colored by continent/region, with shades within each group, rather than
# arbitrary distinct hues, so the regional grouping is visible at a glance.
LANG_GROUPS = {
    "es": "Europe", "fr": "Europe", "de": "Europe", "it": "Europe", "cs": "Europe",
    "zh": "Asia", "hi": "Asia",
    "ar": "Middle East/Africa",
}


def build_lang_colors():
    cmaps = {"Europe": "Blues", "Asia": "Oranges", "Middle East/Africa": "Greens"}
    groups = {}
    for lang, group in LANG_GROUPS.items():
        groups.setdefault(group, []).append(lang)

    colors = {}
    for group, langs in groups.items():
        n = len(langs)
        # Spread shades across the full colormap range (clamped away from white/black) so
        # adjacent languages within a group stay visually distinguishable.
        span = np.linspace(0.25, 1.0, n) if n > 1 else [0.65]
        shades = plt.get_cmap(cmaps[group])(span)
        colors.update(zip(langs, shades))
    return colors


LANG_COLOR = build_lang_colors()


def plot_scatter(input_csv, output_path):
    df = pd.read_csv(input_csv)
    # Canary has no support for ar/hi/zh; those rows are degenerate (0, 0) judge scores, not
    # real quality scores, so we drop them rather than plot them as if they were data.
    df = df[~((df["llm_judge_style"] == 0) & (df["llm_judge_meaning"] == 0))]

    lo = 35
    hi = max(df["llm_judge_style"].max(), df["llm_judge_meaning"].max()) + 5

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=1, alpha=0.6, zorder=1)

    # Star in the top-right corner marks the ideal point: high style AND high meaning.
    ax.scatter([hi], [hi], marker="*", s=500, color="gold", edgecolors="none", zorder=3, clip_on=False)

    for system, group in df.groupby("system"):
        ax.scatter(
            group["llm_judge_style"], group["llm_judge_meaning"],
            c=[LANG_COLOR[lang] for lang in group["lang"]],
            marker=SYSTEM_MARKER[system], s=SYSTEM_MARKER_SIZE[system], edgecolors="white", linewidths=0.4,
            label=SYSTEM_DISPLAY[system], zorder=2,
        )

    ax.set_xlabel("Style Preservation", fontsize=19)
    ax.set_ylabel("Meaning Preservation", fontsize=19)
    ax.tick_params(axis="both", labelsize=13)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_box_aspect(1)

    system_handles = [
        plt.Line2D([], [], marker=SYSTEM_MARKER[s], color="white", markerfacecolor="white",
                    markeredgecolor="black", markersize=12 * (SYSTEM_MARKER_SIZE[s] / 190) ** 0.5,
                    label=SYSTEM_DISPLAY[s])
        for s in SYSTEM_DISPLAY
    ]
    ideal_handle = plt.Line2D([], [], marker="*", color="white", markerfacecolor="gold",
                               markeredgecolor="none", markersize=16, label="Ideal")
    lang_handles = [
        plt.Line2D([], [], marker="o", color="white", markerfacecolor=LANG_COLOR[l],
                    markeredgecolor="none", markersize=12, label=LANG_DISPLAY[l])
        for l in LANG_DISPLAY
    ]
    legend_fontsize = 16
    legend = fig.legend(handles=[ideal_handle] + system_handles + lang_handles, fontsize=legend_fontsize,
                         loc="center left", bbox_to_anchor=(1.04, 0.54), borderpad=1.0)
    legend.get_frame().set_facecolor("#e8e8e8")
    legend.get_frame().set_edgecolor("none")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved scatter plot to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scatter plot of LLM-judge style vs. meaning preservation.")
    parser.add_argument("--input", default="results/llm_judge/llm_judge_summary.csv")
    parser.add_argument("--output", default="results/llm_judge/style_vs_meaning_scatter.pdf")
    args = parser.parse_args()
    plot_scatter(args.input, args.output)
