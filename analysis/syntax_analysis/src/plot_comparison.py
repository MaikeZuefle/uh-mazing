"""Paper figures comparing disfluency placement in system outputs with the reference translations."""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper_style import (LANG_COLOR, LANG_DISPLAY, LLM_JUDGE_SYSTEM_ORDER, SYSTEM_DISPLAY, SYSTEM_MARKER,
                         SYSTEM_MARKER_SIZE, load_comparison)

HERE = os.path.dirname(os.path.abspath(__file__))
PRONOUN_GROUPS = [("Subject can be omitted", ["cs", "es", "it"]), ("Subject required", ["de", "fr"])]
TVD_CAP = 40
TVD_COMBINED_CAP = 35
APPENDIX_TEXT_SCALE = 1.2
REFERENCE_COLOR = "#222222"
SYSTEM_COLOR = "#9a9a9a"


def style_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", labelsize=13)


def system_handles(color):
    return [plt.Line2D([], [], marker=SYSTEM_MARKER[s], color="white", markerfacecolor=color,
                       markeredgecolor="black" if color == "white" else "none",
                       markersize=12 * (SYSTEM_MARKER_SIZE[s] / 190) ** 0.5, linestyle="none",
                       label=SYSTEM_DISPLAY[s])
            for s in LLM_JUDGE_SYSTEM_ORDER]


def add_legend(fig, handles):
    legend = fig.legend(handles=handles, fontsize=16, loc="center left", bbox_to_anchor=(1.0, 0.5), borderpad=1.0)
    legend.get_frame().set_facecolor("#e8e8e8")
    legend.get_frame().set_edgecolor("none")


def draw_kept_vs_tvd(ax, df, column, cap):
    style_axes(ax)
    ax.scatter([100], [0], marker="*", s=500, color="gold", edgecolors="none", zorder=3, clip_on=False)
    for system_id, group in df.groupby("system_id"):
        ax.scatter(group["kept_pct"], group[column].clip(upper=cap),
                   c=[LANG_COLOR[l] for l in group["lang"]], marker=SYSTEM_MARKER[system_id],
                   s=SYSTEM_MARKER_SIZE[system_id], edgecolors="white", linewidths=0.4, zorder=2, clip_on=False)
    for _, row in df[df[column] > cap].iterrows():
        ax.annotate(f"{row[column]:.0f}%", (row["kept_pct"], cap), xytext=(9, -4),
                    textcoords="offset points", fontsize=12, color="#555555")
    ax.set_xlabel("Disfluencies kept (% of reference)", fontsize=17)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, cap)


def annotate_ideal(ax, text_height, text="Ideal: as many disfluencies\nas the reference, in the same\ngrammatical context"):
    ax.annotate(text,
                xy=(99, 0.03), xycoords=("data", "axes fraction"), xytext=(103, text_height),
                textcoords=("data", "axes fraction"), ha="right", va="bottom", fontsize=14, color="#555555",
                arrowprops={"arrowstyle": "->", "color": "#555555", "linewidth": 1})


def save_with_scatter_legend(fig, df, out_path):
    ideal = plt.Line2D([], [], marker="*", color="white", markerfacecolor="gold", markeredgecolor="none",
                       markersize=16, linestyle="none", label="Ideal")
    langs = [plt.Line2D([], [], marker="o", color="white", markerfacecolor=LANG_COLOR[l], markeredgecolor="none",
                        markersize=12, linestyle="none", label=LANG_DISPLAY[l])
             for l in LANG_DISPLAY if l in set(df["lang"])]
    add_legend(fig, [ideal] + system_handles("white") + langs)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def draw_subject_pronouns(ax, df, groups=PRONOUN_GROUPS):
    style_axes(ax)
    offsets = dict(zip(LLM_JUDGE_SYSTEM_ORDER, [-0.21, -0.07, 0.07, 0.21]))
    positions, x = {}, 0.0
    for group_name, langs in groups:
        start = x
        for lang in langs:
            positions[lang] = x
            x += 1
        ax.text((start + x - 1) / 2, 98, group_name, ha="center", va="top", fontsize=14, color="#555555")
        x += 0.6

    for lang, pos in positions.items():
        rows = df[df["lang"] == lang]
        reference = rows.loc[rows["n_utterances"].idxmax(), "pct_predicates_pron_subj_ref"]
        ax.hlines(reference, pos - 0.32, pos + 0.32, color=REFERENCE_COLOR, linewidth=3, zorder=2)
        for _, row in rows.iterrows():
            ax.scatter(pos + offsets[row["system_id"]], row["pct_predicates_pron_subj_sys"],
                       marker=SYSTEM_MARKER[row["system_id"]], s=0.6 * SYSTEM_MARKER_SIZE[row["system_id"]],
                       color=SYSTEM_COLOR, edgecolors="white", linewidths=0.4, zorder=3)

    ax.set_xticks(list(positions.values()))
    ax.set_xticklabels([LANG_DISPLAY[l] for l in positions], fontsize=15)
    ax.set_xlim(min(positions.values()) - 0.6, max(positions.values()) + 0.6)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Predicates with a pronoun subject (%)", fontsize=17)


def plot_kept_vs_tvd_combined(df, out_path):
    plt.rcParams["font.family"] = "DejaVu Sans"
    df = df.assign(tvd_mean=(df["tvd_before"] + df["tvd_after"]) / 2)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    draw_kept_vs_tvd(ax, df, "tvd_mean", TVD_COMBINED_CAP)
    ax.set_ylabel("% of contexts differing from references", fontsize=17)
    annotate_ideal(ax, 0.72)
    save_with_scatter_legend(fig, df, out_path)


def scale_text(fig, factor):
    for text in fig.findobj(matplotlib.text.Text):
        text.set_fontsize(text.get_fontsize() * factor)


def plot_appendix(df, out_path):
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig = plt.figure(figsize=(16, 5.8))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.1], wspace=0.32)
    before = fig.add_subplot(grid[0])
    after = fig.add_subplot(grid[1], sharey=before)
    pronouns = fig.add_subplot(grid[2])
    for ax, side in zip([before, after], ["before", "after"]):
        draw_kept_vs_tvd(ax, df, f"tvd_{side}", TVD_CAP)
        ax.set_title(f"Word {side} the disfluency", fontsize=17, pad=12)
        ax.set_xlabel("Disfluencies kept (%)", fontsize=17)
    after.tick_params(labelleft=False)
    before.set_ylabel("% of grammatical contexts\ndiffering from references", fontsize=17)
    annotate_ideal(before, 0.74, "Ideal: all disfluencies kept,\nin the same context")
    draw_subject_pronouns(pronouns, df, [("Subject\noptional", ["cs", "es", "it"]), ("Subject\nrequired", ["de", "fr"])])
    pronouns.set_title("Explicit subject pronouns", fontsize=17, pad=12)
    pronouns.set_ylabel("Predicates with a\npronoun subject (%)", fontsize=17)
    scale_text(fig, APPENDIX_TEXT_SCALE)

    ideal = plt.Line2D([], [], marker="*", color="white", markerfacecolor="gold", markeredgecolor="none",
                       markersize=16, linestyle="none", label="Ideal")
    human = plt.Line2D([], [], color=REFERENCE_COLOR, linewidth=3, label="Human")
    langs = [plt.Line2D([], [], marker="o", color="white", markerfacecolor=LANG_COLOR[l], markeredgecolor="none",
                        markersize=12, linestyle="none", label=LANG_DISPLAY[l])
             for l in LANG_DISPLAY if l in set(df["lang"])]
    legend = fig.legend(handles=[ideal, human] + system_handles("white") + langs, fontsize=18, loc="center left",
                        bbox_to_anchor=(0.91, 0.5), handletextpad=0.4, borderpad=0.8, labelspacing=0.35)
    legend.get_frame().set_facecolor("#e8e8e8")
    legend.get_frame().set_edgecolor("none")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison_csv", default=os.path.join(HERE, "../output/generated_translations/comparison.csv"))
    parser.add_argument("--output", default=os.path.join(HERE, "../output/generated_translations/figures"))
    args = parser.parse_args()

    df = load_comparison(args.comparison_csv)
    os.makedirs(args.output, exist_ok=True)
    for ext in ["pdf", "png"]:
        plot_kept_vs_tvd_combined(df, os.path.join(args.output, f"kept_vs_tvd_combined.{ext}"))
        plot_appendix(df, os.path.join(args.output, f"appendix_context_and_pronouns.{ext}"))
    print(f"Wrote figures to {args.output}")


if __name__ == "__main__":
    main()
