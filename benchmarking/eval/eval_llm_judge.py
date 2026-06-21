import os
import pandas as pd

LANGUAGES = ["zh", "es", "hi", "fr", "de", "it", "sw", "cs", "ar"]
SYSTEMS = ["gold_cascaded/tower", "end2end/canary", "speechllm/phi_multimodal", "end2end/gemini-2.5-flash"]


def aggregate_llm_judge(input_csv, output_dir):
    df = pd.read_csv(input_csv)
    df["lang"] = df["lang"].str.lower()

    # Average over rotation: each system sits in every position (1-4) exactly once
    # across its 4 rotations for a given (lang, item_id), so this also cancels position bias.
    per_item = (
        df.groupby(["lang", "item_id", "system"])[["style", "meaning"]]
        .mean()
        .reset_index()
        .rename(columns={"style": "llm_judge_style", "meaning": "llm_judge_meaning"})
    )
    per_item["llm_judge_overall"] = per_item[["llm_judge_style", "llm_judge_meaning"]].mean(axis=1)

    summary = (
        per_item.groupby(["system", "lang"])[["llm_judge_style", "llm_judge_meaning", "llm_judge_overall"]]
        .mean()
        .reset_index()
    )

    os.makedirs(output_dir, exist_ok=True)
    per_item.to_csv(os.path.join(output_dir, "llm_judge_per_item.csv"), index=False)
    summary.to_csv(os.path.join(output_dir, "llm_judge_summary.csv"), index=False)

    for metric in ["llm_judge_style", "llm_judge_meaning", "llm_judge_overall"]:
        table = summary.pivot(index="system", columns="lang", values=metric).reindex(index=SYSTEMS, columns=LANGUAGES)
        table.round(2).to_csv(os.path.join(output_dir, f"{metric}_table.csv"))

    print(f"Saved per-item scores, summary, and per-metric tables to {output_dir}")
    print("\nSummary preview:")
    print(summary.round(2).to_string(index=False))

    return per_item, summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate reference-free LLM-judge scores over rotation/position.")
    parser.add_argument("--input", default="results/llm_judge_scores_without_references.csv",
                         help="Path to the raw llm_judge_scores_without_references.csv")
    parser.add_argument("--output-dir", default="results/llm_judge",
                         help="Directory to write aggregated results to")
    args = parser.parse_args()
    aggregate_llm_judge(args.input, args.output_dir)
