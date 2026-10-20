import collections
import csv
import json
import os
import statistics
import numpy as np
import scipy.stats
import subset2evaluate.evaluate

DEFAULT_LLM_JUDGE_CSV = "../../benchmarking/results/llm_judge/llm_judge_per_item.csv"
DEFAULT_TRANSLATIONS_PER_ITEM_CSV = "../../benchmarking/results/translations/evaluation_results_per_item.csv"
DEFAULT_HUMAN_JSON = "../annotations_human_scoring.json"
DEFAULT_OUTPUT_CSV = "../../benchmarking/results/correlation/correlation_results.csv"

# Maps each llm_judge/human "system" identifier onto the (model_type, model, prompt_type)
# that identifies its rows in evaluation_results_per_item.csv (produced by eval_translations.py).
SYSTEM_TO_PER_ITEM_FILTER = {
    "gold_cascaded/tower": ("gold_cascaded", "tower", "standard"),
    "end2end/canary": ("end2end", "canary", "standard"),
    "speechllm/phi_multimodal": ("speechllm", "phi_multimodal", "standard"),
    "end2end/gemini-2.5-flash": ("commercial_models", "gemini-2.5-flash", "standard-noextra"),
}


def load_human_ratings_and_item_ids(input_json):
    import re
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Key by (lang, wav_filename, system) so that different annotators who received
    # different utterances at the same item_i position are never mixed together.
    # Previously keyed by (lang, item_i, system), but item_i is 0-19 for every
    # annotator batch regardless of which utterance they saw, causing ratings for
    # different utterances to be averaged as if they were the same item.
    ratings = collections.defaultdict(list)
    # item_id_lookup is kept for API compatibility but is now identity: wav -> wav.
    item_id_lookup = {}
    for campaign, sessions in data.items():
        lang = campaign.split("_")[-1]
        for session in sessions:
            if session.get("item") is None or len(session["item"]) != 1:
                continue
            src_html = session["item"][0].get("src", "")
            m = re.search(r"preprocessed_audio_switchboard/([^\"]+?)(?:\.wav)?\"", src_html)
            if not m:
                continue
            wav = m.group(1)
            item_id_lookup[(lang, wav)] = wav

            annotation = session["annotation"][0]
            if not isinstance(annotation, dict):
                continue
            for system, model_kv in annotation.items():
                sliders = model_kv.get("sliders", {})
                style = sliders.get("Style preservation")
                meaning = sliders.get("Meaning preservation")
                if style is None or meaning is None:
                    continue
                ratings[(lang, wav, system)].append((style, meaning))
    return ratings, item_id_lookup


def load_llm_judge_per_item(csv_path):
    scores = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["lang"].lower(), row["item_id"].replace(".wav", ""), row["system"])
            scores[key] = {
                "style": float(row["llm_judge_style"]),
                "meaning": float(row["llm_judge_meaning"]),
                "overall": float(row["llm_judge_overall"]),
            }
    return scores


def load_translation_metrics_per_item(csv_path):
    """Returns chrf_scores, comet_scores: dict[(lang, item_id, system)] = value, restricted to
    the 4 systems in SYSTEM_TO_PER_ITEM_FILTER and re-keyed onto their llm_judge/human system
    identifiers (e.g. 'gold_cascaded/tower') rather than the raw (model_type, model) columns."""
    chrf_scores, comet_scores = {}, {}
    if not os.path.exists(csv_path):
        return chrf_scores, comet_scores

    filter_lookup = {v: k for k, v in SYSTEM_TO_PER_ITEM_FILTER.items()}
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            filter_key = (row["model_type"], row["model"], row["prompt_type"])
            system = filter_lookup.get(filter_key)
            if system is None:
                continue
            key = (row["target_lang"].lower(), row["sample_id"], system)
            if row["chrf_score"]:
                chrf_scores[key] = float(row["chrf_score"])
            if row["comet_score"]:
                comet_scores[key] = float(row["comet_score"])
    return chrf_scores, comet_scores


# segment-level
def segment_level(data_lang):
    docs = collections.defaultdict(list)
    for line in data_lang:
        docs[line["doc_id"]].append(line)

    corrs = [
        scipy.stats.kendalltau(
            [line["score"] for line in doc],
            [line["score_pred"] for line in doc],
            variant="b",
        ).correlation
        for doc in docs.values()
        if len(doc) >= 2
    ]
    return statistics.mean([x for x in corrs if not np.isnan(x)])


# system-level
def system_level(data_lang):
    data_coll = collections.defaultdict(list)
    for line in data_lang:
        data_coll[line["doc_id"]].append(line)
    data_coll = [
        {
            "scores": {
                line["tgt_system"]: {
                    "score": line["score"],
                    "score_pred": line["score_pred"],
                }
                for line in doc
            }
        }
        for doc in data_coll.values()
    ]
    systems = set.union(*[set(doc["scores"].keys()) for doc in data_coll])
    data_coll = [doc for doc in data_coll if set(doc["scores"].keys()) == systems]
    for doc in data_coll:
        doc["scores"] = {system: doc["scores"][system] for system in systems}
    return subset2evaluate.evaluate.eval_subset_spa(data_coll, data_coll, metric=("score", "score_pred"))


def run_correlation(data, metrics):
    results_syslevel = collections.defaultdict(dict)
    results_seglevel = collections.defaultdict(dict)
    for lang in set(line["src_lang"] + line["tgt_lang"] for line in data):
        data_lang = [line for line in data if line["src_lang"] + line["tgt_lang"] == lang]
        for metric in metrics:
            data_lang_metric = [{**line, "score_pred": line["score_pred"][metric]} for line in data_lang]
            results_syslevel[metric][lang] = system_level(data_lang_metric)
            results_seglevel[metric][lang] = segment_level(data_lang_metric)

    print("SEGMENT-LEVEL")
    for metric, scores in sorted(results_seglevel.items()):
        avg_score = statistics.mean(scores.values())
        print(f"{metric:>15}: {avg_score:<5.1%}", end=" | ")
        print(" ".join(f"{lang}:{score:<5.1%}" for lang, score in scores.items()))

    print("SYSTEM-LEVEL")
    for metric, scores in sorted(results_syslevel.items()):
        avg_score = statistics.mean(scores.values())
        print(f"{metric:>15}: {avg_score:<5.1%}", end=" | ")
        print(" ".join(f"{lang}:{score:<5.1%}" for lang, score in scores.items()))

    return results_seglevel, results_syslevel


def build_data(human_axis, human_ratings, item_id_lookup, metric_sources):
    """metric_sources: dict[metric_name] -> dict[(lang, item_id, system)] = value.
    Only rows present in human ratings AND every metric source are included."""
    data = []
    for (lang, wav, system), raw_ratings in human_ratings.items():
        key = (lang, wav, system)
        if not all(key in src for src in metric_sources.values()):
            continue
        if human_axis == "overall":
            human_value = sum((r[0] + r[1]) / 2 for r in raw_ratings) / len(raw_ratings)
        else:
            idx = 0 if human_axis == "style" else 1
            human_value = sum(r[idx] for r in raw_ratings) / len(raw_ratings)
        data.append({
            "doc_id": f"{lang}_{wav}",
            "src_lang": "en",
            "tgt_lang": lang,
            "tgt_system": system,
            "score": human_value,
            "score_pred": {name: src[key] for name, src in metric_sources.items()},
        })
    return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Correlate human Style/Meaning/Overall judgments with automatic metrics.")
    parser.add_argument("--llm-judge-csv", default=DEFAULT_LLM_JUDGE_CSV, help="Per-item LLM-judge scores (from eval_llm_judge.py)")
    parser.add_argument("--llm-judge-label", default="llm_judge", help="Metric label for llm-judge in output (e.g. llm_judge_ref)")
    parser.add_argument("--llm-judge-ref-csv", default=None, help="Optional second per-item LLM-judge CSV (reference-based) shown alongside the first")
    parser.add_argument("--llm-judge-ref-label", default="llm_judge_ref", help="Metric label for the reference-based judge")
    parser.add_argument("--translations-per-item-csv", default=DEFAULT_TRANSLATIONS_PER_ITEM_CSV, help="Per-item chrF/COMET scores (from eval_translations.py --per-item-output)")
    parser.add_argument("--human-json", default=DEFAULT_HUMAN_JSON, help="Human annotation JSON (annotations_human_scoring.json)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_CSV, help="Where to save the long-format correlation results CSV")
    args = parser.parse_args()

    human_ratings, item_id_lookup = load_human_ratings_and_item_ids(args.human_json)
    llm_judge_scores = load_llm_judge_per_item(args.llm_judge_csv)
    llm_judge_ref_scores = load_llm_judge_per_item(args.llm_judge_ref_csv) if args.llm_judge_ref_csv else None
    chrf_scores, comet_scores = load_translation_metrics_per_item(args.translations_per_item_csv)
    if not chrf_scores and not comet_scores:
        print(f"No chrF/COMET per-item scores found in {args.translations_per_item_csv} -- run "
              f"eval_translations.py with --per-item-output (not --only-chrf, COMET needs GPU) "
              f"to produce it. Continuing with only llm_judge.\n")
    elif not comet_scores:
        print(f"No COMET scores found in {args.translations_per_item_csv} (chrF found) -- "
              f"--only-chrf runs don't produce per-item output. Continuing without COMET.\n")

    csv_rows = []
    for human_axis in ["style", "meaning", "overall"]:
        print(f"\n{'=' * 70}\nHuman {human_axis.upper()} vs. automatic metrics\n{'=' * 70}")

        metric_sources = {
            args.llm_judge_label: {k: v[human_axis] for k, v in llm_judge_scores.items()},
        }
        if llm_judge_ref_scores is not None:
            metric_sources[args.llm_judge_ref_label] = {k: v[human_axis] for k, v in llm_judge_ref_scores.items()}
        if chrf_scores:
            metric_sources["chrf"] = chrf_scores
        if comet_scores:
            metric_sources["comet"] = comet_scores

        data = build_data(human_axis, human_ratings, item_id_lookup, metric_sources)
        print(f"Matched (lang, item, system) cells: {len(data)}")
        results_seglevel, results_syslevel = run_correlation(data, metrics=list(metric_sources))

        for granularity, results in [("segment", results_seglevel), ("system", results_syslevel)]:
            for metric, scores in results.items():
                for lang, value in scores.items():
                    csv_rows.append((human_axis, metric, granularity, lang[2:], value))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["human_axis", "metric", "granularity", "lang", "value"])
        writer.writerows(csv_rows)
    print(f"\nSaved {len(csv_rows)} rows to {args.output}")
