"""Compare where disfluencies occur in system outputs with where they occur in the reference translations."""
import argparse
import os
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

from analyse_dependencies import attach_fluent_neighbors, build_spans, context_shift

HERE = os.path.dirname(os.path.abspath(__file__))


def is_punctuation(token):
    return all(unicodedata.category(c).startswith("P") for c in str(token))


def is_pronoun_subject(deprel, upos):
    return deprel.fillna("").str.startswith("nsubj") & (upos == "PRON")


def load_excluded(excluded_csv, system):
    """(lang, utterance) pairs of a system's outputs excluded as repetition loops or not in the target language."""
    excluded = pd.read_csv(excluded_csv)
    excluded = excluded[excluded["system"] == system]
    return set(zip(excluded["lang"].str.lower(), excluded["ID"]))


def without_excluded(df, excluded):
    return df[[key not in excluded for key in zip(df["lang"], df["utt_id"])]] if excluded else df


def load_spans(parse_dir, skip_pronoun_subjects=False, excluded=None):
    """Disfluent spans with their fluent neighbours, ignoring punctuation so its markup cannot split or create spans."""
    dep_df = without_excluded(pd.read_csv(os.path.join(parse_dir, "dependency_results.csv")), excluded)
    fluent_df = without_excluded(pd.read_csv(os.path.join(parse_dir, "fluent_parse_results.csv")), excluded)
    words = dep_df[~dep_df["token"].map(is_punctuation)]
    neighbours = fluent_df
    if skip_pronoun_subjects:
        neighbours = fluent_df[~is_pronoun_subject(fluent_df["deprel"], fluent_df["upos"])]
    spans = attach_fluent_neighbors(build_spans(words), neighbours)
    utts = dep_df.groupby("lang")["utt_id"].unique().apply(set).to_dict()
    return spans, utts, fluent_df


def pct_predicates_with_pronoun_subject(fluent_df, lang, utts):
    """Share of predicates (full verbs, and any word with a subject) whose subject is an explicit pronoun."""
    words = fluent_df[(fluent_df["lang"] == lang) & fluent_df["utt_id"].isin(utts)]
    subjects = words[words["deprel"].fillna("").str.startswith("nsubj")]
    pron_subjects = subjects[subjects["upos"] == "PRON"]
    predicates = (set(zip(words.loc[words["upos"] == "VERB", "utt_id"], words.loc[words["upos"] == "VERB", "orig_token_idx"]))
                  | set(zip(subjects["utt_id"], subjects["head_orig_token_idx"])))
    with_pronoun = set(zip(pron_subjects["utt_id"], pron_subjects["head_orig_token_idx"])) & predicates
    return 100 * len(with_pronoun) / len(predicates) if predicates else np.nan


def overall_context_shift(spans, output_dir):
    context_shift(spans, output_dir)
    return pd.read_csv(os.path.join(output_dir, "context_shift_overall.csv")).set_index("target_lang")


def counts_by_utt(spans, col):
    return {utt: Counter(g[col].fillna("none")) for utt, g in spans.groupby("utt_id")}


def tv_distance(p, q):
    """Total variation distance in percent: the share of mass that would have to change category to turn one distribution into the other."""
    cats = sorted(set(p) | set(q))
    p = np.array([p.get(c, 0) for c in cats], dtype=float)
    q = np.array([q.get(c, 0) for c in cats], dtype=float)
    if not p.sum() or not q.sum():
        return np.nan
    return 50 * np.abs(p / p.sum() - q / q.sum()).sum()


def total(by_utt, sample):
    return sum((by_utt.get(u, Counter()) for u in sample), Counter())


def tvd_with_ci(sys_by_utt, ref_by_utt, utts, n_boot, rng):
    """Raw and bias-corrected TVD with a basic bootstrap interval over utterances, plus the median TVD of reference resamples from the reference."""
    ref_total = total(ref_by_utt, utts)
    point = tv_distance(total(sys_by_utt, utts), ref_total)
    boot, noise = [], []
    for _ in range(n_boot):
        sample = list(rng.choice(utts, size=len(utts), replace=True))
        boot.append(tv_distance(total(sys_by_utt, sample), total(ref_by_utt, sample)))
        noise.append(tv_distance(total(ref_by_utt, sample), ref_total))
    q_low, q_high = np.nanpercentile(boot, [2.5, 97.5])
    return (point, max(2 * point - np.nanmean(boot), 0.0), max(2 * point - q_high, 0.0),
            max(2 * point - q_low, 0.0), np.nanmedian(noise))


def pct(spans, col, value):
    return 100 * (spans[col] == value).mean() if len(spans) else np.nan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference_dir", default=os.path.join(HERE, "../data/reference_parses"))
    parser.add_argument("--systems_dir", default=os.path.join(HERE, "../output/generated_translations/parsed"))
    parser.add_argument("--output", default=os.path.join(HERE, "../output/generated_translations/comparison.csv"))
    parser.add_argument("--excluded_csv", default=os.path.join(HERE, "../data/marked/excluded_outputs.csv"))
    parser.add_argument("--n_boot", type=int, default=1000)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    shift_dir = os.path.join(os.path.dirname(args.output), "context_shift")
    ref_spans, _, ref_fluent = load_spans(args.reference_dir)
    ref_shift = overall_context_shift(ref_spans, os.path.join(shift_dir, "reference"))

    rows = []
    systems = [d for d in os.listdir(args.systems_dir) if os.path.isdir(os.path.join(args.systems_dir, d))]
    for system in tqdm(sorted(systems), desc="Comparing systems"):
        sys_spans, sys_utts, sys_fluent = load_spans(os.path.join(args.systems_dir, system),
                                                     excluded=load_excluded(args.excluded_csv, system))
        sys_shift = overall_context_shift(sys_spans, os.path.join(shift_dir, system))
        for lang in sorted(l for l in sys_utts if l != "en"):
            utts = sorted(sys_utts[lang])
            s = sys_spans[(sys_spans["lang"] == lang) & sys_spans["utt_id"].isin(utts)]
            r = ref_spans[(ref_spans["lang"] == lang) & ref_spans["utt_id"].isin(utts)]
            row = {"system": system, "lang": lang, "n_utterances": len(utts),
                   "n_spans_ref": len(r), "n_spans_sys": len(s)}
            for side in ["before", "after"]:
                col = f"{side}_deprel"
                point, corrected, low, high, noise = tvd_with_ci(counts_by_utt(s, col), counts_by_utt(r, col), utts,
                                                                 args.n_boot, rng)
                row.update({f"tvd_{side}_raw": point, f"tvd_{side}": corrected, f"tvd_{side}_ci_low": low,
                            f"tvd_{side}_ci_high": high, f"tvd_{side}_reference_noise": noise})
            row.update({"pct_before_cc_ref": pct(r, "before_deprel", "cc"),
                        "pct_before_cc_sys": pct(s, "before_deprel", "cc"),
                        "pct_after_nsubj_ref": pct(r, "after_deprel", "nsubj"),
                        "pct_after_nsubj_sys": pct(s, "after_deprel", "nsubj")})
            for name, spans, fluent in [("ref", r, ref_fluent), ("sys", s, sys_fluent)]:
                row[f"pct_predicates_pron_subj_{name}"] = pct_predicates_with_pronoun_subject(fluent, lang, utts)
                row[f"pct_after_pron_subj_{name}"] = (100 * is_pronoun_subject(spans["after_deprel"], spans["after_upos"]).mean()
                                                      if len(spans) else np.nan)
                row[f"pct_after_verb_{name}"] = (100 * spans["after_upos"].isin(["VERB", "AUX"]).mean()
                                                 if len(spans) else np.nan)
            for side in ["before", "after"]:
                col = f"pct_{side}_preserved"
                row[f"{col}_ref"] = ref_shift.loc[lang, col] if lang in ref_shift.index else np.nan
                row[f"{col}_sys"] = sys_shift.loc[lang, col] if lang in sys_shift.index else np.nan
            rows.append(row)

    result = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
