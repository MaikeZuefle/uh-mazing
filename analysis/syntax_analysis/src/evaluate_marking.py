"""Compare the LLM marking of the references with the human disfluency markup, token by token."""
import argparse
import json
import os
import unicodedata

import pandas as pd

from markup import tokenize_and_flag

HERE = os.path.dirname(os.path.abspath(__file__))


def scores(pairs):
    tp = sum(gold and pred for gold, pred in pairs)
    fp = sum(pred and not gold for gold, pred in pairs)
    fn = sum(gold and not pred for gold, pred in pairs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "n_gold_disfluent": tp + fn}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marked_csv", default=os.path.join(HERE, "../data/marked/marked_references.csv"))
    parser.add_argument("--references_csv", default=os.path.join(HERE, "../../../translation_data/references.csv"))
    parser.add_argument("--output", default=os.path.join(HERE, "../output/generated_translations/marking_agreement.csv"))
    args = parser.parse_args()

    refs = pd.read_csv(args.references_csv).set_index("ID")
    marked = pd.read_csv(args.marked_csv)

    pairs = {}
    for _, row in marked.iterrows():
        gold = tokenize_and_flag(str(refs.loc[row["ID"], f"{row['lang']}_disfluent"]))
        predicted = set(json.loads(row["disfluent_token_indices"]))
        for i, (word, is_disfluent) in enumerate(gold):
            is_punct = all(unicodedata.category(c).startswith("P") for c in word)
            pairs.setdefault((row["lang"], is_punct), []).append((is_disfluent, i in predicted))

    rows = []
    for lang in sorted({lang for lang, _ in pairs}) + ["ALL"]:
        all_pairs = [p for (l, _), ps in pairs.items() if lang in (l, "ALL") for p in ps]
        word_pairs = [p for (l, punct), ps in pairs.items() if lang in (l, "ALL") and not punct for p in ps]
        rows.append({"lang": lang, **{f"all_tokens_{k}": v for k, v in scores(all_pairs).items()},
                     **{f"words_only_{k}": v for k, v in scores(word_pairs).items()}})

    result = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
