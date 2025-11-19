import json
import numpy as np

INPUT_JSON = "results/generalized_rules.json"
OUTPUT_JSON = "results/top5_general_rules.json"

def rank_rules_by_weighted_score(json_path, top_k=5):
    with open(json_path) as f:
        data = json.load(f)

    ranked = []
    for rule, vals in data.items():
        freq = vals["frequency"]
        scores = vals["impurity_scores"]
        if not scores:
            continue
        median_imp = float(np.median(scores))
        weighted_score = freq * median_imp
        ranked.append((rule, freq, median_imp, weighted_score))

    ranked.sort(key=lambda x: x[3], reverse=True)
    top_rules = ranked[:top_k]

    return {
        rule: {
            "frequency": freq,
            "median_impurity": median_imp,
            "weighted_score": weighted_score
        }
        for rule, freq, median_imp, weighted_score in top_rules
    }

if __name__ == "__main__":
    top_rules = rank_rules_by_weighted_score(INPUT_JSON, top_k=5)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(top_rules, f, indent=2)
