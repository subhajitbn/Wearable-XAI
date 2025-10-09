import json, os, time
import numpy as np
import pandas as pd
from utils import load_signals_from_csv, create_windows_and_features
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from grid_search import grid_search_for_optimal_window_size


# ---------- 1. Load global (general) rules ----------
def load_general_rules(path):
    with open(path) as f:
        data = json.load(f)
    return list(data.keys())


# ---------- 2. Learn cutoff + probabilities ----------
def learn_rule_stats(X_train, y_train, feature, op):
    vals = X_train[feature].values
    thresholds = np.percentile(vals, np.linspace(5, 95, 50))
    best_acc, best_t, best_stats = 0, None, None

    for t in thresholds:
        mask_L = vals < t if op == "<" else vals > t
        mask_R = ~mask_L
        if mask_L.sum() < 5 or mask_R.sum() < 5:
            continue

        pL = y_train[mask_L].mean() if mask_L.any() else 0
        pR = y_train[mask_R].mean() if mask_R.any() else 0
        preds = np.where(mask_L, int(pL >= 0.5), int(pR >= 0.5))
        acc = accuracy_score(y_train, preds)

        if acc > best_acc:
            best_acc, best_t = acc, t
            best_stats = (pL, pR, mask_L.sum(), mask_R.sum())

    return best_t, best_stats, best_acc


# ---------- 3. Learn personalized rule cutoffs ----------
def learn_rule_cutoffs(baseline, cogload, rules_path, w, s, seed=None):
    fb, fn = create_windows_and_features(baseline, w, s)
    fc, _ = create_windows_and_features(cogload, w, s)
    n = min(len(fb), len(fc))
    X = np.vstack([fb[:n], fc[:n]])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    df = pd.DataFrame(X, columns=fn)

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, stratify=y, random_state=seed)

    rule_keys = load_general_rules(rules_path)
    results = {}

    for rule in rule_keys:
        feat, op = rule.split()
        if feat not in X_train.columns:
            continue
        t, stats, _ = learn_rule_stats(X_train, y_train, feat, op)
        if not stats:
            continue
        pL, pR, nL, nR = stats
        results[rule] = {
            "cutoff": float(t),
            "pL": float(pL),
            "pR": float(pR),
            "nL": int(nL),
            "nR": int(nR),
        }

    return results


# ---------- 4. Apply personalized rules for inference ----------
def apply_rules(X, personalized_rules):
    preds = np.zeros(len(X))
    for rule, params in personalized_rules.items():
        feat, op = rule.split()
        t = params["cutoff"]
        pL, pR = params["pL"], params["pR"]

        if feat not in X.columns:
            continue

        mask_L = X[feat].values < t if op == "<" else X[feat].values > t
        preds_rule = np.where(mask_L, int(pL >= 0.5), int(pR >= 0.5))
        preds += preds_rule  # vote accumulation

    # majority vote over all applicable rules
    preds = (preds >= (len(personalized_rules) / 2)).astype(int)
    return preds


# ---------- 5. Run pipeline for one participant ----------
def run_pipeline(baseline_path, cogload_path, rules_path, seed=None):
    baseline = load_signals_from_csv(baseline_path)
    cogload = load_signals_from_csv(cogload_path)

    window_sizes = [128, 256, 512, 1024, 2048]
    step_sizes = [64, 128, 256, 512]

    model, X, y, feature_names, w, s = grid_search_for_optimal_window_size(
        baseline, cogload, window_sizes, step_sizes, seed=seed
    )

    print(f"✅ Optimal window: {w}, step: {s}")
    rules = learn_rule_cutoffs(baseline, cogload, rules_path, w, s, seed)

    return {"window": w, "step": s, "rules": rules}


# ---------- 6. Run for multiple participants + time inference ----------
def run_all_participants(rules_path, output_json, participants=[0, 1, 2], seed=None):
    all_results = {}
    total_inference_time = 0
    total_acc = []

    for pid in participants:
        print(f"\n▶ Running for participant {pid} ...")
        baseline_path = f"data/pilot/{pid}/baseline/empatica_bvp.csv"
        cogload_path = f"data/pilot/{pid}/cognitive_load/empatica_bvp.csv"

        if not (os.path.exists(baseline_path) and os.path.exists(cogload_path)):
            print(f"⚠️ Skipping participant {pid}: missing data")
            continue

        # 1. Learn personalized rules
        res = run_pipeline(baseline_path, cogload_path, rules_path, seed)
        all_results[str(pid)] = res

        # 2. Inference timing and accuracy
        print(f"⏱ Measuring inference performance for participant {pid} ...")
        baseline = load_signals_from_csv(baseline_path)
        cogload = load_signals_from_csv(cogload_path)
        fb, fn = create_windows_and_features(baseline, res["window"], res["step"])
        fc, _ = create_windows_and_features(cogload, res["window"], res["step"])
        n = min(len(fb), len(fc))
        X = np.vstack([fb[:n], fc[:n]])
        y = np.concatenate([np.zeros(n), np.ones(n)])
        df = pd.DataFrame(X, columns=fn)

        start_time = time.time()
        preds = apply_rules(df, res["rules"])
        elapsed = time.time() - start_time
        acc = accuracy_score(y, preds)

        total_inference_time += elapsed
        total_acc.append(acc)

        print(f"PID {pid}: inference={elapsed:.4f}s, accuracy={acc:.3f}")

    # Save personalized rule sets
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(all_results, f, indent=2)

    avg_acc = np.mean(total_acc)
    print("\n✅ Personalized rule models saved.")
    print(f"🕒 Total inference time (3 participants): {total_inference_time:.4f} s")
    print(f"🎯 Average accuracy: {avg_acc:.3f}")


# ---------- 7. Entry ----------
if __name__ == "__main__":
    run_all_participants(
        rules_path="results/top5_general_rules.json",
        output_json="results/personalized_rules_with_timing.json",
        participants=[0, 1, 2],
        seed=4242
    )
