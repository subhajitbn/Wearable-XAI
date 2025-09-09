import os
import json
from single_run import run_pipeline

# Example: folder with participant subfolders containing baseline.csv & cogload.csv
DATA_DIR = "data/pilot"
OUTPUT_JSON = "results/all_rules.json"

def run_all_participants(seed=None):
    all_results = {}

    for participant in sorted(os.listdir(DATA_DIR)):
        participant_path = os.path.join(DATA_DIR, participant)
        baseline_path = os.path.join(participant_path, "baseline", "empatica_bvp.csv")
        cogload_path = os.path.join(participant_path, "cognitive_load", "empatica_bvp.csv")

        if not (os.path.exists(baseline_path) and os.path.exists(cogload_path)):
            continue  # skip if files missing

        print(f"▶ Running pipeline for participant {participant} ...")

        results = run_pipeline(
            baseline_path,
            cogload_path,
            seed,
            return_results=True
        )

        # Store in dict
        all_results[participant] = {
            "optimal_window_size": results["optimal_window_size"],
            "optimal_step_size": results["optimal_step_size"],
            "accuracy": results["accuracy"],
            "rules": results["rules"]
        }

    # Save all results
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"✅ Saved results for {len(all_results)} participants to {OUTPUT_JSON}")


if __name__ == "__main__":
    run_all_participants(seed=None)
