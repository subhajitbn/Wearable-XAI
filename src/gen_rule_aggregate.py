import re, json
from collections import defaultdict
from impurity import impurity_reduction

def parse_rule(line):
    """Extract condition and impurity stats from a rule line."""
    m = re.search(r'if (.+?) then ([0-9.]+) \(n=(\d+)\) else ([0-9.]+) \(n=(\d+)\)', line)
    if not m:
        return None
    cond, pL, nL, pR, nR = m.groups()
    # extract feature and inequality
    conds = re.findall(r'([a-zA-Z_]+)\s*([<>]=?)\s*[-+]?\d*\.?\d*', cond)
    results = []
    for feat, op in conds:
        op = '>' if '>' in op else '<'
        results.append((f"{feat} {op}", float(pL), int(nL), float(pR), int(nR)))
    return results

def compute_impurities(rules):
    """Compute impurity reductions per generalized inequality direction."""
    scores = defaultdict(list)
    for r in rules:
        parsed = parse_rule(r)
        if not parsed:
            continue
        for cond, pL, nL, pR, nR in parsed:
            ΔG = impurity_reduction(pL, nL, pR, nR)
            scores[cond].append(ΔG)
    return scores

def aggregate_rules(json_path):
    with open(json_path) as f:
        data = json.load(f)
    combined = defaultdict(list)
    for pid, pdata in data.items():
        for cond, vals in compute_impurities(pdata["rules"]).items():
            combined[cond].extend(vals)
    return {
        cond: {"frequency": len(v), "impurity_scores": v}
        for cond, v in combined.items()
    }

if __name__ == "__main__":
    summary = aggregate_rules("results/all_rules.json")
    with open("results/generalized_rules.json", "w") as f:
        json.dump(summary, f, indent=2)
