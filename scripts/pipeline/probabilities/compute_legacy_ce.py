import atexit as _atexit
import time as _time


def _register_runtime_timer():
    _script_start = _time.perf_counter()

    def _print_runtime():
        elapsed = _time.perf_counter() - _script_start
        print(f"\nElapsed time: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")

    _atexit.register(_print_runtime)


if __name__ == "__main__":
    _register_runtime_timer()

import argparse
import json
import numpy as np
import math


# ==========================================
# Safe float loader: accepts e-10, strings, negatives, None
# ==========================================
def safe_float(x):
    try:
        val = float(x)
        # numerical precision correction:
        if abs(val) < 1e-15:
            return 0.0
        if val < 0:
            # clamp tiny negative rounding artifacts:
            if abs(val) < 1e-12:
                return 0.0
            # large negative invalid → raise
            raise ValueError(f"Invalid negative probability value {val}")
        return val
    except:
        return 0.0


# ==========================================
# Extract vector with normalization
# ==========================================
def extract_vector(prob_dict, labels):

    raw = [safe_float(prob_dict.get(lbl, 0.0)) for lbl in labels]

    total = sum(raw)

    # avoid divide-by-zero:
    if total <= 0:
        n = len(labels)
        return [1.0 / n] * n

    # normalize to sum = 1
    return [p / total for p in raw]


# ==========================================
# KL divergence with robust handling
# ==========================================
def kl_divergence(p, q):

    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)

    # epsilon to prevent log(0)
    eps = 1e-12
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)

    return float(np.sum(p * np.log(p / q)))


# ==========================================
# Compute CE per concept
# ==========================================
def compute_ce_for_case(case):

    out = {}

    orig_dist = case.get("ORIGINAL_PROBABILITIES", {})
    if not isinstance(orig_dist, dict):
        return out

    labels = sorted(orig_dist.keys())

    if not labels:
        return out

    orig_vec = extract_vector(orig_dist, labels)

    cf_block = case.get("COUNTERFACTUALS", {})
    if not isinstance(cf_block, dict):
        return out

    for concept, cf_list in cf_block.items():

        if not isinstance(cf_list, list):
            continue

        kl_vals = []
        new_vals = []

        for cf in cf_list:

            pd = cf.get("probability_distribution")
            if not isinstance(pd, dict):
                continue

            cf_vec = extract_vector(pd, labels)

            kl_score = kl_divergence(cf_vec, orig_vec)

            if math.isfinite(kl_score):
                kl_vals.append(kl_score)
                new_vals.append(cf.get("value"))

        if kl_vals:
            out[concept] = {
                "CE": float(np.mean(kl_vals)),
                "new_values": new_vals
            }

    return out


# ==========================================
# MAIN
# ==========================================
DEFAULT_INPUT_LOGPROB = "counterfactuals_probs/legal_concepts_with_logprobs_magistral.json"
DEFAULT_INPUT_CROSS_ENTROPY = "counterfactuals_probs/legal_concepts_with_cross_entropy_magistral.json"

def main(input_json_path: str, output_json_path: str):

    with open(input_json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    result = {}

    for title, case in data.items():
        ce = compute_ce_for_case(case)
        result[title] = ce
        print(f"✔ CE computed → {title}")

    with open(output_json_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)

    print("\nSaved CE:", output_json_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ce-method", choices=["logprob", "cross_entropy"], default="logprob")
    parser.add_argument("--input")
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.input:
        input_path = args.input
    elif args.ce_method == "cross_entropy":
        input_path = DEFAULT_INPUT_CROSS_ENTROPY
    else:
        input_path = DEFAULT_INPUT_LOGPROB

    if args.output:
        output_path = args.output
    else:
        input_name = input_path.split("/")[-1].replace(".json", "")
        output_path = f"CE_calculations/{input_name}_with_ce.json"

    print(f"CE method: {args.ce_method}")
    main(input_path, output_path)
