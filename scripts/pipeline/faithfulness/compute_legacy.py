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
import math

# =========================================================
# PEARSON CORRELATION (NO NUMPY, NO SILENT FAILURES)
# =========================================================

def pearson_corr(xs, ys):
    n = len(xs)
    if n < 2:
        return None

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]

    sx = math.sqrt(sum(d * d for d in dx))
    sy = math.sqrt(sum(d * d for d in dy))

    if sx == 0 or sy == 0:
        return None

    return sum(a * b for a, b in zip(dx, dy)) / (sx * sy)


# =========================================================
# FAITHFULNESS PIPELINE
# =========================================================

def compute_faithfulness(ce_path, ee_path, out_path):
    with open(ce_path, "r", encoding="utf-8") as f:
        ce_data = json.load(f)

    with open(ee_path, "r", encoding="utf-8") as f:
        ee_data = json.load(f)

    question_results = {}
    faithfulness_values = []

    common_titles = sorted(set(ce_data.keys()) & set(ee_data.keys()))

    for title in common_titles:
        ce_block = ce_data.get(title, {})
        ee_block = ee_data.get(title, {})

        # Concept intersection
        concepts = sorted(set(ce_block.keys()) & set(ee_block.keys()))

        CE_vals = []
        EE_vals = []
        used_concepts = []

        for c in concepts:
            ce_entry = ce_block.get(c)
            ee_entry = ee_block.get(c)

            if not isinstance(ce_entry, dict) or not isinstance(ee_entry, dict):
                continue

            ce = ce_entry.get("CE")
            ee = ee_entry.get("EE")

            if ce is None or ee is None:
                continue

            CE_vals.append(float(ce))
            EE_vals.append(float(ee))
            used_concepts.append(c)

        F = pearson_corr(CE_vals, EE_vals)

        question_results[title] = {
            "faithfulness": F,
            "num_concepts": len(used_concepts),
            "concepts_used": used_concepts,
            "CE_values": CE_vals,
            "EE_values": EE_vals
        }

        if F is not None:
            faithfulness_values.append(F)

    dataset_faithfulness = (
        sum(faithfulness_values) / len(faithfulness_values)
        if faithfulness_values else None
    )

    output = {
        "dataset_faithfulness_mean": dataset_faithfulness,
        "num_questions_used": len(faithfulness_values),
        "question_faithfulness": question_results
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("Written faithfulness results →", out_path)
    print("Dataset-level faithfulness =", dataset_faithfulness)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ce",
        required=True,
    )
    parser.add_argument(
        "--ee",
        required=True,
    )
    parser.add_argument(
        "--output",
        required=True,
    )
    args = parser.parse_args()

    compute_faithfulness(
        args.ce,
        args.ee,
        args.output,
    )
