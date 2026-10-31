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
import os


def compute_ee_importance_gold(
    gold_importance_path,
    legal_concepts_path,
    out_path,
    normalize=False,
    max_importance=4.0,
):
    """
    Compute EE from gold explanations when each question only has one
    original_importance block and no counterfactual averaging.

    For each concept c:
      - read its score directly from original_importance[c]
      - optionally normalize by max_importance
    """

    with open(gold_importance_path, "r", encoding="utf-8") as fh:
        importance_data = json.load(fh)

    with open(legal_concepts_path, "r", encoding="utf-8") as fh:
        concepts_data = json.load(fh)

    results = {}

    for title, block in importance_data.items():
        case_concepts = concepts_data.get(title)
        if not isinstance(case_concepts, dict):
            continue

        concepts = [
            k for k in case_concepts.keys()
            if k not in ("ORIGINAL QUESTION", "COUNTERFACTUALS")
        ]
        if not concepts:
            continue

        original_scores = block.get("original_importance", {})
        if not isinstance(original_scores, dict):
            original_scores = {}

        ee_per_concept = {}

        for c in concepts:
            raw_score = original_scores.get(c, 0.0)
            try:
                raw_score = float(raw_score)
            except (TypeError, ValueError):
                raw_score = 0.0

            if normalize:
                ee = raw_score / max_importance if max_importance > 0 else 0.0
            else:
                ee = raw_score

            ee_per_concept[c] = {
                "EE": ee,
                "EE_raw": raw_score,
                "normalized": normalize,
                "max_importance": max_importance if normalize else None,
            }

        results[title] = ee_per_concept

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    print("Written gold EE importance results →", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gold-importance",
        required=True,
        help="JSON with original_importance scores from gold explanations.",
    )
    parser.add_argument(
        "--concepts",
        required=True,
        help="JSON with extracted legal concepts.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Where to write EE results.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=False,
        help="Normalize EE to [0,1] by dividing by --max-importance.",
    )
    parser.add_argument(
        "--max-importance",
        type=float,
        default=4.0,
        help="Maximum possible importance score, used only if --normalize is set.",
    )
    args = parser.parse_args()

    compute_ee_importance_gold(
        gold_importance_path=args.gold_importance,
        legal_concepts_path=args.concepts,
        out_path=args.output,
        normalize=args.normalize,
        max_importance=args.max_importance,
    )