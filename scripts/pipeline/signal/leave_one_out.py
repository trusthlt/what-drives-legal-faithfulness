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
import csv
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple




def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


def normalize_distribution(prob_dict: Dict[str, Any], labels: List[str]) -> List[float]:
    raw = [max(safe_float(prob_dict.get(label), 0.0), 0.0) for label in labels]
    total = sum(raw)
    if total <= 0:
        return [1.0 / len(labels)] * len(labels) if labels else []
    return [value / total for value in raw]


def kl_divergence(p: List[float], q: List[float]) -> float:
    eps = 1e-12
    total = 0.0
    for pi, qi in zip(p, q):
        pi = min(max(pi, eps), 1.0)
        qi = min(max(qi, eps), 1.0)
        total += pi * math.log(pi / qi)
    return total


def mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def leave_one_mean(values: List[float], index: int) -> Optional[float]:
    if len(values) <= 1:
        return None
    return mean(values[:index] + values[index + 1 :])


def normalize_by_max_abs(value: float, max_abs_value: float) -> float:
    if max_abs_value <= 0:
        return 0.0
    return value / max_abs_value


def add_dataset_normalized_fields(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    max_abs_original_ce = max((abs(safe_float(row.get("original_ce"))) for row in rows), default=0.0)
    max_abs_new_ce = max((abs(safe_float(row.get("new_ce"))) for row in rows), default=0.0)
    max_abs_ce_delta = max((abs(safe_float(row.get("ce_delta_original_minus_new"))) for row in rows), default=0.0)
    max_abs_original_ee = max((abs(safe_float(row.get("original_ee"))) for row in rows), default=0.0)
    max_abs_new_ee = max((abs(safe_float(row.get("new_ee"))) for row in rows), default=0.0)
    max_abs_ee_delta = max((abs(safe_float(row.get("ee_delta_original_minus_new"))) for row in rows), default=0.0)

    for row in rows:
        ce_delta_normalized = normalize_by_max_abs(
            safe_float(row.get("ce_delta_original_minus_new")),
            max_abs_ce_delta,
        )
        ee_delta_normalized = normalize_by_max_abs(
            safe_float(row.get("ee_delta_original_minus_new")),
            max_abs_ee_delta,
        )

        row["original_ce_normalized"] = normalize_by_max_abs(
            safe_float(row.get("original_ce")),
            max_abs_original_ce,
        )
        row["new_ce_normalized"] = normalize_by_max_abs(
            safe_float(row.get("new_ce")),
            max_abs_new_ce,
        )
        row["ce_delta_original_minus_new_normalized"] = ce_delta_normalized
        row["abs_ce_delta_normalized"] = abs(ce_delta_normalized)
        row["original_ee_normalized"] = normalize_by_max_abs(
            safe_float(row.get("original_ee")),
            max_abs_original_ee,
        )
        row["new_ee_normalized"] = normalize_by_max_abs(
            safe_float(row.get("new_ee")),
            max_abs_new_ee,
        )
        row["ee_delta_original_minus_new_normalized"] = ee_delta_normalized
        row["abs_ee_delta_normalized"] = abs(ee_delta_normalized)
        row["joint_abs_delta_normalized"] = (
            row["abs_ce_delta_normalized"] + row["abs_ee_delta_normalized"]
        )
        row["signed_alignment_delta_product_normalized"] = (
            ce_delta_normalized * ee_delta_normalized
        )

    return {
        "max_abs_original_ce": max_abs_original_ce,
        "max_abs_new_ce": max_abs_new_ce,
        "max_abs_ce_delta": max_abs_ce_delta,
        "max_abs_original_ee": max_abs_original_ee,
        "max_abs_new_ee": max_abs_new_ee,
        "max_abs_ee_delta": max_abs_ee_delta,
    }


def sort_rows(rows: List[Dict[str, Any]], normalize: bool) -> List[Dict[str, Any]]:
    if normalize:
        return sorted(
            rows,
            key=lambda item: (
                item["joint_abs_delta_normalized"],
                item["abs_ce_delta_normalized"],
                item["abs_ee_delta_normalized"],
            ),
            reverse=True,
        )

    return sorted(
        rows,
        key=lambda item: (
            item["joint_abs_delta"],
            item["abs_ce_delta"],
            item["abs_ee_delta"],
        ),
        reverse=True,
    )


def detect_ee_mode(ee_case: Dict[str, Any], explicit_mode: str) -> str:
    if explicit_mode != "auto":
        return explicit_mode

    cf_block = ee_case.get("counterfactuals", {})
    if isinstance(cf_block, dict):
        for cf_list in cf_block.values():
            if not isinstance(cf_list, list):
                continue
            for item in cf_list:
                if isinstance(item, dict):
                    if "importance" in item:
                        return "importance"
                    if "implied_concepts" in item:
                        return "mentions"

    return "importance"


def ee_value_for_item(item: Dict[str, Any], concept: str, mode: str) -> float:
    if mode == "importance":
        importance = item.get("importance", {})
        return safe_float(importance.get(concept), 0.0) if isinstance(importance, dict) else 0.0

    implied = item.get("implied_concepts", [])
    return 1.0 if isinstance(implied, list) and concept in implied else 0.0


def get_original_ee_value(ee_case: Dict[str, Any], concept: str, mode: str) -> Optional[float]:
    original = ee_case.get("original")
    if mode == "importance" and isinstance(original, dict):
        return safe_float(original.get(concept), 0.0)
    if mode == "mentions" and isinstance(original, list):
        return 1.0 if concept in original else 0.0
    return None


def match_ee_item(
    ee_items: List[Dict[str, Any]],
    prob_item: Dict[str, Any],
    index: int,
    match_by: str,
) -> Tuple[Optional[Dict[str, Any]], str]:
    if match_by in ("value", "value-or-index"):
        value = prob_item.get("value")
        if isinstance(value, str):
            for item in ee_items:
                if isinstance(item, dict) and item.get("value") == value:
                    return item, "value"
        if match_by == "value":
            return None, "missing"

    if index < len(ee_items) and isinstance(ee_items[index], dict):
        return ee_items[index], "index"

    return None, "missing"


def build_ce_scores(prob_case: Dict[str, Any], concept: str) -> Tuple[List[float], List[str]]:
    original_dist = prob_case.get("ORIGINAL_PROBABILITIES", {})
    cf_items = prob_case.get("COUNTERFACTUALS", {}).get(concept, [])

    if not isinstance(original_dist, dict) or not isinstance(cf_items, list):
        return [], []

    labels = sorted(original_dist.keys())
    if not labels:
        return [], []

    original_vec = normalize_distribution(original_dist, labels)
    scores = []
    values = []

    for item in cf_items:
        if not isinstance(item, dict):
            continue
        distribution = item.get("probability_distribution", {})
        if not isinstance(distribution, dict):
            continue
        cf_vec = normalize_distribution(distribution, labels)
        scores.append(kl_divergence(cf_vec, original_vec))
        values.append(str(item.get("value", "")))

    return scores, values


def compute_signal(
    probabilities_data: Dict[str, Any],
    ee_data: Dict[str, Any],
    ee_mode: str,
    match_by: str,
    include_original_ee: bool,
    normalize: bool,
) -> Dict[str, Any]:
    rows = []
    by_question_unsorted: Dict[str, Any] = {}
    skipped = {
        "missing_ee_question": 0,
        "missing_counterfactual_match": 0,
        "single_counterfactual_ce": 0,
        "single_counterfactual_ee": 0,
    }

    for title in sorted(set(probabilities_data.keys())):
        prob_case = probabilities_data.get(title)
        ee_case = ee_data.get(title)

        if not isinstance(prob_case, dict):
            continue
        if not isinstance(ee_case, dict):
            skipped["missing_ee_question"] += 1
            continue

        current_ee_mode = detect_ee_mode(ee_case, ee_mode)
        question_rows = []
        cf_prob_block = prob_case.get("COUNTERFACTUALS", {})
        cf_ee_block = ee_case.get("counterfactuals", {})

        if not isinstance(cf_prob_block, dict) or not isinstance(cf_ee_block, dict):
            continue

        for concept, prob_items in cf_prob_block.items():
            if not isinstance(prob_items, list):
                continue

            ce_scores, ce_values = build_ce_scores(prob_case, concept)
            original_ce = mean(ce_scores)
            if original_ce is None:
                continue

            ee_items = cf_ee_block.get(concept, [])
            if not isinstance(ee_items, list):
                ee_items = []

            matched_ee_values: List[float] = []
            matched_indexes: List[int] = []
            match_methods: Dict[int, str] = {}

            for index, prob_item in enumerate(prob_items):
                if not isinstance(prob_item, dict):
                    continue
                ee_item, match_method = match_ee_item(ee_items, prob_item, index, match_by)
                if ee_item is None:
                    skipped["missing_counterfactual_match"] += 1
                    continue
                matched_indexes.append(index)
                matched_ee_values.append(ee_value_for_item(ee_item, concept, current_ee_mode))
                match_methods[index] = match_method

            original_ee_values = list(matched_ee_values)
            original_ee_contribution = get_original_ee_value(ee_case, concept, current_ee_mode)
            if include_original_ee and original_ee_contribution is not None:
                original_ee_values = [original_ee_contribution] + original_ee_values

            original_ee = mean(original_ee_values)
            if original_ee is None:
                continue

            for local_ee_pos, prob_index in enumerate(matched_indexes):
                new_ce = leave_one_mean(ce_scores, prob_index)
                if new_ce is None:
                    skipped["single_counterfactual_ce"] += 1
                    continue

                leave_one_ee_values = (
                    matched_ee_values[:local_ee_pos] + matched_ee_values[local_ee_pos + 1 :]
                )
                if include_original_ee and original_ee_contribution is not None:
                    leave_one_ee_values = [original_ee_contribution] + leave_one_ee_values

                new_ee = mean(leave_one_ee_values)
                if new_ee is None:
                    skipped["single_counterfactual_ee"] += 1
                    continue

                ce_delta = original_ce - new_ce
                ee_delta = original_ee - new_ee
                abs_ce_delta = abs(ce_delta)
                abs_ee_delta = abs(ee_delta)

                row = {
                    "title": title,
                    "concept": concept,
                    "counterfactual_index": prob_index,
                    "value": ce_values[prob_index] if prob_index < len(ce_values) else str(prob_items[prob_index].get("value", "")),
                    "match_method": match_methods.get(prob_index, "unknown"),
                    "ee_mode": current_ee_mode,
                    "original_ce": original_ce,
                    "new_ce": new_ce,
                    "ce_delta_original_minus_new": ce_delta,
                    "abs_ce_delta": abs_ce_delta,
                    "original_ee": original_ee,
                    "new_ee": new_ee,
                    "ee_delta_original_minus_new": ee_delta,
                    "abs_ee_delta": abs_ee_delta,
                    "joint_abs_delta": abs_ce_delta + abs_ee_delta,
                    "signed_alignment_delta_product": ce_delta * ee_delta,
                }
                rows.append(row)
                question_rows.append(row)

        by_question_unsorted[title] = question_rows

    normalization_constants = None
    if normalize:
        normalization_constants = add_dataset_normalized_fields(rows)

    ranked = sort_rows(rows, normalize)
    by_question = {
        title: sort_rows(question_rows, normalize)
        for title, question_rows in by_question_unsorted.items()
    }

    ranking_description = (
        "Descending by normalized abs(CE delta) + normalized abs(EE delta), "
        "then normalized abs(CE delta), then normalized abs(EE delta)."
        if normalize
        else "Descending by abs(CE delta) + abs(EE delta), then abs(CE delta), then abs(EE delta)."
    )

    return {
        "metadata": {
            "ranking": ranking_description,
            "ce_delta_original_minus_new": "Positive means the removed counterfactual was increasing the concept CE mean.",
            "ee_delta_original_minus_new": "Positive means the removed explanation was increasing the concept EE mean.",
            "normalize": normalize,
            "normalization": (
                {
                    "method": "dataset-level max-absolute scaling, CE and EE scaled separately",
                    "constants": normalization_constants,
                    "ranking_field": "joint_abs_delta_normalized",
                }
                if normalize
                else None
            ),
            "include_original_ee": include_original_ee,
            "match_by": match_by,
            "requested_ee_mode": ee_mode,
            "skipped": skipped,
            "num_ranked_items": len(ranked),
        },
        "ranking": ranked,
        "by_question": by_question,
    }


def normalized_path(path: str) -> str:
    root, ext = os.path.splitext(path)
    if root.endswith("_normalized"):
        return path
    return f"{root}_normalized{ext or ''}"


def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = [
        "title",
        "concept",
        "counterfactual_index",
        "value",
        "match_method",
        "ee_mode",
        "original_ce",
        "new_ce",
        "ce_delta_original_minus_new",
        "abs_ce_delta",
        "original_ee",
        "new_ee",
        "ee_delta_original_minus_new",
        "abs_ee_delta",
        "joint_abs_delta",
        "signed_alignment_delta_product",
        "original_ce_normalized",
        "new_ce_normalized",
        "ce_delta_original_minus_new_normalized",
        "abs_ce_delta_normalized",
        "original_ee_normalized",
        "new_ee_normalized",
        "ee_delta_original_minus_new_normalized",
        "abs_ee_delta_normalized",
        "joint_abs_delta_normalized",
        "signed_alignment_delta_product_normalized",
    ]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description=(
            "Rank individual counterfactual/explanation pairs by leave-one-out "
            "impact on concept-level CE and EE scores."
        )
    )
    parser.add_argument(
        "--probabilities",
        required=True,
    )
    parser.add_argument(
        "--ee",
        required=True,
        help="Detailed EE JSON: importance or binary mention detections.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv-output", required=True)
    parser.add_argument(
        "--ee-mode",
        choices=["auto", "importance", "mentions"],
        default="auto",
        help="How to read the detailed EE file.",
    )
    parser.add_argument(
        "--match-by",
        choices=["value", "index", "value-or-index"],
        default="value-or-index",
        help="How to align CE counterfactuals with EE explanations.",
    )
    parser.add_argument(
        "--include-original-ee",
        action="store_true",
        help="Include the original explanation score/mention when recomputing EE.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help=(
            "Add dataset-level normalized CE/EE fields and rank by normalized joint delta. "
            "Output filenames get '_normalized' appended."
        ),
    )
    args = parser.parse_args()

    if args.normalize:
        args.output = normalized_path(args.output)
        if args.csv_output:
            args.csv_output = normalized_path(args.csv_output)

    with open(args.probabilities, "r", encoding="utf-8") as fh:
        probabilities_data = json.load(fh)
    with open(args.ee, "r", encoding="utf-8") as fh:
        ee_data = json.load(fh)

    result = compute_signal(
        probabilities_data=probabilities_data,
        ee_data=ee_data,
        ee_mode=args.ee_mode,
        match_by=args.match_by,
        include_original_ee=args.include_original_ee,
        normalize=args.normalize,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)

    write_csv(result["ranking"], args.csv_output)

    print(f"Wrote JSON ranking -> {args.output}")
    if args.csv_output:
        print(f"Wrote CSV ranking -> {args.csv_output}")
    print(f"Ranked items: {len(result['ranking'])}")


if __name__ == "__main__":
    main()
