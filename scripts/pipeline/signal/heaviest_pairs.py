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
import os
from typing import Any, Dict, List, Optional




def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_nested_list(data: Dict[str, Any], title: str, concept: str) -> List[Dict[str, Any]]:
    case = data.get(title, {})
    if not isinstance(case, dict):
        return []
    cf_block = case.get("COUNTERFACTUALS", case.get("counterfactuals", {}))
    if not isinstance(cf_block, dict):
        return []
    items = cf_block.get(concept, [])
    return items if isinstance(items, list) else []


def get_by_index(items: List[Dict[str, Any]], index: int) -> Optional[Dict[str, Any]]:
    if 0 <= index < len(items) and isinstance(items[index], dict):
        return items[index]
    return None


def get_by_value(items: List[Dict[str, Any]], value: str) -> Optional[Dict[str, Any]]:
    for item in items:
        if isinstance(item, dict) and item.get("value") == value:
            return item
    return None


def find_matching_item(
    items: List[Dict[str, Any]],
    value: str,
    index: int,
    match_by: str,
) -> Optional[Dict[str, Any]]:
    if match_by in ("value", "value-or-index"):
        item = get_by_value(items, value)
        if item is not None:
            return item
        if match_by == "value":
            return None

    return get_by_index(items, index)


def classify_signal(row: Dict[str, Any], ce_threshold: float, ee_threshold: float) -> str:
    abs_ce = safe_float(row.get("abs_ce_delta"))
    abs_ee = safe_float(row.get("abs_ee_delta"))
    product = safe_float(row.get("signed_alignment_delta_product"))

    high_ce = abs_ce >= ce_threshold
    high_ee = abs_ee >= ee_threshold

    if high_ce and high_ee and product > 0:
        return "faithful signal"
    if high_ce and high_ee and product < 0:
        return "anti-aligned signal"
    if high_ce and not high_ee:
        return "behavior-only signal"
    if high_ee and not high_ce:
        return "explanation-only signal"
    return "weak or redundant counterfactual"


def select_heaviest_rows(
    ranking: List[Dict[str, Any]],
    metric: str,
) -> List[Dict[str, Any]]:
    best_by_concept: Dict[tuple, Dict[str, Any]] = {}

    for row in ranking:
        if not isinstance(row, dict):
            continue
        title = str(row.get("title", ""))
        concept = str(row.get("concept", ""))
        if not title or not concept:
            continue

        key = (title, concept)
        current = best_by_concept.get(key)
        if current is None or safe_float(row.get(metric)) > safe_float(current.get(metric)):
            best_by_concept[key] = row

    return sorted(
        best_by_concept.values(),
        key=lambda item: (
            safe_float(item.get(metric)),
            safe_float(item.get("abs_ce_delta")),
            safe_float(item.get("abs_ee_delta")),
        ),
        reverse=True,
    )


def enrich_row(
    row: Dict[str, Any],
    counterfactuals_data: Dict[str, Any],
    explanations_data: Dict[str, Any],
    probabilities_data: Optional[Dict[str, Any]],
    match_by: str,
    ce_threshold: float,
    ee_threshold: float,
) -> Dict[str, Any]:
    title = str(row.get("title", ""))
    concept = str(row.get("concept", ""))
    value = str(row.get("value", ""))
    index = int(row.get("counterfactual_index", -1))

    counterfactual_items = get_nested_list(counterfactuals_data, title, concept)
    explanation_items = get_nested_list(explanations_data, title, concept)
    probability_items = get_nested_list(probabilities_data or {}, title, concept)

    counterfactual_item = find_matching_item(counterfactual_items, value, index, match_by)
    explanation_item = find_matching_item(explanation_items, value, index, match_by)
    probability_item = find_matching_item(probability_items, value, index, match_by)

    enriched = dict(row)
    enriched["signal_label"] = classify_signal(row, ce_threshold, ee_threshold)
    enriched["counterfactual_question"] = (
        counterfactual_item.get("counterfactual", "") if counterfactual_item else ""
    )
    enriched["model_answer"] = explanation_item.get("answer", "") if explanation_item else ""
    enriched["model_explanation"] = explanation_item.get("explanation", "") if explanation_item else ""
    enriched["probability_distribution"] = (
        probability_item.get("probability_distribution", {}) if probability_item else {}
    )
    enriched["missing_counterfactual_text"] = counterfactual_item is None
    enriched["missing_explanation"] = explanation_item is None
    enriched["missing_probability_distribution"] = probability_item is None
    return enriched


def build_output(
    leave_one_data: Dict[str, Any],
    counterfactuals_data: Dict[str, Any],
    explanations_data: Dict[str, Any],
    probabilities_data: Optional[Dict[str, Any]],
    metric: str,
    match_by: str,
    ce_threshold: float,
    ee_threshold: float,
) -> Dict[str, Any]:
    ranking = leave_one_data.get("ranking", [])
    if not isinstance(ranking, list):
        raise ValueError("Leave-one JSON must contain a list under key 'ranking'.")

    selected = select_heaviest_rows(ranking, metric)
    enriched = [
        enrich_row(
            row=row,
            counterfactuals_data=counterfactuals_data,
            explanations_data=explanations_data,
            probabilities_data=probabilities_data,
            match_by=match_by,
            ce_threshold=ce_threshold,
            ee_threshold=ee_threshold,
        )
        for row in selected
    ]

    by_question: Dict[str, List[Dict[str, Any]]] = {}
    for row in enriched:
        by_question.setdefault(str(row.get("title", "")), []).append(row)

    for rows in by_question.values():
        rows.sort(
            key=lambda item: (
                safe_float(item.get(metric)),
                safe_float(item.get("abs_ce_delta")),
                safe_float(item.get("abs_ee_delta")),
            ),
            reverse=True,
        )

    return {
        "metadata": {
            "description": (
                "For each local question+concept group, this selects the single "
                "counterfactual/explanation pair with the heaviest leave-one-out effect."
            ),
            "local_group": "title + concept",
            "selection_metric": metric,
            "match_by": match_by,
            "ce_threshold_for_labels": ce_threshold,
            "ee_threshold_for_labels": ee_threshold,
            "num_pairs": len(enriched),
        },
        "heaviest_pairs": enriched,
        "by_question": by_question,
    }


def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not path:
        return

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = [
        "title",
        "concept",
        "counterfactual_index",
        "value",
        "signal_label",
        "joint_abs_delta",
        "abs_ce_delta",
        "abs_ee_delta",
        "ce_delta_original_minus_new",
        "ee_delta_original_minus_new",
        "signed_alignment_delta_product",
        "original_ce",
        "new_ce",
        "original_ee",
        "new_ee",
        "model_answer",
        "counterfactual_question",
        "model_explanation",
        "missing_counterfactual_text",
        "missing_explanation",
        "missing_probability_distribution",
    ]

    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description=(
            "Extract the heaviest local counterfactual/explanation pair for each "
            "question+concept from a leave-one-out signal file."
        )
    )
    parser.add_argument(
        "--leave-one",
        required=True,
        help="Output JSON from leave_one_out.py.",
    )
    parser.add_argument(
        "--counterfactuals",
        required=True,
        help="Counterfactual JSON containing rewritten questions.",
    )
    parser.add_argument(
        "--explanations",
        required=True,
        help="Explanations JSON containing generated explanations for counterfactuals.",
    )
    parser.add_argument(
        "--probabilities",
        required=True,
        help="Optional probabilities JSON, used to include probability distributions.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv-output", required=True)
    parser.add_argument(
        "--metric",
        default="joint_abs_delta",
        choices=["joint_abs_delta", "abs_ce_delta", "abs_ee_delta"],
        help="Metric used to select the heaviest pair within each question+concept.",
    )
    parser.add_argument(
        "--match-by",
        choices=["value", "index", "value-or-index"],
        default="value-or-index",
        help="How to match selected rows back to counterfactual/explanation records.",
    )
    parser.add_argument(
        "--ce-threshold",
        type=float,
        default=0.01,
        help="Threshold used only for practical signal labels.",
    )
    parser.add_argument(
        "--ee-threshold",
        type=float,
        default=0.5,
        help="Threshold used only for practical signal labels.",
    )
    args = parser.parse_args()

    probabilities_data = load_json(args.probabilities) if args.probabilities else None
    output = build_output(
        leave_one_data=load_json(args.leave_one),
        counterfactuals_data=load_json(args.counterfactuals),
        explanations_data=load_json(args.explanations),
        probabilities_data=probabilities_data,
        metric=args.metric,
        match_by=args.match_by,
        ce_threshold=args.ce_threshold,
        ee_threshold=args.ee_threshold,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    write_csv(output["heaviest_pairs"], args.csv_output)

    print(f"Wrote JSON -> {args.output}")
    if args.csv_output:
        print(f"Wrote CSV -> {args.csv_output}")
    print(f"Local heaviest pairs: {len(output['heaviest_pairs'])}")


if __name__ == "__main__":
    main()
