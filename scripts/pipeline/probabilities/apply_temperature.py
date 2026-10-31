"""Recompute probabilities in a direct-null score file at a new temperature.

This is a model-free post-processing step for JSON produced by
direct_null.py. It treats ``calibrated_logits`` as the
temperature-independent scores, computes ``logits = calibrated / temperature``,
and then applies a numerically stable softmax.
"""

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple


def temperature_softmax(
    calibrated_logits: Dict[str, Optional[float]],
    temperature: float,
) -> Tuple[Dict[str, Optional[float]], Dict[str, float], Optional[str]]:
    scaled = {
        label: (
            float(value) / temperature
            if value is not None and math.isfinite(float(value))
            else None
        )
        for label, value in calibrated_logits.items()
    }
    finite = {label: value for label, value in scaled.items() if value is not None}
    if not finite:
        return scaled, {label: 0.0 for label in scaled}, None

    maximum = max(finite.values())
    exponentials = {
        label: math.exp(value - maximum) if value is not None else 0.0
        for label, value in scaled.items()
    }
    total = sum(exponentials.values())
    probabilities = {
        label: value / total if total > 0.0 else 0.0
        for label, value in exponentials.items()
    }
    predicted_answer = max(finite, key=finite.get)
    return scaled, probabilities, predicted_answer


def update_original(item: dict, temperature: float) -> bool:
    calibrated = item.get("ORIGINAL_CALIBRATED_LOGITS")
    if not isinstance(calibrated, dict) or not calibrated:
        return False
    logits, probabilities, answer = temperature_softmax(calibrated, temperature)
    item["ORIGINAL_LOGITS"] = logits
    item["ORIGINAL_PROBABILITIES"] = probabilities
    item["ORIGINAL_PREDICTED_ANSWER"] = answer
    item["ORIGINAL_TEMPERATURE"] = temperature
    return True


def update_counterfactual(entry: dict, temperature: float) -> bool:
    calibrated = entry.get("calibrated_logits")
    if not isinstance(calibrated, dict) or not calibrated:
        return False
    logits, probabilities, answer = temperature_softmax(calibrated, temperature)
    entry["logits"] = logits
    entry["probability_distribution"] = probabilities
    entry["predicted_answer"] = answer
    entry["temperature"] = temperature
    return True


def apply_temperature(data: dict, temperature: float) -> dict:
    summary = {
        "questions_seen": 0,
        "originals_updated": 0,
        "originals_missing_calibrated_logits": 0,
        "counterfactuals_seen": 0,
        "counterfactuals_updated": 0,
        "counterfactuals_missing_calibrated_logits": 0,
    }
    for item in data.values():
        if not isinstance(item, dict):
            continue
        summary["questions_seen"] += 1
        if update_original(item, temperature):
            summary["originals_updated"] += 1
        else:
            summary["originals_missing_calibrated_logits"] += 1

        counterfactual_groups = item.get("COUNTERFACTUALS", {})
        if not isinstance(counterfactual_groups, dict):
            continue
        for entries in counterfactual_groups.values():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                summary["counterfactuals_seen"] += 1
                if update_counterfactual(entry, temperature):
                    summary["counterfactuals_updated"] += 1
                else:
                    summary["counterfactuals_missing_calibrated_logits"] += 1
    return summary


def save_json_atomic(data: dict, output_path: str):
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        text=True,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False, allow_nan=False)
            file.write("\n")
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Apply a new temperature to calibrated logits produced by "
            "direct_null.py."
        )
    )
    parser.add_argument("--input", required=True, help="Existing direct-null probability JSON.")
    parser.add_argument("--output", required=True, help="Path for the updated JSON; may equal --input.")
    parser.add_argument("--temperature", required=True, type=float)
    args = parser.parse_args()

    if not math.isfinite(args.temperature) or args.temperature <= 0.0:
        parser.error("--temperature must be a finite number greater than zero")

    with open(args.input, encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("The input JSON root must be an object keyed by question title")

    summary = apply_temperature(data, args.temperature)
    save_json_atomic(data, args.output)
    print(json.dumps({"temperature": args.temperature, **summary}, indent=2))
    print(f"Updated file written to: {args.output}")


if __name__ == "__main__":
    main()
