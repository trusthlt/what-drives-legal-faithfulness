"""Calculate original-question argmax accuracy for the multilevel dataset."""

import argparse
import ast
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional


def normalized_field_map(row: dict) -> Dict[str, str]:
    return {
        re.sub(r"[^a-z0-9]", "", str(key).lower()): key
        for key in row
    }


def get_field(row: dict, *names: str, default=""):
    fields = normalized_field_map(row)
    for name in names:
        key = fields.get(re.sub(r"[^a-z0-9]", "", name.lower()))
        if key is not None:
            return row.get(key, default)
    return default


def parse_option_labels(value: str) -> List[str]:
    if not value:
        return []
    parts = None
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, (list, tuple)):
            parts = [str(item) for item in parsed]
    except (SyntaxError, ValueError):
        pass
    if parts is None:
        parts = str(value).splitlines()
    labels = []
    for part in parts:
        match = re.match(r"\s*([A-Za-z])\s*[.)]", part)
        if match:
            labels.append(match.group(1).upper())
    return labels


def correct_answer_label(raw_answer, option_labels: List[str]) -> Optional[str]:
    value = str(raw_answer or "").strip()
    if not value:
        return None
    if re.fullmatch(r"[A-Za-z]", value):
        return value.upper()
    if re.fullmatch(r"\d+", value):
        # The multilevel dataset stores Correct_answer as a one-based index.
        index = int(value) - 1
        labels = option_labels or [chr(ord("A") + i) for i in range(26)]
        return labels[index] if 0 <= index < len(labels) else None
    match = re.search(r"(?:answer|option)\s*[:\-]?\s*([A-Za-z])\b", value, re.I)
    return match.group(1).upper() if match else None


def load_csv(path: str) -> Dict[str, dict]:
    rows = {}
    with open(path, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file, skipinitialspace=True)
        for row_number, row in enumerate(reader, start=2):
            title = str(get_field(row, "Question_Title", "Title") or "").strip()
            if not title:
                continue
            if title in rows:
                raise ValueError(f"Duplicate CSV question title {title!r} at row {row_number}")
            option_labels = parse_option_labels(get_field(row, "Possible_Answers"))
            raw_answer = get_field(row, "Correct_Answer", "Correct_answer")
            rows[title] = {
                "raw_correct_answer": str(raw_answer or "").strip(),
                "correct_answer": correct_answer_label(raw_answer, option_labels),
                "option_labels": option_labels,
            }
    return rows


def evaluate(csv_rows: Dict[str, dict], probabilities: dict) -> dict:
    csv_titles = set(csv_rows)
    probability_titles = set(probabilities)
    shared = sorted(csv_titles & probability_titles)
    question_results = {}
    correct_count = 0
    invalid_count = 0
    tie_count = 0

    for title in shared:
        csv_item = csv_rows[title]
        probability_item = probabilities.get(title)
        distribution = (
            probability_item.get("ORIGINAL_PROBABILITIES")
            if isinstance(probability_item, dict)
            else None
        )
        result = {
            "raw_correct_answer": csv_item["raw_correct_answer"],
            "correct_answer": csv_item["correct_answer"],
        }
        if not isinstance(distribution, dict) or not distribution:
            result.update(status="invalid", error="Missing or empty ORIGINAL_PROBABILITIES")
            invalid_count += 1
            question_results[title] = result
            continue

        try:
            numeric_distribution = {label: float(value) for label, value in distribution.items()}
        except (TypeError, ValueError):
            result.update(status="invalid", error="ORIGINAL_PROBABILITIES contains a nonnumeric value")
            invalid_count += 1
            question_results[title] = result
            continue
        if any(not math.isfinite(value) for value in numeric_distribution.values()):
            result.update(status="invalid", error="ORIGINAL_PROBABILITIES contains a nonfinite value")
            invalid_count += 1
            question_results[title] = result
            continue
        if csv_item["correct_answer"] is None:
            result.update(status="invalid", error="CSV correct answer could not be parsed")
            invalid_count += 1
            question_results[title] = result
            continue

        maximum = max(numeric_distribution.values())
        tied_labels = sorted(
            label for label, value in numeric_distribution.items()
            if math.isclose(value, maximum, rel_tol=1e-12, abs_tol=1e-15)
        )
        predicted = tied_labels[0]
        ranked = sorted(numeric_distribution.values(), reverse=True)
        margin = ranked[0] - ranked[1] if len(ranked) > 1 else ranked[0]
        is_correct = predicted == csv_item["correct_answer"]
        correct_count += int(is_correct)
        tie_count += int(len(tied_labels) > 1)
        result.update(
            status="ok",
            predicted_answer=predicted,
            correct=is_correct,
            top_probability=maximum,
            top_margin=margin,
            tied_argmax_labels=tied_labels,
            original_probabilities=numeric_distribution,
        )
        question_results[title] = result

    evaluated = len(shared) - invalid_count
    summary = {
        "csv_questions": len(csv_rows),
        "probability_file_questions": len(probabilities),
        "shared_questions": len(shared),
        "csv_questions_missing_from_probability_file": len(csv_titles - probability_titles),
        "probability_questions_missing_from_csv": len(probability_titles - csv_titles),
        "evaluated_questions": evaluated,
        "invalid_questions": invalid_count,
        "correct_questions": correct_count,
        "incorrect_questions": evaluated - correct_count,
        "argmax_ties": tie_count,
        "accuracy": round(correct_count / evaluated, 4) if evaluated else None,
    }
    return {
        "summary": summary,
        "missing_from_probability_file": sorted(csv_titles - probability_titles),
        "missing_from_csv": sorted(probability_titles - csv_titles),
        "questions": question_results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calculate argmax accuracy from multilevel ORIGINAL_PROBABILITIES."
    )
    parser.add_argument("--csv", required=True, help="Multilevel dataset CSV.")
    parser.add_argument("--probs", required=True, help="Probability JSON.")
    parser.add_argument("--output", help="Optional per-question JSON report.")
    args = parser.parse_args()

    csv_rows = load_csv(args.csv)
    with open(args.probs, encoding="utf-8") as file:
        probabilities = json.load(file)
    if not isinstance(probabilities, dict):
        raise ValueError("The probability JSON root must be an object keyed by question title")

    report = evaluate(csv_rows, probabilities)
    print(json.dumps(report["summary"], indent=2))
    accuracy = report["summary"]["accuracy"]
    print(f"Accuracy: {accuracy:.4f}" if accuracy is not None else "Accuracy: n/a")
    if report["summary"]["invalid_questions"]:
        print("\nInvalid questions:")
        for title, result in report["questions"].items():
            if result["status"] == "invalid":
                print(f"  - {title}: {result['error']}")

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, ensure_ascii=False, allow_nan=False)
            file.write("\n")
        print(f"Report written to: {output}")


if __name__ == "__main__":
    main()
