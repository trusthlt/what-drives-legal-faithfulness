import atexit
import argparse
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, Optional


def _register_runtime_timer():
    script_start = time.perf_counter()

    def _print_runtime():
        elapsed = time.perf_counter() - script_start
        print(f"\nElapsed time: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")

    atexit.register(_print_runtime)


if __name__ == "__main__":
    _register_runtime_timer()


def normalize_title_key(value: str) -> str:
    return " ".join((value or "").strip().split()).lower()


def normalize_field_name(name: str) -> str:
    return "".join(ch for ch in (name or "").strip().lower() if ch.isalnum())


def get_csv_field(row: dict, *names: str, default: str = "") -> str:
    lookup = {normalize_field_name(key): key for key in row.keys()}
    for name in names:
        key = lookup.get(normalize_field_name(name))
        if key is not None:
            return row.get(key, default)
    return default


def normalize_answer(value: object) -> str:
    text = str(value or "").strip().upper()
    if re.fullmatch(r"\d+", text):
        index = int(text)
        if 1 <= index <= 10:
            return chr(ord("A") + index - 1)
    match = re.search(r"\b([A-J])\b", text)
    if match:
        return match.group(1)
    match = re.search(r"([A-J])", text)
    return match.group(1) if match else ""


def load_original_answers(explanations_path: str) -> Dict[str, str]:
    with open(explanations_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    answers = {}
    for title, block in data.items():
        if not isinstance(block, dict):
            continue
        original = block.get("original")
        if not isinstance(original, dict):
            continue
        answer = normalize_answer(original.get("answer"))
        if answer:
            answers[normalize_title_key(str(title))] = answer

    return answers


def compare_answers(explanations_path: str, csv_path: str, include_missing: bool):
    predicted_by_title = load_original_answers(explanations_path)

    rows = []
    correct = 0
    evaluated = 0
    missing_predictions = 0

    with open(csv_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile, skipinitialspace=True)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {csv_path}")

        for row in reader:
            title = (get_csv_field(row, "Question_Title", "Title", default="") or "").strip()
            if not title:
                continue

            gold = normalize_answer(get_csv_field(row, "Correct_answer", "Correct Answer", "answer", default=""))
            pred = predicted_by_title.get(normalize_title_key(title), "")

            if not pred:
                missing_predictions += 1
                if not include_missing:
                    continue

            is_correct = bool(pred and gold and pred == gold)
            if pred or include_missing:
                evaluated += 1
                correct += int(is_correct)

            rows.append(
                {
                    "Question_Title": title,
                    "predicted_answer": pred,
                    "correct_answer": gold,
                    "is_correct": is_correct,
                }
            )

    accuracy = (correct / evaluated) if evaluated else 0.0
    summary = {
        "accuracy": accuracy,
        "correct": correct,
        "evaluated": evaluated,
        "missing_predictions": missing_predictions,
        "num_json_original_answers": len(predicted_by_title),
        "include_missing": include_missing,
    }
    return summary, rows


def write_json(path: Optional[str], summary: dict, rows: list):
    if not path:
        return
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "rows": rows}, fh, indent=2, ensure_ascii=False)


def write_csv(path: Optional[str], rows: list):
    if not path:
        return
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fieldnames = ["Question_Title", "predicted_answer", "correct_answer", "is_correct"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--explanations",
        required=True,
        help="Explanation JSON containing original.answer for each question title/id.",
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="CSV with Question_Title and Correct_answer columns.",
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="Count CSV rows missing from the explanation JSON as incorrect.",
    )
    args = parser.parse_args()

    summary, rows = compare_answers(args.explanations, args.csv, args.include_missing)
    write_json(args.output_json, summary, rows)
    write_csv(args.output_csv, rows)

    print(
        "Accuracy: "
        f"{summary['accuracy']:.4f} "
        f"({summary['correct']}/{summary['evaluated']})"
    )
    print(f"Missing predictions: {summary['missing_predictions']}")
    if args.output_json:
        print(f"Written JSON report -> {args.output_json}")
    if args.output_csv:
        print(f"Written CSV report -> {args.output_csv}")


if __name__ == "__main__":
    main()
