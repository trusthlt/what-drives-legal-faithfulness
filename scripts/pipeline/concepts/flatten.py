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

import json
import csv
import argparse
from pathlib import Path


def _add_src_to_path():
    import sys

    for parent in Path(__file__).resolve().parents:
        if (parent / "utils" / "csv_fields.py").is_file():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return


_add_src_to_path()
from utils.csv_fields import get_field

MIN_VALUES_REQUIRED = 4

# Titles to remove if they appear anywhere in the title string (checked manually)
BLOCKED_TITLES = [
    "Foundering fathers.",
    "Smith’s amendment.",
    "By the numbers.",
    "Stooges do the math.",
    "Limiting the limits.",
    "Concurrents.",
    "Line-up.",
    "Quasi-distinctions.",
    "Shaffer survivors?",
    "Barely reasonable.",
    "Unnoticed notice.",
    "Dim prospects for partnership.",
    "Plausibility problems.",
    "Fine distinctions.",
    "Triggering Rule 11.",
    "Fighting an uphill battle.",
    "Fudd-ging Rule 11.",
    "Relevance and discretion.",
]


def flatten_inner_dict(inner: dict) -> dict:
    """
    Flatten inner dict so that list values become key_index : value entries.
    Non-primitive list/dict elements are JSON-serialized.
    """
    out = {}
    for k, v in inner.items():
        if isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, (dict, list)):
                    out[f"{k}_{i}"] = json.dumps(item, ensure_ascii=False)
                else:
                    out[f"{k}_{i}"] = item
        else:
            out[k] = v
    return out


def load_questions_map(csv_path: Path) -> dict:
    mapping = {}
    with csv_path.open(newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            title = get_field(row, "Question_Title")
            question = get_field(row, "Question_Text", "Question_text")
            intro_text = get_field(row, "Intro_Text")
            if title:
                mapping[title] = {
                    "ORIGINAL QUESTION": question,
                    "Intro_Text": intro_text,
                }
    return mapping


def title_is_blocklisted(title: str) -> bool:
    """Return True if title *contains* any blocked title substring."""
    lowered = title.lower()
    return any(b.lower() in lowered for b in BLOCKED_TITLES)


def process(input_json_path: Path, questions_csv_path: Path, output_json_path: Path):
    data = json.loads(input_json_path.read_text(encoding='utf-8'))
    questions = load_questions_map(questions_csv_path)

    out = {}
    removed_few_values = []
    removed_blocklist = []

    for top_key, inner in data.items():

        # ------------------------------------
        # RULE 1 — remove if title is blocked
        # ------------------------------------
        if title_is_blocklisted(top_key):
            removed_blocklist.append(top_key)
            continue

        # ------------------------------------
        # Flatten + attach question
        # ------------------------------------
        if isinstance(inner, dict):
            flattened = flatten_inner_dict(inner)
        else:
            flattened = {"value": inner}

        q = questions.get(top_key)
        if isinstance(q, dict):
            if q.get("ORIGINAL QUESTION") is not None:
                flattened["ORIGINAL QUESTION"] = q["ORIGINAL QUESTION"]
            if q.get("Intro_Text") is not None:
                flattened["Intro_Text"] = q["Intro_Text"]

        # ------------------------------------
        # RULE 2 — remove if < MIN_VALUES_REQUIRED
        # ------------------------------------
        non_empty_values = {
            k: v for k, v in flattened.items()
            if v not in (None, "", [], {})
        }

        value_count = len(non_empty_values)

        if value_count < MIN_VALUES_REQUIRED:
            removed_few_values.append((top_key, value_count))
            continue

        out[top_key] = flattened

    # ------------------------------------
    # Write final JSON
    # ------------------------------------
    output_json_path.write_text(
        json.dumps(out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ------------------------------------
    # Report removals
    # ------------------------------------
    if removed_blocklist:
        print("\nRemoved titles due to blocklist rule:")
        for title in removed_blocklist:
            print(f"  - {title}")

    if removed_few_values:
        print("\nRemoved entries with fewer than 5 values:")
        for title, count in removed_few_values:
            print(f"  - {title}  (values = {count})")
    else:
        print("\nNo entries were removed for value-count rule.")

    print(f"\nFinal kept entries: {len(out)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--csv", "--questions", dest="csv", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    INPUT_JSON = Path(args.input)
    QUESTIONS_CSV = Path(args.csv)
    OUTPUT_JSON = Path(args.output)

    process(INPUT_JSON, QUESTIONS_CSV, OUTPUT_JSON)

    print(f"\nWrote filtered + flattened JSON to: {OUTPUT_JSON}")
