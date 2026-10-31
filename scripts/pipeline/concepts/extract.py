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
import math
import re
from pathlib import Path


MAX_RETRIES = 10
def _add_src_to_path():
    import sys

    for parent in Path(__file__).resolve().parents:
        if (parent / "utils" / "llm_client.py").is_file():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return


_add_src_to_path()
from utils.llm_client import add_llm_cli_args, configure_llm_from_args, get_chat_client
from utils.prompt_loader import format_prompt
from utils.csv_fields import get_field, has_field

llm = None

def _extract_json_object(text: str) -> str:
    start = text.find("{")
    if start == -1:
        return text

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return text[start:]


def _repair_json_text(text: str) -> str:
    repaired = text.strip()
    repaired = repaired.replace("“", '"').replace("”", '"').replace("’", "'")
    repaired = re.sub(r"^```(?:json)?\s*", "", repaired, flags=re.IGNORECASE)
    repaired = re.sub(r"\s*```$", "", repaired)
    repaired = _extract_json_object(repaired)

    # Remove accidental markdown list markers before key/value lines.
    repaired = re.sub(r'^\s*-\s*(?=")', "", repaired, flags=re.MULTILINE)
    repaired = re.sub(r"^\s*-\s*(?=[A-Za-z_])", "", repaired, flags=re.MULTILINE)

    # Quote unquoted JSON keys.
    repaired = re.sub(
        r'^(\s*)([A-Za-z_][A-Za-z0-9 _/\-\(\)\.]*?)\s*:',
        lambda m: f'{m.group(1)}"{m.group(2).strip()}":',
        repaired,
        flags=re.MULTILINE,
    )

    # Remove trailing commas before closing braces.
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    return repaired


def _parse_model_json(text: str) -> dict | None:
    candidates = [text, _extract_json_object(text), _repair_json_text(text)]
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            continue
    return None


def run_magistral(question: str) -> dict:
    messages = [
        {"role": "system", "content": format_prompt("concept_extraction.system")},
        {
            "role": "user",
            "content": format_prompt("concept_extraction.user", question=question),
        },
    ]

    for attempt in range(1, MAX_RETRIES + 1):
        text = llm.generate(
            messages,
            max_new_tokens=1024,
            temperature=0.1,
            do_sample=False,
            top_p=None,
            top_k=None,
        )

        parsed = _parse_model_json(text)
        if parsed is not None:
            return parsed
        else:
            print(f"\n⚠️ JSON parsing failed (attempt {attempt}/{MAX_RETRIES})")
            print("Model output was:")
            print(text)

            if attempt == MAX_RETRIES:
                print("❌ Giving up — returning empty object.")
                return {}
            else:
                print("↻ Retrying...")



# ---------------------------
# CSV processing
# ---------------------------

def process_csv(csv_path, output_file):
    results = {}
    processed = 0
    skipped = 0

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        has_question_label = has_field(reader.fieldnames, "Question_Label")
        if not has_question_label:
            print(
                "Input CSV has no 'Question_Label' column; "
                "processing all rows instead of filtering to Reasoning."
            )

        for i, row in enumerate(reader):
            label = get_field(row, "Question_Label").strip() if has_question_label else "Reasoning"
            if has_question_label and label != "Reasoning":
                skipped += 1
                continue

            title = get_field(row, "Question_Title")
            question = get_field(row, "Question_Text", "Question_text")

            print(f"[{i}] Processing → {title}")

            concepts = run_magistral(question)
            results[title] = concepts
            processed += 1

    output_path = os.path.join(os.getcwd(), output_file)

    with open(output_path, "w", encoding="utf-8") as out:
        json.dump(results, out, indent=4, ensure_ascii=False)

    print(f"\n✔ Saved results to {output_path}")
    if skipped:
        print(f"Processed {processed} Reasoning questions; skipped {skipped} non-Reasoning questions.")
    else:
        print(f"Processed {processed} questions.")


# ---------------------------
# Run
# ---------------------------
# make directory for legal concepts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", "--input", dest="csv", required=True)
    parser.add_argument("--output", required=True)
    add_llm_cli_args(parser)
    args = parser.parse_args()

    configure_llm_from_args(args)
    llm = get_chat_client()

    process_csv(args.csv, args.output)
