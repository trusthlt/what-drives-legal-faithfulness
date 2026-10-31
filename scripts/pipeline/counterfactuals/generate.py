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

import csv
import json
import os
import re
import argparse
from datetime import datetime
from json import JSONDecodeError, JSONDecoder
from pathlib import Path
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

llm = None

# ===============================
# JSON Extraction Utilities
# ===============================

def normalize_title_key(value: str) -> str:
    return " ".join(value.strip().split()).lower()


def load_context_by_title_from_csv(csv_path: str) -> dict:
    context_map = {}
    csv_file = Path(csv_path)

    with csv_file.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, skipinitialspace=True)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {csv_file}")

        field_lookup = {name.strip().lower(): name for name in reader.fieldnames}
        title_col = field_lookup.get("question_title")
        question_col = field_lookup.get("question_text")

        if title_col is None:
            raise ValueError(f"Missing 'Question_Title' column in CSV: {csv_file}")
        if question_col is None:
            raise ValueError(f"Missing 'Question_Text' column in CSV: {csv_file}")

        for row in reader:
            title = (row.get(title_col) or "").strip()
            question = (row.get(question_col) or "").strip()
            if not title:
                continue

            context_map[normalize_title_key(title)] = {
                "question": question,
            }

    return context_map


def strip_markdown_fences(text: str) -> str:
    text = text.strip()

    # Handle ```json ... ``` OR ``` ... ```
    if text.startswith("```"):
        lines = text.splitlines()

        # remove first fence
        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        # remove trailing fence
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    return text


def _extract_json_from_text(text: str):
    # NEW: clean markdown wrappers before parsing
    print("========================================")
    print(text)
    print("========================================")
    
    text = strip_markdown_fences(text).strip()

    # 1) Try direct load
    try:
        return json.loads(text)
    except JSONDecodeError:
        pass

    # 2) Try prefix JSON parse
    try:
        decoder = JSONDecoder()
        obj, idx = decoder.raw_decode(text)
        return obj
    except JSONDecodeError:
        pass

    # 3) Search for first balanced {...}
    start = text.find('{')
    if start == -1:
        raise JSONDecodeError("No JSON object found", text, 0)

    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                candidate = text[start:i+1]
                try:
                    return json.loads(candidate)
                except JSONDecodeError:
                    break

    # 4) Regex fallback
    m = re.search(r'(\{.*\})', text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except JSONDecodeError:
            pass

    raise JSONDecodeError("Could not extract JSON from model output", text, 0)
# ===============================
# Counterfactual Generator
# ===============================


def generate_counterfactuals(
    question: str,
    concept: str,
    original_value: str,
    min_count: int = 1,
    max_count: int = 4,
    retries: int = 2,
):
    system_prompt = format_prompt(
        "counterfactual_generation.system",
        question=question,
        min_count=min_count,
        max_count=max_count,
    )

    user_text = format_prompt(
        "counterfactual_generation.user",
        concept=concept,
        original_value=original_value,
        min_count=min_count,
        max_count=max_count,
    )

    attempt = 0
    while attempt <= retries:
        attempt += 1

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]

        raw = llm.generate(messages)

        try:
            parsed = _extract_json_from_text(raw)
        except JSONDecodeError as ex:
            preview = (raw[:1000] + "...") if len(raw) > 1000 else raw
            print(f"JSON parse error for concept '{concept}': {ex}. Output preview:\n{preview}")

            if attempt <= retries:
                user_text += (
                    "\n\nRetry guidance:"
                    "\n- Return valid JSON only."
                    "\n- Do not add prose outside the JSON."
                    "\n- Keep each value short."
                    "\n- Keep each counterfactual close to the original question."
                )
                continue
            return []

        if (
            not isinstance(parsed, dict)
            or "new_values" not in parsed
            or not isinstance(parsed["new_values"], list)
        ):
            print(f"Unexpected JSON structure for '{concept}': {parsed}")

            if attempt <= retries:
                user_text += (
                    "\n\nRetry guidance:"
                    "\n- Return exactly one JSON object with key 'new_values'."
                    "\n- 'new_values' must be a list of objects with keys 'value' and 'counterfactual'."
                )
                continue
            return []

        clean = []
        for nv in parsed["new_values"]:
            if not isinstance(nv, dict):
                continue

            val = nv.get("value") or nv.get("VALUE")
            cf = nv.get("counterfactual") or nv.get("COUNTERFACTUAL")

            if isinstance(val, str) and isinstance(cf, str):
                val = val.strip()
                cf = cf.strip()
                if val and cf:
                    clean.append({"value": val, "counterfactual": cf})

        if len(clean) > max_count:
            clean = clean[:max_count]

        if len(clean) < min_count:
            print(
                f"Generated {len(clean)} alternatives for '{concept}' "
                f"(need >= {min_count}). Retrying ({attempt}/{retries})..."
            )

            if attempt <= retries:
                user_text += (
                    "\n\nRetry guidance:"
                    "\n- Your previous answer did not produce enough valid alternatives."
                    "\n- Do not invent new facts."
                    "\n- Stay closer to the original wording."
                    "\n- Produce alternatives only if they can be implemented coherently with local edits."
                    f"\n- Return at least {min_count} valid entries."
                )
                continue

        return clean

    return []


# ===============================
# Single Example Processor
# ===============================

def process_one_example(input_json_path: str, output_json_path: str, context_map: dict):
    with open(input_json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, dict) or len(data) == 0:
        print("Input JSON is empty or not a mapping.")
        return

    first_title = next(iter(data.keys()))
    item = data[first_title]

    if not isinstance(item, dict):
        print("First item is not a dict, aborting.")
        return

    csv_context = context_map.get(normalize_title_key(first_title), {})
    question = csv_context.get("question", "")
    if not question:
        print(f"No CSV Question_Text match for '{first_title}', aborting.")
        return

    result = {k: v for k, v in item.items()}
    cf_map = {}

    for concept, original_value in item.items():
        if concept == "ORIGINAL QUESTION":
            continue
        if original_value is None:
            continue

        orig_val_str = (
            original_value
            if isinstance(original_value, str)
            else json.dumps(original_value, ensure_ascii=False)
        )

        new_values = generate_counterfactuals(
            question=question,
            concept=concept,
            original_value=orig_val_str,
        )

        if new_values:
            cf_map[concept] = new_values

    result["COUNTERFACTUALS"] = cf_map
    out_payload = {first_title: result}

    with open(output_json_path, "w", encoding="utf-8") as out_fh:
        json.dump(out_payload, out_fh, indent=2, ensure_ascii=False)

    print(f"Wrote counterfactuals for '{first_title}' to {output_json_path}")


# ===============================
# Multi-example batch runner
# ===============================

def process_first_n_examples(
    input_json_path: str,
    output_json_path: str,
    context_map: dict,
    n: int = 10,
):
    with open(input_json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, dict) or len(data) == 0:
        print("Input JSON is empty or not a mapping.")
        return

    titles = list(data.keys()) if n is None or n < 0 else list(data.keys())[:n]
    out_payload = {}

    for title in titles:
        item = data.get(title)

        if not isinstance(item, dict):
            print(f"Skipping '{title}': item not a dict.")
            continue

        csv_context = context_map.get(normalize_title_key(title), {})
        question = csv_context.get("question", "")
        if not question:
            print(f"Skipping '{title}': no matching Question_Text in CSV.")
            continue

        result = {k: v for k, v in item.items()}
        cf_map = {}

        for concept, original_value in item.items():
            if concept == "ORIGINAL QUESTION":
                continue
            if original_value is None:
                continue

            orig_val_str = (
                original_value
                if isinstance(original_value, str)
                else json.dumps(original_value, ensure_ascii=False)
            )

            new_values = generate_counterfactuals(
                question=question,
                concept=concept,
                original_value=orig_val_str,
            )

            if new_values:
                cf_map[concept] = new_values

        result["COUNTERFACTUALS"] = cf_map
        out_payload[title] = result

        print(f"Processed '{title}'")

    with open(output_json_path, "w", encoding="utf-8") as out_fh:
        json.dump(out_payload, out_fh, indent=2, ensure_ascii=False)

    print(f"Wrote counterfactuals for {len(out_payload)} examples to {output_json_path}")


# ===============================
# Run
# ===============================

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    default_run_tag = datetime.now().strftime("%d%b")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="Input flattened concepts JSON",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON with generated counterfactuals",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=-1,
        help="Maximum examples to process; -1 processes all",
    )
    parser.add_argument(
        "--csv",
        "--glannon-csv",
        dest="csv",
        required=True,
        help="CSV containing Question_Title and Question_Text",
    )
    add_llm_cli_args(parser)
    args = parser.parse_args()

    configure_llm_from_args(args)
    llm = get_chat_client()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    context_map = load_context_by_title_from_csv(args.csv)

    process_first_n_examples(
        input_json_path=args.input,
        output_json_path=str(output_path),
        context_map=context_map,
        n=args.max_examples,
    )
