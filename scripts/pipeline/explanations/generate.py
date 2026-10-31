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
import csv
import re
import time
import os
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
# Markdown fence cleaner
# ===============================

def strip_markdown_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text



# ===============================
# JSON-safe parse helper
# ===============================

def try_parse_json(text: str):
    text = strip_markdown_fences(text).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    # fallback: extract first {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except Exception:
            pass

    return None
# ===============================
# Parse A/B/C/... answer options
# ===============================

def normalize_title_key(value: str) -> str:
    return " ".join((value or "").strip().split()).lower()


def load_rows_by_title(csv_path: str):
    csv_map = {}

    with open(csv_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile, skipinitialspace=True)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {csv_path}")

        field_lookup = {name.strip().lower(): name for name in reader.fieldnames}
        title_col = field_lookup.get("question_title")
        question_col = field_lookup.get("question_text")
        intro_col = field_lookup.get("intro_text")
        answers_col = field_lookup.get("possible_answers")

        if title_col is None:
            raise ValueError(f"Missing 'Question_Title' column in CSV: {csv_path}")
        if question_col is None:
            raise ValueError(f"Missing 'Question_Text' column in CSV: {csv_path}")
        if intro_col is None:
            raise ValueError(f"Missing 'Intro_Text' column in CSV: {csv_path}")
        if answers_col is None:
            raise ValueError(f"Missing 'Possible_Answers' column in CSV: {csv_path}")

        for row in reader:
            title = (row.get(title_col) or "").strip()
            if not title:
                continue

            csv_map[normalize_title_key(title)] = {
                "Question_Text": (row.get(question_col) or "").strip(),
                "Intro_Text": (row.get(intro_col) or "").strip(),
                "Possible_Answers": row.get(answers_col) or "",
            }

    return csv_map


def parse_possible_answers(field):
    # Remove surrounding brackets if present
    field = field.strip().lstrip("[").rstrip("]")

    # Split ONLY at "'A." "'B." "'C." ... style markers
    parts = re.split(r"\s*'(?=[A-Z]\.)", field)

    options = []

    for part in parts:
        part = part.strip().rstrip(",").rstrip("'").strip()
        if not part:
            continue

        # Extract first capital letter + text
        m = re.match(r"([A-Z])\.\s*(.*)", part)
        if m:
            letter = m.group(1)
            text = m.group(2).strip()
            options.append((letter, text))

    return options



# ===============================
# Ask Magistral for answer + explanation
# ===============================

def get_answer_and_explanation(question, options, title, intro_text="", retry=2):

    options_text = "\n".join([f"{l}. {t}" for l, t in options])
    intro_text = (intro_text or "").strip()

    system_prompt = format_prompt("answer_explanation.system")

    if intro_text:
        # print("===WITH INTRODUCTION===")
        user_prompt = format_prompt(
            "answer_explanation.user_with_intro",
            intro_text=intro_text,
            question=question,
            options_text=options_text,
        )
    else:
        user_prompt = format_prompt(
            "answer_explanation.user",
            question=question,
            options_text=options_text,
        )

    while retry >= 0:

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        raw = llm.generate(messages)
        # print(f"\nModel output for title: {title}\n{raw}\n")
        # print("============================================")

        parsed = try_parse_json(raw)

        if parsed and isinstance(parsed, dict) \
           and "answer" in parsed and "explanation" in parsed:
            return parsed

        retry -= 1
        time.sleep(0.2)

    # print("Explanation generation failed.")
    return None



# ===============================
# MAIN PIPELINE
# ===============================

def generate_explanations(cf_json, csv_path, out_path):

    with open(cf_json, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    # load CSV lookup
    csv_map = load_rows_by_title(csv_path)

    out = {}

    def save_progress():
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, ensure_ascii=False)

    for title, case in data.items():
        # print(f"\nExplaining: {title}")

        row = csv_map.get(normalize_title_key(title))
        if not row:
            continue

        options = parse_possible_answers(row.get("Possible_Answers", ""))
        result = {}
        intro_text = row.get("Intro_Text", "")
        original_question = row.get("Question_Text", "")

        # --- ORIGINAL QUESTION ---
        if not original_question:
            continue
        orig = get_answer_and_explanation(
            original_question,
            options,
            title=title,
            intro_text=intro_text,
        )

        if not orig:
            continue

        result["original"] = orig
        out[title] = result
        save_progress()

        # --- COUNTERFACTUALS ---
        cf_results = {}

        for concept, cf_list in case.get("COUNTERFACTUALS", {}).items():
            cf_results[concept] = []

            for cf in cf_list:
                ans = get_answer_and_explanation(
                    cf["counterfactual"],
                    options,
                    title=title,
                    intro_text=intro_text,
                )

                if ans:
                    cf_results[concept].append({
                        "value": cf["value"],
                        **ans
                    })
                    result["counterfactuals"] = cf_results
                    out[title] = result
                    save_progress()

                time.sleep(0.2)

        result["counterfactuals"] = cf_results
        out[title] = result
        save_progress()

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)

    print("\nWritten explanations →", out_path)



# ===============================
# RUN
# ===============================

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
    )
    parser.add_argument(
        "--csv",
        required=True,
    )
    parser.add_argument(
        "--output",
        required=True,
    )
    add_llm_cli_args(parser)
    args = parser.parse_args()

    configure_llm_from_args(args)
    llm = get_chat_client()

    generate_explanations(
        args.input,
        args.csv,
        args.output,
    )
