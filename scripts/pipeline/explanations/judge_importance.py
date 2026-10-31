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
import re
import csv
import time
import os
from pathlib import Path
from json import JSONDecodeError, JSONDecoder
from typing import Dict, List, Any, Optional
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

def normalize_title_key(value: str) -> str:
    return " ".join((value or "").strip().split()).lower()


def load_rows_by_title(csv_path: str):
    csv_map = {}

    with open(csv_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile, skipinitialspace=True)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {csv_path}")

        field_lookup = {name.strip().lower(): name for name in reader.fieldnames}
        title_col = field_lookup.get("question_title") or field_lookup.get("title")
        analysis_col = field_lookup.get("analysis")
        question_col = field_lookup.get("question_text")

        if title_col is None:
            raise ValueError(f"Missing 'Question_Title' column in CSV: {csv_path}")
        if analysis_col is None:
            raise ValueError(f"Missing 'Analysis' column in CSV: {csv_path}")
        if question_col is None:
            raise ValueError(f"Missing 'Question_Text' column in CSV: {csv_path}")

        for row in reader:
            title = (row.get(title_col) or "").strip()
            if not title:
                continue
            csv_map[normalize_title_key(title)] = {
                "Analysis": (row.get(analysis_col) or "").strip(),
                "Question_Text": (row.get(question_col) or "").strip(),
            }

    return csv_map


def strip_markdown_fences(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


# ===============================
# Robust JSON extraction
# ===============================

def extract_json_from_text(text: str):
    text = strip_markdown_fences(text).strip()

    if not text:
        raise JSONDecodeError("Empty model output", text, 0)

    # direct parse
    try:
        return json.loads(text)
    except JSONDecodeError:
        pass

    # prefix JSON
    try:
        decoder = JSONDecoder()
        obj, _ = decoder.raw_decode(text)
        return obj
    except JSONDecodeError:
        pass

    # balanced braces scan
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except Exception:
                        break

    # regex fallback
    m = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if m:
        return json.loads(m.group(1))

    raise JSONDecodeError("Could not extract JSON", text, 0)
# ===============================
# New: build concept/value menu
# ===============================

def build_concept_value_menu(case_concepts: Dict[str, Any]) -> str:
    """
    case_concepts is your new JSON format for a single title:
    {
      "person": "Marla",
      ...
      "COUNTERFACTUALS": {
         "person": [{"value": "John", "counterfactual": "..."}, ...],
         ...
      }
    }
    """

    cf_block = case_concepts.get("COUNTERFACTUALS", {})
    lines = []

    for concept, original_value in case_concepts.items():
        if concept in ("ORIGINAL QUESTION", "COUNTERFACTUALS", "Intro_Text"):
            continue

        possible_values = []

        if isinstance(cf_block, dict) and concept in cf_block and isinstance(cf_block[concept], list):
            for item in cf_block[concept]:
                v = item.get("value")
                if isinstance(v, str) and v.strip():
                    possible_values.append(v.strip())

        # de-dup while preserving order
        seen = set()
        possible_values = [v for v in possible_values if not (v in seen or seen.add(v))]

        lines.append(f"- concept: {concept}")
        lines.append(f"  original_value: {str(original_value)}")

        if possible_values:
            lines.append(f"  possible_counterfactual_values: {possible_values}")
        else:
            lines.append(f"  possible_counterfactual_values: []")

    return "\n".join(lines)


def extract_concept_names(case_concepts: Dict[str, Any]) -> List[str]:
    return [
        k for k in case_concepts.keys()
        if k not in ("ORIGINAL QUESTION", "COUNTERFACTUALS", "Intro_Text")
    ]


# ===============================
# Concept Importance Scoring (Magistral)
# ===============================

def score_concept_importance(
    explanation: str,
    concept_names: List[str],
    concept_value_menu: str,
    original_question: str = "",
    counterfactual_concept: str = "",
    counterfactual_value: str = "",
    retries=2,
):

    system_prompt = format_prompt("concept_importance.system")

    cf_context = ""
    if counterfactual_concept:
        cf_context = format_prompt(
            "concept_importance.counterfactual_context",
            counterfactual_concept=counterfactual_concept,
            counterfactual_value=counterfactual_value,
        )

    user_prompt = format_prompt(
        "concept_importance.user",
        original_question=(original_question or "").strip(),
        counterfactual_context=cf_context,
        explanation=explanation,
        concept_names=concept_names,
        concept_value_menu=concept_value_menu,
    )

    while retries >= 0:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        raw = llm.generate(messages)

        try:
            parsed = extract_json_from_text(raw)
        except Exception:
            retries -= 1
            time.sleep(0.2)
            continue

        importance = parsed.get("importance", {})

        if isinstance(importance, dict):
            cleaned = {}
            for concept in concept_names:
                raw_score = importance.get(concept, 0)
                try:
                    score = int(raw_score)
                except Exception:
                    score = 0
                if score < 0:
                    score = 0
                if score > 4:
                    score = 4
                cleaned[concept] = score
            return cleaned

        retries -= 1

    print("Concept importance scoring failed for explanation:")
    print(explanation[:300], "...\n")
    return {concept: 0 for concept in concept_names}


# ===============================
# MAIN PIPELINE
# ===============================

def process_explanations(
    explanations_json_path: str,
    legal_concepts_json_path: str,
    csv_path: str,
    output_json_path: str,
):
    with open(explanations_json_path, "r", encoding="utf-8") as fh:
        explanations_data = json.load(fh)

    with open(legal_concepts_json_path, "r", encoding="utf-8") as fh:
        concepts_data = json.load(fh)

    # load source rows from CSV (title-linked)
    csv_map = load_rows_by_title(csv_path)

    output = {}

    for title, expl_block in explanations_data.items():
        print(f"Scoring concept importance for: {title}")

        case_concepts = concepts_data.get(title)
        if not isinstance(case_concepts, dict):
            continue

        concept_names = extract_concept_names(case_concepts)
        if not concept_names:
            continue

        concept_value_menu = build_concept_value_menu(case_concepts)

        entry = {}

        row = csv_map.get(normalize_title_key(title), {})
        original_question = row.get("Question_Text", "")

        # ORIGINAL EXPLANATION (gold from CSV)
        gold = row.get("Analysis", "")
        if isinstance(gold, str):
            entry["original"] = score_concept_importance(
                explanation=gold,
                concept_names=concept_names,
                concept_value_menu=concept_value_menu,
                original_question=original_question,
            )
        else:
            entry["original"] = {c: 0 for c in concept_names}

        # COUNTERFACTUAL EXPLANATIONS (Magistral output)
        cf_out = {}

        for concept, cf_list in expl_block.get("counterfactuals", {}).items():
            if not isinstance(cf_list, list):
                continue

            cf_out[concept] = []

            for item in cf_list:
                expl = item.get("explanation")
                val = item.get("value")

                if not isinstance(expl, str):
                    continue

                importance = score_concept_importance(
                    explanation=expl,
                    concept_names=concept_names,
                    concept_value_menu=concept_value_menu,
                    original_question=original_question,
                    counterfactual_concept=concept,
                    counterfactual_value=(val if isinstance(val, str) else str(val)),
                )

                cf_out[concept].append({
                    "value": val,
                    "importance": importance,
                })

        entry["counterfactuals"] = cf_out
        output[title] = entry

    with open(output_json_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    print("\nWritten concept importance scores →", output_json_path)


# ===============================
# ENTRY POINT
# ===============================

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--explanations",
        required=True,
    )
    parser.add_argument(
        "--concepts",
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

    process_explanations(
        args.explanations,
        args.concepts,
        args.csv,
        args.output,
    )
