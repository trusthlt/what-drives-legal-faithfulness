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
from utils.csv_fields import get_field

llm = None


# ===============================
# Markdown fence cleaner
# ===============================

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
# Concept formatting utilities
# ===============================

def get_concept_names(case_concepts: Dict[str, Any]) -> List[str]:
    return [
        k for k in case_concepts.keys()
        if k not in ("ORIGINAL QUESTION", "COUNTERFACTUALS", "Intro_Text")
    ]


def build_concepts_with_values_block(case_concepts: Dict[str, Any]) -> str:
    """
    Produces a prompt-friendly concept list with:
    - concept name
    - original value
    - possible counterfactual values (from COUNTERFACTUALS)
    """
    cf_block = case_concepts.get("COUNTERFACTUALS", {})
    lines = []

    for concept, original_value in case_concepts.items():
        if concept in ("ORIGINAL QUESTION", "COUNTERFACTUALS", "Intro_Text"):
            continue

        possible_values = []
        if isinstance(cf_block, dict) and isinstance(cf_block.get(concept), list):
            for entry in cf_block[concept]:
                v = entry.get("value")
                if isinstance(v, str) and v.strip():
                    possible_values.append(v.strip())

        # deduplicate while preserving order
        seen = set()
        possible_values = [v for v in possible_values if not (v in seen or seen.add(v))]

        lines.append(f"- concept: {concept}")
        lines.append(f"  original_value: {str(original_value)}")

    return "\n".join(lines)


# ===============================
# Concept Importance Scoring (Magistral)
# ===============================

def detect_implied_concepts(
    explanation: str,
    concept_names: List[str],
    concepts_with_values_block: str,
    retries: int = 2,
) -> Dict[str, int]:

    system_prompt = format_prompt("gold_concept_importance.system")
    user_prompt = format_prompt(
        "gold_concept_importance.user",
        explanation=explanation,
        concepts_with_values_block=concepts_with_values_block,
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

        # Backward-compatibility fallback if model returns old list format.
        implied = parsed.get("implied_concepts", [])
        if isinstance(implied, list):
            cleaned = {concept: 0 for concept in concept_names}
            for concept in implied:
                if isinstance(concept, str) and concept in cleaned:
                    cleaned[concept] = 1
            return cleaned

        retries -= 1

    print("Concept importance scoring failed for explanation:")
    print(explanation[:300], "...\n")
    return {concept: 0 for concept in concept_names}


# ===============================
# MAIN PIPELINE
# ===============================

def process_gold_explanations(
    explanations_json_path: str,
    legal_concepts_json_path: str,
    csv_path: str,
    output_json_path: str,
):

    # Only process titles that appear in this file
    with open(explanations_json_path, "r", encoding="utf-8") as fh:
        explanations_data = json.load(fh)

    # Concept values per Title (new format contains COUNTERFACTUALS)
    with open(legal_concepts_json_path, "r", encoding="utf-8") as fh:
        concepts_data = json.load(fh)

    # Gold explanations from CSV
    gold_map = {}
    with open(csv_path, newline="", encoding="utf-8") as csvfile:
        for row in csv.DictReader(csvfile):
            title = get_field(row, "Question_Title")
            analysis = get_field(row, "Analysis")
            if title and analysis:
                gold_map[title] = analysis

    output = {}

    for title in explanations_data.keys():
        print(f"\nProcessing gold EE for: {title}")

        case_concepts = concepts_data.get(title)
        if not isinstance(case_concepts, dict):
            print(" -> skipped (no concept entry)")
            continue

        concept_names = get_concept_names(case_concepts)
        if not concept_names:
            print(" -> skipped (no extracted concepts)")
            continue

        gold_expl = gold_map.get(title)
        if not isinstance(gold_expl, str):
            print(" -> skipped (no gold explanation)")
            continue

        concepts_with_values_block = build_concepts_with_values_block(case_concepts)

        importance = detect_implied_concepts(
            explanation=gold_expl,
            concept_names=concept_names,
            concepts_with_values_block=concepts_with_values_block,
        )

        output[title] = {
            "original_importance": importance
        }

    with open(output_json_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    print("\n✓ Written gold EE importance scores →", output_json_path)


# ===============================
# ENTRY POINT
# ===============================

if __name__ == "__main__":
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

    process_gold_explanations(
        args.explanations,
        args.concepts,
        args.csv,
        args.output,
    )
