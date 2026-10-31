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
import inspect
import json
import os
import re
from dataclasses import dataclass
from json import JSONDecodeError, JSONDecoder
from pathlib import Path
from typing import Any, Dict, List, Optional


def _add_src_to_path():
    import sys

    for parent in Path(__file__).resolve().parents:
        if (parent / "utils" / "prompt_loader.py").is_file():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return


_add_src_to_path()
from utils.prompt_loader import format_prompt


DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B"

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


@dataclass
class ImportanceTask:
    title: str
    kind: str
    explanation: str
    concept_names: List[str]
    concept_value_menu: str
    original_question: str = ""
    counterfactual_concept: str = ""
    counterfactual_value: str = ""
    attempt: int = 1


class OfflineVLLMChatClient:
    def __init__(
        self,
        model: str,
        tokenizer: Optional[str],
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        max_model_len: Optional[int],
        dtype: str,
        tokenizer_mode: str,
        trust_remote_code: bool,
        limit_mm_per_prompt: Optional[dict],
        enforce_eager: bool,
        gdn_prefill_backend: Optional[str],
        extra_llm_kwargs: Optional[dict],
        guided_json_schema: Optional[dict],
    ):
        from vllm import LLM, SamplingParams

        llm_kwargs = {
            "model": model,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "dtype": dtype,
            "tokenizer_mode": tokenizer_mode,
            "trust_remote_code": trust_remote_code,
            "enforce_eager": enforce_eager,
        }
        if tokenizer:
            llm_kwargs["tokenizer"] = tokenizer
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len
        if limit_mm_per_prompt is not None:
            llm_kwargs["limit_mm_per_prompt"] = limit_mm_per_prompt
        if gdn_prefill_backend:
            # Qwen3.6 uses hybrid GDN layers. On Hopper/Blackwell, vLLM can
            # auto-select FlashInfer GDN prefill, which requires the optional
            # flashinfer package. Triton/FLA avoids that dependency.
            llm_kwargs["gdn_prefill_backend"] = gdn_prefill_backend
        if extra_llm_kwargs:
            llm_kwargs.update(extra_llm_kwargs)

        llm_kwargs = filter_supported_llm_kwargs(LLM, llm_kwargs)

        print(f"Loading offline vLLM model: {model}")
        if tokenizer:
            print(f"Using tokenizer: {tokenizer}")
        print(f"vLLM init kwargs: {llm_kwargs}")
        self.llm = LLM(**llm_kwargs)
        sampling_kwargs = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": max_tokens,
        }
        if guided_json_schema is not None:
            try:
                from vllm.sampling_params import GuidedDecodingParams

                sampling_kwargs["guided_decoding"] = GuidedDecodingParams(json=guided_json_schema)
            except Exception as exc:
                print(f"[WARN] Guided JSON decoding is not available in this vLLM install: {exc}")

        sampling_kwargs = filter_supported_sampling_kwargs(SamplingParams, sampling_kwargs)
        self.sampling_params = SamplingParams(**sampling_kwargs)

    def generate_batch(self, batch_messages: List[List[Dict[str, str]]]) -> List[str]:
        if hasattr(self.llm, "chat"):
            outputs = self.llm.chat(
                batch_messages,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )
        else:
            prompts = [render_messages(messages) for messages in batch_messages]
            outputs = self.llm.generate(
                prompts,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )

        return [output.outputs[0].text.strip() if output.outputs else "" for output in outputs]


def filter_supported_llm_kwargs(llm_cls, kwargs: dict) -> dict:
    signature = inspect.signature(llm_cls.__init__)
    params = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return kwargs

    supported = set(params.keys()) - {"self"}
    dropped = sorted(key for key in kwargs if key not in supported)
    if dropped:
        print(f"[WARN] Dropping unsupported vLLM LLM kwargs: {', '.join(dropped)}")
    return {key: value for key, value in kwargs.items() if key in supported}


def filter_supported_sampling_kwargs(sampling_cls, kwargs: dict) -> dict:
    signature = inspect.signature(sampling_cls.__init__)
    params = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return kwargs

    supported = set(params.keys()) - {"self"}
    dropped = sorted(key for key in kwargs if key not in supported)
    if dropped:
        print(f"[WARN] Dropping unsupported vLLM SamplingParams kwargs: {', '.join(dropped)}")
    return {key: value for key, value in kwargs.items() if key in supported}


# ===============================
# Shared parsing and CSV helpers
# ===============================

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


def load_rows_by_title(csv_path: str):
    csv_map = {}

    with open(csv_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile, skipinitialspace=True)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {csv_path}")

        for row in reader:
            title = (get_csv_field(row, "Question_Title", "Title", default="") or "").strip()
            if not title:
                continue
            csv_map[normalize_title_key(title)] = {
                "Analysis": (get_csv_field(row, "Analysis", default="") or "").strip(),
                "Question_Text": (get_csv_field(row, "Question_Text", "Question_text", default="") or "").strip(),
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


def extract_json_from_text(text: str):
    text = strip_markdown_fences(text).strip()
    if not text:
        raise JSONDecodeError("Empty model output", text, 0)

    try:
        return json.loads(text)
    except JSONDecodeError:
        pass

    try:
        decoder = JSONDecoder()
        obj, _ = decoder.raw_decode(text)
        return obj
    except JSONDecodeError:
        pass

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
                        return json.loads(text[start : i + 1])
                    except Exception:
                        break

    match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if match:
        return json.loads(match.group(1))

    raise JSONDecodeError("Could not extract JSON", text, 0)


def parse_json_arg(value: Optional[str], arg_name: str):
    if value is None or value == "":
        return None
    try:
        return json.loads(value)
    except JSONDecodeError as exc:
        raise ValueError(f"{arg_name} must be valid JSON. Got: {value}") from exc


# ===============================
# Prompt construction
# ===============================

def build_concept_value_menu(case_concepts: Dict[str, Any]) -> str:
    cf_block = case_concepts.get("COUNTERFACTUALS", {})
    lines = []

    for concept, original_value in case_concepts.items():
        if concept in ("ORIGINAL QUESTION", "COUNTERFACTUALS", "Intro_Text"):
            continue

        possible_values = []
        if isinstance(cf_block, dict) and concept in cf_block and isinstance(cf_block[concept], list):
            for item in cf_block[concept]:
                if not isinstance(item, dict):
                    continue
                value = item.get("value")
                if isinstance(value, str) and value.strip():
                    possible_values.append(value.strip())

        seen = set()
        possible_values = [value for value in possible_values if not (value in seen or seen.add(value))]

        lines.append(f"- concept: {concept}")
        lines.append(f"  original_value: {str(original_value)}")
        lines.append(f"  possible_counterfactual_values: {possible_values if possible_values else []}")

    return "\n".join(lines)


def extract_concept_names(case_concepts: Dict[str, Any]) -> List[str]:
    return [
        key for key in case_concepts.keys()
        if key not in ("ORIGINAL QUESTION", "COUNTERFACTUALS", "Intro_Text")
    ]


def build_importance_json_schema() -> dict:
    # The concept keys are dynamic, so the schema enforces the outer shape and
    # integer score range. The cleaner below still enforces exact concept names.
    return {
        "type": "object",
        "properties": {
            "importance": {
                "type": "object",
                "additionalProperties": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 4,
                },
            }
        },
        "required": ["importance"],
        "additionalProperties": False,
    }


def build_importance_skeleton(concept_names: List[str]) -> str:
    return json.dumps({"importance": {concept: 0 for concept in concept_names}}, ensure_ascii=False)


def build_messages(task: ImportanceTask, strict_json_prompt: bool, qwen_no_think: bool) -> List[Dict[str, str]]:
    if strict_json_prompt:
        system_prompt = (
            "You are a deterministic JSON scoring function for legal concept importance.\n"
            "Return exactly one valid JSON object and nothing else.\n"
            "Do not explain. Do not analyze aloud. Do not include markdown.\n"
            "Your first character must be { and your last character must be }.\n"
            "The JSON must have exactly this outer shape: {\"importance\": {\"concept_name\": 0}}.\n"
            "Scores must be integers from 0 to 4."
        )
    else:
        system_prompt = format_prompt("concept_importance.system")

    cf_context = ""
    if task.counterfactual_concept:
        cf_context = format_prompt(
            "concept_importance.counterfactual_context",
            counterfactual_concept=task.counterfactual_concept,
            counterfactual_value=task.counterfactual_value,
        )

    user_prompt = format_prompt(
        "concept_importance.user",
        original_question=(task.original_question or "").strip(),
        counterfactual_context=cf_context,
        explanation=task.explanation,
        concept_names=task.concept_names,
        concept_value_menu=task.concept_value_menu,
    )
    if strict_json_prompt:
        user_prompt += (
            "\n\nIMPORTANT OUTPUT CONTRACT:\n"
            "- Return only JSON.\n"
            "- Do not write any reasoning or commentary.\n"
            "- Include every concept exactly once.\n"
            f"- Required concept keys: {json.dumps(task.concept_names, ensure_ascii=False)}\n"
            f"- Output skeleton to fill with scores: {build_importance_skeleton(task.concept_names)}\n"
            "Return the JSON object now."
        )
    if qwen_no_think:
        # Qwen3-style models can otherwise spend the completion budget on a
        # hidden/visible reasoning preamble. This text-only scorer needs the
        # final JSON directly.
        user_prompt += "\n\n/no_think"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def render_messages(messages: List[Dict[str, str]]) -> str:
    rendered = []
    for message in messages:
        role = message.get("role", "user").upper()
        rendered.append(f"{role}:\n{message.get('content', '')}")
    rendered.append("ASSISTANT:\n")
    return "\n\n".join(rendered)


def clean_importance_response(raw: str, concept_names: List[str]) -> Dict[str, int]:
    raw = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL | re.IGNORECASE).strip()
    parsed = extract_json_from_text(raw)
    importance = parsed.get("importance", {}) if isinstance(parsed, dict) else {}
    if not isinstance(importance, dict):
        raise ValueError("Model JSON did not contain an importance object.")

    cleaned = {}
    for concept in concept_names:
        raw_score = importance.get(concept, 0)
        try:
            score = int(raw_score)
        except Exception:
            score = 0
        cleaned[concept] = max(0, min(4, score))
    return cleaned


def zero_importance(concept_names: List[str]) -> Dict[str, int]:
    return {concept: 0 for concept in concept_names}


# ===============================
# Task construction and output
# ===============================

def build_tasks(explanations_data: dict, concepts_data: dict, csv_map: dict) -> List[ImportanceTask]:
    tasks = []

    for title, expl_block in explanations_data.items():
        case_concepts = concepts_data.get(title)
        if not isinstance(case_concepts, dict):
            print(f"Skipping '{title}': no matching concept block.")
            continue

        concept_names = extract_concept_names(case_concepts)
        if not concept_names:
            print(f"Skipping '{title}': no concepts.")
            continue

        concept_value_menu = build_concept_value_menu(case_concepts)
        row = csv_map.get(normalize_title_key(title), {})
        original_question = row.get("Question_Text", "")
        gold = row.get("Analysis", "")

        if isinstance(gold, str) and gold.strip():
            tasks.append(
                ImportanceTask(
                    title=title,
                    kind="original",
                    explanation=gold,
                    concept_names=concept_names,
                    concept_value_menu=concept_value_menu,
                    original_question=original_question,
                )
            )

        if not isinstance(expl_block, dict):
            continue

        for concept, cf_list in expl_block.get("counterfactuals", {}).items():
            if not isinstance(cf_list, list):
                continue
            for item in cf_list:
                if not isinstance(item, dict):
                    continue
                explanation = item.get("explanation")
                if not isinstance(explanation, str):
                    continue
                value = item.get("value")
                tasks.append(
                    ImportanceTask(
                        title=title,
                        kind="counterfactual",
                        explanation=explanation,
                        concept_names=concept_names,
                        concept_value_menu=concept_value_menu,
                        original_question=original_question,
                        counterfactual_concept=concept,
                        counterfactual_value=value if isinstance(value, str) else str(value),
                    )
                )

    return tasks


def apply_task_result(output: dict, task: ImportanceTask, importance: Dict[str, int]):
    entry = output.setdefault(task.title, {})
    if task.kind == "original":
        entry["original"] = importance
        entry.setdefault("counterfactuals", {})
        return

    cf_out = entry.setdefault("counterfactuals", {})
    concept_out = cf_out.setdefault(task.counterfactual_concept, [])
    concept_out.append(
        {
            "value": task.counterfactual_value,
            "importance": importance,
        }
    )


def save_output(output: dict, output_path: str):
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)


def process_explanations_offline(
    explanations_json_path: str,
    concepts_json_path: str,
    csv_path: str,
    output_json_path: str,
    client: OfflineVLLMChatClient,
    batch_size: int,
    retries: int,
    save_every: int,
    strict_json_prompt: bool,
    qwen_no_think: bool,
):
    with open(explanations_json_path, "r", encoding="utf-8") as fh:
        explanations_data = json.load(fh)

    with open(concepts_json_path, "r", encoding="utf-8") as fh:
        concepts_data = json.load(fh)

    csv_map = load_rows_by_title(csv_path)
    tasks = build_tasks(explanations_data, concepts_data, csv_map)
    print(f"Built {len(tasks)} importance scoring prompts.")

    output = {}
    pending = tasks
    completed = 0
    failed = 0
    since_save = 0

    progress = None
    if tqdm is not None:
        progress = tqdm(total=len(tasks), desc="Importance scored", unit="prompt")

    try:
        while pending:
            batch = pending[:batch_size]
            pending = pending[batch_size:]
            raw_outputs = client.generate_batch(
                [build_messages(task, strict_json_prompt=strict_json_prompt, qwen_no_think=qwen_no_think) for task in batch]
            )

            retry_tasks = []
            finished_this_batch = 0
            for task, raw in zip(batch, raw_outputs):
                try:
                    importance = clean_importance_response(raw, task.concept_names)
                    apply_task_result(output, task, importance)
                    completed += 1
                    since_save += 1
                    finished_this_batch += 1
                except Exception:
                    preview = (raw[:1000] + "...") if len(raw) > 1000 else raw
                    print(
                        f"Importance parse error for '{task.title}' / {task.kind} "
                        f"attempt {task.attempt}/{retries + 1}. Output preview:\n{preview}"
                    )
                    if task.attempt <= retries:
                        task.attempt += 1
                        retry_tasks.append(task)
                    else:
                        apply_task_result(output, task, zero_importance(task.concept_names))
                        failed += 1
                        since_save += 1
                        finished_this_batch += 1

            if progress is not None:
                progress.update(finished_this_batch)
                progress.set_postfix(completed=completed, failed=failed, retrying=len(retry_tasks), remaining=len(pending))
            else:
                print(f"Progress: completed={completed}, failed={failed}, remaining={len(pending)}")

            pending.extend(retry_tasks)

            if save_every > 0 and since_save >= save_every:
                save_output(output, output_json_path)
                since_save = 0
    finally:
        if progress is not None:
            progress.close()

    save_output(output, output_json_path)
    print("\nWritten concept importance scores ->", output_json_path)
    print(f"Task summary: completed={completed}, failed={failed}")


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--explanations", required=True)
    parser.add_argument(
        "--concepts",
        required=True,
    )
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=os.environ.get("IMPORTANCE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--tokenizer-mode", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--limit-mm-per-prompt", default='{"image": 0}')
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--gdn-prefill-backend",
        default=None,
        choices=["auto", "flashinfer", "cutedsl", "triton"],
        help="vLLM GDN prefill backend for Qwen3.6-style hybrid models. Use 'triton' if flashinfer is not installed.",
    )
    parser.add_argument(
        "--extra-llm-kwargs",
        default=None,
        help=(
            "Optional JSON object merged into vLLM LLM kwargs. Useful for model-specific "
            "options supported by your installed vLLM version."
        ),
    )
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument(
        "--strict-json-prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use a stricter JSON-only prompt. Enabled by default because Qwen tends to explain before JSON.",
    )
    parser.add_argument(
        "--qwen-no-think",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append /no_think to suppress Qwen reasoning preambles. Enabled by default.",
    )
    parser.add_argument(
        "--guided-json",
        action="store_true",
        help="Request vLLM guided JSON decoding if supported by the installed vLLM version.",
    )
    args = parser.parse_args()

    client = OfflineVLLMChatClient(
        model=args.model,
        tokenizer=args.tokenizer,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        tokenizer_mode=args.tokenizer_mode,
        trust_remote_code=args.trust_remote_code,
        limit_mm_per_prompt=parse_json_arg(args.limit_mm_per_prompt, "--limit-mm-per-prompt"),
        enforce_eager=args.enforce_eager,
        gdn_prefill_backend=args.gdn_prefill_backend,
        extra_llm_kwargs=parse_json_arg(args.extra_llm_kwargs, "--extra-llm-kwargs"),
        guided_json_schema=build_importance_json_schema() if args.guided_json else None,
    )

    process_explanations_offline(
        explanations_json_path=args.explanations,
        concepts_json_path=args.concepts,
        csv_path=args.csv,
        output_json_path=args.output,
        client=client,
        batch_size=args.batch_size,
        retries=args.retries,
        save_every=args.save_every,
        strict_json_prompt=args.strict_json_prompt,
        qwen_no_think=args.qwen_no_think,
    )
