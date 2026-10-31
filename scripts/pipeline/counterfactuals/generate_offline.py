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
import re
from dataclasses import dataclass
from datetime import datetime
from json import JSONDecodeError, JSONDecoder
from pathlib import Path
from typing import Dict, List, Optional


def _add_src_to_path():
    import sys

    for parent in Path(__file__).resolve().parents:
        if (parent / "utils" / "prompt_loader.py").is_file():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return


_add_src_to_path()
from utils.prompt_loader import format_prompt


DEFAULT_MODEL = "mistralai/Magistral-Small-2509"

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


@dataclass
class CounterfactualTask:
    title: str
    concept: str
    original_value: str
    question: str
    user_text: str
    attempt: int = 1


class OfflineVLLMChatClient:
    def __init__(
        self,
        model: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        max_model_len: Optional[int],
        dtype: str,
    ):
        from vllm import LLM, SamplingParams

        llm_kwargs = {
            "model": model,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "dtype": dtype,
        }
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len

        print(f"Loading offline vLLM model: {model}")
        print(f"vLLM init kwargs: {llm_kwargs}")
        self.llm = LLM(**llm_kwargs)
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
        )

    def generate_batch(self, batch_messages: List[List[Dict[str, str]]]) -> List[str]:
        # vLLM exposes an offline chat API for chat-style conversations. If a
        # local vLLM version lacks chat(), fall back to rendering prompts and
        # using generate(). Both paths stay fully offline and do not use a server.
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


# ===============================
# CSV/context helpers
# ===============================

def normalize_title_key(value: str) -> str:
    return " ".join(value.strip().split()).lower()


def normalize_field_name(name: str) -> str:
    return "".join(ch for ch in (name or "").strip().lower() if ch.isalnum())


def get_csv_field(row: dict, *names: str, default: str = "") -> str:
    lookup = {normalize_field_name(key): key for key in row.keys()}
    for name in names:
        key = lookup.get(normalize_field_name(name))
        if key is not None:
            return row.get(key, default)
    return default


def load_context_by_title_from_csv(csv_path: str) -> dict:
    context_map = {}
    csv_file = Path(csv_path)

    with csv_file.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, skipinitialspace=True)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {csv_file}")

        for row in reader:
            title = (get_csv_field(row, "Question_Title", default="") or "").strip()
            question = (get_csv_field(row, "Question_Text", "Question_text", default="") or "").strip()
            if not title:
                continue

            context_map[normalize_title_key(title)] = {
                "question": question,
            }

    return context_map


# ===============================
# JSON extraction and validation
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


def _extract_json_from_text(text: str):
    text = strip_markdown_fences(text).strip()

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
    if start == -1:
        raise JSONDecodeError("No JSON object found", text, 0)

    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except JSONDecodeError:
                    break

    match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except JSONDecodeError:
            pass

    raise JSONDecodeError("Could not extract JSON from model output", text, 0)


def clean_counterfactual_response(raw: str, concept: str, min_count: int, max_count: int):
    parsed = _extract_json_from_text(raw)
    if (
        not isinstance(parsed, dict)
        or "new_values" not in parsed
        or not isinstance(parsed["new_values"], list)
    ):
        raise ValueError(f"Unexpected JSON structure for '{concept}': {parsed}")

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
        raise ValueError(f"Generated {len(clean)} alternatives for '{concept}' but need >= {min_count}.")
    return clean


# ===============================
# Prompt construction
# ===============================

def build_system_prompt(question: str, min_count: int, max_count: int) -> str:
    return format_prompt(
        "counterfactual_generation.system",
        question=question,
        min_count=min_count,
        max_count=max_count,
    )


def build_user_prompt(concept: str, original_value: str, min_count: int, max_count: int) -> str:
    return format_prompt(
        "counterfactual_generation.user",
        concept=concept,
        original_value=original_value,
        min_count=min_count,
        max_count=max_count,
    )


def build_messages(task: CounterfactualTask, min_count: int, max_count: int) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": build_system_prompt(task.question, min_count, max_count),
        },
        {
            "role": "user",
            "content": task.user_text,
        },
    ]


def render_messages(messages: List[Dict[str, str]]) -> str:
    rendered = []
    for message in messages:
        role = message.get("role", "user").upper()
        rendered.append(f"{role}:\n{message.get('content', '')}")
    rendered.append("ASSISTANT:\n")
    return "\n\n".join(rendered)


def retry_user_text(task: CounterfactualTask, reason: str, min_count: int) -> str:
    if "json" in reason.lower() or "structure" in reason.lower():
        return (
            task.user_text
            + "\n\nRetry guidance:"
            + "\n- Return valid JSON only."
            + "\n- Do not add prose outside the JSON."
            + "\n- Keep each value short."
            + "\n- Keep each counterfactual close to the original question."
        )

    return (
        task.user_text
        + "\n\nRetry guidance:"
        + "\n- Your previous answer did not produce enough valid alternatives."
        + "\n- Do not invent new facts."
        + "\n- Stay closer to the original wording."
        + "\n- Produce alternatives only if they can be implemented coherently with local edits."
        + f"\n- Return at least {min_count} valid entries."
    )


# ===============================
# Batch runner
# ===============================

def build_tasks(data: dict, context_map: dict, titles: List[str], min_count: int, max_count: int) -> List[CounterfactualTask]:
    tasks = []
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

            tasks.append(
                CounterfactualTask(
                    title=title,
                    concept=concept,
                    original_value=orig_val_str,
                    question=question,
                    user_text=build_user_prompt(concept, orig_val_str, min_count, max_count),
                )
            )
    return tasks


def process_first_n_examples(
    input_json_path: str,
    output_json_path: str,
    context_map: dict,
    client: OfflineVLLMChatClient,
    n: int = 10,
    batch_size: int = 8,
    min_count: int = 1,
    max_count: int = 4,
    retries: int = 2,
):
    with open(input_json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, dict) or len(data) == 0:
        print("Input JSON is empty or not a mapping.")
        return

    titles = list(data.keys()) if n is None or n < 0 else list(data.keys())[:n]
    out_payload = {
        title: {k: v for k, v in data[title].items()}
        for title in titles
        if isinstance(data.get(title), dict)
    }
    for title in out_payload:
        out_payload[title]["COUNTERFACTUALS"] = {}

    pending = build_tasks(data, context_map, titles, min_count, max_count)
    print(f"Built {len(pending)} counterfactual prompts for {len(titles)} examples.")

    completed = 0
    failed = 0
    progress = None
    if tqdm is not None:
        progress = tqdm(total=len(pending), desc="Concepts finished", unit="concept")

    try:
        while pending:
            batch = pending[:batch_size]
            pending = pending[batch_size:]
            messages_batch = [build_messages(task, min_count, max_count) for task in batch]
            raw_outputs = client.generate_batch(messages_batch)

            retry_tasks = []
            finished_this_batch = 0
            for task, raw in zip(batch, raw_outputs):
                try:
                    cleaned = clean_counterfactual_response(raw, task.concept, min_count, max_count)
                    out_payload[task.title]["COUNTERFACTUALS"][task.concept] = cleaned
                    completed += 1
                    finished_this_batch += 1
                except (JSONDecodeError, ValueError) as ex:
                    preview = (raw[:1000] + "...") if len(raw) > 1000 else raw
                    print(
                        f"Counterfactual parse/validation error for '{task.title}' / '{task.concept}' "
                        f"attempt {task.attempt}/{retries + 1}: {ex}\nOutput preview:\n{preview}"
                    )
                    if task.attempt <= retries:
                        task.attempt += 1
                        task.user_text = retry_user_text(task, str(ex), min_count)
                        retry_tasks.append(task)
                    else:
                        failed += 1
                        finished_this_batch += 1

            if progress is not None:
                progress.update(finished_this_batch)
                progress.set_postfix(completed=completed, failed=failed, retrying=len(retry_tasks), remaining=len(pending))
            else:
                print(f"Progress: completed={completed}, failed={failed}, remaining={len(pending)}")

            pending.extend(retry_tasks)
    finally:
        if progress is not None:
            progress.close()

    with open(output_json_path, "w", encoding="utf-8") as out_fh:
        json.dump(out_payload, out_fh, indent=2, ensure_ascii=False)

    print(f"Wrote counterfactuals for {len(out_payload)} examples to {output_json_path}")
    print(f"Task summary: completed={completed}, failed={failed}")


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
    parser.add_argument("--model", default=os.environ.get("MAGISTRAL_MODEL", DEFAULT_MODEL))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--max-count", type=int, default=4)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    context_map = load_context_by_title_from_csv(args.csv)

    client = OfflineVLLMChatClient(
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
    )

    process_first_n_examples(
        input_json_path=args.input,
        output_json_path=str(output_path),
        context_map=context_map,
        client=client,
        n=args.max_examples,
        batch_size=args.batch_size,
        min_count=args.min_count,
        max_count=args.max_count,
        retries=args.retries,
    )
