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
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


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
class ExplanationTask:
    title: str
    kind: str
    question: str
    options: List[Tuple[str, str]]
    intro_text: str
    concept: Optional[str] = None
    value: Optional[str] = None
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
        lora_path: Optional[str],
        lora_name: str,
        lora_id: int,
        max_lora_rank: int,
        tokenizer_mode: str,
        trust_remote_code: bool,
        limit_mm_per_prompt: Optional[dict],
    ):
        # vLLM V1 is currently brittle for this text-only Magistral/Pixtral
        # merged-model path. Set before importing/initializing vLLM unless the
        # caller explicitly exported a different value.
        os.environ.setdefault("VLLM_USE_V1", "0")

        from vllm import LLM, SamplingParams

        llm_kwargs = {
            "model": model,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "dtype": dtype,
            "tokenizer_mode": tokenizer_mode,
            "trust_remote_code": trust_remote_code,
        }
        if limit_mm_per_prompt is not None:
            llm_kwargs["limit_mm_per_prompt"] = limit_mm_per_prompt
        if tokenizer:
            llm_kwargs["tokenizer"] = tokenizer
        self.lora_request = None
        if lora_path:
            from vllm.lora.request import LoRARequest

            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_lora_rank"] = max_lora_rank
            self.lora_request = LoRARequest(lora_name, lora_id, lora_path)

        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len

        print(f"Loading offline vLLM model: {model}")
        if tokenizer:
            print(f"Using tokenizer: {tokenizer}")
        if lora_path:
            print(f"Using LoRA adapter: name={lora_name}, id={lora_id}, path={lora_path}, max_lora_rank={max_lora_rank}")
        print(f"vLLM init kwargs: {llm_kwargs}")
        self.llm = LLM(**llm_kwargs)
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
        )

    def generate_batch(self, batch_messages: List[List[Dict[str, str]]]) -> List[str]:
        if hasattr(self.llm, "chat"):
            outputs = self.llm.chat(
                batch_messages,
                sampling_params=self.sampling_params,
                lora_request=self.lora_request,
                use_tqdm=False,
            )
        else:
            prompts = [render_messages(messages) for messages in batch_messages]
            outputs = self.llm.generate(
                prompts,
                sampling_params=self.sampling_params,
                lora_request=self.lora_request,
                use_tqdm=False,
            )

        return [output.outputs[0].text.strip() if output.outputs else "" for output in outputs]


# ===============================
# JSON and CSV helpers
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


def try_parse_json(text: str):
    text = strip_markdown_fences(text).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass

    return None


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
            title = (get_csv_field(row, "Question_Title", default="") or "").strip()
            if not title:
                continue

            csv_map[normalize_title_key(title)] = {
                "Question_Text": (get_csv_field(row, "Question_Text", "Question_text", default="") or "").strip(),
                "Intro_Text": (get_csv_field(row, "Intro_Text", default="") or "").strip(),
                "Possible_Answers": get_csv_field(row, "Possible_Answers", default="") or "",
            }

    return csv_map


def parse_possible_answers(field):
    field = (field or "").strip().lstrip("[").rstrip("]")
    parts = re.split(r"\s*'(?=[A-Z]\.)", field)

    options = []
    for part in parts:
        part = part.strip().rstrip(",").rstrip("'").strip()
        if not part:
            continue
        match = re.match(r"([A-Z])\.\s*(.*)", part)
        if match:
            options.append((match.group(1), match.group(2).strip()))

    return options


# ===============================
# Prompt and parsing
# ===============================

def build_messages(task: ExplanationTask) -> List[Dict[str, str]]:
    options_text = "\n".join([f"{letter}. {text}" for letter, text in task.options])
    system_prompt = format_prompt("answer_explanation.system")

    if task.intro_text:
        user_prompt = format_prompt(
            "answer_explanation.user_with_intro",
            intro_text=task.intro_text,
            question=task.question,
            options_text=options_text,
        )
    else:
        user_prompt = format_prompt(
            "answer_explanation.user",
            question=task.question,
            options_text=options_text,
        )

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


def parse_answer_explanation(raw: str):
    parsed = try_parse_json(raw)
    if parsed and isinstance(parsed, dict) and "answer" in parsed and "explanation" in parsed:
        return {
            "answer": str(parsed["answer"]).strip(),
            "explanation": str(parsed["explanation"]).strip(),
        }
    return None


# ===============================
# Task construction and runner
# ===============================

def build_tasks(data: dict, csv_map: dict) -> List[ExplanationTask]:
    tasks = []
    for title, case in data.items():
        row = csv_map.get(normalize_title_key(title))
        if not row:
            print(f"Skipping '{title}': no matching CSV row.")
            continue

        options = parse_possible_answers(row.get("Possible_Answers", ""))
        intro_text = row.get("Intro_Text", "")
        original_question = row.get("Question_Text", "")
        if not original_question:
            print(f"Skipping '{title}': empty Question_Text.")
            continue

        tasks.append(
            ExplanationTask(
                title=title,
                kind="original",
                question=original_question,
                options=options,
                intro_text=intro_text,
            )
        )

        if not isinstance(case, dict):
            continue

        for concept, cf_list in case.get("COUNTERFACTUALS", {}).items():
            if not isinstance(cf_list, list):
                continue
            for cf in cf_list:
                if not isinstance(cf, dict):
                    continue
                question = cf.get("counterfactual", "")
                value = cf.get("value", "")
                if not question:
                    continue
                tasks.append(
                    ExplanationTask(
                        title=title,
                        kind="counterfactual",
                        question=question,
                        options=options,
                        intro_text=intro_text,
                        concept=concept,
                        value=value,
                    )
                )

    return tasks


def apply_task_result(out: dict, task: ExplanationTask, parsed: dict):
    title_result = out.setdefault(task.title, {})
    if task.kind == "original":
        title_result["original"] = parsed
        return

    cf_results = title_result.setdefault("counterfactuals", {})
    concept_results = cf_results.setdefault(task.concept, [])
    concept_results.append(
        {
            "value": task.value,
            **parsed,
        }
    )


def save_progress(out: dict, out_path: str):
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)


# ===============================
# vLLM merged-model preparation
# ===============================

SERVING_METADATA_FILES = [
    # Magistral/Mistral3 models need more than just config + weights when
    # loaded from a local merged directory. We symlink local merged weights
    # and copy only missing metadata from the original base snapshot.
    "params.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "chat_template.json",
    "tokenizer.json",
    "tokenizer.model",
    "tekken.json",
]


DEFAULT_PIXTRAL_PREPROCESSOR_CONFIG = {
    "crop_size": None,
    "data_format": "channels_first",
    "default_to_square": True,
    "device": None,
    "disable_grouping": None,
    "do_center_crop": None,
    "do_convert_rgb": True,
    "do_normalize": True,
    "do_rescale": True,
    "do_resize": True,
    "image_mean": [0.48145466, 0.4578275, 0.40821073],
    "image_processor_type": "PixtralImageProcessor",
    "image_std": [0.26862954, 0.26130258, 0.27577711],
    "input_data_format": None,
    "patch_size": {"height": 14, "width": 14},
    "processor_class": "PixtralProcessor",
    "resample": 3,
    "rescale_factor": 0.00392156862745098,
    "return_tensors": None,
    "size": {"longest_edge": 1540},
}


DEFAULT_PIXTRAL_PROCESSOR_CONFIG = {
    "image_break_token": "[IMG_BREAK]",
    "image_end_token": "[IMG_END]",
    "image_token": "[IMG]",
    "patch_size": 14,
    "processor_class": "PixtralProcessor",
}


def _safe_link_or_copy(src: Path, dst: Path, copy_files: bool):
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_files:
        shutil.copy2(src, dst)
    else:
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)


def _copy_missing_base_metadata(base_model: str, prepared_dir: Path, allow_download: bool):
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Preparing a merged model needs huggingface_hub so the script can "
            "locate the base model metadata files."
        ) from exc

    base_snapshot = Path(snapshot_download(base_model, local_files_only=not allow_download))
    copied = []
    for filename in SERVING_METADATA_FILES:
        src = base_snapshot / filename
        dst = prepared_dir / filename
        if src.is_file() and not dst.exists():
            shutil.copy2(src, dst)
            copied.append(filename)
    if copied:
        print(f"Copied missing base serving metadata: {', '.join(copied)}")


def _write_json_if_missing(path: Path, data: dict):
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")
    return True


def _ensure_pixtral_processor_metadata(prepared_dir: Path):
    created = []
    if _write_json_if_missing(prepared_dir / "preprocessor_config.json", DEFAULT_PIXTRAL_PREPROCESSOR_CONFIG):
        created.append("preprocessor_config.json")
    if _write_json_if_missing(prepared_dir / "processor_config.json", DEFAULT_PIXTRAL_PROCESSOR_CONFIG):
        created.append("processor_config.json")
    if created:
        print(
            "Created fallback Pixtral processor metadata: "
            + ", ".join(created)
            + ". This is needed because vLLM loads Magistral through a Pixtral/Mistral3 processor path."
        )


def prepare_merged_model_for_vllm(
    merged_model: str,
    base_model: str,
    prepared_dir: Optional[str],
    output_path: str,
    copy_model_files: bool,
    allow_download: bool,
) -> str:
    merged_dir = Path(merged_model).expanduser().resolve()
    if not merged_dir.is_dir():
        raise ValueError(f"--model must be a local merged model directory when --prepare-merged-model is used: {merged_model}")

    if prepared_dir:
        target = Path(prepared_dir).expanduser().resolve()
    else:
        output_stem = Path(output_path).expanduser().resolve()
        target = output_stem.parent / f"{output_stem.stem}_vllm_merged_model"

    target.mkdir(parents=True, exist_ok=True)

    for src in merged_dir.iterdir():
        if src.is_file():
            _safe_link_or_copy(src, target / src.name, copy_model_files)

    _copy_missing_base_metadata(base_model, target, allow_download)
    _ensure_pixtral_processor_metadata(target)

    print(f"Prepared vLLM merged model directory: {target}")
    print("This directory uses the merged weights plus missing tokenizer/processor metadata from the base model.")
    return str(target)


def parse_json_arg(value: Optional[str], arg_name: str):
    if value is None or value == "":
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{arg_name} must be valid JSON. Got: {value}") from exc


def generate_explanations_offline(
    cf_json: str,
    csv_path: str,
    out_path: str,
    client: OfflineVLLMChatClient,
    batch_size: int,
    retries: int,
    save_every: int,
):
    with open(cf_json, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    csv_map = load_rows_by_title(csv_path)
    tasks = build_tasks(data, csv_map)
    print(f"Built {len(tasks)} explanation prompts.")

    out = {}
    pending = tasks
    completed = 0
    failed = 0
    since_save = 0

    progress = None
    if tqdm is not None:
        progress = tqdm(total=len(tasks), desc="Explanations finished", unit="prompt")

    try:
        while pending:
            batch = pending[:batch_size]
            pending = pending[batch_size:]
            raw_outputs = client.generate_batch([build_messages(task) for task in batch])

            retry_tasks = []
            finished_this_batch = 0
            for task, raw in zip(batch, raw_outputs):
                parsed = parse_answer_explanation(raw)
                if parsed:
                    apply_task_result(out, task, parsed)
                    completed += 1
                    since_save += 1
                    finished_this_batch += 1
                else:
                    preview = (raw[:1000] + "...") if len(raw) > 1000 else raw
                    print(
                        f"Explanation parse error for '{task.title}' / {task.kind} "
                        f"attempt {task.attempt}/{retries + 1}. Output preview:\n{preview}"
                    )
                    if task.attempt <= retries:
                        task.attempt += 1
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

            if save_every > 0 and since_save >= save_every:
                save_progress(out, out_path)
                since_save = 0
    finally:
        if progress is not None:
            progress.close()

    save_progress(out, out_path)
    print("\nWritten explanations →", out_path)
    print(f"Task summary: completed={completed}, failed={failed}")


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
    parser.add_argument("--model", default=os.environ.get("MAGISTRAL_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Optional tokenizer path/name passed to vLLM. Useful when serving a merged local model.",
    )
    parser.add_argument(
        "--tokenizer-mode",
        default="auto",
        help="vLLM tokenizer mode. Use 'mistral' for Magistral if auto selects the wrong tokenizer.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--prepare-merged-model",
        action="store_true",
        help=(
            "Create a lightweight vLLM-ready directory for a local merged model by "
            "symlinking merged files and copying missing serving metadata from the base model."
        ),
    )
    parser.add_argument(
        "--base-model-for-serving",
        default=DEFAULT_MODEL,
        help="Base model used only to copy missing serving metadata when --prepare-merged-model is set.",
    )
    parser.add_argument(
        "--prepared-model-dir",
        default=None,
        help="Directory for the prepared vLLM merged model. Defaults next to --output.",
    )
    parser.add_argument(
        "--copy-prepared-model-files",
        action="store_true",
        help="Copy merged files instead of symlinking them into --prepared-model-dir.",
    )
    parser.add_argument(
        "--allow-serving-metadata-download",
        action="store_true",
        help="Allow downloading missing base serving metadata if it is not already in the HF cache.",
    )
    parser.add_argument(
        "--limit-mm-per-prompt",
        default='{"image": 0}',
        help=(
            "JSON passed to vLLM limit_mm_per_prompt. Default disables image inputs "
            "because this script is text-only and Magistral is loaded through a multimodal processor path."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--lora-path", default=None, help="Optional LoRA adapter checkpoint path for offline vLLM.")
    parser.add_argument("--lora-name", default="magistral-lora", help="LoRA adapter name passed to vLLM.")
    parser.add_argument("--lora-id", type=int, default=1, help="Numeric LoRA adapter id passed to vLLM.")
    parser.add_argument("--max-lora-rank", type=int, default=64, help="Maximum LoRA rank for vLLM.")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=25)
    args = parser.parse_args()

    model_path = args.model
    if args.prepare_merged_model:
        model_path = prepare_merged_model_for_vllm(
            merged_model=args.model,
            base_model=args.base_model_for_serving,
            prepared_dir=args.prepared_model_dir,
            output_path=args.output,
            copy_model_files=args.copy_prepared_model_files,
            allow_download=args.allow_serving_metadata_download,
        )
    limit_mm_per_prompt = parse_json_arg(args.limit_mm_per_prompt, "--limit-mm-per-prompt")

    client = OfflineVLLMChatClient(
        model=model_path,
        tokenizer=args.tokenizer,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        lora_path=args.lora_path,
        lora_name=args.lora_name,
        lora_id=args.lora_id,
        max_lora_rank=args.max_lora_rank,
        tokenizer_mode=args.tokenizer_mode,
        trust_remote_code=args.trust_remote_code,
        limit_mm_per_prompt=limit_mm_per_prompt,
    )

    generate_explanations_offline(
        cf_json=args.input,
        csv_path=args.csv,
        out_path=args.output,
        client=client,
        batch_size=args.batch_size,
        retries=args.retries,
        save_every=args.save_every,
    )
