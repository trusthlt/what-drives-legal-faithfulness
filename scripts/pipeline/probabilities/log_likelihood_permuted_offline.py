import atexit
import argparse
import csv
import itertools
import json
import math
import os
import random
import re
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _register_runtime_timer():
    script_start = time.perf_counter()

    def _print_runtime():
        elapsed = time.perf_counter() - script_start
        print(f"\nElapsed time: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")

    atexit.register(_print_runtime)


if __name__ == "__main__":
    _register_runtime_timer()


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


class OfflineVLLMLogLikelihoodClient:
    def __init__(
        self,
        model: str,
        tokenizer: Optional[str],
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        max_model_len: Optional[int],
        dtype: str,
        prompt_logprobs: int,
        batch_size: int,
        trust_remote_code: bool,
        limit_mm_per_prompt: Optional[dict],
        extra_llm_kwargs: Optional[dict],
    ):
        from vllm import LLM, SamplingParams

        llm_kwargs = {
            "model": model,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "dtype": dtype,
            "trust_remote_code": trust_remote_code,
        }
        if tokenizer:
            llm_kwargs["tokenizer"] = tokenizer
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len
        if limit_mm_per_prompt is not None:
            llm_kwargs["limit_mm_per_prompt"] = limit_mm_per_prompt
        if extra_llm_kwargs:
            llm_kwargs.update(extra_llm_kwargs)

        print(f"Loading offline vLLM model: {model}")
        if tokenizer:
            print(f"Using tokenizer: {tokenizer}")
        print(f"vLLM init kwargs: {llm_kwargs}")
        self.llm = LLM(**llm_kwargs)
        self.tokenizer = self._get_tokenizer()
        self.batch_size = batch_size
        self.prompt_logprobs = prompt_logprobs
        self.scoring_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            prompt_logprobs=prompt_logprobs,
        )

    def _get_tokenizer(self):
        if hasattr(self.llm, "get_tokenizer"):
            return self.llm.get_tokenizer()
        if hasattr(self.llm, "llm_engine") and hasattr(self.llm.llm_engine, "tokenizer"):
            return self.llm.llm_engine.tokenizer
        raise RuntimeError("Could not access the vLLM tokenizer.")

    def chat_prompt_token_ids(self, prompt: str) -> List[int]:
        messages = [{"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            tokenized = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            if isinstance(tokenized, dict):
                tokenized = tokenized.get("input_ids", [])
            return _flatten_token_ids(tokenized)

        fallback = f"USER:\n{prompt}\n\nASSISTANT:\n"
        return self.encode_text(fallback)

    def encode_text(self, text: str) -> List[int]:
        if hasattr(self.tokenizer, "encode"):
            try:
                return list(self.tokenizer.encode(text, add_special_tokens=False))
            except TypeError:
                return list(self.tokenizer.encode(text))
        encoded = self.tokenizer(text)
        return _flatten_token_ids(encoded["input_ids"])

    def sequence_logprobs_batch(self, prompt_continuations: List[Tuple[str, str]]) -> List[Tuple[float, int]]:
        full_token_prompts = []
        prompt_lens = []
        for prompt, continuation in prompt_continuations:
            prompt_ids = self.chat_prompt_token_ids(prompt)
            continuation_ids = self.encode_text(continuation)
            full_token_prompts.append({"prompt_token_ids": prompt_ids + continuation_ids})
            prompt_lens.append(len(prompt_ids))

        results = []
        for start in range(0, len(full_token_prompts), self.batch_size):
            batch_prompts = full_token_prompts[start : start + self.batch_size]
            batch_prompt_lens = prompt_lens[start : start + self.batch_size]
            outputs = self.llm.generate(batch_prompts, sampling_params=self.scoring_params, use_tqdm=False)

            for output, prompt_len in zip(outputs, batch_prompt_lens):
                token_ids = list(getattr(output, "prompt_token_ids", []) or [])
                prompt_logprobs = getattr(output, "prompt_logprobs", None)
                if not token_ids or prompt_logprobs is None:
                    results.append((float("-inf"), 0))
                    continue

                full_len = len(token_ids)
                cont_len = full_len - prompt_len
                if cont_len <= 0:
                    results.append((float("-inf"), 0))
                    continue

                total = 0.0
                ok = True
                for pos in range(prompt_len, full_len):
                    token_id = token_ids[pos]
                    logprob = _lookup_prompt_logprob(prompt_logprobs[pos], token_id)
                    if logprob is None:
                        ok = False
                        break
                    total += float(logprob)

                results.append((total if ok else float("-inf"), cont_len))

        return results


def _lookup_prompt_logprob(entry, token_id: int) -> Optional[float]:
    if entry is None:
        return None
    if isinstance(entry, dict):
        item = entry.get(token_id)
        if item is None:
            item = entry.get(str(token_id))
        if item is None and len(entry) == 1:
            item = next(iter(entry.values()))
        return _extract_logprob(item)
    return _extract_logprob(entry)


def _flatten_token_ids(tokenized) -> List[int]:
    if hasattr(tokenized, "tolist"):
        tokenized = tokenized.tolist()
    if tokenized and isinstance(tokenized[0], list):
        if len(tokenized) != 1:
            raise ValueError(f"Expected one tokenized prompt, got {len(tokenized)}")
        tokenized = tokenized[0]
    return [int(token_id) for token_id in tokenized]


def _extract_logprob(item) -> Optional[float]:
    if item is None:
        return None
    if isinstance(item, (int, float)):
        return float(item)
    if isinstance(item, dict):
        value = item.get("logprob")
        return float(value) if value is not None else None
    value = getattr(item, "logprob", None)
    return float(value) if value is not None else None


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


def load_rows_by_title(input_csv_path: str):
    csv_map = {}
    with open(input_csv_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile, skipinitialspace=True)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {input_csv_path}")

        for row in reader:
            title = (get_csv_field(row, "Question_Title", "Title", default="") or "").strip()
            if not title:
                continue
            csv_map[normalize_title_key(title)] = {
                "Question_Text": (get_csv_field(row, "Question_Text", "Question_text", default="") or "").strip(),
                "Intro_Text": (get_csv_field(row, "Intro_Text", default="") or "").strip(),
                "Possible_Answers": get_csv_field(row, "Possible_Answers", default="") or "",
            }
    return csv_map


def parse_possible_answers(field: str) -> List[Tuple[str, str]]:
    if not field:
        return []

    field = field.strip().lstrip("[").rstrip("]")
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


def _build_semantic_mc_prompt(question: str, options: List[Tuple[str, str]], intro_text: str = "") -> str:
    intro_text = (intro_text or "").strip()
    options_text = "\n".join(f"{label}. {text}" for label, text in options)

    if intro_text:
        return format_prompt(
            "semantic_mc_answer.user_with_intro",
            intro_text=intro_text,
            question=question,
            options_text=options_text,
        )

    return format_prompt(
        "semantic_mc_answer.user",
        question=question,
        options_text=options_text,
    )


def _build_null_prompt(options: List[Tuple[str, str]], intro_text: str = "") -> str:
    intro_text = (intro_text or "").strip()
    options_text = "\n".join(f"{label}. {text}" for label, text in options)

    if intro_text:
        return format_prompt(
            "semantic_mc_null.user_with_intro",
            intro_text=intro_text,
            options_text=options_text,
        )

    return format_prompt("semantic_mc_null.user", options_text=options_text)


def _get_permutations(options: List[Tuple[str, str]], n_permutations: int, seed: int = 0) -> List[List[Tuple[str, str]]]:
    if not options:
        return []

    all_perms = list(itertools.permutations(options))
    if n_permutations >= len(all_perms):
        return [list(perm) for perm in all_perms]

    rng = random.Random(seed)
    indexes = list(range(len(all_perms)))
    rng.shuffle(indexes)
    return [list(all_perms[index]) for index in indexes[:n_permutations]]


def offline_option_distribution(
    client: OfflineVLLMLogLikelihoodClient,
    question: str,
    options: List[Tuple[str, str]],
    intro_text: str = "",
    n_permutations: int = 8,
    calibration_lambda: float = 1.0,
    temperature: float = 1.0,
    seed: int = 0,
) -> Dict[str, float]:
    if not question or not options:
        return {}

    original_labels = [label for label, _ in options]
    original_label_to_text = {label: text for label, text in options}
    score_sums = {label: 0.0 for label in original_labels}
    score_counts = {label: 0 for label in original_labels}

    scoring_items = []
    item_meta = []
    perms = _get_permutations(options, n_permutations=n_permutations, seed=seed)

    for perm in perms:
        prompt = _build_semantic_mc_prompt(question=question, options=perm, intro_text=intro_text)
        null_prompt = _build_null_prompt(options=perm, intro_text=intro_text)

        for original_label in original_labels:
            continuation = " " + original_label_to_text[original_label]
            scoring_items.append((prompt, continuation))
            item_meta.append(("raw", original_label))
            scoring_items.append((null_prompt, continuation))
            item_meta.append(("null", original_label))

    scored = client.sequence_logprobs_batch(scoring_items)

    pending_raw = {}
    for (kind, label), (logprob, token_count) in zip(item_meta, scored):
        if token_count <= 0 or not math.isfinite(logprob):
            continue
        score = logprob / token_count
        if kind == "raw":
            pending_raw[label] = score
        else:
            raw_score = pending_raw.pop(label, None)
            if raw_score is None:
                continue
            debiased_score = raw_score - calibration_lambda * score
            score_sums[label] += debiased_score
            score_counts[label] += 1

    avg_scores = {
        label: (score_sums[label] / score_counts[label] if score_counts[label] > 0 else float("-inf"))
        for label in original_labels
    }
    finite_scores = [score for score in avg_scores.values() if math.isfinite(score)]
    if not finite_scores:
        return {label: 0.0 for label in original_labels}

    max_score = max(finite_scores)
    safe_temp = max(float(temperature), 1e-8)
    exps = {
        label: (math.exp((score - max_score) / safe_temp) if math.isfinite(score) else 0.0)
        for label, score in avg_scores.items()
    }
    total = sum(exps.values())
    if total <= 0:
        return {label: 0.0 for label in original_labels}

    return {label: exps[label] / total for label in original_labels}


def enrich_with_logprobs(
    input_json_path: str,
    input_csv_path: str,
    output_json_path: str,
    client: OfflineVLLMLogLikelihoodClient,
    n_permutations: int,
    calibration_lambda: float,
    temperature: float,
    seed: int,
    save_every: int,
):
    with open(input_json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    csv_map = load_rows_by_title(input_csv_path)
    out_payload = {}
    processed = 0

    progress = tqdm(total=len(data), desc="Questions scored", unit="question") if tqdm is not None else None
    try:
        for title, item in data.items():
            print(f"\nProcessing -> {title}")

            if not isinstance(item, dict):
                out_payload[title] = item
                continue

            row = csv_map.get(normalize_title_key(title), {})
            options = parse_possible_answers(row.get("Possible_Answers", ""))
            intro = item.get("Intro_Text") or row.get("Intro_Text", "") or ""
            result = dict(item)

            original_question = (
                item.get("ORIGINAL QUESTION")
                or item.get("Question_text")
                or item.get("Question_Text")
                or row.get("Question_Text", "")
            )
            if original_question and options:
                result["ORIGINAL_PROBABILITIES"] = offline_option_distribution(
                    client=client,
                    question=original_question,
                    options=options,
                    intro_text=intro,
                    n_permutations=n_permutations,
                    calibration_lambda=calibration_lambda,
                    temperature=temperature,
                    seed=seed,
                )
            else:
                result["ORIGINAL_PROBABILITIES"] = {}

            new_counterfactuals = {}
            for concept, cf_list in item.get("COUNTERFACTUALS", {}).items():
                new_counterfactuals[concept] = []
                for cf in cf_list:
                    entry = dict(cf)
                    counterfactual_question = cf.get("counterfactual")
                    if counterfactual_question and options:
                        entry["probability_distribution"] = offline_option_distribution(
                            client=client,
                            question=counterfactual_question,
                            options=options,
                            intro_text=intro,
                            n_permutations=n_permutations,
                            calibration_lambda=calibration_lambda,
                            temperature=temperature,
                            seed=seed,
                        )
                    else:
                        entry["probability_distribution"] = {}
                    new_counterfactuals[concept].append(entry)

            result["COUNTERFACTUALS"] = new_counterfactuals
            out_payload[title] = result
            processed += 1

            if progress is not None:
                progress.update(1)
            if save_every > 0 and processed % save_every == 0:
                save_json(out_payload, output_json_path)
    finally:
        if progress is not None:
            progress.close()

    save_json(out_payload, output_json_path)
    print("\nDone - offline vLLM calibrated probabilities saved")
    print("Wrote:", output_json_path)



# ===============================
# vLLM merged-model preparation
# ===============================

SERVING_METADATA_FILES = [
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
    with path.open("w", encoding="utf-8") as fh:
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

def save_json(payload: dict, output_json_path: str):
    output_path = Path(output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as outfh:
        json.dump(payload, outfh, indent=2, ensure_ascii=False)


def parse_json_arg(value: Optional[str], arg_name: str):
    if value is None or value == "":
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{arg_name} must be valid JSON. Got: {value}") from exc


def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
    )
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=os.environ.get("MAGISTRAL_MODEL", DEFAULT_MODEL))
    parser.add_argument("--tokenizer", default=None)
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
    parser.add_argument("--n-permutations", type=int, default=8)
    parser.add_argument("--calibration-lambda", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--prompt-logprobs", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--limit-mm-per-prompt", default=None)
    parser.add_argument("--extra-llm-kwargs", default=None)
    parser.add_argument("--save-every", type=int, default=5)
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

    client = OfflineVLLMLogLikelihoodClient(
        model=model_path,
        tokenizer=args.tokenizer,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        prompt_logprobs=args.prompt_logprobs,
        batch_size=args.batch_size,
        trust_remote_code=args.trust_remote_code,
        limit_mm_per_prompt=parse_json_arg(args.limit_mm_per_prompt, "--limit-mm-per-prompt"),
        extra_llm_kwargs=parse_json_arg(args.extra_llm_kwargs, "--extra-llm-kwargs"),
    )

    enrich_with_logprobs(
        input_json_path=args.input,
        input_csv_path=args.csv,
        output_json_path=args.output,
        client=client,
        n_permutations=args.n_permutations,
        calibration_lambda=args.calibration_lambda,
        temperature=args.temperature,
        seed=args.seed,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
