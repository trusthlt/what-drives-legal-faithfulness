"""Score original and counterfactual questions with direct null calibration.

The script performs no rationale generation. For each original/counterfactual
question it teacher-forces every answer candidate, subtracts the corresponding
score from a question-free ("N/A") prompt, applies temperature scaling, and
softmaxes the resulting option-level logits.
"""

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _load_sibling(filename: str, module_name: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


debug = _load_sibling(
    "inspect_scoring.py",
    "debug_original_answer_scoring_offline",
)
offline = debug.offline

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def finite_or_none(value: Optional[float]) -> Optional[float]:
    return float(value) if value is not None and math.isfinite(value) else None


def normalize_sequence_scores(
    raw_scores: Dict[str, Optional[float]],
    token_counts: Dict[str, int],
    full_answer: bool,
) -> Dict[str, Optional[float]]:
    scores = {}
    for label, raw_score in raw_scores.items():
        if raw_score is None or not math.isfinite(raw_score):
            scores[label] = None
        elif full_answer:
            count = token_counts.get(label, 0)
            scores[label] = raw_score / count if count > 0 else None
        else:
            scores[label] = raw_score
    return scores


def score_real_question(
    client,
    question: str,
    options: List[Tuple[str, str]],
    intro: str,
    candidates: Dict[str, str],
    null_raw_scores: Dict[str, Optional[float]],
    null_scores: Dict[str, Optional[float]],
    null_token_counts: Dict[str, int],
    calibration_lambda: float,
    temperature: float,
    full_answer: bool,
) -> dict:
    messages = debug.build_answer_messages(
        question,
        options,
        intro,
        full_answer=full_answer,
    )
    raw_scores, token_counts = debug.score_candidate_continuations(
        client,
        messages,
        assistant_prefix='{ "answer": "',
        candidates=candidates,
    )
    matched_scores = normalize_sequence_scores(raw_scores, token_counts, full_answer)
    calibrated_scores = {
        label: (
            matched_scores[label] - calibration_lambda * null_scores[label]
            if matched_scores.get(label) is not None and null_scores.get(label) is not None
            else None
        )
        for label in candidates
    }
    safe_temperature = max(float(temperature), 1e-8)
    logits = {
        label: (
            calibrated_scores[label] / safe_temperature
            if calibrated_scores.get(label) is not None
            else None
        )
        for label in candidates
    }
    probabilities = debug.stable_softmax(logits)
    finite_labels = [label for label, value in logits.items() if value is not None]
    answer = max(finite_labels, key=lambda label: logits[label]) if finite_labels else None

    return {
        "continuations": candidates,
        "raw_sequence_log_likelihoods": {
            label: finite_or_none(raw_scores.get(label)) for label in candidates
        },
        "matched_logits": {
            label: finite_or_none(matched_scores.get(label)) for label in candidates
        },
        "null_raw_sequence_log_likelihoods": {
            label: finite_or_none(null_raw_scores.get(label)) for label in candidates
        },
        "null_logits": {
            label: finite_or_none(null_scores.get(label)) for label in candidates
        },
        "calibrated_logits": {
            label: finite_or_none(calibrated_scores.get(label)) for label in candidates
        },
        "logits": {label: finite_or_none(logits.get(label)) for label in candidates},
        "probabilities": probabilities,
        "predicted_answer": answer,
        "token_counts": token_counts,
        "null_token_counts": null_token_counts,
    }


def add_original_scores(result: dict, scored: dict):
    result["ORIGINAL_CONTINUATIONS"] = scored["continuations"]
    result["ORIGINAL_RAW_SEQUENCE_LOG_LIKELIHOODS"] = scored["raw_sequence_log_likelihoods"]
    result["ORIGINAL_MATCHED_LOGITS"] = scored["matched_logits"]
    result["ORIGINAL_NULL_RAW_SEQUENCE_LOG_LIKELIHOODS"] = scored[
        "null_raw_sequence_log_likelihoods"
    ]
    result["ORIGINAL_NULL_LOGITS"] = scored["null_logits"]
    result["ORIGINAL_CALIBRATED_LOGITS"] = scored["calibrated_logits"]
    result["ORIGINAL_LOGITS"] = scored["logits"]
    result["ORIGINAL_PROBABILITIES"] = scored["probabilities"]
    result["ORIGINAL_PREDICTED_ANSWER"] = scored["predicted_answer"]
    result["ORIGINAL_TOKEN_COUNTS"] = scored["token_counts"]
    result["ORIGINAL_NULL_TOKEN_COUNTS"] = scored["null_token_counts"]


def add_counterfactual_scores(entry: dict, scored: dict):
    entry["continuations"] = scored["continuations"]
    entry["raw_sequence_log_likelihoods"] = scored["raw_sequence_log_likelihoods"]
    entry["matched_logits"] = scored["matched_logits"]
    entry["null_raw_sequence_log_likelihoods"] = scored[
        "null_raw_sequence_log_likelihoods"
    ]
    entry["null_logits"] = scored["null_logits"]
    entry["calibrated_logits"] = scored["calibrated_logits"]
    entry["logits"] = scored["logits"]
    entry["probability_distribution"] = scored["probabilities"]
    entry["predicted_answer"] = scored["predicted_answer"]
    entry["token_counts"] = scored["token_counts"]
    entry["null_token_counts"] = scored["null_token_counts"]


def save_json(payload: dict, output_path: str):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def enrich(
    data: dict,
    csv_rows: dict,
    client,
    output_path: str,
    calibration_lambda: float,
    temperature: float,
    full_answer: bool,
    save_every: int,
    debug_examples: int,
) -> dict:
    output = {}
    progress = tqdm(total=len(data), desc="Questions scored", unit="question") if tqdm else None
    try:
        for index, (title, item) in enumerate(data.items(), start=1):
            if not isinstance(item, dict):
                output[title] = item
                continue

            result = dict(item)
            row = csv_rows.get(offline.normalize_title_key(title), {})
            options = offline.parse_possible_answers(row.get("Possible_Answers", ""))
            intro = item.get("Intro_Text") or row.get("Intro_Text", "") or ""
            original_question = (
                item.get("ORIGINAL QUESTION")
                or item.get("Question_text")
                or item.get("Question_Text")
                or row.get("Question_Text", "")
            )

            try:
                if not options:
                    raise ValueError("Possible_Answers could not be parsed from the CSV")
                if not original_question:
                    raise ValueError("Original question text is missing")

                candidates = {
                    label: (text if full_answer else label)
                    for label, text in options
                }
                null_messages = debug.build_answer_messages(
                    "N/A",
                    options,
                    intro,
                    full_answer=full_answer,
                )
                null_raw_scores, null_token_counts = debug.score_candidate_continuations(
                    client,
                    null_messages,
                    assistant_prefix='{ "answer": "',
                    candidates=candidates,
                )
                null_scores = normalize_sequence_scores(
                    null_raw_scores,
                    null_token_counts,
                    full_answer,
                )

                def score(question_text: str) -> dict:
                    return score_real_question(
                        client=client,
                        question=question_text,
                        options=options,
                        intro=intro,
                        candidates=candidates,
                        null_raw_scores=null_raw_scores,
                        null_scores=null_scores,
                        null_token_counts=null_token_counts,
                        calibration_lambda=calibration_lambda,
                        temperature=temperature,
                        full_answer=full_answer,
                    )

                add_original_scores(result, score(original_question))
                new_counterfactuals = {}
                for concept, counterfactuals in item.get("COUNTERFACTUALS", {}).items():
                    new_counterfactuals[concept] = []
                    for counterfactual in counterfactuals:
                        entry = dict(counterfactual)
                        question_text = counterfactual.get("counterfactual")
                        if question_text:
                            add_counterfactual_scores(entry, score(question_text))
                        else:
                            entry["scoring_error"] = "Counterfactual question text is missing"
                        new_counterfactuals[concept].append(entry)
                result["COUNTERFACTUALS"] = new_counterfactuals
            except Exception as error:
                result["SCORING_ERROR"] = f"{type(error).__name__}: {error}"
                print(f"\n[{index}/{len(data)}] {title}: {result['SCORING_ERROR']}")

            output[title] = result
            if index <= debug_examples:
                print(f"\n{'=' * 24} DEBUG EXAMPLE {index}/{debug_examples} {'=' * 24}")
                print(json.dumps({title: result}, indent=2, ensure_ascii=False))
                print(f"{'=' * 72}\n")
            if progress:
                progress.update(1)
            if save_every > 0 and index % save_every == 0:
                save_json(output, output_path)
    finally:
        if progress:
            progress.close()

    save_json(output, output_path)
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Score original and counterfactual questions using direct null calibration."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=os.environ.get("MAGISTRAL_MODEL", offline.DEFAULT_MODEL))
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--prepare-merged-model", action="store_true")
    parser.add_argument("--base-model-for-serving", default=offline.DEFAULT_MODEL)
    parser.add_argument("--prepared-model-dir", default=None)
    parser.add_argument("--copy-prepared-model-files", action="store_true")
    parser.add_argument("--allow-serving-metadata-download", action="store_true")
    parser.add_argument("--full-answer", action="store_true")
    parser.add_argument("--calibration-lambda", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
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
    parser.add_argument("--debug-examples", type=int, default=3)
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as file:
        data = json.load(file)
    csv_rows = offline.load_rows_by_title(args.csv)

    model_path = args.model
    if args.prepare_merged_model:
        model_path = offline.prepare_merged_model_for_vllm(
            merged_model=args.model,
            base_model=args.base_model_for_serving,
            prepared_dir=args.prepared_model_dir,
            output_path=args.output,
            copy_model_files=args.copy_prepared_model_files,
            allow_download=args.allow_serving_metadata_download,
        )

    os.environ.setdefault("VLLM_USE_V1", "0")
    client = offline.OfflineVLLMLogLikelihoodClient(
        model=model_path,
        tokenizer=args.tokenizer,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        prompt_logprobs=args.prompt_logprobs,
        batch_size=args.batch_size,
        trust_remote_code=args.trust_remote_code,
        limit_mm_per_prompt=offline.parse_json_arg(
            args.limit_mm_per_prompt,
            "--limit-mm-per-prompt",
        ),
        extra_llm_kwargs=offline.parse_json_arg(
            args.extra_llm_kwargs,
            "--extra-llm-kwargs",
        ),
    )

    print(f"Continuation type: {'full answer text' if args.full_answer else 'answer label'}")
    print(f"Calibration lambda: {args.calibration_lambda}")
    print(f"Temperature: {args.temperature}")
    enrich(
        data=data,
        csv_rows=csv_rows,
        client=client,
        output_path=args.output,
        calibration_lambda=args.calibration_lambda,
        temperature=args.temperature,
        full_answer=args.full_answer,
        save_every=args.save_every,
        debug_examples=args.debug_examples,
    )
    print(f"Done. Report written to: {args.output}")


if __name__ == "__main__":
    main()
