"""Diagnose how answer-scoring debiasing changes original-question predictions.

This script intentionally ignores counterfactual questions. It uses the original
questions from a counterfactual JSON, reproduces the explanation prompt,
teacher-forces each answer label, and compares the argmax before and after direct
null-prompt calibration with the original answer in an explanation JSON.
"""

import argparse
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _load_offline_module():
    path = Path(__file__).with_name("log_likelihood_offline.py")
    spec = importlib.util.spec_from_file_location("offline_loglikelihood", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load offline scorer from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


offline = _load_offline_module()


def normalize_answer(value) -> Optional[str]:
    if value is None:
        return None
    for char in str(value).strip().upper():
        if char.isalpha():
            return char
    return None


def is_included(value: str) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def load_csv_rows(path: str) -> Tuple[Dict[str, dict], int]:
    rows = {}
    excluded = 0
    with open(path, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file, skipinitialspace=True)
        required = {"Question_Title", "Question_Text", "Possible_Answers", "Included"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

        for row in reader:
            title = (row.get("Question_Title") or "").strip()
            if not title:
                continue
            if not is_included(row.get("Included", "")):
                excluded += 1
                continue
            rows[offline.normalize_title_key(title)] = {
                "title": title,
                "question": (row.get("Question_Text") or "").strip(),
                "intro": (row.get("Intro_Text") or "").strip(),
                "possible_answers": row.get("Possible_Answers") or "",
                "correct_answer": normalize_answer(row.get("Correct_Answer")),
            }
    return rows, excluded


def stable_softmax(scores: Dict[str, Optional[float]]) -> Dict[str, float]:
    finite = {key: value for key, value in scores.items() if value is not None and math.isfinite(value)}
    if not finite:
        return {key: 0.0 for key in scores}
    maximum = max(finite.values())
    exps = {
        key: math.exp(value - maximum) if key in finite else 0.0
        for key, value in scores.items()
    }
    total = sum(exps.values())
    return {key: value / total for key, value in exps.items()}


def make_stage(
    scores: Dict[str, Optional[float]],
    reference_answer: Optional[str],
    previous_answer: Optional[str],
) -> dict:
    probabilities = stable_softmax(scores)
    finite_labels = [
        label for label, score in scores.items()
        if score is not None and math.isfinite(score)
    ]
    answer = max(finite_labels, key=lambda label: scores[label]) if finite_labels else None
    return {
        # These are option-level logits (sequence scores), not individual
        # vocabulary-token logits. Softmaxing them gives the values below.
        "logits": scores,
        "probabilities": probabilities,
        "answer": answer,
        "matches_explanation_answer": (
            answer == reference_answer if answer is not None and reference_answer is not None else None
        ),
        "changed_from_previous_stage": (
            answer != previous_answer if previous_answer is not None and answer is not None else None
        ),
    }


def build_answer_messages(
    question: str,
    options: List[Tuple[str, str]],
    intro: str,
    full_answer: bool = False,
) -> List[dict]:
    """Reproduce the messages used by the explanation-generation script."""
    options_text = "\n".join(f"{label}. {text}" for label, text in options)
    system_prompt = offline.format_prompt(
        "answer_explanation_full_answer.system" if full_answer else "answer_explanation.system"
    )
    if (intro or "").strip():
        user_prompt = offline.format_prompt(
            "answer_explanation.user_with_intro",
            intro_text=intro.strip(),
            question=question,
            options_text=options_text,
        )
    else:
        user_prompt = offline.format_prompt(
            "answer_explanation.user",
            question=question,
            options_text=options_text,
        )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def messages_to_token_ids(client, messages: List[dict]) -> List[int]:
    tokenized = client.tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    if isinstance(tokenized, dict):
        tokenized = tokenized.get("input_ids", [])
    return offline._flatten_token_ids(tokenized)


def score_candidate_continuations(
    client,
    messages: List[dict],
    assistant_prefix: str,
    candidates: Dict[str, str],
) -> Tuple[Dict[str, Optional[float]], Dict[str, int]]:
    message_ids = messages_to_token_ids(client, messages)
    assistant_prefix_ids = client.encode_text(assistant_prefix)
    requests = []
    prefix_lengths = []
    labels = list(candidates)
    for label, candidate in candidates.items():
        # Tokenize the prefix and answer together so leading whitespace and
        # tokenizer merges at their character boundary are represented exactly.
        assistant_full_ids = client.encode_text(assistant_prefix + candidate)
        common_length = 0
        for prefix_token, full_token in zip(assistant_prefix_ids, assistant_full_ids):
            if prefix_token != full_token:
                break
            common_length += 1
        requests.append({"prompt_token_ids": message_ids + assistant_full_ids})
        prefix_lengths.append(len(message_ids) + common_length)

    scores = {}
    token_counts = {}
    for start in range(0, len(requests), client.batch_size):
        batch = requests[start : start + client.batch_size]
        batch_prefix_lengths = prefix_lengths[start : start + client.batch_size]
        outputs = client.llm.generate(
            batch,
            sampling_params=client.scoring_params,
            use_tqdm=False,
        )
        batch_labels = labels[start : start + client.batch_size]
        for label, output, prefix_length in zip(batch_labels, outputs, batch_prefix_lengths):
            token_ids = list(getattr(output, "prompt_token_ids", []) or [])
            prompt_logprobs = getattr(output, "prompt_logprobs", None)
            continuation_length = len(token_ids) - prefix_length
            if not token_ids or prompt_logprobs is None or continuation_length <= 0:
                scores[label] = None
                continue
            total = 0.0
            valid = True
            for position in range(prefix_length, len(token_ids)):
                logprob = offline._lookup_prompt_logprob(prompt_logprobs[position], token_ids[position])
                if logprob is None:
                    valid = False
                    break
                total += float(logprob)
            scores[label] = total if valid else None
            token_counts[label] = continuation_length
    return scores, token_counts


def score_original_question(
    client,
    question: str,
    options: List[Tuple[str, str]],
    intro: str,
    reference_answer: Optional[str],
    calibration_lambda: float,
    temperature: float,
    full_answer: bool,
) -> dict:
    labels = [label for label, _ in options]
    candidates = {
        label: (text if full_answer else label)
        for label, text in options
    }
    matched_messages = build_answer_messages(question, options, intro, full_answer=full_answer)
    null_matched_messages = build_answer_messages("N/A", options, intro, full_answer=full_answer)

    matched_raw_scores, matched_token_counts = score_candidate_continuations(
        client,
        matched_messages,
        assistant_prefix='{ "answer": "',
        candidates=candidates,
    )
    null_raw_scores, null_token_counts = score_candidate_continuations(
        client,
        null_matched_messages,
        assistant_prefix='{ "answer": "',
        candidates=candidates,
    )
    # A label is normally one token. Full answers vary substantially in length,
    # so compare their mean token log-probabilities instead of favoring the
    # shortest option through raw sequence-log-likelihood sums.
    matched_scores = {
        label: (
            matched_raw_scores[label] / matched_token_counts[label]
            if full_answer and matched_raw_scores.get(label) is not None
            else matched_raw_scores.get(label)
        )
        for label in labels
    }
    null_scores = {
        label: (
            null_raw_scores[label] / null_token_counts[label]
            if full_answer and null_raw_scores.get(label) is not None
            else null_raw_scores.get(label)
        )
        for label in labels
    }

    calibrated_scores = {
        label: (
            matched_scores[label] - calibration_lambda * null_scores[label]
            if matched_scores.get(label) is not None and null_scores.get(label) is not None
            else None
        )
        for label in labels
    }
    safe_temperature = max(float(temperature), 1e-8)
    temperature_logits = {
        label: (
            calibrated_scores[label] / safe_temperature
            if calibrated_scores[label] is not None
            else None
        )
        for label in labels
    }

    stages = {}
    previous = None
    for name, scores in (
        ("matched_prompt_label_logits", matched_scores),
        ("null_calibrated_matched_prompt_label_logits", calibrated_scores),
        ("temperature_scaled_label_logits", temperature_logits),
    ):
        stages[name] = make_stage(scores, reference_answer, previous)
        previous = stages[name]["answer"]

    transitions = {}
    stage_names = list(stages)
    for before, after in zip(stage_names, stage_names[1:]):
        before_answer = stages[before]["answer"]
        after_answer = stages[after]["answer"]
        transitions[f"{before}_to_{after}"] = {
            "before": before_answer,
            "after": after_answer,
            "changed": (
                before_answer != after_answer
                if before_answer is not None and after_answer is not None
                else None
            ),
        }

    return {
        "continuations": candidates,
        "score_normalization": "mean_logprob_per_token" if full_answer else "sum_logprob",
        "raw_sequence_log_likelihoods": {
            "matched_prompt": matched_raw_scores,
            "null_matched_prompt": null_raw_scores,
        },
        "label_token_counts": {
            "matched_prompt": matched_token_counts,
            "null_matched_prompt": null_token_counts,
        },
        "null_label_logits": null_scores,
        "stages": stages,
        "transitions": transitions,
        "initial_answer": stages["matched_prompt_label_logits"]["answer"],
        "final_answer": stages["temperature_scaled_label_logits"]["answer"],
        "initial_misalignment": stages["matched_prompt_label_logits"]["matches_explanation_answer"] is False,
        "final_misalignment": stages["temperature_scaled_label_logits"]["matches_explanation_answer"] is False,
    }


def print_question_report(title: str, result: dict):
    print(f"\n{'=' * 88}\n{title}\n{'=' * 88}")
    print(
        f"Explanation JSON answer: {result['explanation_json']['answer']} | "
        f"CSV correct answer: {result['csv']['correct_answer']}"
    )
    print("\nScoring stages:")
    for name, stage in result["stages"].items():
        score_text = ", ".join(
            f"{label}={score:.6f}" if score is not None else f"{label}=None"
            for label, score in stage["logits"].items()
        )
        print(
            f"  {name:<25} answer={stage['answer']} "
            f"changed={stage['changed_from_previous_stage']} "
            f"matches_explanation={stage['matches_explanation_answer']}"
        )
        print(f"    logits: {score_text}")
    print(
        f"Final misalignment: {result['final_misalignment']} "
        f"({result['final_answer']} vs {result['explanation_json']['answer']})"
    )


def build_summary(questions: Dict[str, dict], input_counts: dict) -> dict:
    successful = [result for result in questions.values() if result.get("status") == "ok"]
    failed_details = {
        title: result.get("error", "Unknown error")
        for title, result in questions.items()
        if result.get("status") != "ok"
    }
    stage_names = (
        "matched_prompt_label_logits",
        "null_calibrated_matched_prompt_label_logits",
        "temperature_scaled_label_logits",
    )
    stage_summary = {}
    for name in stage_names:
        comparable = [
            result["stages"][name]
            for result in successful
            if result["stages"][name]["matches_explanation_answer"] is not None
        ]
        matches = sum(stage["matches_explanation_answer"] for stage in comparable)
        stage_summary[name] = {
            "comparable_questions": len(comparable),
            "matches_explanation_answer": matches,
            "misalignments": len(comparable) - matches,
        }

    transition_counts = {}
    for result in successful:
        for name, transition in result["transitions"].items():
            entry = transition_counts.setdefault(name, {"comparable_questions": 0, "changed": 0})
            if transition["changed"] is not None:
                entry["comparable_questions"] += 1
                entry["changed"] += int(transition["changed"])

    return {
        **input_counts,
        "successful_questions": len(successful),
        "failed_questions": len(failed_details),
        "failed_question_details": failed_details,
        "stage_alignment": stage_summary,
        "stage_answer_changes": transition_counts,
    }


def save_report(path: str, metadata: dict, summary: dict, questions: dict):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(
            {"metadata": metadata, "summary": summary, "questions": questions},
            file,
            indent=2,
            ensure_ascii=False,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Debug teacher-forced answer-label changes across matched prompting and debiasing stages."
    )
    parser.add_argument("--counterfactuals", required=True)
    parser.add_argument("--explanations", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=os.environ.get("MAGISTRAL_MODEL", offline.DEFAULT_MODEL))
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--prepare-merged-model", action="store_true")
    parser.add_argument("--base-model-for-serving", default=offline.DEFAULT_MODEL)
    parser.add_argument("--prepared-model-dir", default=None)
    parser.add_argument("--copy-prepared-model-files", action="store_true")
    parser.add_argument("--allow-serving-metadata-download", action="store_true")
    parser.add_argument("--calibration-lambda", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--full-answer",
        action="store_true",
        help="Teacher-force and score each complete answer-option text instead of its letter.",
    )
    parser.add_argument(
        "--explanation-max-tokens",
        type=int,
        default=1024,
        help="Deprecated compatibility option; rationales are no longer generated.",
    )
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
    parser.add_argument("--max-questions", type=int, default=None)
    args = parser.parse_args()

    with open(args.counterfactuals, encoding="utf-8") as file:
        counterfactuals = json.load(file)
    with open(args.explanations, encoding="utf-8") as file:
        explanations = json.load(file)
    csv_rows, csv_excluded = load_csv_rows(args.csv)

    cf_by_normalized = {offline.normalize_title_key(title): (title, item) for title, item in counterfactuals.items()}
    exp_by_normalized = {offline.normalize_title_key(title): item for title, item in explanations.items()}
    shared = sorted(set(csv_rows) & set(cf_by_normalized) & set(exp_by_normalized))
    if args.max_questions is not None:
        shared = shared[: max(args.max_questions, 0)]
    if not shared:
        raise SystemExit("No included questions are shared by the CSV and both JSON files.")

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

    # Match the offline explanation generator unless the launch environment
    # explicitly requests another vLLM engine version.
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
        limit_mm_per_prompt=offline.parse_json_arg(args.limit_mm_per_prompt, "--limit-mm-per-prompt"),
        extra_llm_kwargs=offline.parse_json_arg(args.extra_llm_kwargs, "--extra-llm-kwargs"),
    )

    input_counts = {
        "counterfactual_json_questions": len(counterfactuals),
        "explanation_json_questions": len(explanations),
        "included_csv_questions": len(csv_rows),
        "csv_questions_excluded_by_Included": csv_excluded,
        "shared_questions_selected": len(shared),
        "included_csv_missing_from_counterfactual_json": len(set(csv_rows) - set(cf_by_normalized)),
        "included_csv_missing_from_explanation_json": len(set(csv_rows) - set(exp_by_normalized)),
    }
    metadata_out = {
        "counterfactuals": args.counterfactuals,
        "explanations": args.explanations,
        "csv": args.csv,
        "model": model_path,
        "calibration_lambda": args.calibration_lambda,
        "temperature": args.temperature,
        "continuation_type": "full_answer_text" if args.full_answer else "answer_label",
        "score_normalization": "mean_logprob_per_token" if args.full_answer else "sum_logprob",
        "rationale_generation_enabled": False,
        "counterfactual_questions_scored": False,
        "stage_order": [
            "matched_prompt_label_logits",
            "null_calibrated_matched_prompt_label_logits",
            "temperature_scaled_label_logits",
        ],
    }

    questions_out = {}
    for index, normalized_title in enumerate(shared, start=1):
        csv_row = csv_rows[normalized_title]
        title, cf_item = cf_by_normalized[normalized_title]
        explanation_item = exp_by_normalized[normalized_title] or {}
        original_reference = explanation_item.get("original") or {}
        reference_answer = normalize_answer(original_reference.get("answer"))
        options = offline.parse_possible_answers(csv_row["possible_answers"])
        question = (
            (cf_item or {}).get("ORIGINAL QUESTION")
            or (cf_item or {}).get("Question_text")
            or (cf_item or {}).get("Question_Text")
            or csv_row["question"]
        )
        intro = (cf_item or {}).get("Intro_Text") or csv_row["intro"]

        base = {
            "status": "ok",
            "csv": {
                "correct_answer": csv_row["correct_answer"],
                "question": question,
                "intro": intro,
                "options": {label: text for label, text in options},
            },
            "explanation_json": {
                "answer": reference_answer,
                "explanation": original_reference.get("explanation"),
            },
        }
        try:
            if not question:
                raise ValueError("Original question text is empty")
            if not options:
                raise ValueError("Possible_Answers could not be parsed")
            if reference_answer is None:
                raise ValueError("Explanation JSON original answer is missing or invalid")
            diagnostic = score_original_question(
                client=client,
                question=question,
                options=options,
                intro=intro,
                reference_answer=reference_answer,
                calibration_lambda=args.calibration_lambda,
                temperature=args.temperature,
                full_answer=args.full_answer,
            )
            base.update(diagnostic)
            questions_out[title] = base
            print_question_report(title, base)
        except Exception as error:
            base["status"] = "error"
            base["error"] = f"{type(error).__name__}: {error}"
            base["failure_context"] = {
                "has_question_text": bool(question),
                "parsed_option_count": len(options),
                "parsed_option_labels": [label for label, _ in options],
                "explanation_answer_raw": original_reference.get("answer"),
                "explanation_answer_normalized": reference_answer,
            }
            questions_out[title] = base
            print(f"\n[{index}/{len(shared)}] {title}: ERROR: {base['error']}")

        if args.save_every > 0 and index % args.save_every == 0:
            summary = build_summary(questions_out, input_counts)
            save_report(args.output, metadata_out, summary, questions_out)

    summary = build_summary(questions_out, input_counts)
    save_report(args.output, metadata_out, summary, questions_out)
    print(f"\n{'=' * 88}\nFINAL SUMMARY\n{'=' * 88}")
    print(json.dumps(summary, indent=2))
    if summary["failed_question_details"]:
        print(f"\n{'=' * 88}\nFAILED QUESTIONS\n{'=' * 88}")
        for title, reason in summary["failed_question_details"].items():
            print(f"- {title}: {reason}")
    print(f"Report written to: {args.output}")


if __name__ == "__main__":
    main()
