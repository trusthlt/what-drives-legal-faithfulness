import argparse
import ast
import csv
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    Mistral3ForConditionalGeneration,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed as set_transformers_seed,
)
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

THINK_OPEN = "[THINK]"
THINK_CLOSE = "[/THINK]"

# =========================
# PROMPTS (THINK + INTRO ONLY)
# =========================

SYSTEM_PROMPT_THINK_WITH_INTRO_OLD = (
    "You are a legal expert. "
    "You will be given introductory text that may help answer the question. "
    "A user will ask you to solve a task. You should write a complete legal reasoning trace "
    "that applies the relevant law to the facts, explains why the incorrect options are incorrect, "
    "and justifies the correct answer. End the reasoning trace "
    "with 'Final answer: X', where X is the single letter (A, B, C, D, E,...) of the correct answer. "
    "No explanation outside [THINK]...[/THINK]."
    "Do not repeat the same point. End with \"Final answer: X\" before [/THINK]."
    "\n\n"
    "Preserve the style and substance of the provided legal explanation. "
    "Your thinking process must follow the template:\n"
    "[THINK]\n"
    "Your legal reasoning, including the analysis of the correct answer and relevant incorrect options\n"
    "Final answer: X\n"
    "[/THINK]\n"
)

SYSTEM_PROMPT_THINK_WITH_INTRO = (
    "You are an expert in US Civil Procedure.\n\n"

    "Given an introductory legal text, a question, and multiple possible answers, "
    "produce a complete legal reasoning trace that:\n"
    "- identifies the legally correct answer,\n"
    "- explains why the correct answer is correct,\n"
    "- discusses the relevant incorrect options when useful,\n"
    "- avoids repetition and unnecessary restatement.\n\n"

    "Output exactly one reasoning trace enclosed in [THINK] and [/THINK]. "
    "The final line inside the reasoning trace must be:\n"
    "Final answer: X\n"
    "where X is the single correct answer letter."
)


USER_PROMPT_THINK_WITH_INTRO_OLD = (
    "Introductory text:\n{intro}\n\n"
    "Question:\n{question}\n\n"
    "Possible answers:\n{answers}\n\n"
    "Write a complete Glannon-style legal reasoning trace. "
    "Your thinking process must follow the template below: "
    "[THINK] Your legal reasoning, including the analysis of the correct answer and relevant incorrect options. "
    "Final answer: X[/THINK]."
)

USER_PROMPT_THINK_WITH_INTRO = (
    "Introductory text:\n"
    "{intro}\n\n"

    "Question:\n"
    "{question}\n\n"

    "Possible answers:\n"
    "{answers}\n\n"

    "Produce the reasoning trace in the required format."
)

# =========================
# DATA
# =========================

@dataclass
class QAItem:
    intro_text: str
    question_title: str
    question_text: str
    possible_answers: str
    analysis: str
    correct_answer: str


def safe_str(x) -> str:
    return "" if x is None else str(x)


def normalize_field_name(name: str) -> str:
    return "".join(ch for ch in safe_str(name).lower() if ch.isalnum())


def get_csv_field(row: dict, *names: str, default: str = "") -> str:
    lookup = {normalize_field_name(key): key for key in row.keys()}
    for name in names:
        key = lookup.get(normalize_field_name(name))
        if key is not None:
            return row.get(key, default)
    return default


def normalize_answers(raw: str) -> str:
    raw = safe_str(raw).strip()
    if not raw:
        return ""
    if raw.startswith("["):
        try:
            items = ast.literal_eval(raw)
            if isinstance(items, list):
                return "\n".join(str(x).strip() for x in items)
        except Exception:
            pass
    return raw


def normalize_correct_answer(raw: str) -> str:
    raw = safe_str(raw).strip().upper()
    for ch in raw:
        if "A" <= ch <= "Z":
            return ch
    return ""


def normalize_analysis(raw: str) -> str:
    txt = safe_str(raw).strip()
    return txt


def load_csv(csv_path: str) -> List[QAItem]:
    rows: List[QAItem] = []
    skipped = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            correct = normalize_correct_answer(get_csv_field(r, "Correct_Answer", "Correct_answer"))
            if not correct:
                skipped += 1
                continue

            rows.append(
                QAItem(
                    intro_text=safe_str(get_csv_field(r, "Intro_Text")),
                    question_title=safe_str(get_csv_field(r, "Question_Title")),
                    question_text=safe_str(get_csv_field(r, "Question_Text", "Question_text")),
                    possible_answers=normalize_answers(get_csv_field(r, "Possible_Answers")),
                    analysis=safe_str(get_csv_field(r, "Analysis")),
                    correct_answer=correct,
                )
            )

    if skipped:
        print(f"[WARN] Skipped {skipped} rows with empty/invalid Correct_answer.")
    return rows


def build_messages(row: QAItem) -> List[Dict[str, str]]:
    user_prompt = USER_PROMPT_THINK_WITH_INTRO.format(
        intro=safe_str(row.intro_text).strip(),
        question=safe_str(row.question_text).strip(),
        answers=safe_str(row.possible_answers).strip(),
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT_THINK_WITH_INTRO},
        {"role": "user", "content": user_prompt},
    ]


def build_assistant_target(row: QAItem) -> str:
    """
    [THINK]
    analysis

    Final answer: ANSWER
    [/THINK]
    """
    ans = normalize_correct_answer(row.correct_answer)
    if not ans:
        return ""

    reasoning = normalize_analysis(row.analysis)
    return f"{THINK_OPEN}\n{reasoning}\n\nFinal answer: {ans}\n{THINK_CLOSE}"


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    set_transformers_seed(seed)


# =========================
# DATASET
# =========================

class QADataset(Dataset):
    def __init__(
        self,
        rows: List[QAItem],
        tokenizer,
        max_length: int,
    ):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        row = self.rows[idx]

        messages_prompt = build_messages(row)
        assistant_target = build_assistant_target(row)

        if not assistant_target.strip():
            raise ValueError(f"Empty assistant target at idx={idx}")

        messages_full = messages_prompt + [{"role": "assistant", "content": assistant_target}]

        prompt_tok = self.tokenizer.apply_chat_template(
            messages_prompt,
            return_dict=True,
            truncation=True,
            max_length=self.max_length,
        )
        full_tok = self.tokenizer.apply_chat_template(
            messages_full,
            return_dict=True,
            continue_final_message=True,
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = full_tok["input_ids"]
        attention_mask = full_tok["attention_mask"]

        prompt_len = len(prompt_tok["input_ids"])
        labels = [-100] * prompt_len + input_ids[prompt_len:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class QACollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) for x in batch)

        input_ids = []
        attention_mask = []
        labels = []

        for item in batch:
            pad_len = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.tokenizer.pad_token_id] * pad_len)
            attention_mask.append(item["attention_mask"] + [0] * pad_len)
            labels.append(item["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# =========================
# EVAL
# =========================

def extract_letter(text: str) -> str:
    text = safe_str(text).strip()

    matches = re.findall(r"Final answer:\s*([A-J])", text)
    if matches:
        return matches[-1]
    return ""


class TrainAccuracyCallback(TrainerCallback):
    def __init__(self, tokenizer, rows: List[QAItem], max_new_tokens: int):
        self.tokenizer = tokenizer
        self.rows = rows
        self.max_new_tokens = max_new_tokens

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return
        was_training = model.training
        outputs = evaluate_model(model, self.tokenizer, self.rows, self.max_new_tokens)
        if was_training:
            model.train()
        epoch_label = f"{state.epoch:.0f}" if state.epoch is not None else "?"
        print(f"Train accuracy after epoch {epoch_label}: {outputs['accuracy']:.4f} ({outputs['correct']}/{outputs['total']})")


def evaluate_model(model, tokenizer, rows: List[QAItem], max_new_tokens: int) -> Dict[str, float]:
    model.eval()
    correct = 0
    invalid = 0
    total = 0
    outputs = []

    for r in rows:
        messages = build_messages(r)
        tokenized = tokenizer.apply_chat_template(messages, return_dict=True)

        input_ids = torch.tensor(tokenized["input_ids"], device=model.device).unsqueeze(0)
        attention_mask = torch.tensor(tokenized["attention_mask"], device=model.device).unsqueeze(0)

        with torch.inference_mode():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.2,
                no_repeat_ngram_size=8,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        gen = out[0][input_ids.shape[-1]:]
        text = tokenizer.decode(gen, skip_special_tokens=True).strip()
        pred = extract_letter(text)

        outputs.append({
            "question_title": r.question_title,
            "gold": r.correct_answer,
            "pred": pred,
            "raw": text,
        })

        if pred == r.correct_answer:
            correct += 1
        if not pred:
            invalid += 1
        total += 1

    acc = correct / total if total else 0.0
    return {
        "accuracy": acc,
        "total": total,
        "correct": correct,
        "invalid": invalid,
        "outputs": outputs,
    }


def save_eval(outputs: Dict[str, float], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)


def save_pretrain_outputs_txt(outputs: Dict[str, float], out_path: str) -> None:
    lines = []
    for item in outputs["outputs"]:
        lines.append(f"Title: {item['question_title']}")
        lines.append(f"Gold: {item['gold']}")
        lines.append(f"Pred: {item['pred']}")
        lines.append(f"Raw: {item['raw']}")
        lines.append("")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _model_device(model):
    device = getattr(model, "device", None)
    if device is not None:
        return device
    return next(model.parameters()).device


def evaluate_loss(
    model,
    tokenizer,
    rows: List[QAItem],
    max_length: int,
    batch_size: int,
) -> float:
    if not rows:
        return 0.0

    was_training = model.training
    model.eval()
    dataset = QADataset(rows, tokenizer, max_length)
    collator = QACollator(tokenizer)
    loader = DataLoader(dataset, batch_size=max(1, batch_size), shuffle=False, collate_fn=collator)
    device = _model_device(model)

    weighted_loss = 0.0
    total_examples = 0
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            batch_size_actual = int(batch["input_ids"].shape[0])
            weighted_loss += float(outputs.loss.item()) * batch_size_actual
            total_examples += batch_size_actual

    if was_training:
        model.train()
    return weighted_loss / total_examples if total_examples else 0.0


def parameter_counts(model) -> Dict[str, float]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable": trainable,
        "total": total,
        "trainable_percent": (100.0 * trainable / total) if total else 0.0,
    }


def summarize_eval(outputs: Dict[str, float], loss: float | None = None) -> Dict[str, float]:
    summary = {
        "accuracy": outputs.get("accuracy"),
        "total": outputs.get("total"),
        "correct": outputs.get("correct"),
        "invalid": outputs.get("invalid"),
    }
    if loss is not None:
        summary["loss"] = loss
    return summary


def write_run_log(run_log: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(run_log, f, indent=2, ensure_ascii=False)


def json_safe(value):
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(k): json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(v) for v in value]
        return str(value)


def ensure_think_tokens(tokenizer) -> int:
    vocab = tokenizer.get_vocab()
    additional = [tok for tok in (THINK_OPEN, THINK_CLOSE) if tok not in vocab]
    if not additional:
        return 0
    return tokenizer.add_special_tokens({"additional_special_tokens": additional})


# =========================
# MAIN
# =========================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--out-dir", default="magistral_fine_tuned_think")
    parser.add_argument("--model-id", default="mistralai/Magistral-Small-2509")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--held-out",
        action="store_true",
        default=False,
        help=(
            "Train only on --train-csv and evaluate on --test-csv. "
            "By default, train and test CSVs are combined for training/evaluation."
        ),
    )

    parser.add_argument("--do-pre-eval", action="store_true")
    parser.add_argument("--do-train", action="store_true")
    parser.add_argument("--do-post-eval", action="store_true")

    parser.add_argument("--output-model-dir", default="magistral_fine_tuned_think")
    parser.add_argument("--run-log", default=None, help="Path for a JSON summary of arguments, losses, and accuracies.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--logging-steps", type=int, default=20)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="q_proj,gate_proj,v_proj,o_proj,up_proj,down_proj")

    args = parser.parse_args()
    set_global_seed(args.seed)
    run_log_path = args.run_log or os.path.join(args.out_dir, "run_log.json")
    run_log = {
        "args": vars(args),
        "script_parameters": vars(args),
        "datasets": {},
        "pre_train": {},
        "training": {},
        "post_train": {},
    }

    train_rows_primary = load_csv(args.train_csv)
    train_rows_extra = load_csv(args.test_csv)
    if args.held_out:
        train_rows = train_rows_primary
        eval_rows = train_rows_extra
        eval_label = "held-out test data"
        print(
            f"Loaded held-out split: {len(train_rows)} training rows from {args.train_csv}; "
            f"{len(eval_rows)} test rows from {args.test_csv}"
        )
    else:
        train_rows = train_rows_primary + train_rows_extra
        eval_rows = train_rows
        eval_label = "combined training data"
        print(
            f"Loaded training rows: {len(train_rows_primary)} from {args.train_csv} + "
            f"{len(train_rows_extra)} from {args.test_csv} = {len(train_rows)} total"
        )

    run_log["datasets"] = {
        "held_out": args.held_out,
        "train_csv": args.train_csv,
        "test_csv": args.test_csv,
        "train_rows_primary": len(train_rows_primary),
        "train_rows_extra": len(train_rows_extra),
        "train_rows_used_for_training": len(train_rows),
        "eval_rows": len(eval_rows),
        "eval_label": eval_label,
    }

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        tokenizer_type="mistral",
    )
    added_tokens = ensure_think_tokens(tokenizer)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"Using model path: {args.model_id}")
    print("Using adapters path: None")

    model = Mistral3ForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if added_tokens:
        model.resize_token_embeddings(len(tokenizer))

    if args.do_pre_eval:
        pre_eval = evaluate_model(model, tokenizer, eval_rows, args.max_new_tokens)
        pre_eval_loss = evaluate_loss(
            model,
            tokenizer,
            eval_rows,
            args.max_length,
            args.per_device_train_batch_size,
        )
        save_eval(pre_eval, os.path.join(args.out_dir, "eval_pre.json"))
        save_pretrain_outputs_txt(pre_eval, "pre_train_outputs.txt")
        run_log["pre_train"]["eval"] = summarize_eval(pre_eval, pre_eval_loss)
        print(
            f"Pre-train accuracy ({eval_label}): {pre_eval['accuracy']:.4f} "
            f"({pre_eval['correct']}/{pre_eval['total']}), loss: {pre_eval_loss:.4f}"
        )
        write_run_log(run_log, run_log_path)

    if args.do_train:
        train_ds = QADataset(
            train_rows,
            tokenizer,
            args.max_length,
        )
        data_collator = QACollator(tokenizer)

        model.config.use_cache = False
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=[m.strip() for m in args.lora_target_modules.split(",") if m.strip()],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        peft_parameter_counts = parameter_counts(model)

        training_args = TrainingArguments(
            output_dir=args.output_model_dir,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            save_total_limit=2,
            bf16=True,
            fp16=False,
            report_to=[],
            seed=args.seed,
            data_seed=args.seed,
        )
        run_log["training"]["training_arguments"] = json_safe(training_args.to_dict())
        write_run_log(run_log, run_log_path)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            data_collator=data_collator,
            callbacks=[
                TrainAccuracyCallback(tokenizer, eval_rows, args.max_new_tokens)
            ],
        )

        train_result = trainer.train()
        run_log["training"].update({
            "trainer_metrics": train_result.metrics,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "seed": args.seed,
            "lora": {
                "r": args.lora_r,
                "alpha": args.lora_alpha,
                "dropout": args.lora_dropout,
                "target_modules": [m.strip() for m in args.lora_target_modules.split(",") if m.strip()],
            },
            "parameters": peft_parameter_counts,
        })
        trainer.save_model(args.output_model_dir)
        tokenizer.save_pretrained(args.output_model_dir)
        train_eval = evaluate_model(model, tokenizer, train_rows, args.max_new_tokens)
        train_eval_loss = evaluate_loss(
            model,
            tokenizer,
            train_rows,
            args.max_length,
            args.per_device_train_batch_size,
        )
        save_eval(train_eval, os.path.join(args.out_dir, "eval_train_post.json"))
        run_log["post_train"]["train"] = summarize_eval(train_eval, train_eval_loss)
        print(
            f"Post-train accuracy (train set): {train_eval['accuracy']:.4f} "
            f"({train_eval['correct']}/{train_eval['total']}), loss: {train_eval_loss:.4f}"
        )
        write_run_log(run_log, run_log_path)

    if args.do_post_eval:
        base_eval_model = Mistral3ForConditionalGeneration.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        eval_model = PeftModel.from_pretrained(base_eval_model, args.output_model_dir)
        print(f"Using model path: {args.model_id}")
        print(f"Using adapters path: {args.output_model_dir}")
        post_eval = evaluate_model(eval_model, tokenizer, eval_rows, args.max_new_tokens)
        post_eval_loss = evaluate_loss(
            eval_model,
            tokenizer,
            eval_rows,
            args.max_length,
            args.per_device_train_batch_size,
        )
        save_eval(post_eval, os.path.join(args.out_dir, "eval_post.json"))
        run_log["post_train"]["eval"] = summarize_eval(post_eval, post_eval_loss)
        print(
            f"Post-train accuracy ({eval_label}): {post_eval['accuracy']:.4f} "
            f"({post_eval['correct']}/{post_eval['total']}), loss: {post_eval_loss:.4f}"
        )
        write_run_log(run_log, run_log_path)

    write_run_log(run_log, run_log_path)
    print(f"Run log written to {run_log_path}")


if __name__ == "__main__":
    main()
