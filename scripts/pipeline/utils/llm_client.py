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

import json
import os
from pathlib import Path


DEFAULT_HF_MODEL = "mistralai/Magistral-Small-2509"
DEFAULT_ENV_PATH = Path(__file__).with_name(".env")
DEFAULT_VLLM_BASE_URL = "http://localhost:8000/v1"


def load_local_env(path=DEFAULT_ENV_PATH):
    if not path.is_file():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and not os.environ.get(key):
            os.environ[key] = value


def add_llm_cli_args(parser):
    parser.add_argument(
        "--vllm",
        action="store_true",
        help="Use a vLLM OpenAI-compatible server instead of loading the HF model locally.",
    )
    parser.add_argument(
        "--vllm-base-url",
        default=None,
        help=f"vLLM OpenAI-compatible base URL. Default: {DEFAULT_VLLM_BASE_URL}",
    )
    parser.add_argument(
        "--vllm-model",
        default=None,
        help="Model name exposed by the vLLM server. Falls back to LLM_MODEL/OPENAI_MODEL.",
    )
    parser.add_argument(
        "--vllm-api-key",
        default=None,
        help="API key for the vLLM server. Default: EMPTY.",
    )


def configure_llm_from_args(args):
    if not getattr(args, "vllm", False):
        return

    os.environ["LLM_PROVIDER"] = "vllm"
    os.environ["LLM_BASE_URL"] = (
        getattr(args, "vllm_base_url", None)
        or os.environ.get("LLM_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or DEFAULT_VLLM_BASE_URL
    )
    os.environ["LLM_API_KEY"] = (
        getattr(args, "vllm_api_key", None)
        or os.environ.get("LLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or "EMPTY"
    )

    model = (
        getattr(args, "vllm_model", None)
        or os.environ.get("LLM_MODEL")
        or os.environ.get("OPENAI_MODEL")
    )
    if model:
        os.environ["LLM_MODEL"] = model


class HuggingFaceChatClient:
    def __init__(self):
        self.tokenizer, self.model, self.info = load_hf_model()

    def generate(self, messages, max_new_tokens=1024, temperature=0.0, do_sample=False, **kwargs):
        import torch

        tokenized = self.tokenizer.apply_chat_template(
            messages,
            return_dict=True,
        )
        input_ids = torch.tensor(
            tokenized["input_ids"], device=self.model.device
        ).unsqueeze(0)
        attention_mask = torch.tensor(
            tokenized["attention_mask"], device=self.model.device
        ).unsqueeze(0)

        generate_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if temperature is not None:
            generate_kwargs["temperature"] = temperature
        for key in ("top_p", "top_k"):
            if key in kwargs:
                generate_kwargs[key] = kwargs[key]

        with torch.inference_mode():
            output = self.model.generate(**generate_kwargs)

        generated = output[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


class OpenAIResponsesClient:
    def __init__(self):
        load_local_env()

        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("LLM_BASE_URL")
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = os.environ.get("OPENAI_MODEL") or os.environ.get("LLM_MODEL") or "gpt-5.4"
        print(f"Using OpenAI responses model: {self.model}")

    def generate(self, messages, max_new_tokens=1024, temperature=0.0, **_kwargs):
        instructions = "\n\n".join(
            message["content"] for message in messages if message.get("role") == "system"
        )
        input_messages = [
            {"role": message.get("role", "user"), "content": message.get("content", "")}
            for message in messages
            if message.get("role") != "system"
        ]
        request = {
            "model": self.model,
            "input": input_messages,
            "max_output_tokens": max_new_tokens,
        }
        if instructions:
            request["instructions"] = instructions
        if temperature is not None and not self.model.startswith("gpt-5"):
            request["temperature"] = temperature

        response = self.client.responses.create(**request)
        return response.output_text.strip()


class OpenAIChatCompletionsClient:
    def __init__(self):
        load_local_env()

        from openai import OpenAI

        api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        base_url = os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = os.environ.get("LLM_MODEL") or os.environ.get("OPENAI_MODEL")
        if not self.model:
            raise EnvironmentError("Set LLM_MODEL for the chat-completions provider.")
        print(f"Using chat-completions model: {self.model}")

    def generate(self, messages, max_new_tokens=1024, temperature=0.0, **_kwargs):
        request = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_new_tokens,
        }
        if temperature is not None:
            request["temperature"] = temperature

        response = self.client.chat.completions.create(**request)
        return response.choices[0].message.content.strip()


class OpenAILogprobClient(OpenAIChatCompletionsClient):
    def first_token_distribution(
        self,
        messages,
        labels,
        temperature=0.0,
        top_logprobs=20,
    ):
        import math

        request = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 1,
            "temperature": temperature,
            "logprobs": True,
            "top_logprobs": top_logprobs,
        }

        response = self.client.chat.completions.create(**request)
        choice = response.choices[0]
        content_logprobs = getattr(choice.logprobs, "content", None) if choice.logprobs else None
        if not content_logprobs:
            raise RuntimeError("vLLM response did not include token logprobs.")

        token_logprobs = {}
        for item in content_logprobs[0].top_logprobs:
            token = getattr(item, "token", "")
            logprob = getattr(item, "logprob", None)
            normalized = token.strip()
            if normalized and logprob is not None:
                token_logprobs[normalized] = max(
                    token_logprobs.get(normalized, float("-inf")),
                    float(logprob),
                )

        scores = {label: token_logprobs.get(label, float("-inf")) for label in labels}
        finite_scores = [score for score in scores.values() if math.isfinite(score)]
        if not finite_scores:
            raise RuntimeError(
                "None of the answer labels appeared in vLLM top logprobs. "
                "Try increasing --vllm-top-logprobs or check the prompt format."
            )

        max_score = max(finite_scores)
        exp_scores = {
            label: (math.exp(score - max_score) if math.isfinite(score) else 0.0)
            for label, score in scores.items()
        }
        total = sum(exp_scores.values())
        if total <= 0:
            return {label: 0.0 for label in labels}

        return {label: exp_scores[label] / total for label in labels}


def get_logprob_client():
    configure_provider = (
        os.environ.get("LLM_PROVIDER")
        or os.environ.get("MODEL_PROVIDER")
        or ""
    ).strip().lower()
    if configure_provider not in {"openai-chat", "chat", "openai-compatible", "vllm", "ollama"}:
        raise ValueError("Logprob client currently requires an OpenAI-compatible/vLLM provider.")
    return OpenAILogprobClient()


def load_hf_model():
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, Mistral3ForConditionalGeneration

    model_id = os.environ.get("MAGISTRAL_MODEL") or os.environ.get("HF_MODEL") or DEFAULT_HF_MODEL
    base_model_id = os.environ.get("MAGISTRAL_BASE_MODEL") or os.environ.get("HF_BASE_MODEL")
    tokenizer_id = os.environ.get("MAGISTRAL_TOKENIZER") or os.environ.get("HF_TOKENIZER")

    model_ref = os.path.expanduser(model_id)
    adapter_config_path = Path(model_ref) / "adapter_config.json"
    is_lora_adapter = bool(base_model_id) or adapter_config_path.is_file()

    if is_lora_adapter:
        adapter_cfg = {}
        if adapter_config_path.is_file():
            with adapter_config_path.open("r", encoding="utf-8") as f:
                adapter_cfg = json.load(f)

        base_model_id = base_model_id or adapter_cfg.get("base_model_name_or_path") or DEFAULT_HF_MODEL
        if not base_model_id:
            raise ValueError(
                "LoRA adapter detected but base model is unknown. "
                "Set MAGISTRAL_BASE_MODEL or HF_BASE_MODEL."
            )

        tokenizer_source = tokenizer_id or base_model_id
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, tokenizer_type="mistral")
        base_model = Mistral3ForConditionalGeneration.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, model_ref).eval()
        info = {"model_path": base_model_id, "adapter_path": model_ref}
    else:
        tokenizer_source = tokenizer_id or model_ref
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, tokenizer_type="mistral")
        model = Mistral3ForConditionalGeneration.from_pretrained(
            model_ref,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        info = {"model_path": model_ref, "adapter_path": None}

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Using model path: {info['model_path']}")
    print(f"Using adapters path: {info['adapter_path']}")
    return tokenizer, model, info


def get_chat_client():
    provider = (
        os.environ.get("LLM_PROVIDER")
        or os.environ.get("MODEL_PROVIDER")
        or "hf"
    ).strip().lower()

    if provider in {"hf", "huggingface", "transformers", "magistral", "local"}:
        return HuggingFaceChatClient()
    if provider in {"openai", "responses"}:
        return OpenAIResponsesClient()
    if provider in {"openai-chat", "chat", "openai-compatible", "vllm", "ollama"}:
        return OpenAIChatCompletionsClient()

    raise ValueError(
        "Unsupported LLM_PROVIDER. Use hf, openai, or openai-compatible."
    )
