import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer, Mistral3ForConditionalGeneration


def merge_lora_adapter(
    base_model: str,
    adapter_path: str,
    output_dir: str,
    dtype: str,
    device_map: str | None,
    safe_serialization: bool,
    max_shard_size: str,
):
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype.lower())
    if torch_dtype is None:
        raise ValueError(f"Unsupported dtype: {dtype}. Use bfloat16, float16, or float32.")

    adapter = Path(adapter_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not adapter.is_dir():
        raise FileNotFoundError(f"Adapter directory not found: {adapter}")
    if not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Missing adapter_config.json in adapter directory: {adapter}")

    print(f"Base model: {base_model}")
    print(f"Adapter path: {adapter}")
    print(f"Output dir: {output}")
    print(f"dtype: {torch_dtype}")
    print(f"device_map: {device_map}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, tokenizer_type="mistral")
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs = {"torch_dtype": torch_dtype}
    if device_map and device_map.lower() != "none":
        model_kwargs["device_map"] = device_map

    base = Mistral3ForConditionalGeneration.from_pretrained(base_model, **model_kwargs)
    model = PeftModel.from_pretrained(base, str(adapter))
    print("Merging LoRA adapter into base model...")
    merged = model.merge_and_unload()

    output.mkdir(parents=True, exist_ok=True)
    print("Saving merged model...")
    merged.save_pretrained(
        str(output),
        safe_serialization=safe_serialization,
        max_shard_size=max_shard_size,
    )
    tokenizer.save_pretrained(str(output))
    print(f"Merged model written to: {output}")


def main():
    parser = argparse.ArgumentParser(description="Merge a no-think Magistral LoRA adapter into the base model.")
    parser.add_argument("--base-model", default="mistralai/Magistral-Small-2509")
    parser.add_argument(
        "--adapter-path",
        required=True,
        help="Path to the PEFT LoRA adapter directory or checkpoint directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="magistral_r64_42_no_think_merged",
        help="Directory where the merged standalone model will be saved.",
    )
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"])
    parser.add_argument("--device-map", default="auto", help="Use 'none' to disable device_map.")
    parser.add_argument("--no-safe-serialization", action="store_true", help="Save PyTorch .bin files instead of safetensors.")
    parser.add_argument(
        "--max-shard-size",
        default="200GB",
        help="Passed to save_pretrained. A large value avoids a Transformers sharding bug for vision_tower keys.",
    )
    args = parser.parse_args()

    merge_lora_adapter(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        output_dir=args.output_dir,
        dtype=args.dtype,
        device_map=args.device_map,
        safe_serialization=not args.no_safe_serialization,
        max_shard_size=args.max_shard_size,
    )


if __name__ == "__main__":
    main()
