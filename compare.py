import argparse
import json
import textwrap
from pathlib import Path
from typing import List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate side-by-side outputs from a base model and its LoRA-adapted variant."
    )
    parser.add_argument("--prompt", type=str, help="Single prompt to compare.")
    parser.add_argument(
        "--prompts_file",
        type=str,
        help="Optional path to a text file containing one prompt per line.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="EleutherAI/pythia-2.8b",
        help="Hugging Face model id or local path for the baseline model.",
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default="outputs/pythia-2_8b-lora/resumed_notes/checkpoint-1445",
        help="Path to the LoRA adapter directory to compare against the base model.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for generation. Set to -1 for stochastic runs each time.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "mps", "cpu"],
        default="auto",
        help="Computation device preference. 'auto' picks CUDA > MPS > CPU.",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        help="Optional path to write the comparison results as JSON Lines.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        help="Optional path to write the comparison results as a single JSON array.",
    )
    return parser.parse_args()


def select_device(preference: str) -> str:
    if preference == "cpu":
        return "cpu"
    if preference == "mps":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_models(base_model_id: str, adapter_dir: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = None
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif device == "mps":
        dtype = torch.float32  # fp32 is the most stable choice on macOS MPS

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    lora_base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    lora_model = PeftModel.from_pretrained(lora_base, adapter_dir)

    base_model.eval()
    lora_model.eval()

    # Keep models on CPU by default. We'll move them on-demand in generate().
    if device != "cuda":
        base_model.to("cpu")
        lora_model.to("cpu")

    return tokenizer, base_model, lora_model


def generate(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> str:
    prompt = prompt.strip()
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or eos_id

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Move model to the requested device for generation.
    original_device = next(model.parameters()).device
    if str(original_device) != device:
        model.to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            num_return_sequences=1,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
        )

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Move model back to CPU if we borrowed an accelerator.
    if device in {"mps", "cuda"}:
        model.to("cpu")
        if device == "mps":
            torch.mps.empty_cache()

    return text


def collect_prompts(args: argparse.Namespace) -> List[str]:
    prompts: List[str] = []
    if args.prompts_file:
        path = Path(args.prompts_file)
        if not path.exists():
            raise FileNotFoundError(f"Prompts file not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            prompts.extend([line.rstrip("\n") for line in fh if line.strip()])
    if args.prompt:
        prompts.append(args.prompt)
    if not prompts:
        raise ValueError("Provide --prompt or --prompts_file with at least one prompt.")
    return prompts


def main():
    args = parse_args()
    prompts = collect_prompts(args)
    device = select_device(args.device)

    if args.seed >= 0:
        torch.manual_seed(args.seed)

    tokenizer, base_model, lora_model = load_models(args.base_model, args.adapter_dir, device)

    results = []

    for idx, prompt in enumerate(prompts, 1):
        tuned_text = generate(
            lora_model,
            tokenizer,
            prompt,
            device,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
            args.top_k,
            args.repetition_penalty,
        )
        base_text = generate(
            base_model,
            tokenizer,
            prompt,
            device,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
            args.top_k,
            args.repetition_penalty,
        )

        entry = {
            "prompt": prompt,
            "fine_tuned": tuned_text,
            "base": base_text,
        }
        results.append(entry)

        print(f"\n=== Prompt {idx} ===")
        print(prompt)
        print("--- Fine-tuned ---")
        print(textwrap.fill(tuned_text, width=100))
        print("--- Base ---")
        print(textwrap.fill(base_text, width=100))

    if args.output_jsonl:
        out_path = Path(args.output_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            for entry in results:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"\nSaved JSONL comparisons to {out_path}")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=2)
        print(f"\nSaved JSON comparisons to {out_path}")


if __name__ == "__main__":
    main()
