"""CLI helper to sample from base vs. LoRA fine-tuned Pythia models."""

import argparse
from dataclasses import dataclass
from typing import List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ModelBundle:
    tokenizer: AutoTokenizer
    base_model: AutoModelForCausalLM
    tuned_model: AutoModelForCausalLM
    device: str


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_models(base_model_id: str, adapter_dir: str, device: str) -> ModelBundle:
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype: Optional[torch.dtype] = None
    device_map = None
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        device_map = "auto"
    elif device == "mps":
        dtype = torch.float32

    base = AutoModelForCausalLM.from_pretrained(base_model_id, dtype=dtype, device_map=device_map)
    tuned_base = AutoModelForCausalLM.from_pretrained(base_model_id, dtype=dtype, device_map=device_map)
    tuned = PeftModel.from_pretrained(tuned_base, adapter_dir)

    base.eval()
    tuned.eval()

    if device == "cuda":
        base.to("cuda")
        tuned.to("cuda")
    else:
        base.to("cpu")
        tuned.to("cpu")

    return ModelBundle(tokenizer, base, tuned, device)


def generate_text(bundle: ModelBundle, prompt: str, *, use_base: bool, max_new_tokens: int,
                  temperature: float, top_p: float, top_k: int, repetition_penalty: float,
                  num_return_sequences: int) -> List[str]:
    model = bundle.base_model if use_base else bundle.tuned_model
    tokenizer = bundle.tokenizer

    prompt = prompt.strip()
    if not prompt:
        raise ValueError("Prompt is empty.")

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or eos_id

    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {k: v.to(bundle.device) for k, v in encoded.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **encoded,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
        )

    return [tokenizer.decode(outputs[i], skip_special_tokens=True) for i in range(outputs.size(0))]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample from base vs. LoRA fine-tuned Pythia models.")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to generate from.")
    parser.add_argument("--base_model", type=str, default="EleutherAI/pythia-2.8b")
    parser.add_argument("--adapter_dir", type=str,
                        default="outputs/pythia-2_8b-lora/resumed_notes/checkpoint-1445")
    parser.add_argument("--use_base", action="store_true", help="Use the base model instead of the LoRA adapter.")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--num_return_sequences", type=int, default=1)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    device = select_device()
    bundle = load_models(args.base_model, args.adapter_dir, device)

    outputs = generate_text(
        bundle,
        prompt=args.prompt,
        use_base=args.use_base,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        num_return_sequences=args.num_return_sequences,
    )

    if len(outputs) == 1:
        print(outputs[0])
    else:
        for idx, text in enumerate(outputs, 1):
            print(f"\n=== Sample {idx} ===\n{text}\n")


if __name__ == "__main__":
    main()

