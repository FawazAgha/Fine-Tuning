import os
import math
import json
import argparse
import logging
from dataclasses import dataclass
from typing import Optional, Dict, List

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune EleutherAI/pythia-2.8b with (Q)LoRA")
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-2.8b", help="Base model")
    parser.add_argument("--train_file", type=str, required=True, help="Path to JSONL train file")
    parser.add_argument("--validation_file", type=str, default=None, help="Optional JSONL validation file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for checkpoints")

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lora", action="store_true", help="Enable LoRA adapters")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_targets", type=str, default="query_key_value,dense",
                        help="Comma-separated target module names for LoRA")

    parser.add_argument("--qlora", action="store_true", help="Enable 4-bit QLoRA (Linux + CUDA only)")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    return parser.parse_args()


def setup_logging():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


def is_bitsandbytes_available() -> bool:
    try:
        import bitsandbytes as bnb  # noqa: F401
        return torch.cuda.is_available() and (torch.version.cuda is not None)
    except Exception:
        return False


def get_quant_config(load_4bit: bool):
    if not load_4bit:
        return None
    try:
        from transformers import BitsAndBytesConfig
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    except Exception:
        return None


def prepare_model_and_tokenizer(args):
    logging.info("Loading tokenizer and model: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_bnb = args.qlora and is_bitsandbytes_available()
    quant_config = get_quant_config(load_4bit=use_bnb)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        quantization_config=quant_config,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if args.lora:
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        except Exception as e:
            raise RuntimeError("PEFT is required for LoRA. Please ensure 'peft' is installed.") from e

        if use_bnb:
            model = prepare_model_for_kbit_training(model)

        target_modules = [t.strip() for t in args.lora_targets.split(",") if t.strip()]
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info("LoRA enabled. Trainable params: %s", f"{trainable_params:,}")

    return model, tokenizer, use_bnb


def load_and_prepare_datasets(tokenizer, args):
    data_files = {"train": args.train_file}
    if args.validation_file:
        data_files["validation"] = args.validation_file

    raw_datasets = load_dataset("json", data_files=data_files)

    def build_text(example: Dict) -> str:
        if "text" in example and example["text"]:
            return example["text"]
        if "prompt" in example and "response" in example:
            return f"{example['prompt'].strip()}\n\n{example['response'].strip()}"
        raise ValueError("Each JSONL line must have either 'text' or both 'prompt' and 'response'.")

    def tokenize_fn(batch: Dict[str, List[str]]):
        texts = [build_text(ex) for ex in batch[batch.keys().__iter__().__next__()]]  # hack to iterate batch rows
        # The above trick is to get number of rows. We'll instead rebuild properly:
        # Batch is a dict of columns; reconstruct rows first
        rows = []
        num_rows = None
        for k, v in batch.items():
            if num_rows is None:
                num_rows = len(v)
            else:
                assert num_rows == len(v)
        for i in range(num_rows or 0):
            row = {k: v[i] for k, v in batch.items()}
            rows.append(row)
        texts = [build_text(r) for r in rows]
        return tokenizer(texts, truncation=True, max_length=args.block_size)

    tokenized = raw_datasets.map(
        tokenize_fn,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Tokenizing",
    )

    def group_texts(examples: Dict[str, List[int]]):
        # Concatenate and split into blocks of block_size
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])
        total_length = (total_length // args.block_size) * args.block_size
        result = {
            k: [t[i: i + args.block_size] for i in range(0, total_length, args.block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized.map(
        group_texts,
        batched=True,
        desc="Grouping into blocks",
    )

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"] if "validation" in lm_datasets else None
    return train_dataset, eval_dataset


def main():
    args = parse_args()
    setup_logging()
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    model, tokenizer, used_bnb = prepare_model_and_tokenizer(args)
    train_dataset, eval_dataset = load_and_prepare_datasets(tokenizer, args)

    eval_strategy = "steps" if eval_dataset is not None else "no"

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16 = torch.cuda.is_available() and not bf16

    optim = "paged_adamw_8bit" if (args.qlora and used_bnb) else "adamw_torch"

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size),
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        evaluation_strategy=eval_strategy,
        eval_steps=args.eval_steps if eval_strategy != "no" else None,
        save_total_limit=args.save_total_limit,
        fp16=fp16,
        bf16=bf16,
        dataloader_num_workers=2,
        report_to=["tensorboard"],
        optim=optim,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    resume = args.resume_from_checkpoint
    train_result = trainer.train(resume_from_checkpoint=resume)
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    if eval_dataset is not None:
        eval_metrics = trainer.evaluate()
        metrics.update({f"eval_{k}": v for k, v in eval_metrics.items()})

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logging.info("Training complete. Output saved to %s", args.output_dir)


if __name__ == "__main__":
    main()

