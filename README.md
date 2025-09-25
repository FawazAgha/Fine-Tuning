# Fine‑tuning Pythia

This repo is my Pythia fine‑tuning project. I tried to build a language model from scratch in PyTorch and quickly realized I don’t have the compute or the data to get anything coherent. So I imported the Pythia‑2.8B model and fine‑tuned it on specific data , in my case, class notes.

## What’s inside
- `train.py` – trainer script with LoRA/QLoRA, automatic device selection, and fp32 on MPS to keep Apple Silicon stable.
             (ran into nan issues when running on fp16 even when decresing learning rates and implementing other fixes)
- `scripts/prepare_texts.py` – sweeps `data/text/**/*.txt` into `data/train.jsonl`. (my notes were in .txt)
- `sample.py` – quick CLI to sample from either the base model or the fine-tuned adapter.
- `compare.py` – prints base vs. LoRA generations side-by-side and can dump them to JSON/JSONL.
- `data/` – scratchpad for training corpora (`data/text/`) or curated JSONL files I want to keep.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
If I need private HF models, I run `huggingface-cli login` once per machine.

## Data pipeline
1. Drop class notes (or any .txt files) under `data/text/`.
2. Generate the JSONL the trainer expects:
   ```bash
   ./.venv/bin/python scripts/prepare_texts.py --input_dir data/text --output_file data/train.jsonl
   ```
   Each line becomes a training chunk; no special formatting needed beyond plain text.

## Training
- **Pythia‑2.8B LoRA (baseline run)**
  ```bash
  ./.venv/bin/python train.py \
    --model_name EleutherAI/pythia-2.8b \
    --train_file data/train.jsonl \
    --output_dir outputs/pythia-2_8b-lora \
    --lora --block_size 512 --batch_size 2 --grad_accum 8 \
    --max_steps 2000 --logging_steps 50
  ```
- **Pythia‑1B LoRA (faster iteration)**
  ```bash
  ./.venv/bin/python train.py \
    --model_name EleutherAI/pythia-1b \
    --train_file data/train.jsonl \
    --output_dir outputs/pythia-1b-lora \
    --lora --block_size 1024 --batch_size 4 --grad_accum 2 \
    --max_steps 2400 --logging_steps 100
  ```
- Need to resume? Point `--resume_from_checkpoint` at the last checkpoint and increase `--max_steps` to the new total.

## Sampling quick hits
- Fine-tuned adapter (default):
  ```bash
  ./.venv/bin/python sample.py --prompt "Explain stacks vs queues" --max_new_tokens 200
  ```
- Base model instead:
  ```bash
  ./.venv/bin/python sample.py --prompt "Explain stacks vs queues" --use_base
  ```
- Multiple samples at once: add `--num_return_sequences 3`.

## Side‑by‑side checks
```bash
TOKENIZERS_PARALLELISM=false ./.venv/bin/python compare.py \
  --prompt "What’s the difference between a stack and a queue?" \
  --device cpu \
  --output_json comparisons.json
```
I stick to `--device cpu` here so I’m not loading two 2.8B models onto MPS at the same time. (caused my computer to crash)

## Notes‑to‑self
- Keep `TOKENIZERS_PARALLELISM=false` around when spawning dataloader workers to avoid warnings.
- LoRA adapters are small enough to version in Git; merged full models aren’t (and I don’t store them here).
- FP32 on MPS is slower but saves me from NaNs. If outputs look repetitive, lower `--temperature` or skip `--top_k`.
- Smaller checkpoints (`pythia-410m`, `pythia-1.4b`) work with the exact same scripts if I want faster experiments.

That’s the whole workflow: drop notes into `data/text`, rebuild `data/train.jsonl`, run `train.py`, and try things with `sample.py` or `compare.py`. Simple and repeatable.
