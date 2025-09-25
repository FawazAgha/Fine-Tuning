# Fine-tuning Pythia, but chill

This repo is my playground for teaching `EleutherAI/pythia-*` models new tricks with LoRA. It runs great on my M1 Pro (32 GB) and also works on Linux/CUDA boxes if you have a beefy GPU.

## What’s here
- `train.py` – main trainer with LoRA/QLoRA options, auto device picking, and fp32 on MPS for stability.
- `scripts/prepare_texts.py` – vacuums up all `.txt` files and spits out a JSONL (`data/train.jsonl`).
- `sample.py` – quick CLI to sample from either the base model or the fine-tuned adapter.
- `compare.py` – prints base vs. fine-tuned generations side-by-side. Use `--device cpu` unless you love MPS OOMs.
- `data/` – drop your raw `.txt` class notes under `data/text/`, or keep curated JSONL files around.

## Setup (one-time)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you plan to pull anything from the Hugging Face Hub that isn’t public, run `huggingface-cli login` first.

## Prep your data
1. Toss raw notes into `data/text/` (subfolders are fine).
2. Build the training file:
   ```bash
   ./.venv/bin/python scripts/prepare_texts.py --input_dir data/text --output_file data/train.jsonl
   ```
   Each line in the JSONL ends up as a training chunk.

## Training cheatsheet
- **Full 2.8B + LoRA (resume-friendly example):**
  ```bash
  ./.venv/bin/python train.py \
    --model_name EleutherAI/pythia-2.8b \
    --train_file data/train.jsonl \
    --output_dir outputs/pythia-2_8b-lora \
    --lora --block_size 512 --batch_size 2 --grad_accum 8 \
    --max_steps 2000 --logging_steps 50
  ```

- **Smaller + faster (pythia-1b):**
  ```bash
  ./.venv/bin/python train.py \
    --model_name EleutherAI/pythia-1b \
    --train_file data/train.jsonl \
    --output_dir outputs/pythia-1b-lora \
    --lora --block_size 1024 --batch_size 4 --grad_accum 2 \
    --max_steps 2400 --logging_steps 100
  ```

- Resuming? Point `--resume_from_checkpoint` at the checkpoint folder and bump `--max_steps` to the new total.

## Sampling
Grab a response from the fine-tuned adapter:
```bash
./.venv/bin/python sample.py --prompt "Explain stacks vs queues" --max_new_tokens 200
```

Need the raw base model instead? Add `--use_base`. Multiple completions? `--num_return_sequences 3`.

## Compare base vs. LoRA
```bash
TOKENIZERS_PARALLELISM=false ./.venv/bin/python compare.py \
  --prompt "What’s the difference between a stack and a queue?" \
  --device cpu \
  --output_json comparisons.json
```
The script prints a side-by-side diff and drops a JSON array you can open anywhere. Stay on CPU unless you have ridiculous headroom; loading both 2.8B models onto MPS eats ~22 GB.

## Little tips
- Keep `TOKENIZERS_PARALLELISM=false` around when spawning lots of workers to dodge warnings.
- LoRA adapters are tiny, so feel free to version them in Git if you want. Merged full models are many GB—don’t even try to store those here.
- macOS: fp32 is slower but avoids NaNs. If you do hit randomness, lower `--temperature` or drop `--top_k`.
- Want a lighter run? Try `pythia-410m` or 1.4B with the same scripts; nothing else changes.

That’s it. Drop in new text, regenerate `data/train.jsonl`, fire off `train.py`, and sample away. Feel free to tweak, break, and rerun as much as you like.
