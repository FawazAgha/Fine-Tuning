Pythia 2.8B Fine-Tuning (LoRA-ready)

This folder contains a minimal setup to fine‑tune `EleutherAI/pythia-2.8b` on your data using Hugging Face Transformers. It supports classic LoRA and, when available on CUDA Linux, QLoRA (4‑bit) to reduce memory.

Quick start

- Prepare data under `data/` as JSONL with either a single `text` field per line, or a `prompt` + `response` pair per line. See `data/sample.jsonl`.
- Install requirements and run the training script.

Environment notes

- Linux + CUDA: Full support for LoRA and QLoRA (bitsandbytes).
- macOS (CPU/MPS): Use standard LoRA (no bitsandbytes); QLoRA is not supported.

Install

1) (Recommended) Create a fresh Python 3.10+ environment.
2) Install dependencies:

   pip install -r requirements.txt

3) (Optional) Login to Hugging Face if you want to push to the Hub:

   huggingface-cli login

Data format

- Plain LM: one JSON object per line containing `text`.
- Prompt/response: one JSON object per line containing `prompt` and `response` (they will be concatenated for causal LM).

Example: see `data/sample.jsonl`.

Run a small test

- LoRA without 4‑bit (works on CPU/MPS/GPU):

  python train.py \
    --model_name EleutherAI/pythia-2.8b \
    --train_file data/sample.jsonl \
    --output_dir outputs/pythia-2_8b-lora \
    --epochs 1 --batch_size 1 --lora

- QLoRA (Linux + CUDA only):

  python train.py \
    --model_name EleutherAI/pythia-2.8b \
    --train_file data/sample.jsonl \
    --output_dir outputs/pythia-2_8b-qlora \
    --epochs 1 --batch_size 1 --lora --qlora

Common tips

- Set `--block_size` smaller (e.g., 512) if memory is tight.
- Increase `--gradient_accumulation_steps` to emulate larger batch sizes.
- Use `--gradient_checkpointing` to lower memory at some compute cost.
- If you have evaluation data, provide `--validation_file` to enable periodic eval.

Outputs

- Model checkpoints and logs are written to `--output_dir`.
- With LoRA/QLoRA, checkpoints contain adapter weights. You can merge them later or load adapters at inference.

