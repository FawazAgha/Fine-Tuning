#!/usr/bin/env python
import argparse
import json
import os
from pathlib import Path


def read_text_file(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8") as f:
            text = f.read()
    except UnicodeDecodeError:
        with path.open("r", encoding="latin-1", errors="ignore") as f:
            text = f.read()
    # Normalize newlines and strip BOM/whitespace
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip("\ufeff\n\t ")
    return text


def main():
    parser = argparse.ArgumentParser(description="Convert plain .txt files to JSONL for causal LM training")
    parser.add_argument("--input_dir", type=str, default="data/text", help="Directory containing .txt files")
    parser.add_argument("--output_file", type=str, default="data/train.jsonl", help="Output JSONL path")
    parser.add_argument("--min_chars", type=int, default=32, help="Skip samples shorter than this many chars")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists() or not in_dir.is_dir():
        raise SystemExit(f"Input directory not found: {in_dir}")

    txt_files = sorted(
        [p for p in in_dir.rglob("*.txt") if p.is_file() and not p.name.startswith(".~")]
    )
    if not txt_files:
        raise SystemExit(f"No .txt files found under {in_dir}")

    written = 0
    with out_path.open("w", encoding="utf-8") as out:
        for p in txt_files:
            text = read_text_file(p)
            if len(text) < args.min_chars:
                continue
            # One JSON object per file. Trainer packs tokens later.
            obj = {"text": text}
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} samples to {out_path}")


if __name__ == "__main__":
    main()

