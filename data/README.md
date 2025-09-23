Place your training/eval data here.

Supported formats (JSON Lines):

1) Single-field causal LM:
   {"text": "Your full training text goes here."}

2) Prompt/response pairs (concatenated for causal LM):
   {"prompt": "Question?", "response": "Answer."}

See `sample.jsonl` for a minimal example.
