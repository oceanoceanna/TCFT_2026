# Teaching Large Language Models When Not to Know: Learning Temporal Critique for Ex-Ante Reasoning

Code for **Teaching Large Language Models When Not to Know: Learning Temporal Critique for Ex-Ante Reasoning** (NeurIPS 2026).

## Layout

- `qa/`
  - `data/`: prepared `train.jsonl`, `valid.jsonl`, `test.jsonl`
  - `generate_critiques.py`: generate QA critiques from leakage prompts
  - `train_lora_sft_from_critiques.py`: train a LoRA SFT model on QA critique data
  - `eval_lora_open.py`: generate a critique from the question only, then revise the original answer using the question and generated critique

- `wiki/`
  - `data/`: prepared `train.jsonl`, `valid.jsonl`, `test.jsonl`
  - `generate_5_claims_api.py`: call API to generate 5 claims
  - `generate_5_claims_lora.py`: generate 5 claims with a LoRA model
  - `generate_wiki_critiques_api.py`: generate critiques for 5-claim outputs
  - `train_lora_sft_from_critiques.py`: train a LoRA SFT model on wiki critique data
  - `eval_5claims_validity.py`: check generated claims using year-based regular expressions; no model or API is used for this check
