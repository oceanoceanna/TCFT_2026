# Teaching Large Language Models When Not to Know: Learning Temporal Critique for Ex-Ante Reasoning

Code for **Teaching Large Language Models When Not to Know: Learning Temporal Critique for Ex-Ante Reasoning** (NeurIPS 2026).

## Layout

- `qa/`
  - `data/`: prepared `train.jsonl`, `valid.jsonl`, `test.jsonl`
  - `augment_post_event_data.py`: reproducibly construct post-event QA examples with provenance embedded in each new row
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

## QA data

The train, validation, and test files contain 10,472, 800, and 800 examples.
This includes 2,400, 300, and 300 added examples whose cutoff is 1, 14, or 30
days after the event, evenly divided among these offsets in each split.
Their answers give the recorded event date, with a complete critique ending
in `Conclusion: right`.

The 3,000 added examples have an optional `augmentation` object containing
the generation method and seed, source split and line, original split SHA-256,
output line at generation time, day offset, and critique template number.
Source and output line numbers are one-based. Existing question, answer, and
critique fields retain their meaning. Provenance is stored directly in these
three JSONL files; there is no separate augmentation manifest.

The added critiques are derived from 12 text templates and the recorded date
relationships. Source event dates have not been independently fact-checked.
