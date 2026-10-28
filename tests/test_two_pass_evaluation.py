import __future__
import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_script(relative_path):
    path = ROOT / relative_path
    module = types.ModuleType(path.stem)
    module.__file__ = str(path)
    exec(compile(path.read_text(), str(path), "exec", flags=__future__.annotations.compiler_flag), module.__dict__)
    return module


class FakeTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]


class FakeModel:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def merge_and_unload(self):
        return self

    def eval(self):
        return self

    def to(self, device):
        return self


def fake_dependencies():
    return {
        "torch": types.SimpleNamespace(bfloat16="test-dtype"),
        "peft": types.SimpleNamespace(PeftModel=FakeModel),
        "transformers": types.SimpleNamespace(AutoModelForCausalLM=FakeModel, AutoTokenizer=FakeTokenizer),
        "tqdm.auto": types.SimpleNamespace(tqdm=lambda values, **kwargs: values),
    }


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.qa = load_script("qa/eval_lora_open.py")
        self.row = {
            "leakage_prompt": "QUESTION_SENTINEL: answer as of 2020-01-01.",
            "leakage_answer_raw": "ORIGINAL_ANSWER_SENTINEL",
            "critique": "REFERENCE_CRITIQUE_SENTINEL",
            "true_date": "2020-01-02",
            "ask_date": "2020-01-01",
            "event": "HIDDEN_EVENT_SENTINEL",
            "critique_model": "private-source-model",
            "api_model": "private-answer-model",
            "adapter_path": "private-adapter-path",
        }

    def test_two_pass_inputs_and_pairing(self):
        second = {**self.row, "leakage_prompt": "SECOND_QUESTION", "leakage_answer_raw": "SECOND_ANSWER"}
        critiques = ["NEW_CRITIQUE_A", "NEW_CRITIQUE_B"]
        answers = ["I don't know.", "I don't know."]
        for chat in (False, True):
            with self.subTest(chat=chat), patch.object(self.qa, "generate_batch", side_effect=[critiques, answers]) as generate:
                result = self.qa.generate_two_pass_batch(
                    object(), FakeTokenizer(), [self.row, second], max_length=1024,
                    max_critique_tokens=384, max_new_tokens=160, use_chat_template=chat,
                )
                self.assertEqual(result, (critiques, answers))
                self.assertEqual(generate.call_count, 2)
                first_prompts = generate.call_args_list[0].args[2]
                self.assertIn(self.row["leakage_prompt"], first_prompts[0])
                for row, prompt in zip([self.row, second], first_prompts):
                    for key in ("leakage_answer_raw", "critique", "true_date", "event"):
                        self.assertNotIn(row[key], prompt)
                revision_prompts = generate.call_args_list[1].args[2]
                for index, (row, prompt) in enumerate(zip([self.row, second], revision_prompts)):
                    self.assertIn(row["leakage_prompt"], prompt)
                    self.assertIn(row["leakage_answer_raw"], prompt)
                    self.assertIn(critiques[index], prompt)
                    self.assertNotIn(critiques[1 - index], prompt)
                    self.assertNotIn(row["critique"], prompt)
                    self.assertNotIn(row["true_date"], prompt)

    def test_empty_critique_stops_before_second_pass(self):
        with patch.object(self.qa, "generate_batch", return_value=[" "]) as generate:
            with self.assertRaisesRegex(ValueError, "nonempty critique"):
                self.qa.generate_two_pass_batch(
                    object(), FakeTokenizer(), [self.row], max_length=1024,
                    max_critique_tokens=384, max_new_tokens=160,
                )
            self.assertEqual(generate.call_count, 1)

    def test_long_prompt_is_not_silently_truncated(self):
        class LongTokenizer:
            def __call__(self, prompts, **kwargs):
                assert kwargs["truncation"] is False
                return {"input_ids": types.SimpleNamespace(shape=(1, 1025))}

        with patch.dict(sys.modules, fake_dependencies()):
            with self.assertRaisesRegex(ValueError, "exceeding --max-length"):
                self.qa.generate_batch(object(), LongTokenizer(), ["long input"], 1024, 160)

    def test_qa_main_writes_new_critique_answer_and_anonymous_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.jsonl"
            output = root / "predictions" / "output.jsonl"
            summary = root / "separate-metrics" / "summary.json"
            source.write_text(json.dumps(self.row) + "\n")
            argv = ["eval", "--model-id", "private-base-model", "--adapter-path", "private-adapter",
                    "--test-jsonl", str(source), "--output-jsonl", str(output), "--summary-json", str(summary)]
            with patch.dict(sys.modules, fake_dependencies()), patch.object(sys, "argv", argv):
                with patch.object(self.qa, "generate_batch", side_effect=[["NEW_CRITIQUE"], ["I don't know."]]) as generate:
                    with contextlib.redirect_stdout(io.StringIO()) as stdout:
                        self.qa.main()
            self.assertEqual(generate.call_count, 2)
            record = json.loads(output.read_text())
            self.assertEqual(record["generated_critique"], "NEW_CRITIQUE")
            self.assertEqual(record["generated_text"], "I don't know.")
            self.assertEqual(record["leakage_answer_raw"], self.row["leakage_answer_raw"])
            self.assertEqual(record["critique"], self.row["critique"])
            self.assertFalse(record["leakage_detected"])
            self.assertEqual(json.loads(summary.read_text())["generation_passes"], 2)
            for text in (output.read_text(), summary.read_text(), stdout.getvalue()):
                self.assertNotIn("private-", text)

    def test_api_generators_do_not_reintroduce_model_metadata(self):
        cases = [
            ("qa/generate_critiques.py", "--input", "--output"),
            ("wiki/generate_5_claims_api.py", "--input-jsonl", "--output-jsonl"),
            ("wiki/generate_wiki_critiques_api.py", "--input-jsonl", "--output-jsonl"),
        ]
        row = {**self.row, "title": "Example", "cutoff_year": 2020,
               "prompt": "Example question", "raw_response": "1. Example fact.", "claims": []}
        for path, input_flag, output_flag in cases:
            with self.subTest(script=path), tempfile.TemporaryDirectory() as tmp:
                module = load_script(path)
                source, output = Path(tmp) / "input.jsonl", Path(tmp) / "output.jsonl"
                source.write_text(json.dumps(row) + "\n")
                argv = [path, input_flag, str(source), output_flag, str(output), "--model", "private-model"]
                with patch.object(sys, "argv", argv), patch.object(module, "call_openai", return_value="Conclusion: right"):
                    module.main()
                self.assertNotIn("private-", output.read_text())

    def test_wiki_validity_strips_input_model_metadata(self):
        module = load_script("wiki/eval_5claims_validity.py")
        record = module.evaluate_row({**self.row, "input": "As of January 1st, 2020",
                                      "generated_raw_response": "1. An event occurred in 2010."})
        self.assertNotIn("private-", json.dumps(record))
        self.assertEqual(record["before_valid_count"], 1)

    def test_wiki_lora_output_and_summary_omit_model_paths(self):
        with patch.dict(sys.modules, fake_dependencies()), tempfile.TemporaryDirectory() as tmp:
            module = load_script("wiki/generate_5_claims_lora.py")
            source, output = Path(tmp) / "input.jsonl", Path(tmp) / "output.jsonl"
            source.write_text(json.dumps({**self.row, "prompt": "Example question"}) + "\n")
            argv = ["generate", "--model-id", "private-model", "--adapter-path", "private-adapter",
                    "--input-jsonl", str(source), "--output-jsonl", str(output)]
            with patch.object(sys, "argv", argv), patch.object(module, "generate_batch", return_value=["1. A fact."]):
                with contextlib.redirect_stdout(io.StringIO()) as stdout:
                    module.main()
            self.assertNotIn("private-", output.read_text())
            self.assertNotIn("private-", output.with_suffix(".summary.json").read_text())
            self.assertNotIn("private-", stdout.getvalue())

    def test_historical_scoring_is_preserved(self):
        self.assertFalse(self.qa.detect_leakage("2020-01-02", "2020-01-02", "2020-01-03"))
        self.assertTrue(self.qa.historical_date_exact_match("2020-01-02", "2020-01-02", "2020-01-03"))
        self.assertTrue(self.qa.detect_leakage("2020-01-02", "2020-01-02", "2020-01-01"))


if __name__ == "__main__":
    unittest.main()
