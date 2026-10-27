# SPDX-License-Identifier: Apache-2.0
"""Focused checks for dataset construction, CLI orchestration and output control."""

import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import deai
from dependency_perturb import load_parser
from output_control import control_errors
from style_data import _replace_known, EXPANSIONS, prepare, synthetic_sources
from text_pipeline import build_plan, prompt_token_ids


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nlp = load_parser()

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return list(range(sum(len(msg["content"].split()) for msg in messages) + 5))

    def test_contraction_and_reordered_source(self):
        source, changed = _replace_known("They don't agree.", EXPANSIONS)
        self.assertTrue(changed)
        self.assertEqual(source, "They do not agree.")
        variants = synthetic_sources("The team worked in the garden after lunch.", self.nlp)
        self.assertIn("In the garden, after lunch, the team worked.", variants)

    def test_split_before_augmentation_and_length(self):
        text = "\n\n".join(
            f"The team worked in the garden after lunch. They rested at home near"
            f" the river. The report mentioned project {index}."
            for index in range(30)
        )
        pairs, report = prepare(text, self.nlp, self.Tokenizer())
        self.assertEqual(report["units"], 30)
        self.assertGreater(report["train_pairs"], report["valid_pairs"])
        train = {item["messages"][1]["content"] for item in pairs["train"]}
        valid = {item["messages"][1]["content"] for item in pairs["valid"]}
        self.assertFalse(train & valid)
        self.assertEqual(len(train | valid), 30)
        for examples in pairs.values():
            for row in examples:
                self.assertLessEqual(
                    len(self.Tokenizer().apply_chat_template(row["messages"])), 2048
                )
                self.assertNotEqual(row["messages"][0]["content"].split("\n\n", 1)[1],
                                    row["messages"][1]["content"])

    def test_rejects_tiny_or_untransformable_target(self):
        with self.assertRaisesRegex(ValueError, "至少"):
            prepare("A brief note.", self.nlp, self.Tokenizer())
        target = "\n\n".join(f"She saw project {i}." for i in range(20))
        with self.assertRaisesRegex(ValueError, "覆盖不足"):
            prepare(target, self.nlp, self.Tokenizer())

    def test_output_control_opt_in_and_unsafe_sentences(self):
        text = 'The teacher reviewed the document. "The doctor reviewed a document."'
        self.assertEqual(control_errors(text, self.nlp), text)
        with patch("output_control._selected", return_value=True):
            modified = control_errors(text, self.nlp, .05, .05)
        self.assertNotEqual(modified, text)
        self.assertIn('"The doctor reviewed a document."', modified)
        with self.assertRaises(ValueError):
            control_errors(text, self.nlp, spelling=.5)

    def test_paragraph_packing_without_server_api(self):
        text = ("The team worked in the garden after lunch. " * 120).strip()
        plan = build_plan(self.Tokenizer(), text + "\n\n" + text)
        self.assertGreater(len(plan), 2)
        self.assertEqual(sum(separator == "\n\n" for _, separator in plan), 1)
        self.assertTrue(all(len(prompt_token_ids(self.Tokenizer(), unit)) <= 900
                            for unit, _ in plan))


class CLITests(unittest.TestCase):
    def test_template_inference_without_style_or_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template = root / "template"
            template.mkdir()
            (template / "adapter_config.json").write_text(
                '{"fine_tune_type":"full"}', encoding="utf-8"
            )
            (template / "adapters.safetensors").write_bytes(b"fixture")
            source = root / "input.txt"
            source.write_text("The team worked.", encoding="utf-8")
            with patch.object(deai, "load_full", return_value=(object(), object())) as load:
                with patch.object(deai, "_generate", return_value="Result.") as generate:
                    with patch.object(deai.subprocess, "run") as train:
                        deai.run(["--input", str(source), "--adapter-path", str(template),
                                  "--run-dir", str(root / "result")])
            load.assert_called_once()
            generate.assert_called_once()
            train.assert_not_called()
            self.assertEqual((root / "result/result.txt").read_text(encoding="utf-8"),
                             "Result.")

    def test_prepare_only_writes_independent_training_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            style, source, run_dir = root / "style.txt", root / "input.txt", root / "run"
            style.write_text("\n\n".join(
                f"The team worked in the garden after lunch. They rested at home "
                f"near the river. The report mentioned project {i}."
                for i in range(30)
            ), encoding="utf-8")
            source.write_text("The team worked in the garden.", encoding="utf-8")
            with patch.object(deai.subprocess, "run") as train:
                deai.run(["--style", str(style), "--input", str(source),
                          "--run-dir", str(run_dir), "--prepare-only"])
            train.assert_not_called()
            report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
            self.assertGreaterEqual(report["train_pairs"], 12)
            self.assertEqual(report["weak_template_pairs"]["valid"], 0)
            self.assertTrue((run_dir / "data/train.jsonl").is_file())
            self.assertFalse((run_dir / "weights").exists())
            with self.assertRaisesRegex(ValueError, "已存在"):
                deai.run(["--style", str(style), "--input", str(source),
                          "--run-dir", str(run_dir), "--prepare-only"])

    def test_seed_flows_to_training_config(self):
        self.assertEqual(deai._config(20, 123)["seed"], 123)

    @unittest.skipUnless(os.environ.get("DEAI_E2E"), "set DEAI_E2E=1 for real MLX training")
    def test_one_step_training_and_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            style, source, run_dir = root / "style.txt", root / "input.txt", root / "run"
            style.write_text("\n\n".join(
                f"The team worked in the garden after lunch. They rested at home "
                f"near the river. The report mentioned project {i}."
                for i in range(30)
            ), encoding="utf-8")
            source.write_text("The team worked in the garden.", encoding="utf-8")
            deai.run(["--style", str(style), "--input", str(source),
                      "--run-dir", str(run_dir), "--steps", "1"])
            self.assertTrue((run_dir / "weights/adapters.safetensors").is_file())
            self.assertTrue((run_dir / "result.txt").read_text(encoding="utf-8").strip())

    @unittest.skipUnless(os.environ.get("DEAI_E2E"), "set DEAI_E2E=1 for real MLX inference")
    def test_real_release_template_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template = root / "template"
            template.mkdir()
            shutil.copyfile(deai.ROOT / "template/adapter_config.json",
                            template / "adapter_config.json")
            (template / "adapters.safetensors").symlink_to(
                deai.ROOT / "output/adapters.safetensors"
            )
            source = root / "input.txt"
            source.write_text("The team worked in the garden after lunch.", encoding="utf-8")
            deai.run(["--input", str(source), "--adapter-path", str(template),
                      "--run-dir", str(root / "result")])
            self.assertTrue((root / "result/result.txt").read_text(
                encoding="utf-8"
            ).strip())


if __name__ == "__main__":
    unittest.main()
