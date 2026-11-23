"""Check attention direction, option positions, voting, and full-set accounting."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from coco_classification import evaluate_option_attention as evaluator


def fixture():
    prepared = SimpleNamespace(
        inputs={"input_ids": torch.tensor([[11, 11, 12, 13, 14, 15]])},
        prompt_start=0, image_start=4, letters=(1, 3), grid=SimpleNamespace(count=2),
    )
    sample = SimpleNamespace(categories=[
        SimpleNamespace(name="car", letter="A"), SimpleNamespace(name="person", letter="B"),
    ])
    tokenizer = SimpleNamespace(decode=lambda ids: {11: "A", 13: "B"}[ids[0]])
    weights = torch.zeros(1, 3, 6, 6)
    # Two heads favor B, but mean attention favors A for the first patch.
    for head, a, b in ((0, .02, .03), (1, .02, .03), (2, .50, .00)):
        weights[0, head, 4, 1] = a
        weights[0, head, 4, 3] = b
        weights[0, head, 4, 0] = 1 - a - b  # Another A occurrence: must be ignored.
        weights[0, head, 5, 1] = .01
        weights[0, head, 5, 3] = .04
        weights[0, head, 5, 0] = .95
    return prepared, sample, tokenizer, weights


class OptionAttentionTests(unittest.TestCase):
    def test_exact_option_positions_ignore_repeated_letter_elsewhere(self):
        prepared, sample, tokenizer, _ = fixture()
        self.assertEqual(evaluator.option_positions(sample, prepared, tokenizer), ([1, 3], [11, 13]))

    def test_invalid_option_identity_or_position_is_rejected(self):
        prepared, sample, tokenizer, _ = fixture()
        prepared.letters = (1, 4)
        with self.assertRaises(ValueError):
            evaluator.option_positions(sample, prepared, tokenizer)
        prepared.letters = (1, 3)
        prepared.inputs["input_ids"][0, 3] = 11
        with self.assertRaises(ValueError):
            evaluator.option_positions(sample, prepared, tokenizer)

    def test_outgoing_attention_mean_then_rank_without_renormalization(self):
        prepared, _, _, weights = fixture()
        scores, mass = evaluator.option_attention_scores(weights, prepared, [1, 3])
        np.testing.assert_allclose(scores, [[.18, .02], [.01, .04]], atol=1e-7)
        np.testing.assert_allclose(mass, [1, 1], atol=1e-7)
        np.testing.assert_array_equal(evaluator.rank_options(scores), [[0, 1], [1, 0]])
        self.assertLess(scores[0].sum(), 1)  # Full-context denominator was preserved.
        # Reverse-direction (option query -> image key) attention was all zero.
        self.assertTrue((weights[0, :, [1, 3], 4:] == 0).all())

    def test_attention_must_be_returned(self):
        prepared, _, _, _ = fixture()
        with self.assertRaises(RuntimeError):
            evaluator.option_attention_scores(None, prepared, [1, 3])

    def test_patch_ties_have_stable_alphabetical_order(self):
        np.testing.assert_array_equal(evaluator.rank_options(np.array([[.2, .2, .1]])), [[0, 1, 2]])

    def test_object_uses_fractional_patch_votes_before_attention_support(self):
        scores = np.array([[.11, .10], [0, .9], [0, .9]])
        winner, votes, support = evaluator.instance_vote(scores, evaluator.rank_options(scores), np.array([.9, .1, .1]))
        self.assertEqual(winner, 0)
        np.testing.assert_allclose(votes, [.9, .2])
        self.assertGreater(support[1], support[0])

    def test_object_vote_ties_use_attention_then_alphabetical_order(self):
        scores = np.array([[.2, .1], [0, .8]])
        winner, _, _ = evaluator.instance_vote(scores, evaluator.rank_options(scores), np.ones(2))
        self.assertEqual(winner, 1)
        scores = np.array([[.2, .1], [.1, .2]])
        winner, _, _ = evaluator.instance_vote(scores, evaluator.rank_options(scores), np.ones(2))
        self.assertEqual(winner, 0)

    def test_fractional_mask_coverage_and_empty_instance_rejection(self):
        grid = SimpleNamespace(x_edges=(0, 2, 4), y_edges=(0, 2))
        mask = np.array([[1, 0, 1, 0], [1, 0, 0, 0]], dtype=np.uint8)
        np.testing.assert_allclose(evaluator.patch_coverage(mask, grid), [.5, .25])
        with self.assertRaises(ValueError):
            evaluator.instance_vote(np.ones((2, 1)), np.zeros((2, 1), dtype=int), np.zeros(2))

    def test_inference_captures_once_and_always_removes_hook(self):
        prepared, sample, tokenizer, weights = fixture()

        class Attention(torch.nn.Module):
            def forward(self):
                return None, weights

        class Core(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(16, 2)
                self.attention = Attention()
                self.language_model = SimpleNamespace(
                    layers=[SimpleNamespace(self_attn=self.attention)], embed_tokens=self.embed_tokens,
                )
                self.fail = False

            def forward(self, **kwargs):
                assert not torch.is_grad_enabled()
                assert kwargs["use_cache"] is False
                self.attention()
                if self.fail:
                    raise RuntimeError("forward failed")

        core = Core()
        probe = SimpleNamespace(
            prepare=lambda sample: prepared, processor=SimpleNamespace(tokenizer=tokenizer),
            load_model=lambda: SimpleNamespace(model=core),
        )
        _, scores, _, positions, _ = evaluator.read_option_attention(probe, sample, 1)
        self.assertEqual(positions, [1, 3])
        np.testing.assert_allclose(scores, [[.18, .02], [.01, .04]], atol=1e-7)
        self.assertFalse(core.attention._forward_hooks)
        core.fail = True
        with self.assertRaisesRegex(RuntimeError, "forward failed"):
            evaluator.read_option_attention(probe, sample, 1)
        self.assertFalse(core.attention._forward_hooks)

    def test_no_gt_image_is_accounted_for_without_model_call(self):
        dataset = SimpleNamespace(coco=SimpleNamespace(
            imgToAnns={9: []}, imgs={9: {"file_name": "9.jpg"}},
        ))
        record = evaluator.evaluate_image(dataset, None, 9, 32, None)
        self.assertEqual(record["status"], "no_gt_masks")
        summary = evaluator.summarize([record], 1, 1, 32)
        self.assertTrue(summary["complete"])
        self.assertTrue(summary["full_dataset"])
        self.assertIsNone(summary["overall"]["accuracy"])

    def test_single_class_and_multiclass_metrics_are_separate(self):
        rows = [
            {"class": "car", "prediction": "car", "image_id": 1, "area": 100, "image_class_count": 1},
            {"class": "person", "prediction": "car", "image_id": 2, "area": 2000, "image_class_count": 2},
        ]
        records = [{"instances": rows, "status": "evaluated"}]
        summary = evaluator.summarize(records, 5000, 2, 32)
        self.assertEqual(summary["overall"]["accuracy"], .5)
        self.assertEqual(summary["single_class"]["accuracy"], 1)
        self.assertEqual(summary["multi_class"]["accuracy"], 0)
        self.assertEqual(summary["overall"]["uniform_guess"], .75)
        self.assertFalse(summary["full_dataset"])

    @unittest.skipUnless((evaluator.DEFAULT_EXPLORER / "probe.py").is_file(), "Gradio source unavailable")
    def test_gradio_prompt_is_reused_and_crowds_are_only_prompt_candidates(self):
        import json
        from PIL import Image

        explorer = evaluator.load_explorer(evaluator.DEFAULT_EXPLORER)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (4, 2)).save(root / "1.png")
            annotations = {
                "images": [{"id": 1, "file_name": "1.png", "width": 4, "height": 2}],
                "categories": [{"id": 1, "name": "car"}, {"id": 2, "name": "person"}],
                "annotations": [
                    {"id": 11, "image_id": 1, "category_id": 1, "area": 4, "bbox": [0, 0, 2, 2], "iscrowd": 0, "segmentation": [[0, 0, 2, 0, 2, 2, 0, 2]]},
                    {"id": 12, "image_id": 1, "category_id": 2, "area": 4, "bbox": [2, 0, 2, 2], "iscrowd": 1, "segmentation": [[2, 0, 4, 0, 4, 2, 2, 2]]},
                ],
            }
            annotation_path = root / "annotations.json"
            annotation_path.write_text(json.dumps(annotations))
            dataset = explorer.CocoImages(root, annotation_path)
            sample = dataset.load(1)
            self.assertEqual(sample.prompt, "Find every instance of the listed categories in the image. Match each object region to the label after its category name.\n\ncar = A\nperson = B")
            prepared = SimpleNamespace(
                image_start=4, inputs={"input_ids": torch.zeros(1, 6, dtype=torch.long)},
                grid=SimpleNamespace(rows=1, cols=2, x_edges=(0, 2, 4), y_edges=(0, 2)),
            )
            root.joinpath("patches").mkdir()
            with patch.object(evaluator, "read_option_attention", return_value=(
                prepared, np.array([[.03, .02], [.01, .04]]), np.ones(2), [1, 3], [11, 13],
            )):
                record = evaluator.evaluate_image(dataset, None, 1, 32, root / "patches")
            self.assertEqual(record["classes"], ["car", "person"])
            self.assertEqual(len(record["instances"]), 1)
            self.assertEqual(record["instances"][0]["prediction"], "car")
            with np.load(root / record["patch_file"]) as data:
                np.testing.assert_array_equal(data["votes"], [0, 1])
                np.testing.assert_allclose(data["non_option_attention_mass"], [.95, .95])


if __name__ == "__main__":
    unittest.main()
