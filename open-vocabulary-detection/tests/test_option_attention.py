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

    def test_multi_character_label_can_be_one_token(self):
        prepared, sample, _, _ = fixture()
        sample.categories[1].letter = "AB"
        tokenizer = SimpleNamespace(decode=lambda ids: {11: " A", 13: " AB"}[ids[0]])
        self.assertEqual(evaluator.option_positions(sample, prepared, tokenizer), ([1, 3], [11, 13]))

    def test_seeded_shuffle_reassigns_every_option_and_preserves_the_token_set(self):
        labels = [f"option-{index}" for index in range(80)]
        shuffled = evaluator.shuffle_options(labels, 42)
        self.assertEqual(shuffled, evaluator.shuffle_options(labels, 42))
        self.assertCountEqual(shuffled, labels)
        self.assertTrue(all(a != b for a, b in zip(shuffled, labels, strict=True)))

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

    def test_per_head_predictions_match_independent_instance_votes(self):
        prepared, _, _, weights = fixture()
        regions = [
            ({"id": 2, "category_id": 1}, np.array([.9, .1])),
            ({"id": 3, "category_id": 2}, np.array([.5, .5])),
        ]
        coverage = torch.tensor(np.stack([weight for _, weight in regions]))
        choices, support = evaluator.head_readout(weights[0], prepared, torch.tensor([1, 3]), coverage)
        coco = SimpleNamespace(cats={1: {"name": "car"}, 2: {"name": "person"}})
        record = {"image_id": 1, "classes": ["car", "person"], "status": "evaluated"}
        output = evaluator.evaluate_layer_heads(record, coco, regions, choices.numpy(), support.numpy())
        scores = weights[0, :, 4:, :][:, :, [1, 3]].numpy()
        for region, (_, fraction) in zip(output["instances"], regions, strict=True):
            expected = [record["classes"][evaluator.instance_vote(
                head, evaluator.rank_options(head), fraction,
            )[0]] for head in scores]
            self.assertEqual(region["head_predictions"], expected)
        summary = evaluator.summarize_heads([output], 1, 1, 4, 3)
        self.assertEqual(len(summary["by_head"]), 3)
        self.assertEqual(summary["overall"]["accuracy"], .5)

    def test_qwen_batch_right_pads_text_and_concatenates_vision_in_order(self):
        def image(ids, patches):
            return SimpleNamespace(inputs={
                "input_ids": torch.tensor([ids]),
                "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
                "mm_token_type_ids": torch.ones(1, len(ids), dtype=torch.long),
                "pixel_values": torch.full((patches, 2), patches),
                "image_grid_thw": torch.tensor([[1, 1, patches]]),
            })

        batch = evaluator.collate_inputs([image([10, 11], 2), image([21], 3)], SimpleNamespace(pad_token_id=99))
        self.assertEqual(batch["input_ids"].tolist(), [[10, 11], [21, 99]])
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1], [1, 0]])
        self.assertEqual(batch["mm_token_type_ids"].tolist(), [[1, 1], [1, 0]])
        self.assertEqual(batch["pixel_values"][:, 0].tolist(), [2, 2, 3, 3, 3])
        self.assertEqual(batch["image_grid_thw"].tolist(), [[1, 1, 2], [1, 1, 3]])

    def test_linear_patch_projection_uses_the_original_convolution_weights(self):
        projection = torch.nn.Conv3d(3, 5, kernel_size=(2, 4, 4), stride=(2, 4, 4))
        visual = SimpleNamespace(patch_embed=SimpleNamespace(proj=projection))
        patches = torch.randn(7, 3, 2, 4, 4)
        expected = projection(patches).flatten(1)
        evaluator.use_linear_patch_projection(visual)
        torch.testing.assert_close(visual.patch_embed.proj(patches), expected)

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

    def test_inference_captures_multiple_layers_in_one_pass_and_removes_hooks(self):
        prepared, sample, tokenizer, weights = fixture()

        class Attention(torch.nn.Module):
            def __init__(self, weights):
                super().__init__()
                self.weights = weights

            def forward(self):
                return None, self.weights

        class Core(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(16, 2)
                self.attentions = torch.nn.ModuleList([
                    Attention(weights), Attention(weights.flip(-1)),
                ])
                self.language_model = SimpleNamespace(
                    layers=[SimpleNamespace(self_attn=attention) for attention in self.attentions],
                    embed_tokens=self.embed_tokens,
                )
                self.calls = 0
                self.fail = False

            def forward(self, **kwargs):
                assert not torch.is_grad_enabled()
                assert kwargs["use_cache"] is False
                self.calls += 1
                self.attentions[0]()
                if self.fail:
                    raise RuntimeError("forward failed")
                self.attentions[1]()

        core = Core()
        probe = SimpleNamespace(
            prepare=lambda sample: prepared, processor=SimpleNamespace(tokenizer=tokenizer),
            load_model=lambda: SimpleNamespace(model=core),
        )
        _, captured, positions, _ = evaluator.read_option_attention(probe, sample, (1, 2))
        self.assertEqual(positions, [1, 3])
        self.assertEqual(core.calls, 1)
        self.assertEqual(set(captured), {1, 2})
        np.testing.assert_allclose(captured[1][0], [[.18, .02], [.01, .04]], atol=1e-7)
        np.testing.assert_array_equal(captured[2][0], np.zeros((2, 2)))
        self.assertTrue(all(not attention._forward_hooks for attention in core.attentions))
        core.fail = True
        with self.assertRaisesRegex(RuntimeError, "forward failed"):
            evaluator.read_option_attention(probe, sample, (1, 2))
        self.assertTrue(all(not attention._forward_hooks for attention in core.attentions))

    def test_no_gt_image_is_accounted_for_without_model_call(self):
        dataset = SimpleNamespace(coco=SimpleNamespace(
            imgToAnns={9: []}, imgs={9: {"file_name": "9.jpg"}},
        ))
        record = evaluator.evaluate_image(dataset, None, 9, {32: None})[32]
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
        for row in rows:
            row["candidate_class_count"] = 80
        summary = evaluator.summarize(records, 5000, 2, 32)
        self.assertEqual(summary["overall"]["uniform_guess"], 1 / 80)
        self.assertEqual(summary["single_class"]["total"], 1)
        self.assertEqual(summary["multi_class"]["total"], 1)

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
                image_start=4, prompt_start=0, letters=(1, 3),
                inputs={"input_ids": torch.tensor([[0, 11, 0, 13, 0, 0]])},
                grid=SimpleNamespace(rows=1, cols=2, x_edges=(0, 2, 4), y_edges=(0, 2)),
            )
            tokenizer = SimpleNamespace(decode=lambda ids: {11: "A", 13: "B"}[ids[0]])
            probe = SimpleNamespace(prepare=lambda _: prepared, processor=SimpleNamespace(tokenizer=tokenizer))
            patch_dirs = {layer: root / f"layer_{layer:02d}" / "patches" for layer in (4, 8)}
            for directory in patch_dirs.values():
                directory.mkdir(parents=True)
            with patch.object(evaluator, "forward_option_attention", return_value=[
                {4: (np.array([[.03, .02], [.01, .04]]), np.ones(2)),
                 8: (np.array([[.02, .03], [.06, .01]]), np.ones(2))},
            ]):
                records = evaluator.evaluate_image(dataset, probe, 1, patch_dirs)
            record = records[4]
            self.assertEqual(record["classes"], ["car", "person"])
            self.assertEqual(len(record["instances"]), 1)
            self.assertEqual(record["instances"][0]["prediction"], "car")
            self.assertEqual(records[8]["instances"][0]["prediction"], "person")
            with np.load(patch_dirs[4].parent / record["patch_file"]) as data:
                np.testing.assert_array_equal(data["votes"], [0, 1])
                np.testing.assert_allclose(data["non_option_attention_mass"], [.95, .95])
            with np.load(patch_dirs[8].parent / records[8]["patch_file"]) as data:
                np.testing.assert_array_equal(data["votes"], [1, 0])


if __name__ == "__main__":
    unittest.main()
