from dataclasses import replace
from types import SimpleNamespace
from threading import Lock
import unittest

import numpy as np
import torch
from PIL import Image

from qwen_token_explorer.app import build_app
from qwen_token_explorer.probe import Category, Grid, Prepared, QwenProbe, Sample, Scores, VocabularyReadout, cosine_scores
from qwen_token_explorer.views import (HEAD_OUTPUTS, LAYER_HIDDEN, OPTION_COS, POST_RMS, TOKEN_ATTN, TOKEN_COS, VALUE_ONLY, frame_for,
                   relationship_values, relationship_views, selected_cosines, token_views, vocabulary_view)


def fixture():
    grid = Grid(2, 3, 96, 64, (0, 32, 64, 96), (0, 32, 64))
    mask = np.zeros((64, 96), dtype=bool)
    sample = Sample(1, Image.new("RGB", (96, 64), (70, 90, 100)), "cat = A",
                    (Category("cat", "A", (0, 3), (6, 7), mask, mask, 1),))
    prepared = Prepared({}, grid, 2, 0, ("cat", "A"), ((0,),), (1,))
    states = np.array([[1, 0, 0], [.8, .6, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [.6, .8, 0]],
                      dtype=np.float32)
    weights = np.tril(np.arange(1, 37, dtype=np.float32).reshape(6, 6)) / 1000
    image_attention = {4: np.stack([weights, weights * 2]), 8: np.stack([weights * 3, weights * 4])}
    prompt_attention = {layer: np.full((2, 6, 2), .01, np.float32) for layer in (4, 8)}
    full = torch.tensor(np.concatenate([[[1, 0, 0], [0, 1, 0]], states])).float()
    alternate = full * torch.tensor([1., 2., 3.]) + torch.tensor([.2, .1, .3])
    later = full + torch.tensor([.4, .2, .1])
    scores = Scores(
        prompt_attention, image_attention,
        {4: cosine_scores(full, prepared), 8: cosine_scores(later, prepared)},
        {4: cosine_scores(torch.stack([full, alternate]), prepared),
         8: cosine_scores(torch.stack([later, alternate + .4]), prepared)},
        post_rms_cosine={4: cosine_scores(alternate, prepared), 8: cosine_scores(later, prepared)},
        value_cosine={4: cosine_scores(torch.stack([alternate, later]), prepared)},
    )
    return replace(frame_for(sample, prepared), scores=scores), states


class RelationshipTests(unittest.TestCase):
    def setUp(self):
        self.frame, self.states = fixture()

    def test_cosine_values_and_ranking_exclude_source(self):
        values = relationship_values(self.frame, 0, TOKEN_COS, "4", "0")
        np.testing.assert_allclose(values, [1, .8, 0, 0, -1, .6])
        image, legend, neighbors = relationship_views(self.frame, 0, TOKEN_COS, "4", "0", True)
        self.assertEqual(image.size, (96, 64))
        self.assertIn("source token 0", legend)
        self.assertIn("<td>1</td><td>0</td><td>1</td><td>+0.8000</td>", neighbors)
        self.assertNotIn("<td>0</td><td>0</td><td>0</td>", neighbors)
        self.assertIn("<td>4</td><td>1</td><td>1</td><td>-1.0000</td>", neighbors)

    def test_attention_direction_layer_head_and_mean(self):
        expected = self.frame.scores.image_attention[8][1, 3, :]
        np.testing.assert_array_equal(relationship_values(self.frame, 3, TOKEN_ATTN, "8", "1"), expected)
        mean = relationship_values(self.frame, 3, TOKEN_ATTN, "8", "Mean")
        np.testing.assert_allclose(mean, self.frame.scores.image_attention[8][:, 3, :].mean(axis=0))
        np.testing.assert_array_equal(mean[4:], 0)
        self.assertFalse(np.array_equal(expected, self.frame.scores.image_attention[8][1, :, 3]))
        self.assertFalse(np.array_equal(expected, relationship_values(self.frame, 3, TOKEN_ATTN, "4", "1")))

    def test_attention_neighbors_and_first_token(self):
        _, _, first = relationship_views(self.frame, 0, TOKEN_ATTN, "4", "Mean", True)
        self.assertIn("no earlier image tokens", first)
        _, _, neighbors = relationship_views(self.frame, 3, TOKEN_ATTN, "8", "1", True)
        self.assertIn("layer 8 · head 1", neighbors)
        self.assertIn("causally masked", neighbors)
        self.assertNotIn("<td>4</td>", neighbors)
        self.assertNotIn("<td>5</td>", neighbors)

    def test_cosine_changes_with_source_layer_and_head(self):
        first = relationship_values(self.frame, 0, TOKEN_COS, "4", "0")
        self.assertFalse(np.allclose(first, relationship_values(self.frame, 0, TOKEN_COS, "8", "0")))
        self.assertFalse(np.allclose(first, relationship_values(self.frame, 0, TOKEN_COS, "4", "1")))
        self.assertFalse(np.array_equal(first, relationship_values(self.frame, 5, TOKEN_COS, "4", "0")))
        for token in (0, 3, 5):
            self.assertAlmostEqual(float(relationship_values(self.frame, token, TOKEN_COS, "4", "0")[token]), 1)

    def test_mean_pools_cosines_not_vectors(self):
        maps = [relationship_values(self.frame, 0, TOKEN_COS, "4", str(head)) for head in (0, 1)]
        actual = relationship_values(self.frame, 0, TOKEN_COS, "4", "Mean")
        np.testing.assert_allclose(actual, np.mean(maps, axis=0), atol=1e-7)
        vectors = self.frame.scores.head_cosine[4].image_vectors.mean(axis=0)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        self.assertFalse(np.allclose(actual, vectors @ vectors[0]))
        word, letter = selected_cosines(self.frame, "4", "Mean", HEAD_OUTPUTS)
        np.testing.assert_allclose(word, self.frame.scores.head_cosine[4].word_cosine.mean(axis=0))
        np.testing.assert_allclose(letter, self.frame.scores.head_cosine[4].letter_cosine.mean(axis=0))

    def test_layer_hidden_state_uses_layer_but_combines_heads(self):
        first = relationship_values(self.frame, 0, TOKEN_COS, "4", "0", LAYER_HIDDEN)
        np.testing.assert_array_equal(first, relationship_values(self.frame, 0, TOKEN_COS, "4", "Mean", LAYER_HIDDEN))
        self.assertFalse(np.allclose(first, relationship_values(self.frame, 0, TOKEN_COS, "8", "0", LAYER_HIDDEN)))
        head_cos = selected_cosines(self.frame, "4", "1", HEAD_OUTPUTS)
        layer_cos = selected_cosines(self.frame, "4", "1", LAYER_HIDDEN)
        self.assertFalse(np.allclose(head_cos[0], layer_cos[0]))

    def test_waiting_and_ready_views_have_correct_output_counts(self):
        args = (3, OPTION_COS, "0", "4", "Mean", True, True, False, TOKEN_COS)
        waiting = token_views(replace(self.frame, scores=None), *args)
        ready = token_views(self.frame, *args)
        self.assertEqual(len(waiting), 11)
        self.assertEqual(len(ready), 11)
        self.assertIsNone(waiting[7])
        self.assertEqual(ready[7].size, (96, 64))
        self.assertIn("Token 3", ready[4])

    def test_post_rms_view_uses_the_mlp_input_for_both_cosines(self):
        normalized = self.frame.scores.post_rms_cosine[4]
        word, letter = selected_cosines(self.frame, "4", "0", POST_RMS)
        np.testing.assert_array_equal(word, normalized.word_cosine)
        np.testing.assert_array_equal(letter, normalized.letter_cosine)
        cosine = relationship_values(self.frame, 2, TOKEN_COS, "4", "1", POST_RMS)
        expected = np.clip(normalized.image_vectors @ normalized.image_vectors[2], -1, 1)
        np.testing.assert_allclose(cosine, expected)
        np.testing.assert_array_equal(cosine, relationship_values(self.frame, 2, TOKEN_COS, "4", "0", POST_RMS))
        self.assertFalse(np.allclose(cosine, relationship_values(self.frame, 2, TOKEN_COS, "4", "0", LAYER_HIDDEN)))

    def test_value_only_selects_shared_kv_head_and_averages_scores(self):
        scores = self.frame.scores
        grouped = replace(scores, attention={4: np.repeat(scores.attention[4], 2, axis=0)})
        frame = replace(self.frame, scores=grouped)
        values = scores.value_cosine[4]
        maps = []
        for query_head, kv_head in enumerate((0, 0, 1, 1)):
            word, letter = selected_cosines(frame, "4", str(query_head), VALUE_ONLY)
            np.testing.assert_array_equal(word, values.word_cosine[kv_head])
            np.testing.assert_array_equal(letter, values.letter_cosine[kv_head])
            vectors = values.image_vectors[kv_head]
            similarity = relationship_values(frame, 2, TOKEN_COS, "4", str(query_head), VALUE_ONLY)
            np.testing.assert_allclose(similarity, np.clip(vectors @ vectors[2], -1, 1))
            maps.append(similarity)
        self.assertFalse(np.allclose(maps[0], maps[2]))
        mean_map = relationship_values(frame, 2, TOKEN_COS, "4", "Mean", VALUE_ONLY)
        np.testing.assert_allclose(mean_map, (maps[0] + maps[2]) / 2, atol=1e-7)
        word, letter = selected_cosines(frame, "4", "Mean", VALUE_ONLY)
        np.testing.assert_allclose(word, values.word_cosine.mean(axis=0))
        np.testing.assert_allclose(letter, values.letter_cosine.mean(axis=0))

    def test_gradio_draw_and_click_callbacks_share_one_forward(self):
        calls = []
        dataset = SimpleNamespace(random_image=lambda previous: self.frame.sample)
        probe = SimpleNamespace(full_layers=(4, 8), head_count=2, layer_count=8, model=True,
                                prepare=lambda sample: self.frame.prepared,
                                run=lambda prepared: calls.append(prepared) or self.frame.scores,
                                vocabulary=lambda scores, token, layer: VocabularyReadout(
                                    layer, token, np.arange(10), tuple(f"token {i}" for i in range(10)),
                                    np.full(10, .03), 17, 1.))
        app = build_app(dataset, probe)
        self.addCleanup(app.close)
        events = app.config["dependencies"]
        draw = app.fns[events[0]["id"]]
        updates = list(draw.fn(None, True, True, False))
        self.assertEqual(len(updates), 2)
        self.assertTrue(all(len(update) == len(draw.outputs) for update in updates))
        self.assertEqual(len(calls), 1)
        components = app.config["components"]
        basis = next(c for c in components if c["props"].get("elem_id") == "cosine-representation")
        self.assertEqual(basis["props"]["value"], HEAD_OUTPUTS)
        choices = [value for _, value in basis["props"]["choices"]]
        self.assertEqual(choices, [HEAD_OUTPUTS, LAYER_HIDDEN, POST_RMS, VALUE_ONLY])
        self.assertEqual(updates[-1][events[0]["outputs"].index(basis["id"])]["value"], HEAD_OUTPUTS)
        self.assertIn("mean of head cosine scores", next(v for v in updates[-1] if isinstance(v, str) and "Cosine:" in v))
        relationship_id = next(c["id"] for c in components
                               if c["props"].get("elem_id") == "image-relationship-map")
        event = next(e for e in events if any(t[0] == relationship_id and t[1] == "select"
                                              for t in e["targets"]))
        selected = app.fns[event["id"]].fn(
            updates[-1][0], OPTION_COS, "0", "4", "Mean", True, True, False, TOKEN_COS,
            HEAD_OUTPUTS, "Final",
            SimpleNamespace(index=(70, 45)),
        )
        self.assertEqual(len(selected), len(event["outputs"]))
        self.assertEqual(selected[0], 5)
        self.assertIn("Token 5", selected[5])
        self.assertIn("Source token <b>5</b>", selected[10])
        self.assertIn("Image token 5", selected[12])
        self.assertEqual(len(calls), 1)

    def test_probe_captures_image_query_to_image_key_and_removes_hooks(self):
        probe, layers = fake_probe(self.states)
        result = probe.run(self.frame.prepared)
        for number, layer in enumerate(layers, start=1):
            np.testing.assert_array_equal(result.image_attention[number], layer.self_attn.weights[0, :, 2:8, 2:8])
            np.testing.assert_array_equal(result.attention[number], layer.self_attn.weights[0, :, 2:8, :2])
            np.testing.assert_allclose(np.linalg.norm(result.head_cosine[number].image_vectors, axis=-1), 1, atol=1e-6)
            expected_heads = layer.self_attn.joined[0].reshape(8, 2, 3).transpose(0, 1)
            expected = cosine_scores(expected_heads, self.frame.prepared)
            np.testing.assert_allclose(result.head_cosine[number].image_vectors, expected.image_vectors)
            expected_layer = cosine_scores(layer.last_output[0], self.frame.prepared)
            np.testing.assert_allclose(result.layer_cosine[number].word_cosine, expected_layer.word_cosine)
            expected_norm = cosine_scores(layer.mlp_input[0], self.frame.prepared)
            actual_norm = result.post_rms_cosine[number]
            np.testing.assert_allclose(actual_norm.image_vectors, expected_norm.image_vectors)
            np.testing.assert_allclose(actual_norm.word_cosine, expected_norm.word_cosine)
            np.testing.assert_allclose(actual_norm.letter_cosine, expected_norm.letter_cosine)
            raw_residual = cosine_scores(layer.pre_mlp[0], self.frame.prepared)
            self.assertFalse(np.allclose(actual_norm.image_vectors, raw_residual.image_vectors))
            values = layer.self_attn.values[0].unsqueeze(0)
            expected_values = cosine_scores(values, self.frame.prepared)
            actual_values = result.value_cosine[number]
            np.testing.assert_allclose(actual_values.image_vectors, expected_values.image_vectors)
            np.testing.assert_allclose(actual_values.word_cosine, expected_values.word_cosine)
            np.testing.assert_allclose(actual_values.letter_cosine, expected_values.letter_cosine)
            self.assertEqual(len(layer.self_attn._forward_hooks), 0)
            self.assertEqual(len(layer.self_attn.o_proj._forward_pre_hooks), 0)
            self.assertEqual(len(layer.post_attention_layernorm._forward_hooks), 0)
            self.assertEqual(len(layer.self_attn.v_proj._forward_hooks), 0)
            self.assertEqual(len(layer._forward_hooks), 0)
        self.assertEqual(probe.model.model.calls, 1)
        self.assertFalse(np.allclose(result.head_cosine[1].image_vectors, result.head_cosine[2].image_vectors))

    def test_probe_cleans_all_hooks_after_failure(self):
        probe, layers = fake_probe(self.states, fail=True)
        with self.assertRaisesRegex(RuntimeError, "test forward failure"):
            probe.run(self.frame.prepared)
        for layer in layers:
            self.assertFalse(layer._forward_hooks)
            self.assertFalse(layer.self_attn._forward_hooks)
            self.assertFalse(layer.self_attn.o_proj._forward_pre_hooks)
            self.assertFalse(layer.post_attention_layernorm._forward_hooks)
            self.assertFalse(layer.self_attn.v_proj._forward_hooks)

    def test_vocabulary_uses_full_softmax_and_caches_top_ten(self):
        probe, _ = fake_probe(self.states)
        scores = probe.run(self.frame.prepared)
        before = probe.model.model.calls
        result = probe.vocabulary(scores, 0, "Final")
        hidden = torch.from_numpy(scores.vocabulary_states["Final"][0])
        expected = torch.softmax(probe.model.lm_head.weight.detach() @ hidden, dim=-1)
        np.testing.assert_allclose(result.probabilities, expected[result.token_ids], atol=1e-7)
        self.assertEqual(len(result.token_ids), 10)
        self.assertEqual(result.vocabulary_size, 17)
        self.assertLess(float(result.probabilities.sum()), 1.)
        self.assertAlmostEqual(result.total_probability, 1., places=6)
        self.assertIs(result, probe.vocabulary(scores, 0, "Final"))
        self.assertEqual(probe.model.model.calls, before)
        self.assertIn("Softmax over 17 vocabulary entries", vocabulary_view(result))

    def test_vocabulary_final_skips_norm_and_last_layer_matches_final(self):
        probe, _ = fake_probe(self.states)
        scores = probe.run(self.frame.prepared)
        norm = probe.model.model.language_model.norm
        before = norm.calls
        final = probe.vocabulary(scores, 2, "Final")
        self.assertEqual(norm.calls, before)
        last = probe.vocabulary(scores, 2, "2")
        self.assertEqual(norm.calls, before + 1)
        np.testing.assert_array_equal(final.token_ids, last.token_ids)
        np.testing.assert_allclose(final.probabilities, last.probabilities, atol=1e-7)
        self.assertIn("Layer 2 · logit lens", vocabulary_view(last))

    def test_vocabulary_captures_layers_without_full_attention(self):
        probe, _ = fake_probe(self.states)
        probe.full_layers = (2,)
        scores = probe.run(self.frame.prepared)
        self.assertEqual(set(scores.vocabulary_states), {"1", "2", "Final"})
        self.assertEqual(set(scores.layer_cosine), {2})
        first = probe.vocabulary(scores, 0, "1")
        final = probe.vocabulary(scores, 0, "Final")
        self.assertFalse(np.allclose(first.probabilities, final.probabilities))


def fake_probe(states, fail=False):
    hidden = torch.tensor(np.concatenate([[[1, 0, 0], [0, 1, 0]], states], axis=0)).float().unsqueeze(0)

    class Attention(torch.nn.Module):
        def __init__(self, shift):
            super().__init__()
            self.shift = shift
            logits = torch.arange(128, dtype=torch.float32).reshape(1, 2, 8, 8) / (20 + shift)
            self.weights = logits.softmax(dim=-1)
            self.v_proj = torch.nn.Linear(3, 3, bias=False)
            self.o_proj = torch.nn.Linear(6, 3, bias=False)
            with torch.no_grad():
                self.v_proj.weight.copy_(torch.diag(torch.tensor([.5, 2., 3.])))
                self.o_proj.weight.copy_(torch.arange(18).reshape(3, 6) / 100)

        def forward(self, hidden):
            # Deliberately different per-head vectors to catch the wrong capture site.
            self.values = self.v_proj(hidden)
            self.joined = torch.cat([hidden + self.shift * .1, self.values + .3], dim=-1)
            return self.o_proj(self.joined), self.weights

    class Block(torch.nn.Module):
        def __init__(self, shift):
            super().__init__()
            self.self_attn = Attention(shift)
            self.post_attention_layernorm = torch.nn.LayerNorm(3)

        def forward(self, hidden):
            attention, _ = self.self_attn(hidden)
            self.pre_mlp = hidden + attention
            self.mlp_input = self.post_attention_layernorm(self.pre_mlp)
            mlp_update = self.mlp_input * torch.tensor([.1, .2, .3])
            self.last_output = self.pre_mlp + mlp_update
            return self.last_output

    layers = [Block(i) for i in range(2)]

    class Norm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.2, .8, 1.1]), requires_grad=False)
            self.calls = 0

        def forward(self, hidden):
            self.calls += 1
            return hidden * self.weight

    class Body:
        language_model = SimpleNamespace(layers=layers, norm=Norm())
        calls = 0

        def __call__(self, **kwargs):
            self.calls += 1
            output = hidden
            for layer in layers:
                output = layer(output)
            if fail:
                raise RuntimeError("test forward failure")
            return SimpleNamespace(last_hidden_state=self.language_model.norm(output))

    probe = object.__new__(QwenProbe)
    lm_head = torch.nn.Linear(3, 17, bias=False)
    with torch.no_grad():
        lm_head.weight.copy_(torch.linspace(-.2, .2, 51).reshape(17, 3))
    probe.model = SimpleNamespace(model=Body(), lm_head=lm_head)
    probe.full_layers = (1, 2)
    probe.head_count = 2
    probe.kv_head_count = 1
    probe.layer_count = 2
    probe._vocabulary_lock = Lock()
    probe.processor = SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda ids, **kwargs: f"token {ids[0]}"))
    return probe, layers


if __name__ == "__main__":
    unittest.main()
