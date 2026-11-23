from __future__ import annotations

import random
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from string import ascii_uppercase
from threading import Lock

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from transformers import AutoConfig, AutoProcessor, Qwen3_5ForConditionalGeneration


@dataclass(frozen=True)
class Category:
    name: str
    letter: str
    name_span: tuple[int, int]
    letter_span: tuple[int, int]
    mask: np.ndarray
    outline: np.ndarray
    instances: int


@dataclass(frozen=True)
class Sample:
    image_id: int
    image: Image.Image
    prompt: str
    categories: tuple[Category, ...]


class CocoImages:
    def __init__(self, images: Path, annotations: Path):
        self.coco = COCO(str(annotations))
        self.images = images
        self.image_ids = [
            image_id
            for image_id, records in self.coco.imgToAnns.items()
            if any(ann.get("segmentation") for ann in records)
            and (images / self.coco.imgs[image_id]["file_name"]).is_file()
        ]
        if not self.image_ids:
            raise ValueError(f"No images with segmentation annotations found in {images}")

    def random_image(self, previous_id: int | None) -> Sample:
        image_id = random.choice(self.image_ids)
        while image_id == previous_id and len(self.image_ids) > 1:
            image_id = random.choice(self.image_ids)
        return self.load(image_id)

    def load(self, image_id: int) -> Sample:
        record = self.coco.imgs[image_id]
        with Image.open(self.images / record["file_name"]) as source:
            image = source.convert("RGB")
        masks: dict[str, np.ndarray] = {}
        outlines: dict[str, np.ndarray] = {}
        counts: dict[str, int] = {}
        for ann in self.coco.imgToAnns[image_id]:
            if not ann.get("segmentation"):
                continue
            name = self.coco.cats[ann["category_id"]]["name"]
            mask = self.coco.annToMask(ann).astype(bool)
            if not mask.any():
                continue
            padded = np.pad(mask, 1)
            boundary = mask & ~(
                padded[:-2, 1:-1] & padded[2:, 1:-1] &
                padded[1:-1, :-2] & padded[1:-1, 2:]
            )
            if name in masks:
                masks[name] |= mask
                outlines[name] |= boundary
                counts[name] += 1
            else:
                masks[name] = mask
                outlines[name] = boundary
                counts[name] = 1

        prompt = (
            "Find every instance of the listed categories in the image. "
            "Match each object region to the label after its category name.\n"
        )
        categories = []
        for name, letter in zip(sorted(masks), ascii_uppercase):
            prompt += "\n"
            start = len(prompt)
            prompt += f"{name} = {letter}"
            categories.append(Category(
                name, letter, (start, start + len(name)), (len(prompt) - 1, len(prompt)),
                masks[name], outlines[name], counts[name],
            ))
        if not categories or len(categories) != len(masks):
            raise ValueError(f"Image {image_id} needs between 1 and 26 masked categories")
        return Sample(image_id, image, prompt, tuple(categories))


@dataclass(frozen=True)
class Grid:
    rows: int
    cols: int
    processed_width: int
    processed_height: int
    x_edges: tuple[int, ...]
    y_edges: tuple[int, ...]

    @property
    def count(self) -> int:
        return self.rows * self.cols

    def bounds(self, index: int) -> tuple[int, int, int, int]:
        row, col = divmod(index, self.cols)
        return self.x_edges[col], self.y_edges[row], self.x_edges[col + 1], self.y_edges[row + 1]

    def index_at(self, x: float, y: float) -> int:
        col = min(self.cols - 1, max(0, bisect_right(self.x_edges, x) - 1))
        row = min(self.rows - 1, max(0, bisect_right(self.y_edges, y) - 1))
        return row * self.cols + col


@dataclass(frozen=True)
class Prepared:
    inputs: dict
    grid: Grid
    image_start: int
    prompt_start: int
    prompt_tokens: tuple[str, ...]
    names: tuple[tuple[int, ...], ...]
    letters: tuple[int, ...]


@dataclass(frozen=True)
class CosineScores:
    word_cosine: np.ndarray
    letter_cosine: np.ndarray
    image_vectors: np.ndarray


def cosine_scores(states: torch.Tensor, prepared: Prepared) -> CosineScores:
    """Cache normalized image vectors and class scores; leading axes may be heads."""
    image_rows = slice(prepared.image_start, prepared.image_start + prepared.grid.count)
    prompt_cols = slice(prepared.prompt_start, prepared.prompt_start + len(prepared.prompt_tokens))
    image = torch.nn.functional.normalize(states[..., image_rows, :], dim=-1)
    prompt = states[..., prompt_cols, :]
    words = torch.stack([prompt[..., list(span), :].mean(dim=-2) for span in prepared.names], dim=-2)
    letters = prompt[..., list(prepared.letters), :]
    return CosineScores(
        (image @ torch.nn.functional.normalize(words, dim=-1).transpose(-1, -2)).clamp(-1, 1).numpy(),
        (image @ torch.nn.functional.normalize(letters, dim=-1).transpose(-1, -2)).clamp(-1, 1).numpy(),
        image.numpy(),
    )


@dataclass(frozen=True)
class VocabularyReadout:
    layer: str
    image_token: int
    token_ids: np.ndarray
    tokens: tuple[str, ...]
    probabilities: np.ndarray
    vocabulary_size: int
    total_probability: float


@dataclass(frozen=True)
class Scores:
    attention: dict[int, np.ndarray]
    image_attention: dict[int, np.ndarray]
    layer_cosine: dict[int, CosineScores]
    head_cosine: dict[int, CosineScores]
    vocabulary_states: dict[str, np.ndarray] = field(default_factory=dict)
    vocabulary_cache: OrderedDict = field(default_factory=OrderedDict, compare=False, repr=False)
    post_rms_cosine: dict[int, CosineScores] = field(default_factory=dict)
    value_cosine: dict[int, CosineScores] = field(default_factory=dict)


class QwenProbe:
    def __init__(self, checkpoint: Path):
        self.checkpoint = checkpoint
        self.config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
        self.processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
        self.full_layers = tuple(
            i + 1 for i, kind in enumerate(self.config.text_config.layer_types)
            if kind == "full_attention"
        )
        self.head_count = self.config.text_config.num_attention_heads
        self.kv_head_count = self.config.text_config.num_key_value_heads
        self.layer_count = self.config.text_config.num_hidden_layers
        self._vocabulary_lock = Lock()
        self.model = None

    def prepare(self, sample: Sample) -> Prepared:
        messages = [{"role": "user", "content": [
            {"type": "text", "text": sample.prompt}, {"type": "image"},
        ]}]
        rendered = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        inputs = self.processor(
            text=[rendered], images=[sample.image], add_special_tokens=False, return_tensors="pt",
        )
        tokenizer = self.processor.tokenizer
        text_only = tokenizer(rendered, return_offsets_mapping=True, add_special_tokens=False)
        ids = inputs["input_ids"][0].tolist()
        image_positions = [i for i, token in enumerate(ids) if token == self.config.image_token_id]
        if not image_positions:
            raise ValueError("Qwen's processor did not produce image tokens")
        image_start = image_positions[0]
        if ids[:image_start] != text_only["input_ids"][:image_start]:
            raise ValueError("Image preprocessing changed the prompt token prefix")
        if image_positions != list(range(image_start, image_start + len(image_positions))):
            raise ValueError("Expected one contiguous block of image tokens")

        offset = rendered.index(sample.prompt)
        offsets = text_only["offset_mapping"][:image_start]

        def positions(span: tuple[int, int]) -> tuple[int, ...]:
            lo, hi = span
            return tuple(i for i, (start, end) in enumerate(offsets)
                         if start < offset + hi and end > offset + lo)

        prompt = positions((0, len(sample.prompt)))
        names = tuple(positions(category.name_span) for category in sample.categories)
        letter_positions = tuple(positions(category.letter_span) for category in sample.categories)
        if any(not span for span in names) or any(len(span) != 1 for span in letter_positions):
            raise ValueError("Could not locate a class name or a single-token option in the prompt")
        prompt_start = prompt[0]
        prompt_end = prompt[-1] + 1
        prompt_tokens = tuple(repr(tokenizer.decode([token])) for token in ids[prompt_start:prompt_end])

        frames, height, width = inputs["image_grid_thw"][0].tolist()
        patch_size = self.config.vision_config.patch_size
        merge = self.config.vision_config.spatial_merge_size
        rows, cols = height // merge, width // merge
        if frames != 1 or rows * cols != len(image_positions):
            raise ValueError("Merged token grid does not match the number of image tokens")
        grid = Grid(
            rows, cols, width * patch_size, height * patch_size,
            tuple(i * sample.image.width // cols for i in range(cols + 1)),
            tuple(i * sample.image.height // rows for i in range(rows + 1)),
        )
        return Prepared(
            dict(inputs), grid, image_start, prompt_start, prompt_tokens,
            tuple(tuple(i - prompt_start for i in span) for span in names),
            tuple(span[0] - prompt_start for span in letter_positions),
        )

    def load_model(self):
        if self.model is not None:
            return self.model
        count = torch.cuda.device_count()
        if not count:
            raise RuntimeError("The probe needs at least one visible CUDA GPU")
        free_gib = [torch.cuda.mem_get_info(i)[0] / 2**30 for i in range(count)]
        minimum = 4 + 7 / count
        if min(free_gib) < minimum:
            raise RuntimeError(f"Need {minimum:.1f} GiB free per visible GPU to load the model. "
                               f"Found {', '.join(f'{x:.1f}' for x in free_gib)} GiB.")
        layers = self.config.text_config.num_hidden_layers
        device_map = {
            "model.visual": 0,
            "model.language_model.embed_tokens": 0,
            "model.language_model.rotary_emb": 0,
            "model.language_model.norm": count - 1,
            "lm_head": 0,
            **{f"model.language_model.layers.{i}": i * count // layers for i in range(layers)},
        }
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            self.checkpoint, local_files_only=True, dtype=torch.bfloat16,
            device_map=device_map, attn_implementation="eager",
        ).eval()
        self.model = model.requires_grad_(False)
        print("Decoder devices:", sorted({str(next(layer.parameters()).device)
                                          for layer in model.model.language_model.layers}), flush=True)
        return self.model

    @torch.inference_mode()
    def run(self, prepared: Prepared) -> Scores:
        model = self.load_model()
        image_rows = slice(prepared.image_start, prepared.image_start + prepared.grid.count)
        prompt_cols = slice(prepared.prompt_start, prepared.prompt_start + len(prepared.prompt_tokens))
        attention = {}
        image_attention = {}
        layer_cosine = {}
        head_cosine = {}
        post_rms_cosine = {}
        value_cosine = {}
        vocabulary_states = {}

        def capture(layer):
            def hook(_module, _args, output):
                weights = output[1]
                if weights is None:
                    raise RuntimeError(f"Layer {layer} did not return eager attention weights")
                attention[layer] = weights[0, :, image_rows, prompt_cols].float().cpu().numpy()
                image_attention[layer] = weights[0, :, image_rows, image_rows].float().cpu().numpy()
            return hook

        def capture_layer(layer):
            def hook(_module, _args, output):
                hidden = output[0] if isinstance(output, tuple) else output
                if layer in self.full_layers:
                    cpu_hidden = hidden[0].float().cpu()
                    layer_cosine[layer] = cosine_scores(cpu_hidden, prepared)
                    vocabulary_states[str(layer)] = cpu_hidden[image_rows].numpy().copy()
                else:
                    vocabulary_states[str(layer)] = hidden[0, image_rows].float().cpu().numpy()
            return hook

        def capture_heads(layer):
            def hook(_module, args):
                # Qwen passes concatenated, gated head outputs into o_proj.
                # Split here, before the learned output projection mixes heads.
                joined = args[0][0].float().cpu()
                heads = joined.reshape(joined.shape[0], self.head_count, -1).transpose(0, 1)
                head_cosine[layer] = cosine_scores(heads, prepared)
            return hook

        def capture_post_rms(layer):
            def hook(_module, _args, output):
                # Qwen feeds the output of this norm directly to the MLP.
                post_rms_cosine[layer] = cosine_scores(output[0].float().cpu(), prepared)
            return hook

        def capture_values(layer):
            def hook(_module, _args, output):
                # v_proj returns each token's values before grouped-query expansion or attention.
                projected = output[0].float().cpu()
                values = projected.reshape(projected.shape[0], self.kv_head_count, -1).transpose(0, 1)
                value_cosine[layer] = cosine_scores(values, prepared)
            return hook

        layers = model.model.language_model.layers
        hooks = []
        try:
            for layer in self.full_layers:
                block = layers[layer - 1]
                hooks.append(block.self_attn.register_forward_hook(capture(layer)))
                hooks.append(block.self_attn.o_proj.register_forward_pre_hook(capture_heads(layer)))
                hooks.append(block.post_attention_layernorm.register_forward_hook(capture_post_rms(layer)))
                hooks.append(block.self_attn.v_proj.register_forward_hook(capture_values(layer)))
            for layer, block in enumerate(layers, start=1):
                hooks.append(block.register_forward_hook(capture_layer(layer)))
            inputs = {key: tensor.to("cuda:0") for key, tensor in prepared.inputs.items()}
            output = model.model(**inputs, use_cache=False, return_dict=True)
            # The model's last_hidden_state already includes its final learned norm.
            vocabulary_states["Final"] = output.last_hidden_state[0, image_rows].float().cpu().numpy()
        finally:
            for hook in hooks:
                hook.remove()
        if any(set(captured) != set(self.full_layers)
               for captured in (attention, image_attention, layer_cosine, head_cosine, post_rms_cosine, value_cosine)):
            raise RuntimeError("Some full-attention layers did not produce attention or cosine data")
        if set(vocabulary_states) != {"Final", *map(str, range(1, len(layers) + 1))}:
            raise RuntimeError("Some decoder layers did not produce vocabulary states")
        return Scores(attention, image_attention, layer_cosine, head_cosine,
                      vocabulary_states=vocabulary_states, post_rms_cosine=post_rms_cosine,
                      value_cosine=value_cosine)

    @torch.inference_mode()
    def vocabulary(self, scores: Scores, token: int, layer: str = "Final") -> VocabularyReadout:
        """Project one cached image state; probabilities are normalized over the entire vocabulary."""
        key = (str(layer), int(token))
        with self._vocabulary_lock:
            if key in scores.vocabulary_cache:
                scores.vocabulary_cache.move_to_end(key)
                return scores.vocabulary_cache[key]
            model = self.load_model()
            hidden = torch.from_numpy(scores.vocabulary_states[key[0]][token:token + 1])
            if layer != "Final":
                norm = model.model.language_model.norm
                norm_weight = next(norm.parameters())
                hidden = norm(hidden.to(device=norm_weight.device, dtype=norm_weight.dtype))
            head_weight = model.lm_head.weight
            logits = model.lm_head(hidden.to(device=head_weight.device, dtype=head_weight.dtype)).float()[0]
            probabilities = torch.softmax(logits, dim=-1)
            values, ids = torch.topk(probabilities, min(10, probabilities.numel()))
            token_ids = ids.cpu().numpy()
            tokenizer = self.processor.tokenizer
            tokens = tuple(tokenizer.decode([int(token_id)], skip_special_tokens=False,
                                             clean_up_tokenization_spaces=False) for token_id in token_ids)
            result = VocabularyReadout(
                key[0], key[1], token_ids, tokens, values.cpu().numpy(),
                probabilities.numel(), float(probabilities.sum().cpu()),
            )
            scores.vocabulary_cache[key] = result
            while len(scores.vocabulary_cache) > 128:
                scores.vocabulary_cache.popitem(last=False)
            return result
