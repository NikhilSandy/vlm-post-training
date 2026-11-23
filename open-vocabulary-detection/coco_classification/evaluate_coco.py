"""Evaluate COCO instance classification with the Qwen Vision Token Explorer readout."""

import argparse
from collections import defaultdict
from itertools import product
import json
from pathlib import Path
import random
from string import ascii_uppercase
import sys
import time

import numpy as np
from PIL import Image
import torch
from pycocotools.coco import COCO


PROMPT_PREFIX = (
    "Find every instance of the listed categories in the image. "
    "Match each object region to the label after its category name.\n"
)


def is_single_token(tokenizer, label):
    line = f" = {label}\n"
    start = line.index(label)
    offsets = tokenizer(line, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    return sum(lo < start + len(label) and hi > start for lo, hi in offsets) == 1


def make_sample(probe_module, image_id, image, names, labels):
    prompt = PROMPT_PREFIX
    categories = []
    for name, label in zip(names, labels, strict=False):
        prompt += "\n"
        start = len(prompt)
        prompt += f"{name} = {label}"
        categories.append(probe_module.Category(
            name, label, (start, start + len(name)), (len(prompt) - len(label), len(prompt)),
            None, None, 0,
        ))
    if len(categories) != len(names):
        raise ValueError("Insufficient tokenizer-verified option labels")
    return probe_module.Sample(image_id, image, prompt, tuple(categories))


def word_attention_scores(weights, prepared):
    image_rows = slice(prepared.image_start, prepared.image_start + prepared.grid.count)
    prompt_cols = slice(prepared.prompt_start, prepared.prompt_start + len(prepared.prompt_tokens))
    prompt = weights[0, :, image_rows, prompt_cols].float().cpu().numpy().mean(axis=0)
    return np.column_stack([prompt[:, list(span)].sum(axis=1) for span in prepared.names])


def class_scores_for_sample(probe, probe_module, sample, layer, readouts):
    prepared = probe.prepare(sample)
    ids = prepared.inputs["input_ids"][0]
    option_ids = [ids[prepared.prompt_start + position].item() for position in prepared.letters]
    if len(set(option_ids)) != len(option_ids):
        raise ValueError("The prompt contains duplicate option token IDs")
    model = probe.load_model()
    captured = {}

    def capture_cosine(_module, args):
        joined = args[0][0].float().cpu()
        heads = joined.reshape(joined.shape[0], probe.head_count, -1).transpose(0, 1)
        similarities = probe_module.cosine_scores(heads, prepared)
        captured["cosine"] = similarities.word_cosine.mean(axis=0)

    def capture_attention(_module, _args, output):
        weights = output[1]
        if weights is None:
            raise RuntimeError(f"Layer {layer} did not return eager attention weights")
        captured["word_attention"] = word_attention_scores(weights, prepared)

    block = model.model.language_model.layers[layer - 1]
    hooks = []
    if "cosine" in readouts:
        hooks.append(block.self_attn.o_proj.register_forward_pre_hook(capture_cosine))
    if "word_attention" in readouts:
        hooks.append(block.self_attn.register_forward_hook(capture_attention))
    try:
        with torch.inference_mode():
            inputs = {key: tensor.to("cuda:0") for key, tensor in prepared.inputs.items()}
            model.model(**inputs, use_cache=False, return_dict=True)
    finally:
        for hook in hooks:
            hook.remove()
    return prepared, captured


def patch_coverage(mask, grid):
    integral = np.pad(mask, ((1, 0), (1, 0))).cumsum(0, dtype=np.int32).cumsum(1, dtype=np.int32)
    x = np.array(grid.x_edges)
    y = np.array(grid.y_edges)
    corners = integral[np.ix_(y, x)]
    counts = corners[1:, 1:] - corners[:-1, 1:] - corners[1:, :-1] + corners[:-1, :-1]
    areas = np.outer(np.diff(y), np.diff(x))
    return np.divide(counts, areas, out=np.zeros(areas.shape), where=areas > 0).ravel()


def instance_vote(scores, weights):
    patch_labels = scores.argmax(axis=1)
    votes = np.bincount(patch_labels, weights=weights, minlength=scores.shape[1])
    tied = np.flatnonzero(votes == votes.max())
    return int(tied[np.argmax(weights @ scores[:, tied])])


def totals(records, conditions):
    return {
        condition: {
            "correct": sum(row[condition] == row["class"] for row in records),
            "total": len(records),
            "accuracy": sum(row[condition] == row["class"] for row in records) / len(records),
        }
        for condition in conditions
    }


def main():
    home = Path.home()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--explorer-source", type=Path, default=(
        Path(__file__).resolve().parents[1] / "qwen_token_explorer"
    ))
    parser.add_argument("--checkpoint", type=Path, default=home / (
        "Projects/vlm-post-training/person-attribute-detection/Models/Qwen/Qwen3.5-4B"
    ))
    parser.add_argument("--images", type=Path, default=home / "Datasets/coco2017/val2017")
    parser.add_argument("--annotations", type=Path, default=home / (
        "Datasets/coco2017/annotations/instances_val2017.json"
    ))
    parser.add_argument("--classes", nargs=2, help="Restrict the sample and scores to two target classes")
    parser.add_argument("--image-classes-only", action="store_true",
                        help="Run only the per-image class prompt. Evaluate all classes by default.")
    parser.add_argument("--include-single-class", action="store_true",
                        help="Also evaluate images where only one annotated class can vote.")
    readout_group = parser.add_mutually_exclusive_group()
    readout_group.add_argument("--word-attention", action="store_true",
                               help="Score Gradio's mean-head word attention alongside cosine.")
    readout_group.add_argument("--word-attention-only", action="store_true",
                               help="Score only Gradio's mean-head word attention.")
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--stop-after", type=int, help="Evaluate only this many images from the sampled order")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Continue a paired run with the same settings")
    args = parser.parse_args()
    if args.limit < 1 or args.classes is not None and len(set(args.classes)) != 2:
        parser.error("Choose a positive image limit and two distinct classes if filtering")
    if args.stop_after is not None and not 1 <= args.stop_after <= args.limit:
        parser.error("--stop-after must be between 1 and --limit")
    sys.path.insert(0, str(args.explorer_source))
    import probe as explorer

    coco = COCO(str(args.annotations))
    classes = sorted(category["name"] for category in coco.cats.values())
    selected_classes = args.classes or ([] if args.image_classes_only else ["person", "car"])
    target_classes = set(selected_classes or classes)
    if not target_classes <= set(classes):
        parser.error("Target classes must be exact COCO class names")
    target_ids = {category["id"] for category in coco.cats.values() if category["name"] in target_classes}
    required_ids = target_ids if selected_classes else set()
    minimum_classes = 1 if args.include_single_class else 2
    eligible = []
    for image_id, image_info in sorted(coco.imgs.items()):
        annotated_ids = {
            ann["category_id"] for ann in coco.imgToAnns[image_id]
            if ann.get("segmentation") and not ann.get("iscrowd") and ann["area"] > 0
        }
        if len(annotated_ids) >= minimum_classes and required_ids <= annotated_ids:
            if (args.images / image_info["file_name"]).is_file():
                eligible.append(image_id)
    image_ids = random.Random(args.seed).sample(eligible, min(args.limit, len(eligible)))[:args.stop_after]
    if not image_ids:
        raise ValueError("No eligible images")
    print(f"Selected {len(image_ids)} of {len(eligible)} eligible images", flush=True)

    probe = explorer.QwenProbe(args.checkpoint)
    tokenizer = probe.processor.tokenizer
    labels = list(ascii_uppercase)
    if not args.image_classes_only:
        for pair in product(ascii_uppercase, repeat=2):
            label = "".join(pair)
            if is_single_token(tokenizer, label):
                labels.append(label)
            if len(labels) >= len(classes):
                break
    layer = probe.full_layers[-1]
    prompt_conditions = ("image_classes",) if args.image_classes_only else ("image_classes", "all_80_classes")
    readouts = () if args.word_attention_only else ("cosine",)
    if args.word_attention or args.word_attention_only:
        readouts += ("word_attention",)
    score_keys = {
        prompt: {readout: prompt if readout == "cosine" else f"{prompt}_word_attention" for readout in readouts}
        for prompt in prompt_conditions
    }
    conditions = tuple(key for keys in score_keys.values() for key in keys.values())
    records = []
    multiple_token_names = set()
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output / "instances.jsonl"
    completed_images = set()
    if args.resume:
        records = [json.loads(line) for line in predictions_path.read_text().splitlines()]
        completed_order = list(dict.fromkeys(row["image_id"] for row in records))
        if (completed_order != image_ids[:len(completed_order)]
                or len({row["annotation_id"] for row in records}) != len(records)
                or any(row["class"] not in target_classes or not set(conditions) <= row.keys() for row in records)):
            raise ValueError("Existing predictions do not match the selected sample")
        completed_images = set(completed_order)
        print(f"Resuming after {len(completed_images)} images", flush=True)
    with predictions_path.open("a" if args.resume else "x") as predictions:
        for index, image_id in enumerate(image_ids, start=1):
            if image_id in completed_images:
                continue
            with Image.open(args.images / coco.imgs[image_id]["file_name"]) as source:
                image = source.convert("RGB")
            patch_masks = [
                (ann, coco.annToMask(ann)) for ann in coco.imgToAnns[image_id]
                if ann.get("segmentation")
            ]
            image_names = sorted({
                coco.cats[ann["category_id"]]["name"] for ann, mask in patch_masks if mask.any()
            })
            targets = [
                (ann, mask) for ann, mask in patch_masks
                if ann["category_id"] in target_ids and not ann.get("iscrowd") and mask.any()
            ]
            rows = [{
                "image_id": image_id, "annotation_id": ann["id"],
                "class": coco.cats[ann["category_id"]]["name"],
                "area": ann["area"], "image_class_count": len(image_names),
            } for ann, _ in targets]
            reference_grid = None
            for condition in prompt_conditions:
                names = image_names if condition == "image_classes" else classes
                sample = make_sample(explorer, image_id, image, names, labels)
                prepared, scores = class_scores_for_sample(probe, explorer, sample, layer, readouts)
                multiple_token_names.update(
                    name for name, positions in zip(names, prepared.names, strict=True)
                    if len(positions) > 1
                )
                if reference_grid is None:
                    reference_grid = prepared.grid
                    coverages = [patch_coverage(mask, reference_grid) for _, mask in targets]
                elif reference_grid != prepared.grid:
                    raise ValueError("The paired runs used different image token grids")
                for row, coverage in zip(rows, coverages, strict=True):
                    for readout, key in score_keys[condition].items():
                        row[key] = names[instance_vote(scores[readout], coverage)]
            records.extend(rows)
            predictions.write("".join(json.dumps(row) + "\n" for row in rows))
            predictions.flush()
            if index == 1 or index % 8 == 0 or index == len(image_ids):
                print(f"{index}/{len(image_ids)} images, {len(records)} instances, "
                      f"{time.monotonic() - started:.1f}s", flush=True)

    by_class = defaultdict(list)
    by_image = defaultdict(list)
    by_candidate_count = defaultdict(list)
    by_size = defaultdict(list)
    for row in records:
        by_class[row["class"]].append(row)
        by_image[row["image_id"]].append(row)
        by_candidate_count[row["image_class_count"]].append(row)
        size = "small" if row["area"] < 32**2 else "medium" if row["area"] < 96**2 else "large"
        by_size[size].append(row)
    results_by_class = {key: totals(rows, conditions) for key, rows in by_class.items()}
    multi_class_records = [row for row in records if row["image_class_count"] >= 2]
    uniform_guess = {
        key: float(np.mean([1 / row["image_class_count"] for row in records]))
        if condition == "image_classes" else 1 / len(classes)
        for condition, keys in score_keys.items() for key in keys.values()
    }
    summary = {
        "images": image_ids, "dataset_images": len(coco.imgs), "eligible_images": len(eligible),
        "seed": args.seed, "layer": layer,
        "target_classes": sorted(target_classes), "sample_limit": args.limit,
        "prompt_prefix": PROMPT_PREFIX, "conditions": conditions,
        "multi_token_classes": sorted(multiple_token_names),
        "overall": totals(records, conditions),
        "macro_accuracy": {
            condition: float(np.mean([metrics[condition]["accuracy"] for metrics in results_by_class.values()]))
            for condition in conditions
        },
        "by_class": results_by_class,
        "by_image_candidate_count": {key: totals(rows, conditions) for key, rows in by_candidate_count.items()},
        "by_size": {key: totals(rows, conditions) for key, rows in by_size.items()},
        "image_macro_accuracy": {
            condition: float(np.mean([
                totals(image_rows, (condition,))[condition]["accuracy"] for image_rows in by_image.values()
            ]))
            for condition in conditions
        },
        "uniform_guess": uniform_guess,
    }
    if "cosine" in readouts:
        summary["representation"] = "mean per-head class-name cosine, gated before output projection"
    if "all_80_classes" in prompt_conditions:
        summary["all_80_options"] = dict(zip(classes, labels, strict=True))
    if "word_attention" in readouts:
        summary["word_attention"] = "Mean over heads, then sum attention from each patch to all tokens in a class name"
    if args.include_single_class and multi_class_records:
        summary["overall_multi_class"] = totals(multi_class_records, conditions)
        summary["uniform_guess_multi_class"] = {
            key: float(np.mean([1 / row["image_class_count"] for row in multi_class_records]))
            if condition == "image_classes" else 1 / len(classes)
            for condition, keys in score_keys.items() for key in keys.values()
        }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["overall"], indent=2), flush=True)
    if "overall_multi_class" in summary:
        print("Multi-class images:", json.dumps(summary["overall_multi_class"], indent=2), flush=True)


if __name__ == "__main__":
    main()
