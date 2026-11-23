"""Classify COCO instances by image-token attention to single-token options."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import partial
import hashlib
import importlib.util
from itertools import chain, product
import json
from pathlib import Path
from random import Random
from string import ascii_uppercase
import sys
import time

import numpy as np
import torch


DEFAULT_EXPLORER = Path(__file__).resolve().parents[1] / "qwen_token_explorer"


def load_explorer(source):
    """Load the actual Gradio prompt builder without importing its UI."""
    spec = importlib.util.spec_from_file_location("option_attention_explorer", source / "probe.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def option_positions(sample, prepared, tokenizer):
    """Use prompt-span positions, never a global search for matching token IDs."""
    labels = [category.letter for category in sample.categories]
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("Options must be nonempty and distinct")
    # Gradio's prepare() verifies that every option span overlaps exactly one token.
    if len(prepared.letters) != len(labels):
        raise ValueError("Each category must have exactly one option-token position")
    positions = [prepared.prompt_start + position for position in prepared.letters]
    if len(set(positions)) != len(positions) or any(
        not prepared.prompt_start <= position < prepared.image_start for position in positions
    ):
        raise ValueError("Option positions must be distinct and precede the image")
    ids = [int(prepared.inputs["input_ids"][0, position]) for position in positions]
    if len(set(ids)) != len(ids) or any(
        tokenizer.decode([token_id]).strip() != label for token_id, label in zip(ids, labels, strict=True)
    ):
        raise ValueError("Options must map to distinct single tokens in the actual prompt")
    return positions, ids


def shuffle_options(labels, seed):
    if len(labels) < 2:
        raise ValueError("Shuffling requires at least two option labels")
    rng = Random(seed)
    shuffled = list(labels)
    while True:
        rng.shuffle(shuffled)
        if all(old != new for old, new in zip(labels, shuffled, strict=True)):
            return shuffled


def all_class_prompt(explorer, probe, coco, shuffle_seed=None):
    from PIL import Image
    from .evaluate_coco import is_single_token, make_sample

    classes = sorted(category["name"] for category in coco.cats.values())
    if len(classes) != 80 or len(set(classes)) != 80:
        raise ValueError("--all-80-classes requires the full COCO category list")
    tokenizer = probe.processor.tokenizer
    labels = []
    for label in chain(ascii_uppercase, map("".join, product(ascii_uppercase, repeat=2))):
        if is_single_token(tokenizer, label):
            labels.append(label)
        if len(labels) == len(classes):
            break
    if len(labels) != len(classes):
        raise ValueError("The tokenizer needs 80 single-token option labels")
    if shuffle_seed is not None:
        labels = shuffle_options(labels, shuffle_seed)
    template = make_sample(explorer, 0, Image.new("RGB", (256, 256)), classes, labels)
    _, token_ids = option_positions(template, probe.prepare(template), tokenizer)
    return template, dict(zip(classes, token_ids, strict=True))


def load_with_prompt(dataset, template, image_id):
    from PIL import Image

    path = dataset.images / dataset.coco.imgs[image_id]["file_name"]
    with Image.open(path) as source:
        return replace(template, image_id=image_id, image=source.convert("RGB"))


def option_attention_scores(weights, prepared, positions):
    """Filter already-normalized outgoing attention; preserve full-context mass."""
    if weights is None:
        raise RuntimeError("The selected layer did not return eager attention weights")
    if weights.ndim != 4 or weights.shape[0] != 1:
        raise ValueError("Expected attention shaped [1, heads, query tokens, key tokens]")
    if max(positions) >= weights.shape[3]:
        raise ValueError("Attention dimensions do not match option positions")
    scores, mass = reduce_attention(weights[0], prepared, torch.tensor(positions))
    return checked_scores(scores.cpu().numpy(), mass.cpu().numpy())


def reduce_attention(weights, prepared, positions):
    start, count = prepared.image_start, prepared.grid.count
    if weights.shape[1] < start + count:
        raise ValueError("Attention dimensions do not match image and option positions")
    image_rows = weights[:, start:start + count, :]
    scores = image_rows.index_select(-1, positions.to(weights.device)).float().mean(dim=0)
    context_mass = image_rows.sum(dim=-1, dtype=torch.float32).mean(dim=0)
    return scores, context_mass


def checked_scores(scores, context_mass):
    if not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("Invalid option attention shares")
    return scores, context_mass


def read_option_attention(probe, sample, layers):
    prepared = probe.prepare(sample)
    positions, ids = option_positions(sample, prepared, probe.processor.tokenizer)
    captured = forward_option_attention(probe, [(prepared, positions)], layers)
    return prepared, captured[0], positions, ids


def collate_inputs(prepared_images, tokenizer):
    sequences = [prepared.inputs for prepared in prepared_images]
    if len(sequences) == 1:
        return sequences[0]
    padding = {"input_ids": tokenizer.pad_token_id, "attention_mask": 0, "mm_token_type_ids": 0}
    vision_inputs = {"pixel_values", "image_grid_thw"}
    keys = sequences[0].keys()
    if any(inputs.keys() != keys for inputs in sequences) or not keys <= padding.keys() | vision_inputs:
        raise ValueError("Unexpected inputs for a batch of Qwen images")
    length = max(inputs["input_ids"].shape[1] for inputs in sequences)
    batch = {}
    for key in keys:
        tensors = [inputs[key] for inputs in sequences]
        if key in padding:
            tensors = [torch.nn.functional.pad(t, (0, length - t.shape[1]), value=padding[key]) for t in tensors]
        batch[key] = torch.cat(tensors)
    return batch


class VisionPatchProjection(torch.nn.Linear):
    def forward(self, patches):
        return super().forward(patches.flatten(1))


def use_linear_patch_projection(visual):
    """Apply the same weights to independent, already-extracted vision patches."""
    patch_embed = visual.patch_embed
    conv = patch_embed.proj
    if not isinstance(conv, torch.nn.Conv3d):
        return
    if conv.groups != 1 or conv.stride != conv.kernel_size or any(conv.padding) or conv.dilation != (1, 1, 1):
        return
    weight = conv.weight.flatten(1)
    linear = VisionPatchProjection(weight.shape[1], weight.shape[0], bias=conv.bias is not None, device="meta")
    linear.weight = torch.nn.Parameter(weight, requires_grad=False)
    linear.bias = conv.bias
    patch_embed.proj = linear.eval()


def head_readout(weights, prepared, positions, coverage):
    """Extract one patch winner per head and weighted attention for instance ties."""
    start, count = prepared.image_start, prepared.grid.count
    scores = weights[:, start:start + count, :].index_select(-1, positions.to(weights.device))
    patch_winners = scores.argmax(dim=-1)
    support = coverage.to(weights.device) @ scores.double()
    return patch_winners, support


def forward_option_attention(probe, images, layers, head_coverages=None):
    tokenizer = probe.processor.tokenizer
    batch = collate_inputs([prepared for prepared, _ in images], tokenizer)
    if hasattr(probe, "memory_guard"):
        for prepared, _ in images:
            probe.memory_guard.check_inputs(prepared)
        probe.memory_guard.wait("before_image")
    model = probe.load_model()
    if len(images) > 1 and not hasattr(probe, "memory_guard"):
        use_linear_patch_projection(model.model.visual)
    if getattr(probe, "vision_sdpa", False):
        model.model.visual.config._attn_implementation = "sdpa"
    device = model.model.language_model.embed_tokens.weight.device
    positions = [torch.tensor(indices, device=device) for _, indices in images]
    inputs = {key: tensor.to(device) for key, tensor in batch.items()}
    if head_coverages is not None:
        head_coverages = [torch.as_tensor(coverage, device=device) for coverage in head_coverages]
    captured = {}

    def capture(layer):
        def hook(_module, _args, output):
            if layer in captured:
                raise RuntimeError(f"Duplicate attention capture at layer {layer}")
            weights = output[1]
            if weights is None or weights.ndim != 4 or weights.shape[0] != len(images):
                raise RuntimeError(f"Layer {layer} did not return batched eager attention weights")
            if head_coverages is not None:
                readouts = [
                    head_readout(weight, prepared, indices, coverage)
                    for weight, (prepared, _), indices, coverage in
                    zip(weights, images, positions, head_coverages, strict=True)
                ]
                winners, support = zip(*readouts, strict=True)
                captured[layer] = (
                    torch.cat([part.flatten() for part in winners]),
                    torch.cat([part.flatten() for part in support]),
                )
                return
            reduced = [
                reduce_attention(weights[index], prepared, indices)
                for index, ((prepared, _), indices) in enumerate(zip(images, positions, strict=True))
            ]
            captured[layer] = torch.cat([torch.cat((scores, mass[:, None]), dim=1).flatten()
                                         for scores, mass in reduced])
        return hook

    blocks = model.model.language_model.layers
    hooks = []
    try:
        for layer in layers:
            hooks.append(blocks[layer - 1].self_attn.register_forward_hook(capture(layer)))
        with torch.inference_mode():
            model.model(**inputs, use_cache=False, return_dict=True)
    finally:
        for hook in hooks:
            hook.remove()
    missing_layers = set(layers) - captured.keys()
    if missing_layers:
        raise RuntimeError(f"Missing attention captures at layers {sorted(missing_layers)}")
    results = [{} for _ in images]
    for layer, packed in captured.items():
        if head_coverages is not None:
            winners, support = (tensor.cpu().numpy() for tensor in packed)
            patch_offset = support_offset = 0
            heads = probe.head_count
            for result, (prepared, indices), coverage in zip(results, images, head_coverages, strict=True):
                patches = prepared.grid.count
                regions, classes = coverage.shape[0], len(indices)
                patch_end = patch_offset + heads * patches
                support_end = support_offset + heads * regions * classes
                result[layer] = (
                    winners[patch_offset:patch_end].reshape(heads, patches),
                    support[support_offset:support_end].reshape(heads, regions, classes),
                )
                patch_offset, support_offset = patch_end, support_end
            continue
        values = packed.cpu().numpy()
        offset = 0
        for result, (prepared, indices) in zip(results, images, strict=True):
            columns = len(indices) + 1
            end = offset + prepared.grid.count * columns
            row = values[offset:end].reshape(-1, columns)
            result[layer] = checked_scores(row[:, :-1], row[:, -1])
            offset = end
    return results


def patch_coverage(mask, grid):
    """Fraction of each merged image-token cell covered by an original-image mask."""
    integral = np.pad(mask.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    x, y = np.array(grid.x_edges), np.array(grid.y_edges)
    corners = integral[np.ix_(y, x)]
    counts = corners[1:, 1:] - corners[:-1, 1:] - corners[1:, :-1] + corners[:-1, :-1]
    areas = np.outer(np.diff(y), np.diff(x))
    return np.divide(counts, areas, out=np.zeros(areas.shape), where=areas > 0).ravel()


def rank_options(scores):
    """Descending original attention; exact ties follow alphabetical class order."""
    return np.argsort(-scores, axis=1, kind="stable")


def instance_vote(scores, rankings, coverage):
    if coverage.sum() <= 0:
        raise ValueError("An evaluated instance must overlap at least one image patch")
    votes = np.bincount(rankings[:, 0], weights=coverage, minlength=scores.shape[1])
    support = coverage @ scores
    tied = np.flatnonzero(votes == votes.max())
    # First break object-vote ties with raw attention support; then alphabetical order.
    winner = int(tied[np.argmax(support[tied])])
    return winner, votes, support


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def metrics(rows):
    correct = sum(row["prediction"] == row["class"] for row in rows)
    return {
        "correct": correct, "total": len(rows),
        "accuracy": correct / len(rows) if rows else None,
        "uniform_guess": float(np.mean([
            1 / row.get("candidate_class_count", row["image_class_count"]) for row in rows
        ])) if rows else None,
    }


def summarize(image_records, dataset_images, selected_images, layer):
    rows = [row for record in image_records for row in record["instances"]]
    by_class, by_image, by_size = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in rows:
        by_class[row["class"]].append(row)
        by_image[row["image_id"]].append(row)
        size = "small" if row["area"] < 32**2 else "medium" if row["area"] < 96**2 else "large"
        by_size[size].append(row)
    return {
        "dataset_images": dataset_images, "selected_images": selected_images,
        "processed_images": len(image_records), "layer": layer, "head_pooling": "mean",
        "complete": len(image_records) == selected_images,
        "full_dataset": selected_images == dataset_images,
        "image_status_counts": dict(Counter(record["status"] for record in image_records)),
        "overall": metrics(rows),
        "multi_class": metrics([row for row in rows if row["image_class_count"] >= 2]),
        "single_class": metrics([row for row in rows if row["image_class_count"] == 1]),
        "macro_accuracy": float(np.mean([metrics(group)["accuracy"] for group in by_class.values()])) if rows else None,
        "image_macro_accuracy": float(np.mean([metrics(group)["accuracy"] for group in by_image.values()])) if rows else None,
        "by_class": {name: metrics(group) for name, group in sorted(by_class.items())},
        "by_size": {name: metrics(group) for name, group in sorted(by_size.items())},
    }


def evaluate_image(dataset, probe, image_id, patch_dirs):
    record, prepared, positions, regions = prepare_image(dataset, probe, image_id)
    captured = forward_option_attention(probe, [(prepared, positions)], patch_dirs)[0] if prepared else {}
    return finish_image(record, regions, dataset.coco, patch_dirs, captured)


def prepare_image(dataset, probe, image_id, sample_loader=None):
    coco = dataset.coco
    masked = [(ann, coco.annToMask(ann)) for ann in coco.imgToAnns[image_id] if ann.get("segmentation")]
    masked = [(ann, mask) for ann, mask in masked if mask.any()]
    record = {
        "image_id": image_id, "file_name": coco.imgs[image_id]["file_name"],
        "status": "no_gt_masks",
    }
    if not masked:
        return record, None, (), ()
    sample = (sample_loader or dataset.load)(image_id)
    prepared = probe.prepare(sample)
    positions, ids = option_positions(sample, prepared, probe.processor.tokenizer)
    names = [category.name for category in sample.categories]
    regions = [
        (ann, patch_coverage(mask, prepared.grid)) for ann, mask in masked
        if not ann.get("iscrowd") and ann["area"] > 0
    ]
    record.update({
        "status": "evaluated" if regions else "no_evaluable_instances",
        "prompt": sample.prompt, "classes": names,
        "image_class_count": len({ann["category_id"] for ann, _ in masked}),
        "options": [category.letter for category in sample.categories],
        "option_positions": positions, "option_token_ids": ids,
        "image_start": prepared.image_start, "sequence_length": int(prepared.inputs["input_ids"].shape[1]),
        "grid": {
            "rows": prepared.grid.rows, "cols": prepared.grid.cols,
            "x_edges": prepared.grid.x_edges, "y_edges": prepared.grid.y_edges,
        },
    })
    return record, prepared, positions, regions


def finish_image(record, regions, coco, patch_dirs, captured, per_head=False):
    if record["status"] == "no_gt_masks":
        return {layer: dict(record, instances=[]) for layer in patch_dirs}
    if per_head:
        return {
            layer: evaluate_layer_heads(record, coco, regions, *captured[layer])
            for layer in patch_dirs
        }
    return {
        layer: evaluate_layer(record, coco, regions, patch_dir, *captured[layer])
        for layer, patch_dir in patch_dirs.items()
    }


def evaluate_batch(probe, prepared, layers, head_coverages=None):
    if not prepared:
        return []
    try:
        return forward_option_attention(probe, prepared, layers, head_coverages)
    except torch.cuda.OutOfMemoryError:
        if len(prepared) == 1:
            raise
    torch.cuda.empty_cache()
    midpoint = len(prepared) // 2
    print(f"GPU memory limit reached; splitting batch of {len(prepared)}", flush=True)
    left = head_coverages[:midpoint] if head_coverages is not None else None
    right = head_coverages[midpoint:] if head_coverages is not None else None
    return evaluate_batch(probe, prepared[:midpoint], layers, left) + evaluate_batch(probe, prepared[midpoint:], layers, right)


def evaluate_layer_heads(record, coco, regions, patch_winners, support):
    names = np.asarray(record["classes"], dtype=object)
    heads = patch_winners.shape[0]
    # Head-specific offsets keep all of an instance's votes in a single bincount.
    bins = (patch_winners + np.arange(heads)[:, None] * len(names)).ravel()
    correct = np.zeros(heads, dtype=int)
    instances = []
    for region, (ann, coverage) in enumerate(regions):
        if coverage.sum() <= 0:
            raise ValueError("An evaluated instance must overlap at least one image patch")
        votes = np.bincount(bins, weights=np.tile(coverage, heads), minlength=heads * len(names))
        votes = votes.reshape(heads, len(names))
        tied = votes == votes.max(axis=1, keepdims=True)
        winners = np.where(tied, support[:, region, :], -np.inf).argmax(axis=1)
        predictions = names[winners]
        expected = coco.cats[ann["category_id"]]["name"]
        correct += predictions == expected
        instances.append({
            "image_id": record["image_id"], "annotation_id": ann["id"], "class": expected,
            "head_predictions": predictions.tolist(),
        })
    return dict(record, instances=instances, correct_by_head=correct.tolist())


def evaluate_layer(record, coco, regions, patches_dir, scores, context_mass):
    image_id, names = record["image_id"], record["classes"]
    rankings = rank_options(scores)
    patch_path = patches_dir / f"{image_id:012d}.npz"
    temporary = patch_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as output:
        np.savez_compressed(
            output, option_attention=scores, rankings=rankings.astype(np.uint8),
            votes=rankings[:, 0].astype(np.uint8), context_attention_mass=context_mass,
            non_option_attention_mass=context_mass - scores.sum(axis=1),
        )
    temporary.replace(patch_path)
    instances = []
    for ann, coverage in regions:
        winner, votes, support = instance_vote(scores, rankings, coverage)
        instances.append({
            "image_id": image_id, "annotation_id": ann["id"],
            "class": coco.cats[ann["category_id"]]["name"], "prediction": names[winner],
            "area": ann["area"], "image_class_count": record["image_class_count"],
            "candidate_class_count": len(names),
            "vote_weights": votes.tolist(), "attention_support": support.tolist(),
        })
    return dict(record, instances=instances, patch_file=f"patches/{patch_path.name}")


def write_jsonl(output_dir, records):
    for filename, rows in (
        ("images.jsonl", ({key: value for key, value in record.items() if key != "instances"} for record in records)),
        ("instances.jsonl", (row for record in records for row in record["instances"])),
    ):
        temporary = output_dir / (filename + ".tmp")
        with temporary.open("w") as output:
            for row in rows:
                output.write(json.dumps(row, allow_nan=False) + "\n")
        temporary.replace(output_dir / filename)


def summarize_heads(records, dataset_images, selected_images, layer, head_count):
    correct = np.zeros(head_count, dtype=int)
    total = sum(len(record["instances"]) for record in records)
    for record in records:
        if "correct_by_head" in record:
            correct += record["correct_by_head"]
    by_head = {
        head: {"head": head, "correct": int(count), "total": total,
               "accuracy": int(count) / total if total else None}
        for head, count in enumerate(correct, start=1)
    }
    best = int(correct.argmax()) + 1
    return {
        "dataset_images": dataset_images, "selected_images": selected_images,
        "processed_images": len(records), "layer": layer, "head_pooling": "none",
        "complete": len(records) == selected_images,
        "full_dataset": selected_images == dataset_images,
        "image_status_counts": dict(Counter(record["status"] for record in records)),
        "overall": dict(by_head[best], head=best if total else None),
        "by_head": by_head,
    }


def write_summaries(output_dir, layer_dirs, records, dataset_images, selected_images, head_count=None):
    overall = {}
    for layer, directory in layer_dirs.items():
        if head_count is None:
            summary = summarize(records[layer], dataset_images, selected_images, layer)
        else:
            summary = summarize_heads(records[layer], dataset_images, selected_images, layer, head_count)
        atomic_json(directory / "summary.json", summary)
        overall[layer] = summary["overall"]
    if output_dir not in layer_dirs.values():
        atomic_json(output_dir / "summary.json", {
            "layers": list(layer_dirs), "head_pooling": "mean" if head_count is None else "none",
            "dataset_images": dataset_images, "selected_images": selected_images,
            "processed_images": min(map(len, records.values())),
            "complete": all(len(rows) == selected_images for rows in records.values()),
            "full_dataset": selected_images == dataset_images,
            "overall_by_layer": overall,
        })
    return overall


def prepare_window(executor, image_ids, dataset, probe, layer_dirs, resume, sample_loader):
    jobs = []
    for image_id in image_ids:
        existing, pending = {}, {}
        for layer, directory in layer_dirs.items():
            path = directory / "records" / f"{image_id:012d}.json"
            if resume and path.is_file():
                record = json.loads(path.read_text())
                patch_file = record.get("patch_file")
                if record["image_id"] != image_id or (patch_file and not (directory / patch_file).is_file()):
                    raise ValueError(f"Incomplete or mismatched image record: {path}")
                existing[layer] = record
            else:
                pending[layer] = directory / "patches"
        future = executor.submit(prepare_image, dataset, probe, image_id, sample_loader) if pending else None
        jobs.append((existing, pending, future))
    return jobs


def save_image(record, regions, coco, pending, captured, per_head=False):
    result = finish_image(record, regions, coco, pending, captured, per_head)
    for layer, layer_record in result.items():
        filename = f"{record['image_id']:012d}.json"
        # The record is the completion marker and follows the patch artifact.
        atomic_json(pending[layer].parent / "records" / filename, layer_record)
    return result


def bucketed_image_ids(image_ids, coco, batch_size):
    """Group similarly sized images within bounded windows to reduce text padding."""
    if batch_size == 1:
        return image_ids

    def estimated_tokens(image_id):
        image = coco.imgs[image_id]
        categories = {ann["category_id"] for ann in coco.imgToAnns[image_id] if ann.get("segmentation")}
        return round(image["width"] / 32) * round(image["height"] / 32) + 4 * len(categories)

    window_size = 8 * batch_size
    return [image_id for start in range(0, len(image_ids), window_size)
            for image_id in sorted(image_ids[start:start + window_size], key=estimated_tokens)]


def run_batches(args, dataset, probe, image_ids, layer_dirs, records, guard, sample_loader):
    started = time.monotonic()
    image_ids = bucketed_image_ids(image_ids, dataset.coco, args.batch_size)
    last_report = 0
    model_images = 0
    head_count = probe.head_count if args.per_head else None
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        jobs = prepare_window(executor, image_ids[:args.batch_size], dataset, probe, layer_dirs, args.resume, sample_loader)
        for start in range(0, len(image_ids), args.batch_size):
            end = min(start + args.batch_size, len(image_ids))
            ready = [(existing, pending, future.result() if future else None)
                     for existing, pending, future in jobs]
            jobs = prepare_window(
                executor, image_ids[end:end + args.batch_size], dataset, probe, layer_dirs, args.resume, sample_loader,
            )
            images = [(work[1], work[2]) for _, _, work in ready if work and work[1] is not None]
            layers = sorted({layer for _, pending, work in ready if work and work[1] is not None for layer in pending})
            head_coverages = None
            if args.per_head:
                head_coverages = [
                    np.asarray([coverage for _, coverage in work[3]], dtype=np.float64).reshape(-1, work[1].grid.count)
                    for _, _, work in ready if work and work[1] is not None
                ]
            captured = iter(evaluate_batch(probe, images, layers, head_coverages))
            model_images += len(images)
            finished = []
            for existing, pending, work in ready:
                future = None
                if work:
                    record, prepared, _, regions = work
                    scores = next(captured) if prepared is not None else {}
                    future = executor.submit(save_image, record, regions, dataset.coco, pending, scores, args.per_head)
                finished.append((existing, future))
            for existing, future in finished:
                if future:
                    existing.update(future.result())
                for layer, record in existing.items():
                    records[layer].append(record)
            if last_report == 0 or end - last_report >= 100 or end == len(image_ids):
                overall = write_summaries(
                    args.output, layer_dirs, records, len(dataset.coco.imgs), len(image_ids), head_count,
                )
                if guard is not None:
                    guard.write_status("running", f"image_{end}")
                counts = ", ".join(f"L{layer} {row['correct']}/{row['total']}" for layer, row in overall.items())
                elapsed = time.monotonic() - started
                peak = max((torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())), default=0)
                print(f"{end}/{len(image_ids)} images; {counts} correct; {elapsed:.1f}s "
                      f"({end / elapsed:.2f} images/s); peak allocated {peak / 2**30:.2f} GiB", flush=True)
                last_report = end
    elapsed = time.monotonic() - started
    return {
        "batch_size": args.batch_size, "workers": args.workers, "model_images": model_images,
        "seconds": elapsed, "model_images_per_second": model_images / elapsed,
        "gpus": [
            {"name": torch.cuda.get_device_name(i),
             "peak_allocated_gib": torch.cuda.max_memory_allocated(i) / 2**30,
             "peak_reserved_gib": torch.cuda.max_memory_reserved(i) / 2**30}
            for i in range(torch.cuda.device_count())
        ],
    }


def main():
    home = Path.home()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--explorer-source", type=Path, default=DEFAULT_EXPLORER)
    parser.add_argument("--checkpoint", type=Path, default=home / "Projects/vlm-post-training/person-attribute-detection/Models/Qwen/Qwen3.5-4B")
    parser.add_argument("--images", type=Path, default=home / "Datasets/coco2017/val2017")
    parser.add_argument("--annotations", type=Path, default=home / "Datasets/coco2017/annotations/instances_val2017.json")
    parser.add_argument(
        "--layer", type=int,
        help="One-based full-attention layer; default: all even-numbered full-attention layers",
    )
    parser.add_argument("--limit", type=int, help="For smoke checks only; default: every validation image")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Reuse completed per-image records with identical settings")
    parser.add_argument("--all-80-classes", action="store_true", help="Prompt with all 80 COCO classes and distinct single-token options")
    parser.add_argument(
        "--shuffle-options-seed", type=int,
        help="Reassign all 80 option tokens with a seeded shuffle that changes every class assignment",
    )
    parser.add_argument("--batch-size", type=int, help="Images per GPU forward pass (default: 16, or 1 with memory caps)")
    parser.add_argument("--workers", type=int, default=8, help="CPU threads for image preparation and result writing")
    parser.add_argument("--vision-sdpa", action="store_true", help="Use fused scaled dot-product attention in the vision encoder")
    parser.add_argument("--per-head", action="store_true", help="Score each attention head independently and report the best head per layer")
    caps_group = parser.add_mutually_exclusive_group()
    caps_group.add_argument("--gpu-memory-cap-gib", type=float,
                           help="Enable guarded loading with the same allocator cap on each GPU")
    caps_group.add_argument("--gpu-memory-caps-gib", type=float, nargs="+",
                           help="Individual allocator caps in visible-GPU order")
    parser.add_argument("--model-memory-gib", type=float, default=1.5,
                        help="Guarded resident-weight budget per GPU (default: 1.5 GiB)")
    parser.add_argument("--model-memory-budgets-gib", type=float, nargs="+",
                        help="Individual weight budgets in visible-GPU order")
    parser.add_argument("--min-free-gib", type=float, default=4,
                        help="Free VRAM reserve preserved by the guarded run (default: 4 GiB)")
    args = parser.parse_args()
    if args.shuffle_options_seed is not None and not args.all_80_classes:
        parser.error("--shuffle-options-seed requires --all-80-classes")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    cap_settings = args.gpu_memory_caps_gib if args.gpu_memory_caps_gib is not None else args.gpu_memory_cap_gib
    if args.batch_size is None:
        args.batch_size = 16 if cap_settings is None else 1
    if args.batch_size < 1 or args.workers < 1:
        parser.error("--batch-size and --workers must be positive")
    budget_settings = args.model_memory_budgets_gib if args.model_memory_budgets_gib is not None else args.model_memory_gib
    if args.min_free_gib < 0 or args.model_memory_gib <= 0 or any(
        value <= 0 for value in (args.gpu_memory_caps_gib or []) + (args.model_memory_budgets_gib or [])
    ):
        parser.error("Memory budgets must be positive; the free-memory reserve cannot be negative")
    explorer = load_explorer(args.explorer_source)
    dataset = explorer.CocoImages(args.images, args.annotations)
    image_ids = sorted(dataset.coco.imgs)[:args.limit]
    missing = [dataset.coco.imgs[i]["file_name"] for i in image_ids if not (args.images / dataset.coco.imgs[i]["file_name"]).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} selected images; first: {missing[0]}")
    probe = explorer.QwenProbe(args.checkpoint)
    probe.vision_sdpa = args.vision_sdpa
    if args.layer is not None and args.layer not in probe.full_layers:
        parser.error(f"--layer must be one of {probe.full_layers}")
    if args.layer is None:
        layers = [layer for layer in probe.full_layers if layer % 2 == 0]
    else:
        layers = [args.layer]
    if not layers:
        parser.error("The checkpoint has no even-numbered full-attention layers; select one with --layer")
    config = {
        "algorithm": "mean_head_option_attention_patch_vote_v1",
        "batch_size": args.batch_size, "vision_sdpa": args.vision_sdpa,
        "checkpoint": str(args.checkpoint.resolve()), "images": str(args.images.resolve()),
        "annotations": str(args.annotations.resolve()), "image_ids": image_ids,
        "annotations_sha256": hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
        "explorer_source": str(args.explorer_source.resolve()),
        "explorer_sha256": hashlib.sha256((args.explorer_source / "probe.py").read_bytes()).hexdigest(),
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checkpoint_config_sha256": hashlib.sha256((args.checkpoint / "config.json").read_bytes()).hexdigest(),
        "memory_safety": {
            "gpu_memory_cap_gib": args.gpu_memory_cap_gib,
            "gpu_memory_caps_gib": args.gpu_memory_caps_gib,
            "model_memory_gib": args.model_memory_gib, "min_free_gib": args.min_free_gib,
            "model_memory_budgets_gib": args.model_memory_budgets_gib,
            "guard_sha256": hashlib.sha256(Path(__file__).with_name("guarded_cuda.py").read_bytes()).hexdigest(),
        },
    }
    if args.per_head:
        config["per_head"] = probe.head_count
    sample_loader = dataset.load
    if args.all_80_classes:
        template, token_ids = all_class_prompt(explorer, probe, dataset.coco, args.shuffle_options_seed)
        sample_loader = partial(load_with_prompt, dataset, template)
        config["all_80_options"] = {
            category.name: {"label": category.letter, "token_id": token_ids[category.name]}
            for category in template.categories
        }
        if args.shuffle_options_seed is not None:
            config["shuffle_options_seed"] = args.shuffle_options_seed
    if args.layer is None:
        config["layers"] = layers
    else:
        config["layer"] = args.layer
    config_path = args.output / "config.json"
    if args.resume:
        if json.loads(config_path.read_text()) != config:
            raise ValueError("Resume settings or source files differ from the existing run")
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        if any(args.output.iterdir()):
            raise FileExistsError("Output directory must be empty; use --resume for the same run")
        atomic_json(config_path, config)
    guard = None
    if cap_settings is not None:
        from .guarded_cuda import MemoryGuard, enable_guarded_loading
        guard = MemoryGuard(cap_settings, args.min_free_gib, args.output / "memory_status.json")
        enable_guarded_loading(probe, guard, budget_settings)
    layer_dirs = {
        layer: args.output / f"layer_{layer:02d}" if args.layer is None else args.output
        for layer in layers
    }
    for directory in layer_dirs.values():
        (directory / "records").mkdir(parents=True, exist_ok=True)
        if not args.per_head:
            (directory / "patches").mkdir(exist_ok=True)
    records = {layer: [] for layer in layers}
    head_count = probe.head_count if args.per_head else None
    pooling = "independent voting" if args.per_head else "mean pooling"
    print(
        f"Selected {len(image_ids)}/{len(dataset.coco.imgs)} images; "
        f"layers {layers}; {pooling} over {probe.head_count} heads; batch size {args.batch_size}; "
        f"{'all 80 COCO classes' if args.all_80_classes else 'per-image COCO classes'}", flush=True,
    )
    performance = run_batches(args, dataset, probe, image_ids, layer_dirs, records, guard, sample_loader)
    for layer, directory in layer_dirs.items():
        records[layer].sort(key=lambda record: record["image_id"])
        write_jsonl(directory, records[layer])
    overall = write_summaries(args.output, layer_dirs, records, len(dataset.coco.imgs), len(image_ids), head_count)
    if guard is not None:
        guard.write_status("complete", f"image_{len(image_ids)}")
    if not args.resume:
        atomic_json(args.output / "performance.json", performance)
    print(json.dumps(overall if args.layer is None else overall[args.layer], indent=2), flush=True)


if __name__ == "__main__":
    main()
