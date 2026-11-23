"""Classify COCO instances by image-token attention to single-character options."""

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
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
    if not labels or labels != list(ascii_uppercase[:len(labels)]):
        raise ValueError("Options must be unique single characters A through Z")
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
        raise ValueError("Option characters must map to distinct single tokens in the actual prompt")
    return positions, ids


def option_attention_scores(weights, prepared, positions):
    """Filter already-normalized outgoing attention; preserve full-context mass."""
    if weights is None:
        raise RuntimeError("The selected layer did not return eager attention weights")
    if weights.ndim != 4 or weights.shape[0] != 1:
        raise ValueError("Expected attention shaped [1, heads, query tokens, key tokens]")
    start, count = prepared.image_start, prepared.grid.count
    if weights.shape[2] < start + count or max(positions) >= weights.shape[3]:
        raise ValueError("Attention dimensions do not match image and option positions")
    image_rows = weights[0, :, start:start + count, :]
    indices = torch.tensor(positions, device=weights.device, dtype=torch.long)
    scores = image_rows.index_select(-1, indices).float().mean(dim=0).cpu().numpy()
    context_mass = image_rows.sum(dim=-1, dtype=torch.float32).mean(dim=0).cpu().numpy()
    if not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("Invalid option attention shares")
    return scores, context_mass


def read_option_attention(probe, sample, layer):
    prepared = probe.prepare(sample)
    positions, ids = option_positions(sample, prepared, probe.processor.tokenizer)
    if hasattr(probe, "memory_guard"):
        probe.memory_guard.check_inputs(prepared)
        probe.memory_guard.wait("before_image")
    model = probe.load_model()
    captured = []

    def capture(_module, _args, output):
        captured.append(option_attention_scores(output[1], prepared, positions))

    block = model.model.language_model.layers[layer - 1]
    hook = block.self_attn.register_forward_hook(capture)
    try:
        device = model.model.language_model.embed_tokens.weight.device
        inputs = {key: tensor.to(device) for key, tensor in prepared.inputs.items()}
        with torch.inference_mode():
            model.model(**inputs, use_cache=False, return_dict=True)
    finally:
        hook.remove()
    if len(captured) != 1:
        raise RuntimeError("Expected exactly one attention capture per image")
    scores, context_mass = captured[0]
    return prepared, scores, context_mass, positions, ids


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
        "uniform_guess": float(np.mean([1 / row["image_class_count"] for row in rows])) if rows else None,
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


def evaluate_image(dataset, probe, image_id, layer, patches_dir):
    coco = dataset.coco
    masked = [(ann, coco.annToMask(ann)) for ann in coco.imgToAnns[image_id] if ann.get("segmentation")]
    masked = [(ann, mask) for ann, mask in masked if mask.any()]
    record = {
        "image_id": image_id, "file_name": coco.imgs[image_id]["file_name"],
        "status": "no_gt_masks", "instances": [],
    }
    if not masked:
        return record
    # Reuse the Gradio builder verbatim: sorted masked GT classes, A-Z, identical text.
    sample = dataset.load(image_id)
    prepared, scores, context_mass, positions, ids = read_option_attention(probe, sample, layer)
    names = [category.name for category in sample.categories]
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
    for ann, mask in masked:
        if ann.get("iscrowd") or ann["area"] <= 0:
            continue
        coverage = patch_coverage(mask, prepared.grid)
        winner, votes, support = instance_vote(scores, rankings, coverage)
        record["instances"].append({
            "image_id": image_id, "annotation_id": ann["id"],
            "class": coco.cats[ann["category_id"]]["name"], "prediction": names[winner],
            "area": ann["area"], "image_class_count": len(names),
            "vote_weights": votes.tolist(), "attention_support": support.tolist(),
        })
    record.update({
        "status": "evaluated" if record["instances"] else "no_evaluable_instances",
        "prompt": sample.prompt, "classes": names,
        "options": [category.letter for category in sample.categories],
        "option_positions": positions, "option_token_ids": ids,
        "image_start": prepared.image_start, "sequence_length": int(prepared.inputs["input_ids"].shape[1]),
        "grid": {
            "rows": prepared.grid.rows, "cols": prepared.grid.cols,
            "x_edges": prepared.grid.x_edges, "y_edges": prepared.grid.y_edges,
        },
        "patch_file": f"patches/{patch_path.name}",
    })
    return record


def main():
    home = Path.home()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--explorer-source", type=Path, default=DEFAULT_EXPLORER)
    parser.add_argument("--checkpoint", type=Path, default=home / "Projects/vlm-post-training/person-attribute-detection/Models/Qwen/Qwen3.5-4B")
    parser.add_argument("--images", type=Path, default=home / "Datasets/coco2017/val2017")
    parser.add_argument("--annotations", type=Path, default=home / "Datasets/coco2017/annotations/instances_val2017.json")
    parser.add_argument("--layer", type=int, help="One-based full-attention layer; default: last full-attention layer")
    parser.add_argument("--limit", type=int, help="For smoke checks only; default: every validation image")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Reuse completed per-image records with identical settings")
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
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    cap_settings = args.gpu_memory_caps_gib if args.gpu_memory_caps_gib is not None else args.gpu_memory_cap_gib
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
    layer = args.layer if args.layer is not None else probe.full_layers[-1]
    if layer not in probe.full_layers:
        parser.error(f"--layer must be one of {probe.full_layers}")
    config = {
        "algorithm": "mean_head_option_attention_patch_vote_v1", "layer": layer,
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
    records_dir, patches_dir = args.output / "records", args.output / "patches"
    records_dir.mkdir(exist_ok=True)
    patches_dir.mkdir(exist_ok=True)
    records, started = [], time.monotonic()
    print(f"Selected {len(image_ids)}/{len(dataset.coco.imgs)} images; layer {layer}; mean over {probe.head_count} heads", flush=True)
    for index, image_id in enumerate(image_ids, start=1):
        path = records_dir / f"{image_id:012d}.json"
        if args.resume and path.is_file():
            record = json.loads(path.read_text())
            if record["image_id"] != image_id or record.get("patch_file") and not (args.output / record["patch_file"]).is_file():
                raise ValueError(f"Incomplete or mismatched image record: {path}")
        else:
            record = evaluate_image(dataset, probe, image_id, layer, patches_dir)
            # The JSON record is the completion marker; write it after the patch artifact.
            atomic_json(path, record)
        records.append(record)
        if index == 1 or index % 10 == 0 or index == len(image_ids):
            summary = summarize(records, len(dataset.coco.imgs), len(image_ids), layer)
            atomic_json(args.output / "summary.json", summary)
            if guard is not None:
                guard.write_status("complete" if summary["complete"] else "running", f"image_{index}")
            print(f"{index}/{len(image_ids)} images; {summary['overall']['correct']}/{summary['overall']['total']} correct; {time.monotonic() - started:.1f}s", flush=True)
    for filename, rows in (
        ("images.jsonl", ({key: value for key, value in record.items() if key != "instances"} for record in records)),
        ("instances.jsonl", (row for record in records for row in record["instances"])),
    ):
        temporary = args.output / (filename + ".tmp")
        with temporary.open("w") as output:
            for row in rows:
                output.write(json.dumps(row, allow_nan=False) + "\n")
        temporary.replace(args.output / filename)
    print(json.dumps(summary["overall"], indent=2), flush=True)


if __name__ == "__main__":
    main()
