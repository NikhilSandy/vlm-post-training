# COCO option-attention classification

`coco_classification/evaluate_option_attention.py` implements GT-region classification using image-token
attention to option tokens. It reuses `CocoImages.load()` and `QwenProbe.prepare()`
from the bundled `qwen_token_explorer/probe.py` by default. `--explorer-source` can
select an alternate directory containing a compatible `probe.py`.

The default is every image in COCO val2017, with no sampling or class restriction.
Single-class images are included. Every image is
accounted for; those without nonempty GT segmentation masks have status
`no_gt_masks`. Missing image files cause an error rather than silently shrinking
the dataset.

## Algorithm

```text
for each validation image:
    collect nonempty GT segmentation masks
    if none exist: record no_gt_masks and continue

    build the prompt:
        sort the candidate category names alphabetically
        use the masked GT categories by default
        use every COCO category with --all-80-classes
        assign each category a distinct single-token label
        prompt = "Find every instance of the listed categories in the image. "
                 "Match each object region to the label after its category name.\n"
        append "\n{category} = {option}" for each category
        put this text before the image in the user message

    verify each actual option span occupies one distinct token
    record exact option positions in the processed input sequence
    run the frozen model once, without generation

    for each selected full-attention layer:
        for each image patch p and candidate category c:
            score[p, c] = mean_heads(attention[head, p, option_position[c]])
        preserve original attention normalized over the full causal context
        rank options by descending score, without renormalizing over options
        each patch votes for its top option
        exact patch ties follow alphabetical category order

        for each non-crowd, positive-area GT instance with a nonempty mask:
            coverage[p] = fraction of patch p covered by the instance mask
            votes[c] = sum_p coverage[p] * (patch_vote[p] == c)
            predict the class with the largest vote total
            break ties with sum_p coverage[p] * score[p, c], then alphabetical order
            compare with the instance's GT class

report accuracy for each layer, including single-class and multiple-class results
```

All query-to-key attention weights are computed with the model's full-context
softmax. Only the option columns are retained after that computation. Scores use
outgoing image-token queries and option-token keys. Repeated option characters
elsewhere in the prompt are excluded by tracking positions. The layer's attention
weights are averaged across heads before ranking. GT masks determine how patches
vote for each instance and supply the candidate categories in the default mode.
The model receives the prompt and image. Crowd categories remain eligible prompt
candidates, and accuracy is computed on non-crowd instances.

## Run

From this repository root, using the existing runtime:

```bash
CUDA_VISIBLE_DEVICES=1,3 OMP_NUM_THREADS=2 \
  /home/nikhileswara/miniconda3/envs/rtv4/bin/python -B -u \
  open-vocabulary-detection/evaluate_option_attention.py \
  --output outputs/qwen_coco_option_attention_val2017
```

The default evaluates every even-numbered, one-based full-attention layer in one
forward pass per image batch. For Qwen3.5-4B these are layers 4, 8, 12, 16, 20,
24, 28, and 32. The model's other even-numbered layers use linear attention and
do not expose the query-to-key attention weights used for scoring.
`--layer 32` selects only layer 32. `--limit 3` runs
a smoke check; omit it for the full dataset. Choose visible GPUs with adequate free
memory. The default checkpoint, images, annotations, and Gradio source paths can
all be overridden with the corresponding CLI arguments.

`--all-80-classes` gives every image the same alphabetically ordered list of COCO
categories. The evaluator picks labels from `A`–`Z` and uppercase pairs, checks
that each label is one distinct vocabulary token in the rendered prompt, and
stores the mapping and token IDs in `config.json`. The uniform-guess baseline
is 1/80. Single-class and multi-class accuracy still group images by the classes
present in their ground-truth masks.

Add `--shuffle-options-seed 42` to reassign the same 80 tokens across classes.
Every class receives a different token, the class order stays alphabetical, and
the shuffled mapping is fixed across the dataset. The seed and mapping are
saved in `config.json`.

Add `--per-head` to run the patch vote independently for every head. Each layer's
summary includes the top-1 accuracy of all heads and the highest accuracy.
Head numbers are one-based. The run saves per-instance head predictions in head
order, which can be used to recompute the accuracies.

`--batch-size` sets the maximum number of images in each forward pass. It
defaults to 16, or 1 when CUDA memory caps are set. The evaluator splits a batch
and retries if it runs out of GPU memory. Batches group similarly sized images
to reduce padding. Final JSONL records are in ascending image-ID order.
`--workers` controls CPU-side
preparation and writing. `--vision-sdpa` enables PyTorch SDPA in the vision
encoder while keeping eager attention in the decoder for class scoring.
Progress logs show throughput and peak allocated GPU memory.

For the shared machine, use the guarded configuration across all four GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=2 \
  /home/nikhileswara/miniconda3/envs/rtv4/bin/python -B -u \
  open-vocabulary-detection/evaluate_option_attention.py \
  --gpu-memory-caps-gib 3 4 1.5 4 \
  --model-memory-budgets-gib 2.25 3.25 0.75 3.25 --min-free-gib 4 \
  --output outputs/qwen_coco_option_attention_val2017
```

This uses different PyTorch allocator caps for GPUs 0-3: 3, 4, 1.5, and 4 GiB.
Resident-weight budgets are 2.25, 3.25, 0.75, and 3.25 GiB respectively, placing more
weights on the less occupied GPUs. The vision encoder stays intact on GPU 0;
embedding weights and the tied language head stay together on GPU 1. Decoder blocks
are placed within the budgets, with CPU offload for any that do not fit. Before loading,
each image, the vision encoder, and every decoder layer, admission checks preserve
4 GiB of free VRAM plus remaining allocator capacity and 256 MiB of library workspace.
The run waits when another process reduces that headroom. The vision encoder uses
SDPA to avoid materializing its dense attention matrix; decoder attention stays
eager. Unexpected inputs exceeding 1024 sequence tokens or 2048 unmerged vision
patches are rejected before the forward pass, without resizing the images further.

`device_map.json` records GPU placement and CPU offload; `memory_status.json`
records running/waiting state and allocator peaks. CUDA context/library allocations
are outside PyTorch's allocator cap. These safeguards bound this job's allocator
and monitor shared free memory; they cannot reserve memory against an unrelated
process making a concurrent allocation. Only the patch-to-option scores and
attention-mass totals are saved for each selected layer.

For a smaller footprint, the tested CPU-offload configuration is
`--gpu-memory-cap-gib 1.75 --model-memory-gib 1.25 --min-free-gib 4`.

Image records are saved atomically for each layer. To continue a run, repeat its command with
`--resume` and the same settings. Resume checks the selected image IDs, paths,
annotation contents, checkpoint configuration, and evaluator/Gradio source hashes.

## Outputs

The output root contains `config.json` with settings, selected image IDs, and source
hashes, `summary.json` with overall accuracy for each layer, and
`performance.json` with runtime, throughput, and GPU memory for a fresh run.
Each `layer_XX/`
directory contains the files below. With an explicit `--layer`, these files go
directly in the output root.

- `records/{image_id}.json`: image prompt, option positions/IDs, grid, and object
  predictions; also the completion marker used for resume.
- `patches/{image_id}.npz`: `option_attention` shaped `[patches, categories]`,
  `rankings` containing category indices in descending order, top-option `votes`,
  `context_attention_mass`, and `non_option_attention_mass`. Patch order is row-major;
  category indices refer to `classes`/`options` in the image record.
- `summary.json`: periodically updated accuracy and dataset-coverage metrics.
- `images.jsonl` and `instances.jsonl`: consolidated records, written on completion.

Images without GT masks have a JSON record and no patch artifact. Images with only
crowd masks receive patch predictions but contribute no evaluated instances.

## Checks

```bash
cd open-vocabulary-detection
/home/nikhileswara/miniconda3/envs/rtv4/bin/python -B -m unittest -v tests.test_option_attention
/home/nikhileswara/miniconda3/envs/rtv4/bin/python -B -m unittest -v tests.test_guarded_cuda
```
