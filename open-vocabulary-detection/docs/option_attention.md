# COCO option-attention classification

`coco_classification/evaluate_option_attention.py` implements GT-region classification using image-token
attention to option characters. It reuses `CocoImages.load()` and `QwenProbe.prepare()`
from the bundled `qwen_token_explorer/probe.py` by default. `--explorer-source` can
select an alternate directory containing a compatible `probe.py`.

The default is every image in COCO val2017, in ascending image-ID order, with no
sampling or class restriction. Single-class images are included. Every image is
accounted for; those without nonempty GT segmentation masks have status
`no_gt_masks`. Missing image files cause an error rather than silently shrinking
the dataset.

## Algorithm

```text
for each validation image:
    collect nonempty GT segmentation masks
    if none exist: record no_gt_masks and continue

    reuse Gradio's prompt builder:
        sort the unique masked GT category names alphabetically
        assign A, B, C, ... (at most 26 categories)
        prompt = "Find every instance of the listed categories in the image. "
                 "Match each object region to the label after its category name.\n"
        append "\n{category} = {option}" for each category
        put this text before the image in the user message

    verify each actual option span occupies one distinct token
    record exact option positions in the processed input sequence
    run the frozen model once, without generation

    at the selected full-attention layer (default: last):
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

report overall, per-class, size-group, and class-/image-averaged accuracy
report single-class and multiple-class results separately
```

All query-to-key attention weights are computed with the model's full-context
softmax. Only the option columns are retained after that computation. Scores use
outgoing image-token queries and option-token keys. Repeated option characters
elsewhere in the prompt are excluded by tracking positions. The layer's attention
weights are averaged across heads before ranking. GT masks are used for prompt
category selection and evaluation aggregation, and are not model inputs. Crowd
categories remain prompt candidates to match Gradio, but crowd regions are not
scored as instances. These are classification accuracies, not detection mAP.

## Run

From this repository root, using the existing runtime:

```bash
CUDA_VISIBLE_DEVICES=1,3 OMP_NUM_THREADS=2 \
  /home/nikhileswara/miniconda3/envs/rtv4/bin/python -B -u \
  open-vocabulary-detection/evaluate_option_attention.py \
  --output outputs/qwen_coco_option_attention_val2017
```

`--layer 32` explicitly selects a one-based full-attention layer. `--limit 3` runs
a smoke check; omit it for the full dataset. Choose visible GPUs with adequate free
memory. The default checkpoint, images, annotations, and Gradio source paths can
all be overridden with the corresponding CLI arguments.

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
process making a concurrent allocation. Intermediate-layer attention is not saved.

For a smaller footprint, the tested CPU-offload configuration is
`--gpu-memory-cap-gib 1.75 --model-memory-gib 1.25 --min-free-gib 4`.

Completed images are saved atomically. To continue a run, repeat its command with
`--resume` and the same settings. Resume checks the selected image IDs, paths,
annotation contents, checkpoint configuration, and evaluator/Gradio source hashes.

## Outputs

- `config.json`: settings, selected IDs, and source hashes.
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
