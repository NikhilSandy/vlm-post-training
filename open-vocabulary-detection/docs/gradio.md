# Qwen Vision Token Explorer

The Gradio app samples masked images from the local COCO val2017 split and inspects a frozen Qwen3.5-4B forward pass. It uses the class names from the selected image’s masks to build a text-first prompt, and assigns each class a single-token letter.

## Run

```bash
./run.sh
```

Open [the local server](http://127.0.0.1:7861/). The launcher uses the machine’s `rtv4` environment and GPUs 0–3 by default. `CUDA_VISIBLE_DEVICES`, `QWEN_EXPLORER_PYTHON`, and `./run.sh --help` provide device, Python, and path overrides. The first image loads the model, which stays available for the next draw. Preview files use a private directory in shared memory.

## Explore

- **Random COCO image** updates the image, masks, and GT-only prompt, then runs the forward pass.
- Click either main image or enter a zero-based token index. The two highlights, original-image crop, mask coverage, and score tables update together. The grid follows Qwen’s merged vision-token positions.
- **Readout** switches between option cosine margin, word and option cosine, word and option attention, and the sum of attention to option letters. The margin is available when an image has at least two GT classes.
- **Full-attention layer** selects the decoder layer for the cosine and attention maps, including the image-token grid. **Attention head** selects among 16 query heads.
- **Cosine representation** defaults to **Head outputs**, which compares vectors within the selected head after gating and before the output projection. **Mean** averages the 16 head cosine scores in that mode. **Post-RMSNorm** compares the vectors entering the MLP, using the output of the block's learned post-attention RMSNorm. **Layer hidden state** uses the residual stream after the block's MLP update.
- **Value only** compares raw value projections for each patch and text token before attention weights mix token positions. It uses the key/value head shared by the selected query head. In Qwen3.5-4B, query heads 0–3, 4–7, 8–11, and 12–15 share KV heads 0, 1, 2, and 3 respectively. **Mean** averages the four KV-head cosine scores.
- **Compare class word and option maps** shows four maps for the selected class. Each pair uses a shared color scale. Cosine panels and the selected-token scores use the selected cosine representation, layer, and head. The prompt attention table continues to use the selected layer and head's attention weights.
- **Image-token relationships** adds a third clickable grid. Select a source token in any grid, then choose **Image-token cosine similarity** to compare it with every image token, or **Image-token attention** to see its outgoing attention at the selected full-attention layer and head. The nearby table lists the eight strongest other tokens; the source itself is excluded from that ranking.
- **Selected patch vocabulary · top 10** updates when a patch is selected in any grid. Choose **Final** or any of the 32 decoder layers in **Vocabulary layer**. Final states are projected directly through the model's language head. Intermediate states first pass through the model's final learned normalization, then the same language head (logit lens). Softmax uses the entire 248,320-entry vocabulary at temperature 1; the displayed ten probabilities are not renormalized. This is a next-token vocabulary projection at an image-token position, using the combined hidden state rather than one attention head.

Cosine uses normalized vectors from the selected layer and representation. Image tokens, class words, and option letters are compared within the same layer and head space. Multi-token class names are averaged before normalization within each head or layer. Mean cosine pools cosine scores across heads; it does not average head vectors or layers. Option cosine margin is calculated from the selected/pooled option cosine scores. Attention comes from image-token queries to earlier prompt keys. The app runs a forward pass with gradients and generation disabled.

Image-token cosine similarity compares the selected source with every other image token in the selected layer/head representation. Normalized vectors and class cosine scores are cached for all selected layers and heads in one forward pass; source-to-image cosine rows are computed from those cached vectors when a token is selected, avoiding full pairwise matrices for every head. Image-token attention keeps its original probability mass over the full sequence; it is not renormalized over image keys. The decoder is causal, so later image tokens have zero outgoing attention from an earlier source. Clicking a token or changing layer/head does not run the model again.

Raw image-token states from all 32 decoder blocks and final normalized states are cached in that same forward pass. Vocabulary projection is calculated on demand for one selected patch and layer; the top ten results are cached (up to 128 selections per image). This runs the normalization/language head without repeating the decoder or vision encoder. Layer 32's logit-lens projection should match Final, because they use the same final normalization and language head.

Run `python -B -m unittest -v tests.test_relationships` from the project directory to check cosine layer/head selection, mean-of-cosines pooling, head-output capture before mixing, attention direction, hook cleanup, and synchronized Gradio callbacks without loading model weights.
