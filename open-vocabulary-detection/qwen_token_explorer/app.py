from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import gradio as gr

from .probe import CocoImages, QwenProbe
from .views import (COSINE_BASES, HEAD_OUTPUTS, MARGIN, METRICS, TOKEN_COS, TOKEN_METRICS,
                   class_legend, frame_for, token_views, vocabulary_view)


CSS = """
.gradio-container { width: 100% !important; max-width: 1440px !important; min-width: 0 !important; margin: auto !important; }
#explorer-heading { border-bottom: 1px solid var(--border-color-primary); padding: 12px 0 16px; }
#explorer-heading h1 { font-size: clamp(1.55rem, 3vw, 2.75rem); letter-spacing: -.04em; margin-bottom: 4px; }
#explorer-heading p { color: var(--body-text-color-subdued); margin: 0; }
#explorer-status { align-self: center; }
#random-image { min-width: 192px !important; }
#token-grid img, #score-map img, #image-relationship-map img { cursor: crosshair; }
#patch-crop img { image-rendering: pixelated; }
.explorer-legend { display: flex; gap: 5px 12px; flex-wrap: wrap; padding: 5px 0; }
.explorer-legend span { white-space: nowrap; }
.explorer-swatch { width: .68em; height: .68em; border-radius: 50%; display: inline-block; margin-right: 5px; }
.explorer-table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
.explorer-table th, .explorer-table td { text-align: right; padding: 7px 9px; border-bottom: 1px solid var(--border-color-primary); }
.explorer-table th:first-child, .explorer-table td:first-child { text-align: left; }
.explorer-table th { white-space: nowrap; font-weight: 600; }
.explorer-table th:last-child:empty { min-width: 76px; }
.explorer-table-wrap { overflow-x: auto; }
.explorer-bar { height: 7px; border-radius: 3px; background: #269cba; min-width: 1px; }
.explorer-scale { padding: 3px 0; font-size: .83rem; color: var(--body-text-color-subdued); }
.explorer-gradient { height: 6px; border-radius: 3px; margin: 4px 0; }
.explorer-scale-values { display: flex; justify-content: space-between; }
@media (max-width: 920px) {
    #image-workspace, #image-pairs, #explorer-footer, #image-relationships { flex-direction: column; min-width: 0; }
    #image-workspace > *, #image-pairs > *, #explorer-footer > *, #image-relationships > * {
        width: 100% !important; min-width: 0 !important;
    }
    #token-scores, #prompt-attention { min-width: 0 !important; max-width: 100%; }
}
@media (max-width: 540px) {
    .gradio-container .main > .wrap { padding-inline: 12px !important; }
    #explorer-toolbar { flex-direction: column; gap: 3px; }
    #explorer-status { align-self: stretch; }
}
"""


def build_app(dataset: CocoImages, probe: QwenProbe):
    final_layer = str(probe.full_layers[-1])
    with gr.Blocks(title="Qwen Vision Token Explorer", delete_cache=(600, 3600)) as app:
        frame = gr.State(time_to_live=3600)
        gr.Markdown("# Qwen Vision Token Explorer\nCOCO val2017 · Qwen3.5-4B · Frozen forward pass",
                    elem_id="explorer-heading")
        with gr.Row(elem_id="explorer-toolbar"):
            random_button = gr.Button("Random COCO image", variant="primary", scale=0, elem_id="random-image")
            status = gr.Markdown("Draw an annotated COCO image to begin.", elem_id="explorer-status")

        with gr.Row(elem_id="image-workspace"):
            with gr.Column(scale=4, min_width=330):
                with gr.Row(elem_id="image-pairs"):
                    with gr.Column(min_width=300):
                        gr.Markdown("### Ground truth and token grid")
                        grid_image = gr.Image(height=362, format="png", interactive=False,
                                              show_label=False, elem_id="token-grid", buttons=[])
                        category_legend = gr.HTML()
                        with gr.Row():
                            masks = gr.Checkbox(value=True, label="GT masks", interactive=False)
                            grid_lines = gr.Checkbox(value=True, label="Token grid", interactive=False)
                    with gr.Column(min_width=300):
                        gr.Markdown("### Heatmap")
                        map_image = gr.Image(height=362, format="png", interactive=False,
                                             show_label=False, elem_id="score-map", buttons=[])
                        map_legend = gr.HTML()
                        with gr.Row():
                            readout = gr.Dropdown(choices=METRICS, value=MARGIN, label="Readout",
                                                  interactive=False, elem_id="readout")
                            category = gr.Dropdown(choices=[], label="Class", interactive=False,
                                                   elem_id="map-class")
                        with gr.Row():
                            layer = gr.Dropdown(choices=list(map(str, probe.full_layers)),
                                                value=final_layer, label="Full-attention layer",
                                                interactive=False, elem_id="attention-layer")
                            head = gr.Dropdown(choices=["Mean", *map(str, range(probe.head_count))],
                                               value="Mean", label="Attention head", interactive=False,
                                               elem_id="attention-head")
                        cosine_basis = gr.Dropdown(
                            choices=COSINE_BASES, value=HEAD_OUTPUTS, label="Cosine representation",
                            info=("Post-RMSNorm compares MLP inputs. Value only compares raw attention values "
                                  "in the KV head shared by the selected query head. Mean averages the KV-head cosine scores."),
                            interactive=False, elem_id="cosine-representation",
                        )
            with gr.Column(scale=1, min_width=260):
                gr.Markdown("### Selected vision token")
                index = gr.Number(value=0, minimum=0, precision=0, step=1, label="Token index",
                                  interactive=False, elem_id="token-index")
                with gr.Row():
                    previous = gr.Button("Previous", size="sm", min_width=65, interactive=False)
                    following = gr.Button("Next", size="sm", min_width=45, interactive=False)
                crop = gr.Image(label="Patch crop", height=240, format="png", interactive=False,
                                elem_id="patch-crop", buttons=[])
                detail = gr.Markdown("Click a cell in any grid or enter its zero-based index.",
                                     elem_id="token-detail")

        gr.Markdown("### Image-token relationships")
        gr.Markdown("Click a token in this grid or either grid above. The same source token is selected in all three.")
        with gr.Row(elem_id="image-relationships"):
            with gr.Column(scale=3, min_width=300):
                relationship_image = gr.Image(height=400, format="png", interactive=False,
                                              show_label=False, elem_id="image-relationship-map", buttons=[])
                relationship_legend = gr.HTML(elem_id="image-relationship-legend")
            with gr.Column(scale=2, min_width=280):
                relationship = gr.Dropdown(choices=TOKEN_METRICS, value=TOKEN_COS,
                                           label="Image-token readout", interactive=False,
                                           elem_id="image-token-readout")
                related_tokens = gr.HTML(elem_id="related-image-tokens")

        gr.Markdown("### Selected patch vocabulary · top 10")
        vocabulary_layer = gr.Dropdown(
            choices=["Final", *map(str, range(1, probe.layer_count + 1))], value="Final",
            label="Vocabulary layer", interactive=False, elem_id="vocabulary-layer",
            info="Final or any decoder layer. Intermediate layers use the final norm and language head (logit lens).",
        )
        gr.Markdown("Next-token vocabulary projection at the selected image-token position. Uses the combined layer hidden state.")
        vocabulary_table = gr.HTML(elem_id="patch-vocabulary")

        compare = gr.Checkbox(label="Compare class word and option maps", interactive=False,
                              elem_id="compare-maps")
        comparison = gr.Gallery(
            columns=2, rows=2, object_fit="contain", height=690, visible=False,
            show_label=False, allow_preview=False, elem_id="comparison-gallery", format="png",
        )
        gr.Markdown("### Selected token scores")
        scores_table = gr.HTML(elem_id="token-scores")
        with gr.Row(elem_id="explorer-footer"):
            prompt = gr.Textbox(label="GT-only prompt · text precedes image", lines=9,
                                interactive=False, elem_id="gt-prompt")
            with gr.Column(min_width=300):
                with gr.Accordion("Selected token’s attention to the prompt", open=True):
                    attention_table = gr.HTML(elem_id="prompt-attention")

        views = [grid_image, map_image, map_legend, crop, detail, scores_table, attention_table,
                 relationship_image, relationship_legend, related_tokens, comparison, vocabulary_table]
        settings = [readout, category, layer, head, masks, grid_lines, compare, relationship, cosine_basis,
                    vocabulary_layer]
        controls = [index, *settings, previous, following]

        def render_views(current, token, *options):
            *map_options, vocab_layer = options
            *panels, maps = token_views(current, token, *map_options)
            vocab = (vocabulary_view(probe.vocabulary(current.scores, token, vocab_layer))
                     if current.scores is not None else "Vocabulary probabilities appear after the forward pass.")
            return (*panels, gr.update(value=maps, visible=bool(maps)), vocab)

        def draw_random(current, show_masks, show_grid, show_comparison):
            start = perf_counter()
            try:
                sample = dataset.random_image(current.sample.image_id if current else None)
                prepared = probe.prepare(sample)
                current = frame_for(sample, prepared)
            except Exception as exc:
                logging.exception("Could not prepare a COCO image")
                raise gr.Error(str(exc)) from exc

            choices = [(f"{c.letter} · {c.name}", str(i)) for i, c in enumerate(sample.categories)]
            metrics = METRICS if len(choices) > 1 else METRICS[1:]
            default_metric = metrics[0]
            grid = current.grid
            description = (
                f"**COCO {sample.image_id:012}** · {sample.image.width} × {sample.image.height} px · "
                f"{grid.rows} × {grid.cols} merged grid · {grid.count} tokens"
            )

            def render(state, running, message):
                ready = state.scores is not None
                enabled = not running
                return (
                    state, gr.update(interactive=enabled), f"{description}  \n{message}",
                    sample.prompt, class_legend(sample),
                    gr.update(value=0, maximum=grid.count - 1, interactive=enabled),
                    gr.update(choices=metrics, value=default_metric, interactive=ready),
                    gr.update(choices=choices, value="0", interactive=ready),
                    gr.update(value=final_layer, interactive=ready),
                    gr.update(value="Mean", interactive=ready),
                    gr.update(interactive=enabled), gr.update(interactive=enabled),
                    gr.update(interactive=ready), gr.update(value=TOKEN_COS, interactive=ready),
                    gr.update(value=HEAD_OUTPUTS, interactive=ready),
                    gr.update(value="Final", interactive=ready),
                    gr.update(interactive=enabled), gr.update(interactive=enabled),
                    *render_views(state, 0, default_metric, "0", final_layer, "Mean",
                                  show_masks, show_grid, show_comparison, TOKEN_COS, HEAD_OUTPUTS, "Final"),
                )

            phase = "Running Qwen forward pass…" if probe.model else "Loading Qwen for the first image…"
            yield render(current, True, phase)
            try:
                current = replace(current, scores=probe.run(prepared))
            except Exception as exc:
                logging.exception("Qwen forward pass failed")
                yield render(current, False, f"**Qwen error** · {exc}")
                return
            yield render(current, False, f"Forward pass complete · {perf_counter() - start:.1f}s")

        def inspect_token(current, token, *options):
            if current is None:
                return [gr.skip()] * len(views)
            token = min(current.grid.count - 1, max(0, int(token or 0)))
            return render_views(current, token, *options)

        def inspect_click(current, metric, cls, attention_layer, attention_head, show_masks, show_grid,
                          show_comparison, relationship_metric, cosine_representation, vocab_layer,
                          event: gr.SelectData):
            if current is None or event.index is None:
                return [gr.skip()] * (len(views) + 1)
            token = current.grid.index_at(*event.index)
            return token, *render_views(
                current, token, metric, cls, attention_layer, attention_head, show_masks, show_grid,
                show_comparison, relationship_metric, cosine_representation, vocab_layer,
            )

        def move_token(step):
            def move(current, token, *options):
                if current is None:
                    return [gr.skip()] * (len(views) + 1)
                token = min(current.grid.count - 1, max(0, int(token or 0) + step))
                return token, *render_views(current, token, *options)
            return move

        random_button.click(
            draw_random, [frame, masks, grid_lines, compare],
            [frame, random_button, status, prompt, category_legend, *controls, *views],
            concurrency_id="qwen-forward", concurrency_limit=1, show_progress="hidden",
            api_name="random_image",
        )
        index.input(
            move_token(0), [frame, index, *settings], [index, *views],
            queue=False, show_progress="hidden", api_visibility="private",
        )
        for component in settings:
            component.input(
                inspect_token, [frame, index, *settings], views,
                queue=False, show_progress="hidden", api_visibility="private",
            )
        for component in (grid_image, map_image, relationship_image):
            component.select(
                inspect_click, [frame, *settings], [index, *views],
                queue=False, show_progress="hidden", api_visibility="private",
            )
        for component, step in ((previous, -1), (following, 1)):
            component.click(
                move_token(step), [frame, index, *settings], [index, *views],
                queue=False, show_progress="hidden", api_visibility="private",
            )
    return app


def main():
    home = Path.home()
    parser = argparse.ArgumentParser(description="Explore Qwen3.5 image-token states on COCO")
    parser.add_argument("--checkpoint", type=Path, default=(
        home / "Projects/vlm-post-training/person-attribute-detection/Models/Qwen/Qwen3.5-4B"
    ))
    parser.add_argument("--images", type=Path, default=home / "Datasets/coco2017/val2017")
    parser.add_argument("--annotations", type=Path, default=(
        home / "Datasets/coco2017/annotations/instances_val2017.json"
    ))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()
    dataset = CocoImages(args.images, args.annotations)
    print(f"Ready to sample {len(dataset.image_ids)} masked COCO images", flush=True)
    probe = QwenProbe(args.checkpoint)
    app = build_app(dataset, probe)
    app.queue(default_concurrency_limit=1).launch(
        server_name=args.host, server_port=args.port, share=False, inbrowser=False,
        css=CSS, theme=gr.themes.Soft(primary_hue="cyan", secondary_hue="slate"), max_threads=8,
    )


if __name__ == "__main__":
    main()
