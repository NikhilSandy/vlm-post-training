from __future__ import annotations

import html
from dataclasses import dataclass

import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageDraw

from .probe import Prepared, Sample, Scores, VocabularyReadout


MARGIN = "Option cosine margin"
OPTION_COS = "Option cosine"
WORD_COS = "Class word cosine"
OPTION_ATTN = "Option attention"
WORD_ATTN = "Class word attention"
ALL_ATTN = "All-option attention mass"
METRICS = [MARGIN, OPTION_COS, WORD_COS, OPTION_ATTN, WORD_ATTN, ALL_ATTN]
TOKEN_COS = "Image-token cosine similarity"
TOKEN_ATTN = "Image-token attention"
TOKEN_METRICS = [TOKEN_COS, TOKEN_ATTN]
HEAD_OUTPUTS = "Head outputs"
LAYER_HIDDEN = "Layer hidden state"
POST_RMS = "Post-RMSNorm"
VALUE_ONLY = "Value only"
COSINE_BASES = [HEAD_OUTPUTS, LAYER_HIDDEN, POST_RMS, VALUE_ONLY]
PER_HEAD_BASES = (HEAD_OUTPUTS, VALUE_ONLY)
COLORS = (
    "#41e0cf", "#ffc36e", "#ff8f9d", "#8eabff", "#c5a5ff", "#b3e57b", "#ffa2d7",
    "#88e3ff", "#e9d66b", "#f5a58b", "#c5c9d8", "#b1c4f0",
)


@dataclass(frozen=True)
class Frame:
    sample: Sample
    prepared: Prepared
    masked: Image.Image
    outlines: tuple[np.ndarray, ...]
    scores: Scores | None = None

    @property
    def grid(self):
        return self.prepared.grid


def color(index):
    return COLORS[index % len(COLORS)]


def frame_for(sample, prepared):
    base = np.asarray(sample.image).astype(np.float32)
    outlines = []
    for i, category in enumerate(sample.categories):
        mask = category.mask
        edge = category.outline
        rgb = np.array([int(color(i)[p:p + 2], 16) for p in (1, 3, 5)])
        base[mask] = 0.63 * base[mask] + 0.37 * rgb
        base[edge] = rgb
        outlines.append(edge)
    return Frame(sample, prepared, Image.fromarray(base.astype(np.uint8)), tuple(outlines))


def draw_grid(source, grid, index, show_grid):
    image = source.copy()
    draw = ImageDraw.Draw(image, "RGBA")
    if show_grid:
        for x in grid.x_edges[1:-1]:
            draw.line((x, 0, x, image.height), fill=(233, 249, 255, 95))
        for y in grid.y_edges[1:-1]:
            draw.line((0, y, image.width, y), fill=(233, 249, 255, 95))
    x0, y0, x1, y1 = grid.bounds(index)
    draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline="black", width=4)
    draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline="white", width=2)
    return image


def option_margins(scores):
    if scores.shape[1] == 1:
        return None
    winner = scores.argmax(axis=1)
    top_two = np.partition(scores, -2, axis=1)[:, -2:]
    competitors = np.where(
        np.arange(scores.shape[1])[None, :] == winner[:, None],
        top_two[:, :1], top_two[:, 1:],
    )
    return scores - competitors


def attention_view(frame, layer, head):
    weights = frame.scores.attention[int(layer)]
    prompt = weights.mean(axis=0) if head == "Mean" else weights[int(head)]
    letters = prompt[:, frame.prepared.letters]
    words = np.column_stack([prompt[:, list(span)].sum(axis=1) for span in frame.prepared.names])
    return prompt, letters, words


def cosine_head_index(scores, layer, head, basis):
    if basis == VALUE_ONLY:
        query_count = scores.attention[int(layer)].shape[0]
        kv_count = scores.value_cosine[int(layer)].image_vectors.shape[0]
        return int(head) // (query_count // kv_count)
    return int(head)


def cosine_caption(frame, layer, head, basis):
    if basis == POST_RMS:
        return f"layer {layer} · post-RMSNorm (MLP input)"
    if basis == LAYER_HIDDEN:
        return f"layer {layer} · post-MLP hidden state (heads combined)"
    if basis == VALUE_ONLY:
        if head == "Mean":
            return f"layer {layer} · value only · mean across KV heads"
        kv_head = cosine_head_index(frame.scores, layer, head, basis)
        return f"layer {layer} · value only · KV head {kv_head} (query head {head})"
    selection = "mean of head cosine scores" if head == "Mean" else f"head {head} output"
    return f"layer {layer} · {selection}"


def cosine_at_layer(scores, layer, basis):
    if basis == POST_RMS:
        return scores.post_rms_cosine[int(layer)]
    if basis == LAYER_HIDDEN:
        return scores.layer_cosine[int(layer)]
    if basis == VALUE_ONLY:
        return scores.value_cosine[int(layer)]
    return scores.head_cosine[int(layer)]


def selected_cosines(frame, layer, head, basis):
    scores = cosine_at_layer(frame.scores, layer, basis)
    if basis not in PER_HEAD_BASES:
        return scores.word_cosine, scores.letter_cosine
    if head == "Mean":
        return scores.word_cosine.mean(axis=0), scores.letter_cosine.mean(axis=0)
    index = cosine_head_index(frame.scores, layer, head, basis)
    return scores.word_cosine[index], scores.letter_cosine[index]


def heatmap(frame, token, metric, category, margins, letters, words, grid_lines, cosine, scale=None,
            caption=""):
    grid = frame.grid
    values = {
        MARGIN: margins,
        OPTION_COS: cosine[1],
        WORD_COS: cosine[0],
        OPTION_ATTN: letters,
        WORD_ATTN: words,
        ALL_ATTN: letters.sum(axis=1, keepdims=True),
    }[metric]
    if values is None:
        return None, ""
    if metric == MARGIN:
        edge = max(float(np.abs(values).max()), 1e-6)
        lo, hi, palette = -edge, edge, "coolwarm"
    elif metric in (OPTION_COS, WORD_COS):
        lo, hi = float(values.min()), float(values.max())
        hi, palette = max(hi, lo + 1e-6), "viridis"
    else:
        lo, hi, palette = 0.0, max(float(values.max()), 1e-8), "magma"

    if scale is not None:
        lo, hi = scale
    ramp = colormaps[palette]
    normalized = (values[:, 0 if metric == ALL_ATTN else category] - lo) / (hi - lo)
    rgb = ramp(normalized, bytes=True)[:, :3].reshape(grid.rows, grid.cols, 3)
    rgb = np.repeat(np.repeat(rgb, np.diff(grid.y_edges), axis=0), np.diff(grid.x_edges), axis=1)
    blended = (0.3 * np.asarray(frame.sample.image) + 0.7 * rgb).astype(np.uint8)
    label = "All option letters"
    if metric != ALL_ATTN:
        blended[frame.outlines[category]] = (255, 255, 255)
        item = frame.sample.categories[category]
        label = f"{item.letter} · {item.name}"
    if caption:
        label += f" · {caption}"
    overlay = draw_grid(Image.fromarray(blended), grid, token, grid_lines)
    gradient = ", ".join(f"rgb({r}, {g}, {b})" for r, g, b, _ in ramp(np.linspace(0, 1, 7), bytes=True))
    legend = (
        f'<div class="explorer-scale">{html.escape(label)}'
        f'<div class="explorer-gradient" style="background:linear-gradient(to right,{gradient})"></div>'
        f'<div class="explorer-scale-values"><span>{lo:.4g}</span><span>{hi:.4g}</span></div></div>'
    )
    return overlay, legend


def category_label(index, item):
    return (
        f'<i class="explorer-swatch" style="background:{color(index)}"></i>'
        f"{item.letter} · {html.escape(item.name)}"
    )


def class_legend(sample):
    tags = "".join(
        f"<span>{category_label(i, item)} × {item.instances}</span>"
        for i, item in enumerate(sample.categories)
    )
    return f'<div class="explorer-legend">{tags}</div>'


def table(columns, rows):
    headings = "".join(f"<th>{title}</th>" for title in columns)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return (f'<div class="explorer-table-wrap"><table class="explorer-table"><thead><tr>{headings}'
            f'</tr></thead><tbody>{body}</tbody></table></div>')


def vocabulary_view(readout: VocabularyReadout):
    mass = float(readout.probabilities.sum(dtype=np.float64))
    source = "Final normalized state" if readout.layer == "Final" else f"Layer {readout.layer} · logit lens"
    note = (
        f"<p><b>Image token {readout.image_token}</b> · {source}<br>"
        f"Softmax over {readout.vocabulary_size:,} vocabulary entries · temperature 1. "
        f"Top {len(readout.token_ids)} probability mass: {mass:.4%}; remaining: {max(0., 1 - mass):.4%}.</p>"
    )
    rows = [(str(rank), str(int(token_id)), f"<code>{html.escape(repr(token))}</code>", f"{float(probability):.8%}")
            for rank, (token_id, token, probability) in enumerate(
                zip(readout.token_ids, readout.tokens, readout.probabilities), start=1)]
    return note + table(("Rank", "Token ID", "Decoded token", "Probability"), rows)


def compare_readouts(frame, index, category, margins, letters, words, grid_lines, cosine, caption):
    combined = np.concatenate(cosine, axis=1)
    lo, hi = float(combined.min()), float(combined.max())
    cosine_scale = lo, max(hi, lo + 1e-6)
    attention_scale = 0.0, max(float(words.max()), float(letters.max()), 1e-8)
    maps = []
    for pair, scale in (((WORD_COS, OPTION_COS), cosine_scale), ((WORD_ATTN, OPTION_ATTN), attention_scale)):
        for metric in pair:
            overlay, _ = heatmap(frame, index, metric, category, margins, letters, words, grid_lines,
                                 cosine, scale)
            basis = f" · {caption}" if metric in (WORD_COS, OPTION_COS) else ""
            maps.append((overlay, f"{metric}{basis} · {scale[0]:.4g} to {scale[1]:.4g}"))
    return maps


def relationship_values(frame, index, metric, layer, head, basis=HEAD_OUTPUTS):
    if metric == TOKEN_COS:
        vectors = cosine_at_layer(frame.scores, layer, basis).image_vectors
        if basis not in PER_HEAD_BASES:
            return np.clip(vectors @ vectors[index], -1, 1)
        if head == "Mean":
            return np.clip(np.einsum("hnd,hd->hn", vectors, vectors[:, index]).mean(axis=0), -1, 1)
        head_index = cosine_head_index(frame.scores, layer, head, basis)
        return np.clip(vectors[head_index] @ vectors[head_index, index], -1, 1)
    weights = frame.scores.image_attention[int(layer)]
    # Select the query before averaging to avoid averaging the entire N x N map.
    return weights[:, index, :].mean(axis=0) if head == "Mean" else weights[int(head), index, :]


def relationship_views(frame, index, metric, layer, head, grid_lines, basis=HEAD_OUTPUTS):
    values = relationship_values(frame, index, metric, layer, head, basis)
    grid = frame.grid
    candidates = np.arange(grid.count)
    candidates = candidates[candidates != index]
    if metric == TOKEN_COS:
        lo = float(values.min())
        hi = max(float(values.max()), lo + 1e-6)
        palette = "viridis"
        basis_note = {
            POST_RMS: "This compares the token vectors that enter the MLP.",
            VALUE_ONLY: "This compares raw value vectors before attention weights mix tokens.",
            HEAD_OUTPUTS: "Head vectors are captured after gating, before heads are mixed.",
            LAYER_HIDDEN: "The hidden state is captured after the MLP residual addition.",
        }[basis]
        note = (
            f"Source token <b>{index}</b> · cosine similarity to every image token. "
            "The white box marks the source; its similarity to itself is 1. "
            f"Uses {cosine_caption(frame, layer, head, basis)}. "
            f"{basis_note}"
        )
        score_label = "Cosine"
    else:
        lo, hi, palette = 0.0, max(float(values.max()), 1e-8), "magma"
        head_label = "mean over heads" if head == "Mean" else f"head {head}"
        note = (
            f"Source token <b>{index}</b> · layer {layer} · {head_label} · "
            f"{values.sum():.3%} attention to image tokens. "
            "Shows attention from the source to other image tokens. "
            "Later tokens are causally masked (zero); values retain their original attention mass."
        )
        candidates = candidates[candidates < index]
        score_label = "Attention"
    ramp = colormaps[palette]
    rgb = ramp((values - lo) / (hi - lo), bytes=True)[:, :3].reshape(grid.rows, grid.cols, 3)
    rgb = np.repeat(np.repeat(rgb, np.diff(grid.y_edges), axis=0), np.diff(grid.x_edges), axis=1)
    blended = (0.3 * np.asarray(frame.sample.image) + 0.7 * rgb).astype(np.uint8)
    overlay = draw_grid(Image.fromarray(blended), grid, index, grid_lines)
    gradient = ", ".join(f"rgb({r}, {g}, {b})" for r, g, b, _ in ramp(np.linspace(0, 1, 7), bytes=True))
    legend = (
        f'<div class="explorer-scale">{html.escape(metric)} · source token {index}'
        + (f" · {html.escape(cosine_caption(frame, layer, head, basis))}" if metric == TOKEN_COS else "")
        + f'<div class="explorer-gradient" style="background:linear-gradient(to right,{gradient})"></div>'
        + f'<div class="explorer-scale-values"><span>{lo:.4g}</span><span>{hi:.4g}</span></div></div>'
    )
    top = candidates[np.argsort(-values[candidates], kind="stable")[:8]]
    rows = [
        (str(int(token)), str(int(token // grid.cols)), str(int(token % grid.cols)),
         f"{values[token]:+.4f}" if metric == TOKEN_COS else f"{values[token]:.3%}")
        for token in top
    ]
    neighbors = (
        f"<p>{note}</p>"
        + (table(("Other token", "Row", "Column", score_label), rows) if rows
           else "<p>The first image token has no earlier image tokens to attend to.</p>")
    )
    return overlay, legend, neighbors


def token_views(frame, index, metric, category, layer, head, masks, grid_lines, compare=False,
                relationship=TOKEN_COS, cosine_basis=HEAD_OUTPUTS):
    grid = frame.grid
    sample = frame.sample
    row, col = divmod(index, grid.cols)
    x0, y0, x1, y1 = grid.bounds(index)
    image = draw_grid(frame.masked if masks else sample.image, grid, index, grid_lines)
    crop = sample.image.crop((x0, y0, x1, y1))
    scale = 240 / max(crop.size)
    crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
                       Image.Resampling.NEAREST)
    detail = (
        f"**Token {index}** · row {row}, column {col}  \n"
        f"Sequence position **{frame.prepared.image_start + index}**  \n"
        f"Model input {grid.processed_width} × {grid.processed_height} px  \n"
        f"Original crop x {x0}–{x1 - 1}, y {y0}–{y1 - 1} · {x1 - x0} × {y1 - y0} px"
    )
    rows = [
        (category_label(i, item), f"{item.mask[y0:y1, x0:x1].mean():.1%}")
        for i, item in enumerate(sample.categories)
    ]
    columns = ("GT class", "Mask coverage")
    scores = frame.scores
    if scores is None:
        return image, None, "", crop, detail, table(columns, rows), "", None, "", "", []

    cosine = selected_cosines(frame, layer, head, cosine_basis)
    caption = cosine_caption(frame, layer, head, cosine_basis)
    margins = option_margins(cosine[1])
    prompt, letters, words = attention_view(frame, layer, head)
    map_caption = caption if metric in (MARGIN, OPTION_COS, WORD_COS) else (
        f"layer {layer} · " + ("mean over heads" if head == "Mean" else f"head {head}"))
    overlay, legend = heatmap(frame, index, metric, int(category), margins, letters, words, grid_lines,
                              cosine, caption=map_caption)
    for i, item in enumerate(rows):
        rows[i] = (*item, f"{cosine[0][index, i]:+.4f}", f"{cosine[1][index, i]:+.4f}",
                   f"{margins[index, i]:+.4f}" if margins is not None else "—",
                   f"{words[index, i]:.3%}", f"{letters[index, i]:.3%}")
    columns += ("Word cosine", "Option cosine", "Option margin", "Word attention", "Option attention")
    weights = prompt[index]
    top = weights.argsort()[::-1][:12]
    peak = max(float(weights[top[0]]), 1e-8)
    prompt_rows = [
        (f"<code>{html.escape(frame.prepared.prompt_tokens[i])}</code>",
         str(frame.prepared.prompt_start + i), f"{weights[i]:.3%}",
         f'<div class="explorer-bar" style="width:{weights[i] / peak:.1%}"></div>')
        for i in top
    ]
    head_label = "mean over heads" if head == "Mean" else f"head {head}"
    prompt_table = (
        f"<p>Layer {layer} · {head_label} · {letters[index].sum():.3%} attention to all options</p>"
        + table(("Top prompt token", "Sequence position", "Attention", ""), prompt_rows)
    )
    comparison = (
        compare_readouts(frame, index, int(category), margins, letters, words, grid_lines, cosine, caption)
        if compare else []
    )
    related = relationship_views(frame, index, relationship, layer, head, grid_lines, cosine_basis)
    scores_table = f"<p>Cosine: {html.escape(caption)}</p>" + table(columns, rows)
    return image, overlay, legend, crop, detail, scores_table, prompt_table, *related, comparison
