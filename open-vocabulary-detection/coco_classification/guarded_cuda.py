"""Bound this evaluator's CUDA allocator and preserve free VRAM for other jobs."""

import json
from pathlib import Path
import time

import torch


GIB = 2**30


def build_device_map(sizes, layer_count, count, budget):
    """Keep vision intact and reserve GPU 0 staging space for CPU decoder blocks."""
    if count < 2:
        raise ValueError("This guarded placement needs at least two visible GPUs")
    budgets = budget if isinstance(budget, list) else [budget] * count
    if len(budgets) != count:
        raise ValueError("Specify exactly one weight budget per visible GPU")
    embedding_device = 1
    device_map = {
        "model.visual": 0, "model.language_model.embed_tokens": embedding_device,
        "lm_head": embedding_device, "model.language_model.norm": 0,
        "model.language_model.rotary_emb": 0,
    }
    resident = [0] * count
    resident[0] = sizes["model.visual"] + sizes["model.language_model.norm"]
    resident[embedding_device] = sizes["model.language_model.embed_tokens"]
    stage = max(sizes[f"model.language_model.layers.{i}"] for i in range(layer_count))
    if resident[0] + stage > budgets[0] or resident[embedding_device] > budgets[embedding_device]:
        raise ValueError("Weight budget cannot fit the vision encoder, embedding, and staging reserve")
    order = [0] + [i for i in range(count) if i not in (0, embedding_device)] + [embedding_device]
    for index in range(layer_count):
        name = f"model.language_model.layers.{index}"
        fitting = [device for device in order if resident[device] + sizes[name] <= budgets[device] - (stage if device == 0 else 0)]
        if fitting:
            device = fitting[0]
            resident[device] += sizes[name]
            device_map[name] = device
        else:
            device_map[name] = "cpu"
    return device_map


class MemoryGuard:
    def __init__(self, cap_gib, reserve_gib, status_path):
        self.reserve = int(reserve_gib * GIB)
        self.status_path = Path(status_path)
        self.count = torch.cuda.device_count()
        if not self.count:
            raise RuntimeError("The guarded evaluator requires visible CUDA GPUs")
        caps = cap_gib if isinstance(cap_gib, list) else [cap_gib] * self.count
        if len(caps) != self.count:
            raise ValueError("Specify exactly one allocator cap per visible GPU")
        self.caps = [int(cap * GIB) for cap in caps]
        self.wait("before_allocator_setup")
        for device in range(self.count):
            _, total = torch.cuda.mem_get_info(device)
            torch.cuda.set_per_process_memory_fraction(float(self.caps[device] / total), device)
        self.write_status("allocator_capped")

    def snapshot(self):
        return [{
            "device": device,
            "free_bytes": torch.cuda.mem_get_info(device)[0],
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "allocated_bytes": torch.cuda.memory_allocated(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        } for device in range(self.count)]

    def has_headroom(self, snapshot):
        # Leave a further 256 MiB for CUDA/library allocations outside the allocator.
        return all(row["free_bytes"] >= self.reserve + 256 * 2**20 + max(
            0, self.caps[row.get("device", index)] - row["reserved_bytes"]
        ) for index, row in enumerate(snapshot))

    def write_status(self, state, phase=None, snapshot=None):
        value = {
            "state": state, "phase": phase, "updated_unix": time.time(),
            "allocator_cap_bytes_per_gpu": self.caps,
            "minimum_free_bytes": self.reserve,
            "gpus": snapshot if snapshot is not None else self.snapshot(),
        }
        temporary = self.status_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(self.status_path)

    def wait(self, phase):
        waiting = False
        while True:
            snapshot = self.snapshot()
            if self.has_headroom(snapshot):
                if waiting:
                    print(f"VRAM reserve restored; continuing {phase}", flush=True)
                return
            waiting = True
            self.write_status("waiting_for_vram", phase, snapshot)
            print(f"Waiting to preserve other jobs' VRAM reserve ({phase}); "
                  f"free GiB: {[round(row['free_bytes'] / GIB, 2) for row in snapshot]}", flush=True)
            # Do not allocate more CUDA memory or touch other processes while waiting.
            time.sleep(10)

    def check_inputs(self, prepared):
        sequence = int(prepared.inputs["input_ids"].shape[1])
        vision_patches = int(prepared.inputs["image_grid_thw"][0].prod())
        # COCO val2017's maximum original size is 640x640 (1600 vision patches).
        # Reject unexpectedly large processed inputs before allocating model activations.
        if sequence > 1024 or vision_patches > 2048:
            raise RuntimeError(f"Input exceeds guarded COCO envelope: {sequence} tokens, "
                               f"{vision_patches} vision patches; no GPU forward was started")


def enable_guarded_loading(probe, guard, model_budget_gib):
    """Replace only model loading; preserve Gradio's prompt and preprocessing."""
    budgets = model_budget_gib if isinstance(model_budget_gib, list) else [model_budget_gib] * guard.count
    if len(budgets) != guard.count or any(
        budget <= 0 or budget + .5 > cap / GIB for budget, cap in zip(budgets, guard.caps)
    ):
        raise ValueError("Model budget must leave at least 0.5 GiB below the allocator cap")
    probe.memory_guard = guard

    def load_model():
        if probe.model is not None:
            return probe.model
        from accelerate import init_empty_weights
        from accelerate.utils import compute_module_sizes
        from transformers import Qwen3_5ForConditionalGeneration

        guard.wait("before_model_loading")
        with init_empty_weights():
            skeleton = Qwen3_5ForConditionalGeneration(probe.config)
        skeleton.tie_weights()
        device_map = build_device_map(
            compute_module_sizes(skeleton, dtype=torch.bfloat16),
            probe.config.text_config.num_hidden_layers, guard.count, [int(budget * GIB) for budget in budgets],
        )
        del skeleton
        if not set(range(guard.count)) <= set(device_map.values()):
            raise RuntimeError("The guarded device map does not use every visible GPU")
        map_path = guard.status_path.parent / "device_map.json"
        map_path.write_text(json.dumps(device_map, indent=2) + "\n")
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            probe.checkpoint, local_files_only=True, dtype=torch.bfloat16,
            device_map=device_map, low_cpu_mem_usage=True,
            # Vision attention need not materialize a dense attention matrix.
            # Decoder attention remains eager to expose exact option attention.
            attn_implementation={"vision_config": "sdpa", "text_config": "eager"},
        ).eval().requires_grad_(False)
        for index, block in enumerate(model.model.language_model.layers, start=1):
            block.register_forward_pre_hook(
                lambda _module, _args, index=index: guard.wait(f"decoder_layer_{index}")
            )
        model.model.visual.register_forward_pre_hook(
            lambda _module, _args: guard.wait("vision_encoder")
        )
        probe.model = model
        guard.write_status("running", "model_loaded")
        print("Guarded model loaded across all visible GPUs; "
              f"allocator caps {[cap / GIB for cap in guard.caps]} GiB; "
              f"CPU-offloaded decoder layers: {sum(value == 'cpu' for key, value in device_map.items() if '.layers.' in key)}", flush=True)
        return model

    probe.load_model = load_model
