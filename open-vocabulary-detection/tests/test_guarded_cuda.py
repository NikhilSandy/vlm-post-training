"""Exercise admission checks without allocating GPU memory."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from coco_classification.guarded_cuda import GIB, MemoryGuard, build_device_map, enable_guarded_loading


class MemoryGuardTests(unittest.TestCase):
    def make_guard(self):
        guard = object.__new__(MemoryGuard)
        guard.caps, guard.reserve, guard.count = [2 * GIB] * 4, 4 * GIB, 4
        return guard

    def test_admission_reserves_future_allocator_growth_and_library_workspace(self):
        guard = self.make_guard()
        self.assertTrue(guard.has_headroom([{"free_bytes": int(6.25 * GIB), "reserved_bytes": 0}]))
        self.assertFalse(guard.has_headroom([{"free_bytes": int(6.24 * GIB), "reserved_bytes": 0}]))

    def test_own_existing_reservations_are_not_budgeted_twice(self):
        guard = self.make_guard()
        self.assertTrue(guard.has_headroom([{"free_bytes": int(4.75 * GIB), "reserved_bytes": int(1.5 * GIB)}]))
        self.assertFalse(guard.has_headroom([{"free_bytes": int(4.74 * GIB), "reserved_bytes": int(1.5 * GIB)}]))

    def test_every_device_must_have_headroom(self):
        guard = self.make_guard()
        self.assertFalse(guard.has_headroom([
            {"free_bytes": 10 * GIB, "reserved_bytes": 0},
            {"free_bytes": 4 * GIB, "reserved_bytes": 2 * GIB},
        ]))

    def test_individual_gpu_caps_are_used_for_admission(self):
        guard = self.make_guard()
        guard.caps = [3 * GIB, 4 * GIB, int(1.5 * GIB), 4 * GIB]
        self.assertTrue(guard.has_headroom([
            {"device": 0, "free_bytes": int(7.25 * GIB), "reserved_bytes": 0},
            {"device": 2, "free_bytes": int(5.75 * GIB), "reserved_bytes": 0},
        ]))
        self.assertFalse(guard.has_headroom([
            {"device": 1, "free_bytes": int(7.25 * GIB), "reserved_bytes": 0},
        ]))

    def test_wait_does_not_proceed_until_headroom_returns(self):
        guard = self.make_guard()
        guard.snapshot = Mock(side_effect=[
            [{"free_bytes": 5 * GIB, "reserved_bytes": 0}],
            [{"free_bytes": 7 * GIB, "reserved_bytes": 0}],
        ])
        guard.write_status = Mock()
        with patch("coco_classification.guarded_cuda.time.sleep") as sleep:
            guard.wait("test")
        sleep.assert_called_once_with(10)
        self.assertEqual(guard.snapshot.call_count, 2)
        self.assertEqual(guard.write_status.call_args.args[:2], ("waiting_for_vram", "test"))

    def test_large_input_is_rejected_before_forward(self):
        guard = self.make_guard()
        prepared = SimpleNamespace(inputs={
            "input_ids": torch.zeros(1, 1024, dtype=torch.long),
            "image_grid_thw": torch.tensor([[1, 40, 40]]),
        })
        guard.check_inputs(prepared)
        prepared.inputs["input_ids"] = torch.zeros(1, 1025, dtype=torch.long)
        with self.assertRaisesRegex(RuntimeError, "no GPU forward"):
            guard.check_inputs(prepared)
        prepared.inputs["input_ids"] = torch.zeros(1, 100, dtype=torch.long)
        prepared.inputs["image_grid_thw"] = torch.tensor([[1, 100, 100]])
        with self.assertRaises(RuntimeError):
            guard.check_inputs(prepared)

    def test_weight_budget_must_leave_activation_headroom(self):
        guard = self.make_guard()
        with self.assertRaises(ValueError):
            enable_guarded_loading(SimpleNamespace(), guard, 1.6)

    def test_placement_keeps_vision_intact_and_reserves_cpu_staging(self):
        sizes = {
            "model.visual": 600, "model.language_model.norm": 1,
            "model.language_model.embed_tokens": 1180,
            **{f"model.language_model.layers.{i}": 210 for i in range(32)},
        }
        placement = build_device_map(sizes, 32, 4, 1250)
        self.assertEqual(placement["model.visual"], 0)
        self.assertFalse(any(key.startswith("model.visual.") for key in placement))
        self.assertEqual(placement["lm_head"], placement["model.language_model.embed_tokens"])
        self.assertTrue({0, 1, 2, 3, "cpu"} <= set(placement.values()))
        for device in range(4):
            resident = sum(sizes[key] for key, value in placement.items() if value == device and key in sizes)
            self.assertLessEqual(resident + (210 if device == 0 else 0), 1250)

    def test_individual_weight_budgets_can_keep_all_layers_on_gpu(self):
        sizes = {
            "model.visual": 600, "model.language_model.norm": 1,
            "model.language_model.embed_tokens": 1180,
            **{f"model.language_model.layers.{i}": 210 for i in range(32)},
        }
        budgets = [2250, 3250, 750, 3250]
        placement = build_device_map(sizes, 32, 4, budgets)
        self.assertNotIn("cpu", placement.values())
        self.assertEqual(set(placement.values()), {0, 1, 2, 3})
        for device in range(4):
            resident = sum(sizes[key] for key, value in placement.items() if value == device and key in sizes)
            self.assertLessEqual(resident + (210 if device == 0 else 0), budgets[device])


if __name__ == "__main__":
    unittest.main()
