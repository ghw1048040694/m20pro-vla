import unittest
import numpy as np
from m20pro_vla.data.visual_evidence import (
    VisualEvidenceBuffer, VisualEvidenceSpec, sample_visual_evidence)


class VisualEvidenceTests(unittest.TestCase):
    def frames(self, count=1001):
        frames = {camera: np.zeros((count, 10, 10, 3), np.uint8) for camera in ('front', 'rear')}
        frames['front'][120] = [20, 180, 20]
        frames['rear'][200] = [200, 20, 20]
        frames['front'][300] = [200, 180, 20]
        frames['rear'][400] = [20, 190, 20]
        return frames

    def test_prefix_query_equals_online_at_boundaries_and_after_long_occlusion(self):
        frames = self.frames()
        buffer = VisualEvidenceBuffer()
        for tick in range(1001):
            sample = buffer.observe({k: v[tick] for k, v in frames.items()})
            if tick in (0, 119, 120, 121, 200, 300, 400, 401, 1000):
                expected = sample_visual_evidence(frames, tick)
                np.testing.assert_array_equal(sample.valid, expected.valid)
                np.testing.assert_array_equal(sample.age_ticks, expected.age_ticks)
                for camera in frames:
                    np.testing.assert_array_equal(sample.images[camera], expected.images[camera])
        np.testing.assert_array_equal(sample.age_ticks, [800, 600, 700])
        self.assertTrue(np.all(sample.images['rear'][1] == [20, 190, 20]))

    def test_brief_evidence_survives_sparse_history_miss(self):
        frames = self.frames()
        frames['rear'][400] = 0
        sample = sample_visual_evidence(frames, 1000)
        self.assertTrue(sample.valid[1])
        self.assertEqual(sample.age_ticks[1], 880)
        self.assertTrue(np.all(sample.images['front'][1] == [20, 180, 20]))

    def test_future_rgb_does_not_change_current_evidence(self):
        frames = self.frames()
        before = sample_visual_evidence(frames, 350)
        for value in frames.values(): value[351:] = [20, 240, 20]
        after = sample_visual_evidence(frames, 350)
        for camera in frames: np.testing.assert_array_equal(before.images[camera], after.images[camera])
        np.testing.assert_array_equal(before.age_ticks, after.age_ticks)

    def test_exact_existing_eighty_pixel_criterion(self):
        buffer = VisualEvidenceBuffer()
        pair = {k: np.zeros((10, 10, 3), np.uint8) for k in ('front', 'rear')}
        pair['front'].reshape(-1, 3)[:79] = [20, 180, 20]
        self.assertFalse(buffer.observe(pair).valid.any())
        pair['front'].reshape(-1, 3)[79] = [20, 180, 20]
        self.assertTrue(buffer.observe(pair).valid[1])
        buffer.reset()
        pair['front'].reshape(-1, 3)[40:] = 0
        pair['rear'].reshape(-1, 3)[:40] = [20, 180, 20]
        self.assertTrue(buffer.observe(pair).valid[1])
        pair['rear'].reshape(-1, 3)[39] = 0
        buffer.reset()
        self.assertFalse(buffer.observe(pair).valid.any())

    def test_reset_removes_all_evidence_and_missing_slots_are_masked(self):
        buffer = VisualEvidenceBuffer()
        pair = {k: np.full((10, 10, 3), [200, 20, 20], np.uint8) for k in ('front', 'rear')}
        buffer.observe(pair)
        buffer.reset()
        for value in pair.values(): value[:] = 7
        sample = buffer.observe(pair)
        self.assertFalse(sample.valid.any())
        np.testing.assert_array_equal(sample.age_ticks, [-1, -1, -1])
        self.assertTrue(all(np.all(v == 7) for v in sample.images.values()))

    def test_input_and_output_mutations_do_not_pollute_memory(self):
        buffer = VisualEvidenceBuffer()
        pair = {k: np.full((10, 10, 3), [200, 20, 20], np.uint8) for k in ('front', 'rear')}
        sample = buffer.observe(pair)
        for value in sample.images.values(): value[:] = 255
        for value in pair.values(): value[:] = 0
        again = buffer.observe(pair)
        self.assertTrue(np.all(again.images['front'][0] == [200, 20, 20]))
        self.assertEqual(again.age_ticks[0], 1)

    def test_schema_roundtrip_and_privileged_or_invalid_inputs_rejected(self):
        self.assertEqual(VisualEvidenceSpec.from_dict(VisualEvidenceSpec().to_dict()), VisualEvidenceSpec())
        with self.assertRaises(ValueError): VisualEvidenceSpec.from_dict(VisualEvidenceSpec().to_dict() | {'target_xy': [1, 2]})
        with self.assertRaises(ValueError): VisualEvidenceSpec(fps=50)
        pair = {k: np.zeros((10, 10, 3), np.uint8) for k in ('front', 'rear')}
        with self.assertRaises(ValueError): VisualEvidenceBuffer().observe(pair | {'room': np.zeros(2)})
        with self.assertRaises(ValueError): VisualEvidenceBuffer().observe({k: v.astype(float) for k,v in pair.items()})
        buffer = VisualEvidenceBuffer(); buffer.observe(pair)
        with self.assertRaises(ValueError): buffer.observe({k: np.zeros((11, 10, 3), np.uint8) for k in pair})


if __name__ == '__main__': unittest.main()
