from __future__ import annotations

import unittest

import torch

from m20pro_vla.utils import (
    ObservationAugmentationConfig,
    RGBAugmentationConfig,
    apply_observation_augmentation,
    apply_rgb_augmentation,
)


class RGBAugmentationTest(unittest.TestCase):
    def test_rgb_augmentation_is_identity_when_disabled(self) -> None:
        rgb = torch.rand(2, 6, 8, 10)
        config = RGBAugmentationConfig(enabled=False)
        out = apply_rgb_augmentation(rgb, config)
        self.assertTrue(torch.equal(out, rgb))

    def test_rgb_augmentation_preserves_shape_and_bounds(self) -> None:
        rgb = torch.ones(3, 6, 8, 10)
        config = RGBAugmentationConfig(
            enabled=True,
            brightness_jitter=0.2,
            contrast_jitter=0.2,
            noise_std=0.1,
            cutout_prob=1.0,
            cutout_size=4,
        )
        generator = torch.Generator().manual_seed(1234)
        out = apply_rgb_augmentation(rgb, config, generator=generator)
        self.assertEqual(tuple(out.shape), tuple(rgb.shape))
        self.assertTrue(torch.all(out >= 0.0))
        self.assertTrue(torch.all(out <= 1.0))

    def test_observation_augmentation_preserves_shapes(self) -> None:
        rgb = torch.ones(2, 6, 8, 10)
        lidar = torch.full((2, 72), 0.5)
        proprio = torch.zeros(2, 45)
        config = ObservationAugmentationConfig(
            rgb=RGBAugmentationConfig(enabled=True, cutout_prob=0.0),
            lidar_noise_std=0.02,
            lidar_dropout_prob=0.10,
            proprio_noise_std=0.01,
        )
        generator = torch.Generator().manual_seed(4321)
        rgb_out, lidar_out, proprio_out = apply_observation_augmentation(
            rgb=rgb,
            lidar=lidar,
            proprio=proprio,
            config=config,
            generator=generator,
        )
        self.assertEqual(tuple(rgb_out.shape), tuple(rgb.shape))
        self.assertEqual(tuple(lidar_out.shape), tuple(lidar.shape))
        self.assertEqual(tuple(proprio_out.shape), tuple(proprio.shape))
        self.assertTrue(torch.all(lidar_out >= 0.0))
        self.assertTrue(torch.all(lidar_out <= 1.0))


if __name__ == "__main__":
    unittest.main()
