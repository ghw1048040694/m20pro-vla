"""Lightweight training-time augmentations for MuJoCo VLA RGB inputs."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class RGBAugmentationConfig:
    enabled: bool = True
    brightness_jitter: float = 0.08
    contrast_jitter: float = 0.08
    noise_std: float = 0.01
    cutout_prob: float = 0.20
    cutout_size: int = 8


@dataclass(frozen=True)
class ObservationAugmentationConfig:
    rgb: RGBAugmentationConfig = field(default_factory=RGBAugmentationConfig)
    lidar_noise_std: float = 0.01
    lidar_dropout_prob: float = 0.05
    proprio_noise_std: float = 0.005


def _uniform(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> torch.Tensor:
    return torch.rand(shape, device=device, dtype=dtype, generator=generator)


def _normal(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> torch.Tensor:
    return torch.randn(shape, device=device, dtype=dtype, generator=generator)


def _augment_chunk(
    rgb: torch.Tensor,
    config: RGBAugmentationConfig,
    *,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if rgb.ndim != 4:
        raise ValueError(f"Expected BCHW tensor, got shape {tuple(rgb.shape)}")
    batch, channels, height, width = rgb.shape
    if batch == 0:
        return rgb
    result = rgb.clone()
    device = result.device
    dtype = result.dtype
    if config.contrast_jitter > 0.0:
        contrast = 1.0 + (
            _uniform((batch, 1, 1, 1), device=device, dtype=dtype, generator=generator) * 2.0 - 1.0
        ) * float(config.contrast_jitter)
        mean = result.mean(dim=(2, 3), keepdim=True)
        result = (result - mean) * contrast + mean
    if config.brightness_jitter > 0.0:
        brightness = 1.0 + (
            _uniform((batch, 1, 1, 1), device=device, dtype=dtype, generator=generator) * 2.0 - 1.0
        ) * float(config.brightness_jitter)
        result = result * brightness
    if config.noise_std > 0.0:
        result = result + _normal(result.shape, device=device, dtype=dtype, generator=generator) * float(config.noise_std)
    if config.cutout_prob > 0.0 and config.cutout_size > 0:
        cutout_mask = _uniform((batch,), device=device, dtype=dtype, generator=generator) < float(config.cutout_prob)
        if bool(cutout_mask.any()):
            cutout_size = int(max(1, min(config.cutout_size, height, width)))
            starts_y = torch.randint(
                0,
                max(1, height - cutout_size + 1),
                (batch,),
                device=device,
                generator=generator,
            )
            starts_x = torch.randint(
                0,
                max(1, width - cutout_size + 1),
                (batch,),
                device=device,
                generator=generator,
            )
            for index in torch.nonzero(cutout_mask, as_tuple=False).flatten().tolist():
                top = int(starts_y[index])
                left = int(starts_x[index])
                result[index, :, top : top + cutout_size, left : left + cutout_size] = 0.0
    return result.clamp_(0.0, 1.0)


def apply_rgb_augmentation(
    rgb: torch.Tensor,
    config: RGBAugmentationConfig,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply modest domain randomization to a batched RGB tensor."""
    if not config.enabled:
        return rgb
    if rgb.ndim != 4:
        raise ValueError(f"Expected BCHW tensor, got shape {tuple(rgb.shape)}")
    if rgb.shape[1] % 3 == 0 and rgb.shape[1] >= 3:
        chunks = [
            _augment_chunk(chunk, config, generator=generator)
            for chunk in rgb.split(3, dim=1)
        ]
        return torch.cat(chunks, dim=1)
    return _augment_chunk(rgb, config, generator=generator)


def apply_observation_augmentation(
    *,
    rgb: torch.Tensor,
    lidar: torch.Tensor,
    proprio: torch.Tensor,
    config: ObservationAugmentationConfig,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply lightweight training-only perturbations to all sensor streams."""
    if not config.rgb.enabled:
        rgb_out = rgb
    else:
        rgb_out = apply_rgb_augmentation(rgb, config.rgb, generator=generator)
    if lidar.ndim != 2:
        raise ValueError(f"Expected lidar tensor with shape [B, 72], got {tuple(lidar.shape)}")
    if proprio.ndim != 2:
        raise ValueError(f"Expected proprio tensor with shape [B, 45], got {tuple(proprio.shape)}")
    if lidar.shape[0] != rgb.shape[0] or proprio.shape[0] != rgb.shape[0]:
        raise ValueError("Batch size mismatch across observation tensors")
    lidar_out = lidar.clone()
    if config.lidar_noise_std > 0.0:
        lidar_out = lidar_out + _normal(lidar_out.shape, device=lidar_out.device, dtype=lidar_out.dtype, generator=generator) * float(config.lidar_noise_std)
    if config.lidar_dropout_prob > 0.0:
        dropout = _uniform(lidar_out.shape, device=lidar_out.device, dtype=lidar_out.dtype, generator=generator) < float(config.lidar_dropout_prob)
        lidar_out = torch.where(dropout, torch.ones_like(lidar_out), lidar_out)
    lidar_out = lidar_out.clamp_(0.0, 1.0)
    proprio_out = proprio.clone()
    if config.proprio_noise_std > 0.0:
        proprio_out = proprio_out + _normal(proprio_out.shape, device=proprio_out.device, dtype=proprio_out.dtype, generator=generator) * float(config.proprio_noise_std)
    return rgb_out, lidar_out, proprio_out
