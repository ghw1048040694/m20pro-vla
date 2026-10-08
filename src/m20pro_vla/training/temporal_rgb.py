"""Candidate temporal image-prefix preparation; no policy factory patch or CLI."""
import torch
from ..data.temporal_rgb import TemporalRGBSpec


def prepare_temporal_images(policy, batch, spec=TemporalRGBSpec()):
    """Call the policy's real image preprocessing separately for every time slot.

    Returns oldest-first, then front/rear images and masks, plus explicit slot
    descriptors. A future policy adapter must use these in BOTH loss and action
    sampling and persist the spec. This function alone does not integrate memory.
    """
    keys = tuple('observation.images.'+camera for camera in spec.cameras)
    if tuple(policy.config.image_features) != keys:
        raise ValueError('Temporal candidate requires exactly front/rear configured in that order')
    images, masks, slots = [], [], []
    shape = None
    for key in keys:
        image = batch[key]
        if image.ndim != 5 or image.shape[1] != len(spec.offsets) or image.shape[2] != 3:
            raise ValueError('Expected [B,history,C,H,W] RGB with the saved temporal spec')
        if not image.is_floating_point() or not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError('Expected finite RGB in [0,1]')
        if shape is not None and image.shape != shape:
            raise ValueError('Camera shapes must match')
        shape = image.shape
        pad = batch[key+'_is_pad']
        if pad.dtype != torch.bool or pad.shape != image.shape[:2] or pad[:, -1].any():
            raise ValueError('Expected per-slot padding mask and a valid current frame')
    if not torch.equal(batch[keys[0]+'_is_pad'], batch[keys[1]+'_is_pad']):
        raise ValueError('Camera history padding must match')
    for time_index, offset in enumerate(spec.offsets):
        current = {}
        for key in keys:
            current[key] = batch[key][:, time_index]
            current[key+'_padding_mask'] = ~batch[key+'_is_pad'][:, time_index]
        prepared, valid = policy.prepare_images(current)
        if len(prepared) != len(keys) or len(valid) != len(keys):
            raise ValueError('Unexpected camera preprocessing result')
        images.extend(prepared)
        masks.extend(valid)
        slots.extend((offset, camera) for camera in spec.cameras)
    return images, masks, tuple(slots)
