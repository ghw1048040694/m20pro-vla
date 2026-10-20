"""Versioned loss for four physical M20 actions and recorded time steps."""
import json
from pathlib import Path
import torch

CONTRACT_FILE = 'm20_training_objective.json'
PHYSICAL_FLOW_V1 = dict(schema='m20_physical_flow_loss_v1', action_dim=4,
    model_action_dim=32, chunk_size=50, padding_key='action_is_pad',
    reduction='mean_of_per_sample_valid_time_axis_means', axis_weights=[1, 1, 1, 1])


# Preserve the v1 per-consumed-axis scale and independent STOP BCE coefficient.
# The denominator remains four; only gradients for discarded flow outputs vanish.
CONSUMED_MOTION_V2 = dict(schema='m20_consumed_motion_flow_loss_v2', action_dim=4,
    model_action_dim=32, chunk_size=50, padding_key='action_is_pad',
    reduction='mean_of_per_sample_valid_time_sums_divided_by_four',
    consumed_axes=[0, 2], denominator_axes=4,
    required_binary_stop='m20_visual_language_binary_stop_v2',
    lateral_processing='saved_zero_mean_and_std')


def validate_objective(value):
    if value == PHYSICAL_FLOW_V1:
        return json.loads(json.dumps(value))
    if json.dumps(value, sort_keys=True) != json.dumps(CONSUMED_MOTION_V2, sort_keys=True):
        raise ValueError('Unknown M20 training objective')
    return json.loads(json.dumps(value))


def load_objective(directory):
    path = Path(directory)/CONTRACT_FILE
    return validate_objective(json.loads(path.read_text())) if path.is_file() else None


def reduce_physical_flow(losses, action_is_pad, reduction='mean', *, objective=None):
    if losses.ndim != 3 or losses.shape[1:] != (50, 32):
        raise ValueError('Expected original 50 by 32 flow loss')
    if action_is_pad is None or action_is_pad.dtype != torch.bool or action_is_pad.shape != losses.shape[:2]:
        raise ValueError('Recorded action time mask is required')
    valid = ~action_is_pad.to(losses.device)
    counts = valid.sum(dim=1)
    if torch.any(counts == 0):
        raise ValueError('Every sample needs a recorded action')
    objective = validate_objective(PHYSICAL_FLOW_V1 if objective is None else objective)
    physical = losses[:, :, :4] if objective == PHYSICAL_FLOW_V1 else losses[:, :, [0, 2]]
    masked = physical.masked_fill(~valid.unsqueeze(-1), 0)
    per_sample = masked.sum(dim=(1, 2))/(counts*4)
    if reduction == 'none':
        return per_sample
    if reduction != 'mean':
        raise ValueError('Unsupported loss reduction')
    return per_sample.mean()


def physical_flow_forward(policy, batch, noise=None, time=None, reduction='mean'):
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
    if policy.config.max_action_dim != 32 or policy.config.adapt_to_pi_aloha or batch['action'].shape[-2:] != (50, 4):
        raise ValueError('Physical loss requires the original M20 action architecture')
    images, masks = policy.prepare_images(batch)
    losses = policy.model.forward(images, masks, batch[OBS_LANGUAGE_TOKENS],
        batch[OBS_LANGUAGE_ATTENTION_MASK], policy.prepare_state(batch),
        policy.prepare_action(batch), noise, time)
    loss = reduce_physical_flow(losses, batch.get('action_is_pad'), reduction,
        objective=policy.training_objective)
    metrics = dict(loss=float(loss.detach().mean()),
        physical_valid_flow_loss=float(loss.detach().mean()),
        legacy_all32_flow_loss=float(losses.detach().mean()),
        valid_action_fraction=float((~batch['action_is_pad']).float().mean()))
    return loss, metrics


def validate_objective_context(directory, objective, binary):
    """Reject discarding flow outputs unless the saved execution math discards them."""
    if objective is None or validate_objective(objective) == PHYSICAL_FLOW_V1:
        return
    from .binary_stop import validate_binary_stop_spec, _saved_action_stats
    if binary is None or validate_binary_stop_spec(binary)['schema'] != CONSUMED_MOTION_V2['required_binary_stop']:
        raise ValueError('Consumed motion loss requires the visual-language binary STOP head')
    if directory is None:
        return
    pre, _ = _saved_action_stats(directory, 'policy_preprocessor.json', 'normalizer_processor')
    post, _ = _saved_action_stats(directory, 'policy_postprocessor.json', 'unnormalizer_processor')
    if any(float(stats[index][1]) != 0 for stats in (pre, post) for index in (0, 1)):
        raise ValueError('Consumed motion loss requires zero lateral mean/std in saved processors')
