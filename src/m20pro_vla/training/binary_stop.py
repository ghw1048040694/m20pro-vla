"""Versioned observation-only binary STOP head; no policy or global patches."""
from copy import deepcopy
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


CONTRACT_FILE = 'm20_binary_stop.json'
_FIXED = dict(
    schema='m20_observation_binary_stop_v1',
    hidden_dim=960,
    chunk_size=50,
    action_dim=4,
    model_action_dim=32,
    stop_index=3,
    head='linear',
    observation='final_prefix_last_valid_state_token',
    padding_key='action_is_pad',
    reduction='mean_of_per_sample_valid_time_means',
    label_tolerance=1e-5,
    initialization_std=.001,
    initialization_bias=0.,
    initialization_seed=20261011,
    head_learning_rate=.001,
    loss_coefficient=1.,
    normalization='float32_mean_std_forward_add_eps_inverse_std_only',
)
VISUAL_LANGUAGE_SCHEMA = 'm20_visual_language_binary_stop_v2'
_VARIANTS = {
    _FIXED['schema']: _FIXED,
    VISUAL_LANGUAGE_SCHEMA: dict(_FIXED, schema=VISUAL_LANGUAGE_SCHEMA,
        observation='final_prefix_last_valid_visual_language_group0_token'),
}
_VARIABLE = {
    'stop_mean', 'stop_std', 'normalization_epsilon',
    'normalized_stop_zero', 'normalized_stop_one',
    'normalized_stop_output_zero', 'normalized_stop_output_one',
}


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite JSON number')
    return float(value)


def make_binary_stop_spec(stop_mean, stop_std, normalization_epsilon, hidden_dim=960,
                          *, schema='m20_observation_binary_stop_v1'):
    """Build endpoints using the installed processor's actual float32 arithmetic.

    Its forward transform divides by std + epsilon; its inverse multiplies by
    std alone. Store both endpoint pairs, so labels and output probabilities
    each use the correct direction without changing the existing processor.
    """
    if schema not in _VARIANTS:
        raise ValueError('Unknown binary STOP input schema')
    fixed = _VARIANTS[schema]
    if type(hidden_dim) is not int or hidden_dim != fixed['hidden_dim']:
        raise ValueError('This STOP architecture requires hidden_dim=960')
    mean = _number(stop_mean, 'stop_mean')
    std = _number(stop_std, 'stop_std')
    epsilon = _number(normalization_epsilon, 'normalization_epsilon')
    if std <= 0 or epsilon < 0:
        raise ValueError('STOP std must be positive and epsilon nonnegative')
    mean32, std32 = torch.tensor([mean, std], dtype=torch.float32)
    if not torch.isfinite(mean32) or not torch.isfinite(std32) or std32 <= 0:
        raise ValueError('STOP statistics must be finite and positive in float32')
    endpoints = torch.tensor([0., 1.], dtype=torch.float32)
    train = (endpoints - mean32) / (std32 + epsilon)
    output = (endpoints - mean32) / std32
    if not torch.isfinite(train).all() or not torch.isfinite(output).all() or train[1] <= train[0]:
        raise ValueError('STOP normalization endpoints must be finite and distinct')
    return dict(fixed, stop_mean=float(mean32), stop_std=float(std32),
                normalization_epsilon=epsilon,
                normalized_stop_zero=float(train[0]), normalized_stop_one=float(train[1]),
                normalized_stop_output_zero=float(output[0]),
                normalized_stop_output_one=float(output[1]))


def validate_binary_stop_spec(spec):
    if not isinstance(spec, dict) or set(spec) != set(_FIXED) | _VARIABLE:
        raise ValueError('Unknown binary STOP contract fields')
    fixed = _VARIANTS.get(spec.get('schema'))
    if fixed is None or any(type(spec[key]) is not type(value) or spec[key] != value for key, value in fixed.items()):
        raise ValueError('Unknown binary STOP architecture or objective')
    expected = make_binary_stop_spec(spec['stop_mean'], spec['stop_std'],
                                    spec['normalization_epsilon'], spec['hidden_dim'], schema=spec['schema'])
    if any(_number(spec[key], key) != expected[key] for key in _VARIABLE):
        raise ValueError('Binary STOP endpoints disagree with saved normalization statistics')
    return deepcopy(expected)


def load_binary_stop_spec(directory):
    path = Path(directory) / CONTRACT_FILE
    return validate_binary_stop_spec(json.loads(path.read_text())) if path.is_file() else None


def save_binary_stop_spec(directory, spec):
    value = validate_binary_stop_spec(spec)
    (Path(directory) / CONTRACT_FILE).write_text(json.dumps(value, indent=2) + '\n')


def _saved_action_stats(directory, filename, registry_name):
    from safetensors import safe_open
    directory = Path(directory)
    document = json.loads((directory / filename).read_text())
    steps = [step for step in document['steps'] if step.get('registry_name') == registry_name]
    if len(steps) != 1:
        raise ValueError('Exactly one saved action normalization step is required')
    step = steps[0]
    config = step['config']
    if (config.get('norm_map', {}).get('ACTION') != 'MEAN_STD'
            or config.get('features', {}).get('action') != {'type': 'ACTION', 'shape': [4]}):
        raise ValueError('Binary STOP requires saved physical 4D MEAN_STD action processing')
    state_file = step.get('state_file')
    if not isinstance(state_file, str) or Path(state_file).name != state_file:
        raise ValueError('Saved normalization state must be a local checkpoint filename')
    with safe_open(directory / state_file, framework='pt', device='cpu') as saved:
        tensors = [saved.get_tensor('action.' + name) for name in ('mean', 'std')]
    if any(value.dtype != torch.float32 or value.shape != (4,) or not torch.isfinite(value).all()
           for value in tensors):
        raise ValueError('Saved action statistics must be finite float32 four-vectors')
    if torch.any(tensors[1] < 0):
        raise ValueError('Saved action standard deviations cannot be negative')
    return tensors, _number(config['eps'], 'saved normalization epsilon')


def make_binary_stop_spec_from_saved(directory, *, schema='m20_observation_binary_stop_v1'):
    """Read only saved processor metadata and its small action-stat tensors."""
    pre, epsilon = _saved_action_stats(directory, 'policy_preprocessor.json', 'normalizer_processor')
    post, post_epsilon = _saved_action_stats(directory, 'policy_postprocessor.json', 'unnormalizer_processor')
    if epsilon != post_epsilon or any(not torch.equal(left, right) for left, right in zip(pre, post)):
        raise ValueError('Saved action pre/post normalization statistics differ')
    config = json.loads((Path(directory) / 'config.json').read_text())
    if (config.get('max_action_dim') != 32 or config.get('chunk_size') != 50
            or config.get('output_features', {}).get('action') != {'type': 'ACTION', 'shape': [4]}):
        raise ValueError('Saved checkpoint differs from the physical max32 chunk50 architecture')
    return make_binary_stop_spec(float(pre[0][3]), float(pre[1][3]), epsilon, schema=schema)


def validate_saved_normalization(directory, spec):
    spec = validate_binary_stop_spec(spec)
    if spec != make_binary_stop_spec_from_saved(directory, schema=spec['schema']):
        raise ValueError('Binary STOP contract differs from the actual saved action processors')
    return spec


def make_binary_stop_head(spec):
    spec = validate_binary_stop_spec(spec)
    # Head initialization must neither consume nor reset the training/noise RNG.
    with torch.random.fork_rng(devices=[]):
        generator = torch.Generator(device='cpu').manual_seed(spec['initialization_seed'])
        head = nn.Linear(spec['hidden_dim'], spec['chunk_size'], device='cpu', dtype=torch.float32)
        nn.init.normal_(head.weight, mean=0., std=spec['initialization_std'], generator=generator)
        nn.init.constant_(head.bias, spec['initialization_bias'])
    return head


def prefix_stop_logits(head, prefix_out, prefix_pad_masks, spec=None, prefix_attention_masks=None):
    """Read the versioned prefix token, preserving its gradient graph.

    Missing history and padded language create interior mask gaps, so the
    number of valid positions is not the state token's absolute index.
    """
    if (prefix_out.ndim != 3 or prefix_pad_masks is None
            or prefix_pad_masks.dtype != torch.bool
            or prefix_pad_masks.shape != prefix_out.shape[:2]
            or prefix_out.shape[-1] != head.in_features or head.out_features != 50):
        raise ValueError('Expected aligned prefix hidden states and boolean padding mask')
    valid = prefix_pad_masks.to(device=prefix_out.device)
    if spec is not None and validate_binary_stop_spec(spec)['schema'] == VISUAL_LANGUAGE_SCHEMA:
        if (prefix_attention_masks is None or prefix_attention_masks.dtype != torch.bool
                or prefix_attention_masks.shape != valid.shape):
            raise ValueError('Visual-language STOP requires aligned prefix attention groups')
        groups = prefix_attention_masks.to(device=prefix_out.device).cumsum(dim=1)
        state = valid & (groups == 1)
        visual_language = valid & (groups == 0)
        if (not (state.sum(dim=1) == 1).all()
                or (valid & (groups > 1)).any()
                or not visual_language.any(dim=1).all()):
            raise ValueError('Expected visual-language group zero followed by one state token')
        # The official cumulative group mask blocks every group-zero query
        # from the later state token. Keep that state in the moving flow path.
        valid = visual_language
    if not valid.any(dim=1).all():
        raise ValueError('Every observation needs a valid prefix state token')
    positions = torch.arange(prefix_out.shape[1], device=prefix_out.device)
    last = positions.expand_as(valid).masked_fill(~valid, -1).max(dim=1).values
    hidden = prefix_out[torch.arange(prefix_out.shape[0], device=prefix_out.device), last]
    if not torch.isfinite(hidden).all():
        raise ValueError('Nonfinite prefix state hidden')
    return head(hidden.to(dtype=head.weight.dtype))


def _valid_times(logits, action_is_pad):
    if (logits.ndim != 2 or logits.shape[1] != 50 or not logits.is_floating_point()
            or action_is_pad is None or action_is_pad.dtype != torch.bool
            or action_is_pad.shape != logits.shape):
        raise ValueError('Expected B by 50 STOP logits and recorded-time mask')
    valid = ~action_is_pad.to(device=logits.device)
    counts = valid.sum(dim=1)
    if torch.any(counts == 0):
        raise ValueError('Every sample needs a recorded STOP label')
    if not torch.isfinite(logits[valid]).all():
        raise ValueError('Nonfinite recorded STOP logits')
    return valid, counts


def binary_stop_loss(logits, normalized_actions, action_is_pad, spec, reduction='mean'):
    spec = validate_binary_stop_spec(spec)
    valid, counts = _valid_times(logits, action_is_pad)
    if (normalized_actions.ndim != 3 or normalized_actions.shape != (*logits.shape, 4)
            or not normalized_actions.is_floating_point()):
        raise ValueError('Expected the original normalized B by 50 by 4 actions')
    labels = normalized_actions[..., 3].detach().to(device=logits.device, dtype=torch.float32)
    labels = (labels - spec['normalized_stop_zero']) / (
        spec['normalized_stop_one'] - spec['normalized_stop_zero'])
    recorded = labels[valid]
    if (not torch.isfinite(recorded).all()
            or not ((recorded.abs() <= spec['label_tolerance'])
                    | ((recorded - 1).abs() <= spec['label_tolerance'])).all()):
        raise ValueError('Recorded STOP targets must reconstruct binary physical labels')
    labels = labels.masked_fill(~valid, 0).round()
    safe_logits = logits.float().masked_fill(~valid, 0)
    element_loss = F.binary_cross_entropy_with_logits(safe_logits, labels, reduction='none')
    per_sample = element_loss.masked_fill(~valid, 0).sum(dim=1) / counts
    if reduction == 'none':
        return per_sample
    if reduction != 'mean':
        raise ValueError('Unsupported binary STOP loss reduction')
    return per_sample.mean()


# Explicit name used by the policy objective integration.
masked_binary_stop_loss = binary_stop_loss


def replace_stop_column(actions, logits, spec):
    """Map learned probabilities to the existing postprocessor's coordinates."""
    spec = validate_binary_stop_spec(spec)
    if (actions.ndim != 3 or actions.shape[1:] != (50, 32)
            or logits.shape != actions.shape[:2] or not actions.is_floating_point()
            or not logits.is_floating_point() or actions.device != logits.device
            or not torch.isfinite(logits).all()):
        raise ValueError('Expected finite B by 50 STOP logits aligned with max32 actions')
    probability = logits.float().sigmoid()
    normalized = spec['normalized_stop_output_zero'] + (
        spec['normalized_stop_output_one'] - spec['normalized_stop_output_zero']) * probability
    result = actions.clone()
    result[..., 3] = normalized.to(dtype=actions.dtype)
    return result
