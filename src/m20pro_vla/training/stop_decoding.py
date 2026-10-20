"""Opt-in current-observation STOP decoding; default future chunks stay unchanged."""
import json
from pathlib import Path
import torch

CONTRACT_FILE = 'm20_stop_decoding.json'
CURRENT_STOP_V1 = dict(schema='m20_current_observation_stop_decoding_v1',
    source_binary_head_schema='m20_visual_language_binary_stop_v2',
    chunk_size=50, input='first_learned_stop_logit',
    output='repeat_current_probability_for_action_chunk',
    lifetime='until_next_model_replan', votes=False, permanent_latch=False)

def validate_stop_decoding(value, binary_spec):
    if (not isinstance(value, dict) or set(value) != set(CURRENT_STOP_V1)
            or any(type(value[k]) is not type(v) or value[k] != v for k,v in CURRENT_STOP_V1.items())):
        raise ValueError('Unknown STOP decoding contract')
    from .binary_stop import validate_binary_stop_spec
    binary_spec = validate_binary_stop_spec(binary_spec)
    if binary_spec['schema'] != value['source_binary_head_schema']:
        raise ValueError('Current STOP decoding requires the visual-language v2 head')
    return dict(CURRENT_STOP_V1)

def load_stop_decoding(directory, binary_spec):
    path = Path(directory)/CONTRACT_FILE
    return validate_stop_decoding(json.loads(path.read_text()), binary_spec) if path.is_file() else None

def save_stop_decoding(directory, value, binary_spec):
    value = validate_stop_decoding(value, binary_spec)
    (Path(directory)/CONTRACT_FILE).write_text(json.dumps(value,indent=2)+'\n')

def decoded_stop_logits(logits, value, binary_spec):
    if value is None:
        return logits
    validate_stop_decoding(value,binary_spec)
    if (not isinstance(logits,torch.Tensor) or logits.ndim != 2 or logits.shape[1] != 50
            or not logits.is_floating_point() or not torch.isfinite(logits).all()):
        raise ValueError('Expected finite Bx50 learned STOP logits')
    # Only the current label/logit is used. A new actual model observation may
    # immediately resume motion; this adds neither votes nor a permanent latch.
    return logits[:,:1].expand_as(logits)
