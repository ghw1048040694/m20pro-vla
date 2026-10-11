"""Saved physical action and observation contract for reviewed human demos."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np

CONTRACT_FILE = 'm20_execution_contract.json'
HUMAN_CONTRACT = {
    'schema': 'm20_execution_contract_v1',
    'profile': 'human_keyboard_v1',
    'action_units': ['m/s', 'm/s', 'rad/s', 'stop'],
    'forward_limits': [-.25, .50], 'yaw_limits': [-.40, .40],
    'lateral': 0., 'policy_fps': 25, 'control_fps': 50,
    'state': 'm20_smolvla_state_32_no_world_no_previous_action',
    'render': 'interactive-v1: egl-llvmpipe, no shadows/reflections/MSAA',
    'motion': 'keyboard_separate_axes_slew_50hz_v1',
    'stop_distance': .5, 'slow_distance': 1.25,
}


def validate_execution_contract(value):
    if value != HUMAN_CONTRACT:
        raise ValueError('Unsupported or inconsistent M20 execution contract')
    return deepcopy(value)


def load_execution_contract(directory):
    path = Path(directory) / CONTRACT_FILE
    return validate_execution_contract(json.loads(path.read_text())) if path.is_file() else None


def save_execution_contract(directory, value):
    value = validate_execution_contract(value)
    (Path(directory) / CONTRACT_FILE).write_text(json.dumps(value, indent=2)+'\n')


def physical_action(raw, stop_threshold=.5):
    raw = np.asarray(raw, dtype=np.float64).reshape(-1)
    if raw.shape != (4,) or not np.isfinite(raw).all():
        raise ValueError('Expected finite physical 4D body action')
    if raw[3] >= stop_threshold:
        return np.array([0., 0., 0., 1.])
    return np.array([np.clip(raw[0], -.25, .5), 0., np.clip(raw[2], -.4, .4), 0.])
