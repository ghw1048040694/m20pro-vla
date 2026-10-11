"""Explicit high-level encoder coordinates; physical simulator state is untouched."""
import copy
import json
from pathlib import Path
import numpy as np

CONTRACT_FILE = 'm20_state_input.json'
WHEEL_INDICES = (7, 11, 15, 19)

def relative_wheel_spec():
    return dict(schema='m20_episode_relative_wheel_state_v1', state_dim=32,
        wheel_indices=list(WHEEL_INDICES), reference='first_recorded_state_of_episode_or_segment',
        arithmetic='float32_subtract', other_coordinates='unchanged')

def validate_state_input(value):
    if type(value) is not dict or value != relative_wheel_spec():
        raise ValueError('Unknown M20 state input contract')
    return copy.deepcopy(value)

def load_state_input(directory):
    path=Path(directory)/CONTRACT_FILE
    return validate_state_input(json.loads(path.read_text())) if path.exists() else None

def save_state_input(directory, value):
    (Path(directory)/CONTRACT_FILE).write_text(json.dumps(validate_state_input(value),indent=2)+'\n')

def relative_wheel_state(state, reference):
    state=np.asarray(state,dtype=np.float32);reference=np.asarray(reference,dtype=np.float32)
    if state.shape[-1:] != (32,) or reference.shape != (32,) or not np.isfinite(state).all() or not np.isfinite(reference).all():
        raise ValueError('Relative wheel input requires finite 32D state and one episode reference')
    result=state.copy()
    result[...,WHEEL_INDICES]=state[...,WHEEL_INDICES]-reference[list(WHEEL_INDICES)]
    return result

def relative_wheel_metadata(dataset):
    """Clone metadata; replace only four statistics using unchanged cached rows."""
    from lerobot.datasets.compute_stats import aggregate_stats, get_feature_stats
    meta=copy.deepcopy(dataset.meta)
    episodes=[]
    for episode in range(dataset.num_episodes):
        states=dataset._episode_arrays(episode)['state']
        relative=np.asarray(states[:,WHEEL_INDICES],dtype=np.float32)-np.asarray(states[0,list(WHEEL_INDICES)],dtype=np.float32)
        stats=get_feature_stats(relative,axis=0,keepdims=False)
        episodes.append({'observation.state':stats})
    changed=aggregate_stats(episodes)['observation.state']
    for key, value in meta.stats['observation.state'].items():
        if key == 'count':
            if not np.array_equal(value,changed[key]):
                raise ValueError('Relative input must preserve source row counts')
        else:
            value[list(WHEEL_INDICES)]=np.asarray(changed[key],dtype=value.dtype)
    return meta
