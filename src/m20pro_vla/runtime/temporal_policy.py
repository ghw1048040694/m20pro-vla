"""Saved temporal contract loading and causal 25-Hz inference for M20."""
import json
from pathlib import Path
import numpy as np
import torch
from ..data.temporal_rgb import TemporalRGBSpec, TemporalRGBBuffer
from ..training.temporal_policy_candidate import CONTRACT_FILE, temporal_policy_class


def load_m20_policy(checkpoint, device, action_replan_steps):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors
    checkpoint = Path(checkpoint)
    config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    config.device = str(device)
    config.pretrained_path = str(checkpoint)
    config.n_action_steps = int(action_replan_steps)
    temporal = (checkpoint / CONTRACT_FILE).is_file()
    from ..data.state_input import load_state_input
    state_input=load_state_input(checkpoint)
    if state_input is not None and not temporal:
        raise ValueError('Relative wheel state requires a saved temporal architecture')
    cls = temporal_policy_class(SmolVLAPolicy) if temporal else SmolVLAPolicy
    kwargs = {'require_temporal_contract': True} if temporal else {}
    policy = cls.from_pretrained(checkpoint, config=config, local_files_only=True,
                                 strict=True, **kwargs).to(device).eval()
    pre, post = make_pre_post_processors(policy_cfg=config, pretrained_path=str(checkpoint),
        preprocessor_overrides={'device_processor': {'device': str(device)}})
    return policy, pre, post


class TemporalRuntime:
    """Observe every source tick outside the action queue; only RGB/state/task enter policy."""
    def __init__(self, policy, preprocessor, postprocessor, device):
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.device = torch.device(device)
        self.spec = policy.model.spec
        self.buffer = TemporalRGBBuffer(self.spec)
        self.context = None
        self.last_tick = -1
        self.current = None
        self.state_input = getattr(policy, 'state_input', None)
        self.wheel_reference = None

    def observe(self, frames, *, tick, task, episode, state=None):
        context = (str(episode), str(task))
        if type(tick) is not int or tick < 0:
            raise ValueError('Source tick must be a nonnegative integer')
        if context != self.context:
            if tick != 0:
                raise ValueError('New episode/task must start at source tick zero')
            if self.state_input is not None:
                from ..data.state_input import validate_state_input, relative_wheel_state
                validate_state_input(self.state_input)
                if state is None:
                    raise ValueError('Relative wheel runtime requires the first true state at tick zero')
                reference=np.asarray(state,dtype=np.float32)
                relative_wheel_state(reference,reference)
                self.wheel_reference=reference.copy()
            self.buffer.reset()
            self.policy.reset()
            self.context = context
            self.last_tick = -1
        if tick != self.last_tick + 1:
            raise ValueError('Observe exactly once per source tick, including queued actions')
        self.current = self.buffer.observe(frames)
        self.last_tick = tick

    def batch(self, state):
        if self.current is None:
            raise ValueError('Observe RGB before inference')
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (32,) or not np.isfinite(state).all():
            raise ValueError('Expected original finite 32D state')
        if self.state_input is not None:
            from ..data.state_input import relative_wheel_state
            state=relative_wheel_state(state,self.wheel_reference)
        images, valid = self.current
        batch = {'observation.state': torch.from_numpy(state.copy())[None],
                 'task': [self.context[1]], 'robot_type': ['m20pro']}
        for cam in self.spec.cameras:
            key = 'observation.images.' + cam
            batch[key] = torch.from_numpy(images[cam].transpose(0,3,1,2).copy()).float()[None] / 255
            batch[key + '_is_pad'] = torch.from_numpy(~valid.copy())[None]
        return batch

    def predict(self, state):
        with torch.inference_mode(), torch.autocast(device_type=self.device.type,
                enabled=self.device.type == 'cuda' and bool(self.policy.config.use_amp)):
            batch = self.preprocessor(self.batch(state))
            return self.postprocessor(self.policy.select_action(batch))
