"""Explicit project-local policy subclass candidate; no factory/global patch."""
import json
from pathlib import Path
from types import SimpleNamespace
from ..data.temporal_rgb import TemporalRGBSpec
from .temporal_rgb import prepare_temporal_images
from .temporal_flow_candidate import TemporalFlowCoreCandidate

CONTRACT_FILE = 'm20_temporal_policy_candidate.json'


def temporal_policy_class(base_policy_class):
    """Choose this class explicitly in a future official factory/load adapter.

    Base checkpoint transfer loads original keys before wrapping. Temporal
    resume wraps before strict loading and requires its saved contract. No
    inherited policy class, installed module, or user training CLI is modified.
    Factory selection, dataset history offsets, runtime and official full
    optimizer resume still need separate integration/validation.
    """
    class TemporalPolicyCandidate(base_policy_class):
        def __init__(self, config, *, temporal_spec=None, wrap_after_load=False, **kwargs):
            if config.max_state_dim != 32 or config.adapt_to_pi_aloha:
                raise ValueError('Candidate requires original 32D M20 state')
            self._temporal_spec = temporal_spec or TemporalRGBSpec()
            super().__init__(config, **kwargs)
            if not wrap_after_load:
                self._enable_temporal_core()

        def _enable_temporal_core(self):
            if isinstance(self.model, TemporalFlowCoreCandidate):
                raise ValueError('Policy is already temporally wrapped')
            core = self.model
            self.model = TemporalFlowCoreCandidate(core, core.state_proj.out_features, self._temporal_spec)
            self.model.identity.to(device=core.state_proj.weight.device)
            self.model.train(self.training)

        def prepare_images(self, batch):
            base_view = SimpleNamespace(config=self.config, prepare_images=super().prepare_images)
            images, masks, _ = prepare_temporal_images(base_view, batch, self._temporal_spec)
            return images, masks

        def _save_pretrained(self, directory):
            if not isinstance(self.model, TemporalFlowCoreCandidate):
                raise ValueError('Cannot save an unwrapped temporal candidate')
            super()._save_pretrained(directory)
            contract = dict(schema='m20_temporal_policy_candidate_v1', core=self.model.contract())
            (Path(directory)/CONTRACT_FILE).write_text(json.dumps(contract, indent=2)+'\n')

        @classmethod
        def from_pretrained(cls, pretrained_name_or_path, *, temporal_spec=None,
                            require_temporal_contract=False, **kwargs):
            directory = Path(pretrained_name_or_path)
            if not directory.is_dir():
                raise ValueError('Candidate checkpoint loading requires an existing local directory')
            if kwargs.get('strict', True) is not True:
                raise ValueError('Candidate checkpoint loading must be strict')
            saved = directory/CONTRACT_FILE
            if require_temporal_contract and not saved.is_file():
                raise ValueError('Temporal resume requires its saved contract')
            spec = temporal_spec or TemporalRGBSpec()
            contract = None
            if saved.is_file():
                contract = json.loads(saved.read_text())
                if set(contract) != {'schema', 'core'} or contract['schema'] != 'm20_temporal_policy_candidate_v1':
                    raise ValueError('Unknown temporal policy contract')
                stored = TemporalRGBSpec.from_dict(contract['core']['temporal_rgb'])
                if temporal_spec is not None and stored != temporal_spec:
                    raise ValueError('Requested temporal sampling differs from checkpoint')
                spec = stored
            kwargs['strict'] = True
            policy = super().from_pretrained(directory, temporal_spec=spec,
                wrap_after_load=not saved.is_file(), **kwargs)
            if not saved.is_file():
                policy._enable_temporal_core()
            elif policy.model.contract() != contract['core']:
                raise ValueError('Saved core identity/sampling contract differs')
            return policy
    return TemporalPolicyCandidate
