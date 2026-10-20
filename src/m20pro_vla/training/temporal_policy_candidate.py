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
        def __init__(self, config, *, temporal_spec=None, wrap_after_load=False, binary_stop_spec=None, **kwargs):
            if config.max_state_dim != 32 or config.adapt_to_pi_aloha:
                raise ValueError('Candidate requires original 32D M20 state')
            self._temporal_spec = temporal_spec or TemporalRGBSpec()
            super().__init__(config, **kwargs)
            if not wrap_after_load:
                self._enable_temporal_core()
            if binary_stop_spec is not None:
                self.model.enable_binary_stop(binary_stop_spec)

        def _enable_temporal_core(self):
            if isinstance(self.model, TemporalFlowCoreCandidate):
                raise ValueError('Policy is already temporally wrapped')
            core = self.model
            self.model = TemporalFlowCoreCandidate(core, core.state_proj.out_features, self._temporal_spec)
            self.model.identity.to(device=core.state_proj.weight.device)
            self.model.train(self.training)

        def forward(self, batch, noise=None, time=None, reduction='mean'):
            if getattr(self, 'training_objective', None) is None:
                return super().forward(batch, noise=noise, time=time, reduction=reduction)
            from .physical_flow_loss import physical_flow_forward
            loss, metrics = physical_flow_forward(self, batch, noise, time, reduction)
            spec = getattr(self.model, 'binary_stop_spec', None)
            if spec is not None:
                from .binary_stop import masked_binary_stop_loss
                binary = masked_binary_stop_loss(self.model.last_stop_logits,
                    batch['action'], batch['action_is_pad'], spec, reduction=reduction)
                loss = loss + spec['loss_coefficient'] * binary
                metrics.update(binary_stop_loss=float(binary.detach().mean()),
                    loss=float(loss.detach().mean()))
            return loss, metrics

        def get_optim_params(self):
            spec = getattr(self.model, 'binary_stop_spec', None)
            if spec is None:
                return super().get_optim_params()
            head = list(self.model.binary_stop_head.parameters())
            head_ids = {id(p) for p in head}
            base = [p for p in self.parameters() if p.requires_grad and id(p) not in head_ids]
            return [{'params': base}, {'params': head, 'lr': spec['head_learning_rate']}]


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
            objective = getattr(self, 'training_objective', None)
            from .physical_flow_loss import validate_objective_context
            validate_objective_context(None, objective, getattr(self.model, 'binary_stop_spec', None))
            if objective is not None:
                from .physical_flow_loss import CONTRACT_FILE as OBJECTIVE_FILE, validate_objective
                (Path(directory)/OBJECTIVE_FILE).write_text(json.dumps(validate_objective(objective), indent=2)+'\n')
            binary = getattr(self.model, 'binary_stop_spec', None)
            if binary is not None:
                from .binary_stop import validate_binary_stop_spec
                (Path(directory)/'m20_binary_stop.json').write_text(json.dumps(validate_binary_stop_spec(binary), indent=2)+'\n')
            decoding = getattr(self.model, 'stop_decoding_spec', None)
            if decoding is not None:
                from .stop_decoding import save_stop_decoding
                save_stop_decoding(directory, decoding, binary)
            precision = getattr(self, 'optimizer_precision', None)
            if precision is not None:
                from .optimizer_precision import save_precision_spec
                save_precision_spec(directory, precision)
            connector = getattr(self, 'visual_connector_adaptation', None)
            if connector is not None:
                from .visual_connector_adaptation import save_connector_spec
                save_connector_spec(directory, connector)
            state_input = getattr(self, 'state_input', None)
            if state_input is not None:
                from ..data.state_input import save_state_input
                save_state_input(directory, state_input)
            execution = getattr(self, 'execution_contract', None)
            if execution is not None:
                from ..data.execution_contract import save_execution_contract
                save_execution_contract(directory, execution)

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
            from .binary_stop import validate_binary_stop_spec, validate_saved_normalization
            head_saved = directory/'m20_binary_stop.json'
            head_init = directory/'m20_binary_stop_init.json'
            if head_saved.exists() and head_init.exists():
                raise ValueError('Saved head and untrained head request are mutually exclusive')
            binary = None
            if head_saved.exists() or head_init.exists():
                binary = validate_binary_stop_spec(json.loads((head_saved if head_saved.exists() else head_init).read_text()))
                validate_saved_normalization(directory, binary)
                if not saved.exists():
                    raise ValueError('Binary STOP requires the saved temporal policy')
            kwargs['strict'] = True
            policy = super().from_pretrained(directory, temporal_spec=spec,
                wrap_after_load=not saved.is_file(), binary_stop_spec=binary if head_saved.exists() else None, **kwargs)
            if not saved.is_file():
                policy._enable_temporal_core()
            elif policy.model.contract() != contract['core']:
                raise ValueError('Saved core identity/sampling contract differs')
            from ..data.execution_contract import load_execution_contract
            policy.execution_contract = load_execution_contract(directory)
            from .physical_flow_loss import load_objective
            policy.training_objective = load_objective(directory)
            if binary is not None:
                if policy.training_objective is None:
                    raise ValueError('Binary STOP requires the explicit physical flow objective')
                if head_init.exists():
                    policy.model.enable_binary_stop(binary)
            from .physical_flow_loss import validate_objective_context
            validate_objective_context(directory, policy.training_objective, binary)
            from .stop_decoding import load_stop_decoding
            decoding = load_stop_decoding(directory, binary)
            if decoding is not None:
                if head_init.exists():
                    raise ValueError('Runtime STOP decoding requires a saved trained binary head')
                policy.model.stop_decoding_spec = decoding
            from .visual_connector_adaptation import load_connector_spec, enable_visual_connector_adaptation
            connector = load_connector_spec(directory)
            if connector is not None:
                enable_visual_connector_adaptation(policy, connector)
            from .optimizer_precision import load_precision_spec
            policy.optimizer_precision = load_precision_spec(directory)
            from ..data.state_input import load_state_input
            policy.state_input = load_state_input(directory)
            return policy
    return TemporalPolicyCandidate
