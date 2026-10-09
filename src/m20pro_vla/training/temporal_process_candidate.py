"""Explicit process-local temporal opt-in for the existing raw training entry.

This is not a training CLI. The official loop, processors and optimizer remain
its callers' responsibility; real training/runtime/full-resume are unverified.
"""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
from ..data.temporal_raw_candidate import TemporalRawDatasetCandidate
from .temporal_factory_candidate import make_temporal_policy_candidate

TEMPORAL_SETTINGS_ENV = 'M20PRO_TEMPORAL_TRAINING_SPEC'
CONTRACT_FILE = 'm20_temporal_training_contract.json'


def temporal_process_contract(dataset):
    from .raw_smolvla import raw_training_contract
    package = Path(__file__).resolve().parents[1]
    paths = ('data/raw_dataset.py', 'data/temporal_raw_candidate.py', 'data/temporal_rgb.py',
             'training/raw_smolvla.py', 'training/temporal_process_candidate.py',
             'training/temporal_factory_candidate.py', 'training/temporal_policy_candidate.py',
             'training/temporal_flow_candidate.py', 'training/visual_prefix_identity.py',
             'training/temporal_rgb.py')
    return dict(schema='m20_temporal_process_candidate_v1',
        raw=raw_training_contract(dataset.dataset), sampling=dataset.sampling_contract(),
        adapter_sha256={p:hashlib.sha256((package/p).read_bytes()).hexdigest() for p in paths})


@contextmanager
def temporal_process_hooks(official, settings, spec):
    """Temporarily select matching dataset/policy factories inside this process.

    Preserve the original official module attributes on success or failure.
    Never replace installed policy classes or shared policy factory globals.
    """
    from .raw_smolvla import official_raw_dataset_factory
    if spec.fps != 25 or len(spec.offsets) != 3:
        raise ValueError('Temporal process requires three slots at 25Hz')
    original_dataset, original_policy = official.make_dataset, official.make_policy
    selected = {}

    def dataset_factory(cfg):
        if cfg.policy.type != 'smolvla' or cfg.policy.use_peft or getattr(cfg,'peft',None) is not None:
            raise ValueError('Temporal process requires non-PEFT SmolVLA')
        if cfg.policy.max_state_dim != 32 or cfg.policy.adapt_to_pi_aloha:
            raise ValueError('Temporal process requires original 32D state')
        path = Path(cfg.output_dir)/CONTRACT_FILE
        if not cfg.resume and (path.exists() or (Path(cfg.output_dir)/'m20_raw_training_contract.json').exists()):
            raise ValueError('Fresh temporal run cannot overwrite an existing run contract')
        base = official_raw_dataset_factory(cfg, settings)
        dataset = TemporalRawDatasetCandidate(base, spec)
        contract = temporal_process_contract(dataset)
        if cfg.resume:
            if not path.is_file() or json.loads(path.read_text()) != contract:
                raise ValueError('Temporal source/sampling/adapter differs from saved run')
        else:
            with path.open('x') as stream:stream.write(json.dumps(contract,indent=2)+'\n')
        selected.update(dataset=dataset, config=cfg)
        return dataset

    def policy_factory(cfg, ds_meta=None, rename_map=None, **kwargs):
        if not selected or cfg is not selected['config'].policy or ds_meta is not selected['dataset'].meta:
            raise ValueError('Policy must use the selected temporal dataset and config')
        if rename_map or kwargs:
            raise ValueError('Temporal process does not support renamed or extra policy factory arguments')
        return make_temporal_policy_candidate(original_policy,cfg,selected['dataset'],
            require_temporal_contract=selected['config'].resume)

    official.make_dataset, official.make_policy = dataset_factory, policy_factory
    try:yield
    finally:official.make_dataset, official.make_policy = original_dataset, original_policy
