"""Internal process adapter for the existing train-smolvla stage.

Uses the installed official training loop. By default its process-local dataset
factory is replaced; an explicit temporal env opt-in also selects a matching policy.
This module is not a separate user-facing CLI.
"""
import json
import hashlib
import os
from pathlib import Path

RAW_SETTINGS_ENV = 'M20PRO_RAW_TRAINING_SETTINGS'


def raw_training_settings(config):
    settings = config['smolvla']
    return dict(cache_root=str(Path(settings.get('raw_cache_dir', '.runtime/cache/raw_training')).resolve()),
        source_fps=int(settings['source_fps']), frame_stride=int(settings['frame_stride']),
        terminal_stop_repeat=int(settings.get('terminal_stop_repeat', 1)),
        recovery_terminal_stop_repeat=int(settings.get('recovery_terminal_stop_repeat', 1)),
        max_open_episodes=int(settings.get('raw_max_open_episodes', 4)))


def make_raw_dataset(source, settings, policy=None):
    from ..data.raw_dataset import RawM20Dataset
    dataset = RawM20Dataset(source, **settings)
    if policy is not None:
        dataset.delta_indices = {}
        for key in dataset.meta.features:
            offsets = policy.action_delta_indices if key == 'action' else policy.observation_delta_indices
            if offsets is not None:
                dataset.delta_indices[key] = list(offsets)
        if policy.reward_delta_indices is not None:
            raise NotImplementedError('Raw M20 reader has no reward feature')
    return dataset


def raw_training_contract(dataset):
    # Location-independent identities, including every cached artifact digest.
    return dict(schema='m20pro_raw_training_contract_v1',
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), settings=dataset.settings,
        frames=dataset.num_frames, episodes=dataset.num_episodes,
        records=[{key: record[key] for key in ('identity', 'contract', 'episode_id', 'task', 'mode', 'frames', 'artifacts')}
                 for record in dataset.records])


def validate_raw_resume_contract(output, dataset):
    saved = Path(output)/'m20_raw_training_contract.json'
    if not saved.is_file() or json.loads(saved.read_text()) != raw_training_contract(dataset):
        raise ValueError('Raw training source/sampling/backend differs from the saved run')


def official_raw_dataset_factory(cfg, settings):
    if cfg.policy.type != 'smolvla':
        raise NotImplementedError('Raw M20 training adapter currently supports SmolVLA only')
    if cfg.dataset.streaming or cfg.dataset.episodes is not None or cfg.dataset.image_transforms.enable:
        raise NotImplementedError('Raw M20 training does not support streaming, episode filtering or image transforms')
    if cfg.rename_map or cfg.use_rabc:
        raise NotImplementedError('Raw M20 training does not support renamed observations or RA-BC')
    dataset = make_raw_dataset(cfg.dataset.root, settings, cfg.policy)
    output = Path(cfg.output_dir)
    if cfg.resume:
        validate_raw_resume_contract(output, dataset)
    else:
        output.mkdir(parents=True, exist_ok=True)
        (output/'m20_raw_training_contract.json').write_text(json.dumps(raw_training_contract(dataset), indent=2)+'\n')
    if cfg.dataset.use_imagenet_stats:
        from lerobot.datasets.factory import IMAGENET_STATS
        import torch
        for key in dataset.meta.camera_keys:
            for name, value in IMAGENET_STATS.items():
                dataset.meta.stats[key][name] = torch.tensor(value, dtype=torch.float32)
    return dataset


def main():
    settings = json.loads(os.environ[RAW_SETTINGS_ENV])
    import lerobot.scripts.lerobot_train as official
    from .temporal_process_candidate import TEMPORAL_SETTINGS_ENV, temporal_process_hooks
    temporal = os.environ.get(TEMPORAL_SETTINGS_ENV)
    if temporal is None:
        official.make_dataset = lambda cfg: official_raw_dataset_factory(cfg, settings)
        official.main()
    else:
        from ..data.temporal_rgb import TemporalRGBSpec
        spec = TemporalRGBSpec.from_dict(json.loads(temporal))
        with temporal_process_hooks(official, settings, spec):
            official.main()


if __name__ == '__main__':
    main()
