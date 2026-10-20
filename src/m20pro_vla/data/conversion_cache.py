"""Content-verified conversion reuse, with bounded CPU work for new episodes."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
import uuid
from importlib.metadata import version

import numpy as np

from .lerobot_adapter import (
    _episode_pairs, convert_m20_to_lerobot, m20_lerobot_frame_indices, m20_smolvla_state,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _records(source: Path, settings: dict) -> list[dict]:
    records = []
    for npz, metadata in _episode_pairs(source):
        meta = json.loads(metadata.read_text())
        with np.load(npz) as arrays:
            actions = arrays['action']
            indices = m20_lerobot_frame_indices(
                actions, frame_stride=settings['frame_stride'],
                terminal_stop_repeat=settings['terminal_stop_repeat'],
                collection_mode=meta.get('collection_mode', 'legacy'),
                recovery_terminal_stop_repeat=settings['recovery_terminal_stop_repeat'],
            )
            stops = int((actions[indices, 3] >= .5).sum())
        records.append(dict(
            identity=_sha(npz) + ':' + _sha(metadata),
            npz=str(npz.resolve()), metadata=str(metadata.resolve()),
            episode_id=meta['episode_id'], mode=meta.get('collection_mode', 'legacy'),
            frames=len(indices), stop_frames=stops,
            sampled_frames=len(range(0, len(actions), settings['frame_stride'])),
        ))
    if len({r['episode_id'] for r in records}) != len(records):
        raise ValueError('Duplicate episode IDs must be resolved before conversion')
    return records


def _contract(settings: dict) -> dict:
    code = ''.join(inspect.getsource(f) for f in
                   (convert_m20_to_lerobot, m20_lerobot_frame_indices, m20_smolvla_state))
    return dict(settings=settings, lerobot_version=version('lerobot'),
                adapter_sha256=hashlib.sha256(code.encode()).hexdigest())


def _artifacts(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): _sha(p) for directory in ('meta', 'data', 'videos')
            for p in sorted((root / directory).rglob('*')) if p.is_file()}


def register_conversion_cache(*, source: Path, dataset: Path, cache_root: Path,
                              settings: dict) -> Path:
    """Certify a completed local artifact; reject mismatched rows or partial output."""
    import pyarrow.parquet as pq
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    report = json.loads((dataset / 'm20_conversion.json').read_text())
    for key, value in settings.items():
        if report[key] != value:
            raise ValueError(f'Conversion setting mismatch: {key}')
    records = _records(source, settings)
    # Incremental reports carry their actual merge order; legacy reports use sorted NPZ order.
    if 'episode_id_order' in report:
        by_id = {r['episode_id']: r for r in records}
        records = [by_id[eid] for eid in report['episode_id_order']]
        if len(records) != len(by_id) or len({r['episode_id'] for r in records}) != len(by_id):
            raise ValueError('Invalid conversion episode order')
    meta = LeRobotDatasetMetadata(report['repo_id'], root=dataset)
    if meta.total_episodes != len(records) or meta.total_frames != sum(r['frames'] for r in records):
        raise ValueError('Incomplete conversion artifact')
    tables = {}
    for index, record in enumerate(records):
        data_path = dataset / meta.get_data_file_path(index)
        if data_path not in tables:
            tables[data_path] = pq.read_table(data_path).to_pandas()
        rows = tables[data_path]
        rows = rows[rows.episode_index == index].sort_values('frame_index')
        with np.load(record['npz']) as arrays:
            raw = json.loads(Path(record['metadata']).read_text())
            indices = m20_lerobot_frame_indices(
                arrays['action'], frame_stride=settings['frame_stride'],
                terminal_stop_repeat=settings['terminal_stop_repeat'],
                collection_mode=record['mode'],
                recovery_terminal_stop_repeat=settings['recovery_terminal_stop_repeat'],
            )
            np.testing.assert_array_equal(np.stack(rows.action), arrays['action'][indices].astype(np.float32))
            state = m20_smolvla_state(arrays['proprio'], arrays['lidar'])[indices]
            np.testing.assert_array_equal(np.stack(rows['observation.state']), state)
        expected_task = int(meta.tasks.loc[raw['task_text'], 'task_index'])
        if len(rows) != record['frames'] or not (rows.task_index == expected_task).all():
            raise ValueError('Task/frame mismatch in cached conversion')
    # Hash again after verification, so a concurrent source mutation cannot be certified.
    for r in records:
        if r['identity'] != _sha(Path(r['npz'])) + ':' + _sha(Path(r['metadata'])):
            raise ValueError('Source changed during cache certification')
    cache_root.mkdir(parents=True, exist_ok=True)
    receipt = cache_root / f'certificate-{uuid.uuid4().hex}.json'
    payload = dict(schema='m20_conversion_cache_v1', root=str(dataset.resolve()),
                   repo_id=report['repo_id'], contract=_contract(settings), records=records,
                   artifacts=_artifacts(dataset))
    receipt.write_text(json.dumps(payload, indent=2) + '\n')
    return receipt


def _filtered_view(certificate: dict, selected: set[str], output: Path) -> list[dict]:
    """Select/reindex training rows while retaining existing encoded video bytes."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.dataset_tools import (
        _copy_and_reindex_data, _copy_and_reindex_episodes_metadata,
    )
    from lerobot.datasets.utils import write_tasks

    source = LeRobotDataset(certificate['repo_id'], root=Path(certificate['root']))
    indices = [i for i, r in enumerate(certificate['records']) if r['identity'] in selected]
    mapping = {old: new for new, old in enumerate(indices)}
    meta = LeRobotDatasetMetadata.create(
        repo_id='m20/cache-view', root=output, fps=source.meta.fps,
        robot_type=source.meta.robot_type, features=source.meta.features,
        use_videos=bool(source.meta.video_keys),
    )
    meta.tasks = source.meta.tasks.copy()
    write_tasks(meta.tasks, output)
    videos = {}
    for old, new in mapping.items():
        episode = source.meta.episodes[old]
        videos[new] = {}
        for key in source.meta.video_keys:
            for field in ('chunk_index', 'file_index', 'from_timestamp', 'to_timestamp'):
                name = f'videos/{key}/{field}'
                videos[new][name] = episode[name]
            relative = source.meta.get_video_file_path(old, key)
            target = output / relative
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to((source.root / relative).resolve())
    data = _copy_and_reindex_data(source, meta, mapping)
    _copy_and_reindex_episodes_metadata(source, meta, mapping, data, videos or None)
    return [certificate['records'][i] for i in indices]


def convert_m20_incremental(*, source: Path, output: Path, repo_id: str,
                            cache_root: Path, workers: int = 2, **settings) -> dict:
    """Reuse verified finished artifacts and encode only missing episodes on CPUs."""
    from lerobot.datasets.aggregate import aggregate_datasets

    settings = dict(source_fps=50, frame_stride=2, terminal_stop_repeat=1,
                    recovery_terminal_stop_repeat=1, use_videos=True, vcodec='h264') | settings
    if workers < 1 or workers > 4:
        raise ValueError('Conversion workers must be between 1 and 4')
    if output.exists():
        raise FileExistsError(f'LeRobot dataset already exists: {output}')
    records = _records(source, settings)
    remaining = {r['identity'] for r in records}
    contract = _contract(settings)
    certificates = []
    rejected = []
    cache_root.mkdir(parents=True, exist_ok=True)
    for p in cache_root.glob('certificate-*.json'):
        try:
            c = json.loads(p.read_text())
            if c['contract'] != contract:
                continue
            if not c['artifacts'] or any(not (Path(c['root']) / name).is_file()
                                        or _sha(Path(c['root']) / name) != sha
                                        for name, sha in c['artifacts'].items()):
                raise ValueError('Cached artifact changed or is incomplete')
            certificates.append(c)
        except (OSError, ValueError, KeyError) as error:
            rejected.append(dict(certificate=str(p), reason=str(error)))
    certificates.sort(key=lambda c: len(remaining & {r['identity'] for r in c['records']}),
                      reverse=True)
    ordered = []
    roots = []
    reused = 0
    # Stage the merge separately: failed/partial work is never published as training output.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.m20-convert-', dir=output.parent) as temp:
        temp = Path(temp)
        for c in certificates:
            selected = remaining & {r['identity'] for r in c['records']}
            if not selected:
                continue
            root = temp / f'view-{len(roots)}'
            ordered.extend(_filtered_view(c, selected, root))
            roots.append(root)
            reused += len(selected)
            remaining -= selected
        missing = [r for r in records if r['identity'] in remaining]
        groups = [missing[i::workers] for i in range(min(workers, len(missing)))]

        def encode(group: list[dict]) -> tuple[Path, list[dict]]:
            uid = uuid.uuid4().hex
            inputs = temp / f'inputs-{uid}'; inputs.mkdir()
            root = cache_root / f'part-{uid}'
            for r in group:
                for field, suffix in [('npz', '.npz'), ('metadata', '.json')]:
                    (inputs / f"episode_{r['episode_id']:08d}{suffix}").symlink_to(r[field])
            # Writer's sorted order must be recorded, including tasks and episode indices.
            group = sorted(group, key=lambda r: r['episode_id'])
            convert_m20_to_lerobot(source=inputs, output=root, repo_id='m20/cached-part', **settings)
            register_conversion_cache(source=inputs, dataset=root, cache_root=cache_root, settings=settings)
            return root, group

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for root, group in pool.map(encode, groups):
                roots.append(root); ordered.extend(group)
        destination = temp / 'merged'
        aggregate_datasets(repo_ids=['m20/local-part'] * len(roots), roots=roots,
                           aggr_repo_id=repo_id, aggr_root=destination)
        for r in records:
            if r['identity'] != _sha(Path(r['npz'])) + ':' + _sha(Path(r['metadata'])):
                raise ValueError('Source changed during conversion')
        frames = sum(r['frames'] for r in records)
        stops = sum(r['stop_frames'] for r in records)
        modes = {r['mode'] for r in records}
        report = dict(schema='m20pro_lerobot_conversion_v1', source=str(source), output=str(output),
                      repo_id=repo_id, episodes=len(records), frames=frames,
                      sampled_frames_before_terminal_repeat=sum(r['sampled_frames'] for r in records),
                      stop_frames=stops, stop_frame_fraction=stops / frames,
                      frames_by_collection_mode={m: sum(r['frames'] for r in records if r['mode'] == m) for m in modes},
                      stop_frames_by_collection_mode={m: sum(r['stop_frames'] for r in records if r['mode'] == m) for m in modes},
                      episode_frames_min=min(r['frames'] for r in records),
                      episode_frames_max=max(r['frames'] for r in records),
                      fps=settings['source_fps'] // settings['frame_stride'], state_dim=32, action_dim=4,
                      camera_keys=['observation.images.front', 'observation.images.rear'],
                      cached_episodes=reused, encoded_episodes=len(missing), cpu_workers=workers,
                      episode_id_order=[r['episode_id'] for r in ordered], rejected_caches=rejected,
                      **settings)
        info = json.loads((destination / 'meta/info.json').read_text())
        if info['total_frames'] != frames or info['total_episodes'] != len(records):
            raise ValueError('Merged dataset counts differ from input sampling')
        (destination / 'm20_conversion.json').write_text(json.dumps(report, indent=2) + '\n')
        destination.rename(output)
    return report
