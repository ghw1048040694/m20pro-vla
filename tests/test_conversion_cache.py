"""Real LeRobot writer/reader checks for cache reuse and row/video alignment."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from m20pro_vla.data.conversion_cache import convert_m20_incremental, register_conversion_cache
from m20pro_vla.data.lerobot_adapter import convert_m20_to_lerobot, m20_lerobot_frame_indices, m20_smolvla_state


@unittest.skipUnless(importlib.util.find_spec('lerobot'), 'LeRobot required for integration')
class ConversionCacheTest(unittest.TestCase):
    def test_existing_subset_new_parallel_rows_videos_and_invalidation(self):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp); source = temp / 'raw'; source.mkdir()
            settings = dict(source_fps=50, frame_stride=2, terminal_stop_repeat=2,
                            recovery_terminal_stop_repeat=1, use_videos=True, vcodec='h264')

            def episode(eid):
                n = 16
                rgb = np.empty((n, 32, 32, 3), dtype=np.uint8)
                for i in range(n):
                    rgb[i] = (eid * 25 + i * 2) % 256
                action = np.zeros((n, 4), np.float32); action[:10, 0] = .1 * eid; action[10:, 3] = 1
                proprio = np.zeros((n, 45), np.float32); proprio[:, 3] = 1; proprio[:, 23] = eid / 10
                lidar = np.full((n, 72), 3, np.float32)
                np.savez(source / f'episode_{eid:08d}.npz', front_rgb=rgb, rear_rgb=rgb,
                         action=action, proprio=proprio, lidar=lidar)
                (source / f'episode_{eid:08d}.json').write_text(json.dumps(dict(
                    episode_id=eid, task_text=f'Find object {eid}', collection_mode='search')))

            episode(1); episode(2)
            baseline = temp / 'baseline'; cache = temp / 'cache'
            convert_m20_to_lerobot(source=source, output=baseline, repo_id='m20/baseline', **settings)
            register_conversion_cache(source=source, dataset=baseline, cache_root=cache, settings=settings)
            # Selection excludes old episode1. It must never appear in training rows.
            subset = temp / 'subset'; subset.mkdir()
            for suffix in ('npz', 'json'):
                (subset / f'episode_00000002.{suffix}').symlink_to(source / f'episode_00000002.{suffix}')
            for eid in (3, 4):
                episode(eid)
                for suffix in ('npz', 'json'):
                    (subset / f'episode_{eid:08d}.{suffix}').symlink_to(source / f'episode_{eid:08d}.{suffix}')
            output = temp / 'merged'
            report = convert_m20_incremental(source=subset, output=output, repo_id='m20/merged',
                                             cache_root=cache, workers=2, **settings)
            self.assertEqual((report['cached_episodes'], report['encoded_episodes']), (1, 2))
            self.assertEqual(set(report['episode_id_order']), {2, 3, 4})
            dataset = LeRobotDataset('m20/merged', root=output, video_backend='pyav')
            start = 0
            for eid in report['episode_id_order']:
                with np.load(source / f'episode_{eid:08d}.npz') as arrays:
                    idx = m20_lerobot_frame_indices(arrays['action'], frame_stride=2, terminal_stop_repeat=2)
                    for j, original in enumerate(idx):
                        row = dataset[start + j]
                        np.testing.assert_array_equal(row['action'].numpy(), arrays['action'][original])
                        np.testing.assert_array_equal(row['observation.state'].numpy(),
                                                      m20_smolvla_state(arrays['proprio'][original], arrays['lidar'][original]))
                        self.assertEqual(row['task'], f'Find object {eid}')
                        decoded = row['observation.images.front'].numpy() * 255
                        self.assertLess(np.abs(decoded - arrays['front_rgb'][original].transpose(2, 0, 1)).mean(), 4)
                    start += len(idx)
            self.assertEqual(start, len(dataset))
            replay = convert_m20_incremental(source=subset, output=temp / 'replay', repo_id='m20/replay',
                                             cache_root=cache, workers=2, **settings)
            self.assertEqual((replay['cached_episodes'], replay['encoded_episodes']), (3, 0))
            # A changed command must be re-encoded; content hashes, not IDs, select cache hits.
            metadata = source / 'episode_00000002.json'
            raw = json.loads(metadata.read_text()); raw['task_text'] = 'Different command'
            metadata.write_text(json.dumps(raw))
            changed = convert_m20_incremental(source=subset, output=temp / 'changed', repo_id='m20/changed',
                                              cache_root=cache, workers=2, **settings)
            self.assertEqual((changed['cached_episodes'], changed['encoded_episodes']), (2, 1))
            # Changing sampling must invalidate every old cache, preserving the stop budget.
            reweighted = convert_m20_incremental(
                source=subset, output=temp / 'reweighted', repo_id='m20/reweighted',
                cache_root=cache, workers=2, **(settings | dict(terminal_stop_repeat=3)))
            self.assertEqual((reweighted['cached_episodes'], reweighted['encoded_episodes']), (0, 3))
            self.assertEqual(reweighted['stop_frames'], 27)
            # A changed artifact cannot be a silent cache hit, even if source IDs match.
            (baseline / 'meta/stats.json').write_text('{}')
            verified = convert_m20_incremental(
                source=subset, output=temp / 'verified', repo_id='m20/verified',
                cache_root=cache, workers=2, **settings)
            self.assertTrue(verified['rejected_caches'])
            self.assertEqual((verified['cached_episodes'], verified['encoded_episodes']), (3, 0))
            with self.assertRaises(FileExistsError):
                convert_m20_incremental(source=subset, output=output, repo_id='m20/merged', cache_root=cache, **settings)

    def test_worker_limit_prevents_unbounded_cpu_pool(self):
        with self.assertRaises(ValueError):
            convert_m20_incremental(source=Path('.'), output=Path('unused-output'),
                                    repo_id='m20/test', cache_root=Path('unused-cache'), workers=8)
