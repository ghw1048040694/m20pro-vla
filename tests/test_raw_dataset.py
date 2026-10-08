"""CPU integration of experimental raw reader against installed LeRobot."""
import json
from pathlib import Path
import tempfile
import unittest
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader
from m20pro_vla.data.raw_dataset import RawM20Dataset
from m20pro_vla.data.lerobot_adapter import convert_m20_to_lerobot, m20_lerobot_frame_indices


class RawDatasetTest(unittest.TestCase):
    def make_raw(self, root, count=3):
        root.mkdir()
        for eid in range(1,count+1):
            n=16
            rgb=np.empty((n,32,32,3),np.uint8)
            for i in range(n): rgb[i]=20*eid+i
            action=np.zeros((n,4),np.float32); action[:10,0]=eid*.1; action[10:,3]=1
            proprio=np.zeros((n,45),np.float32); proprio[:,3]=1; proprio[:,23]=eid*.01
            lidar=np.full((n,72),3,np.float32)
            np.savez_compressed(root/f'episode_{eid:04}.npz',front_rgb=rgb,rear_rgb=rgb,
                                action=action,proprio=proprio,lidar=lidar)
            (root/f'episode_{eid:04}.json').write_text(json.dumps(dict(episode_id=eid,
                task_text=f'Find {eid}',collection_mode='failure_recovery' if eid==2 else 'search')))

    def test_rows_boundaries_chunks_stats_and_worker_loading(self):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.policies.factory import dataset_to_policy_features
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); source=root/'raw'; self.make_raw(source)
            settings=dict(source_fps=50,frame_stride=2,terminal_stop_repeat=3,recovery_terminal_stop_repeat=1)
            delta={'action':list(range(50)), 'observation.state':[-1,0],
                   'observation.images.front':[-1,0], 'observation.images.rear':[0]}
            raw=RawM20Dataset(source,root/'cache',delta_indices=delta,max_open_episodes=1,**settings)
            output=root/'legacy'
            convert_m20_to_lerobot(source=source,output=output,repo_id='m20/raw-test',**settings)
            official=LeRobotDataset('m20/raw-test',root=output,video_backend='pyav',
                delta_timestamps={k:[i/25 for i in values] for k,values in delta.items()})
            self.assertEqual((len(raw),raw.num_episodes),(len(official),3))
            self.assertEqual(dataset_to_policy_features(raw.meta.features), dataset_to_policy_features(official.meta.features))
            for i in range(len(raw)):
                a,b=raw[i],official[i]
                for key in ('action','observation.state','action_is_pad','observation.state_is_pad',
                            'observation.images.front_is_pad','observation.images.rear_is_pad',
                            'timestamp','episode_index','frame_index','index','task_index'):
                    torch.testing.assert_close(a[key],b[key])
                self.assertEqual(a['task'],b['task'])
                for key in raw.meta.camera_keys:
                    self.assertEqual(a[key].shape,b[key].shape)
                    self.assertLess((a[key]-b[key]).abs().mean().item(), .025)
                self.assertLessEqual(len(raw._maps),1)
            for key,stats in raw.meta.stats.items():
                for name,value in stats.items():
                    np.testing.assert_allclose(value,official.meta.stats[key][name],rtol=1e-5,atol=1e-6)
            clone=pickle.loads(pickle.dumps(raw)); self.assertEqual(len(clone._maps),0)
            plain=RawM20Dataset(source,root/'cache',max_open_episodes=1,**settings)
            loader=DataLoader(plain,batch_size=4,num_workers=2,shuffle=True,
                generator=torch.Generator().manual_seed(1))
            self.assertEqual(sum(len(batch['task']) for batch in loader),len(plain))
            # Same virtual rows must reference exact raw RGB; lossy legacy RGB is only approximate.
            start=0
            for eid in range(1,4):
                with np.load(source/f'episode_{eid:04}.npz') as arrays:
                    idx=m20_lerobot_frame_indices(arrays['action'],frame_stride=2,terminal_stop_repeat=3,
                        recovery_terminal_stop_repeat=1,collection_mode='failure_recovery' if eid==2 else 'search')
                    for j,v in enumerate(idx):
                        np.testing.assert_array_equal(plain[start+j]['observation.images.front'].numpy(),
                            arrays['front_rgb'][v].transpose(2,0,1).astype(np.float32)/255)
                    start+=len(idx)

    def test_cached_immutability_changes_and_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); source=root/'raw'; self.make_raw(source,1)
            first=RawM20Dataset(source,root/'cache')
            original=first.records[0]['root']
            replay=RawM20Dataset(source,root/'cache'); self.assertEqual(replay.records[0]['root'],original)
            self.assertFalse(list((root/'cache').rglob('*.mp4')))
            metadata=source/'episode_0001.json'; value=json.loads(metadata.read_text()); value['task_text']='Find changed'
            metadata.write_text(json.dumps(value))
            changed=RawM20Dataset(source,root/'cache'); self.assertNotEqual(changed.records[0]['root'],original)
            path=Path(changed.records[0]['root'])/'action.npy'
            with path.open('ab') as fp: fp.write(b'corrupt')
            with self.assertRaisesRegex(ValueError,'artifact changed'):
                RawM20Dataset(source,root/'cache')
            with self.assertRaises(ValueError): RawM20Dataset(source,root/'cache',frame_stride=0)
            with self.assertRaises(ValueError): RawM20Dataset(source,root/'cache',max_open_episodes=0)


if __name__=='__main__': unittest.main()
