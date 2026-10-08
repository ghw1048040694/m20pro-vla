"""CPU integration of the unified raw backend with installed LeRobot APIs."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

from m20pro_vla.training.raw_smolvla import official_raw_dataset_factory, raw_training_settings, main, RAW_SETTINGS_ENV
from m20pro_vla.training.smolvla import prepare_smolvla_training_source, smolvla_training_command
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors


class RawSmolVLABackendTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name); self.source=self.root/'raw'; self.source.mkdir()
        for eid in (1,2):
            n=16; image=np.full((n,32,32,3),50+eid,np.uint8)
            action=np.zeros((n,4),np.float32); action[:10,0]=.1; action[10:,3]=1
            np.savez_compressed(self.source/f'episode_{eid}.npz',action=action,
                proprio=np.zeros((n,45),np.float32),lidar=np.ones((n,72),np.float32),front_rgb=image,rear_rgb=image)
            (self.source/f'episode_{eid}.json').write_text(json.dumps(dict(episode_id=eid,
                task_text='Find green bin',collection_mode='search' if eid==1 else 'failure_recovery')))
        self.output=self.root/'output'
        self.policy=SmolVLAConfig(device='cpu',vlm_model_name='HuggingFaceTB/SmolVLM2-500M-Video-Instruct')
        self.cfg=TrainPipelineConfig(dataset=DatasetConfig(repo_id='m20/raw-cpu',root=self.source),
            policy=self.policy,output_dir=self.output)
        self.settings=dict(cache_root=str(self.root/'cache'),source_fps=50,frame_stride=2,
            terminal_stop_repeat=8,recovery_terminal_stop_repeat=1,max_open_episodes=2)

    def test_official_factory_processors_and_process_adapter_without_videos(self):
        dataset=official_raw_dataset_factory(self.cfg,self.settings)
        self.assertEqual((dataset.num_episodes,dataset.num_frames),(2,37))
        self.assertEqual(dataset.delta_indices['action'],list(range(50)))
        batch=next(iter(torch.utils.data.DataLoader(dataset,batch_size=4,num_workers=0)))
        self.assertEqual(batch['action'].shape,(4,50,4))
        self.assertEqual(batch['observation.state'].shape,(4,1,32))
        from lerobot.datasets.utils import dataset_to_policy_features
        from lerobot.configs.types import FeatureType
        features=dataset_to_policy_features(dataset.meta.features)
        self.policy.input_features={k:v for k,v in features.items() if v.type is not FeatureType.ACTION}
        self.policy.output_features={k:v for k,v in features.items() if v.type is FeatureType.ACTION}
        pre,post=make_smolvla_pre_post_processors(self.policy,dataset_stats=dataset.meta.stats)
        processed=pre(batch)
        self.assertEqual(processed['observation.language.tokens'].shape[0],4)
        torch.testing.assert_close(post(processed['action']),batch['action'])
        self.assertFalse(list(self.root.rglob('*.mp4')))
        # The internal child-process entry point uses the official main/loop,
        # with only its dataset factory overridden; no global installation edit.
        import lerobot.scripts.lerobot_train as official
        prior=official.make_dataset
        try:
            with patch.dict('os.environ',{RAW_SETTINGS_ENV:json.dumps(self.settings)}), patch.object(official,'main') as entry:
                main(); entry.assert_called_once()
                self.assertIsNot(official.make_dataset,prior)
        finally:
            official.make_dataset=prior

    def test_resume_contract_and_unsupported_modes_fail_explicitly(self):
        dataset=official_raw_dataset_factory(self.cfg,self.settings)
        self.cfg.resume=True
        restored=official_raw_dataset_factory(self.cfg,self.settings)
        self.assertEqual(dataset.records,restored.records)
        changed=dict(self.settings,terminal_stop_repeat=7)
        with self.assertRaisesRegex(ValueError,'differs'):
            official_raw_dataset_factory(self.cfg,changed)
        metadata=self.source/'episode_1.json'
        value=json.loads(metadata.read_text()); value['task_text']='Changed task'; metadata.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'differs'):
            official_raw_dataset_factory(self.cfg,self.settings)
        self.cfg.dataset.episodes=[0]
        with self.assertRaises(NotImplementedError): official_raw_dataset_factory(self.cfg,self.settings)

    def test_prepare_and_unified_command_do_not_require_converted_dataset(self):
        base=self.root/'base'; base.mkdir(); self.policy.save_pretrained(base)
        (base/'model.safetensors').write_bytes(b'CPU fixture only; not a trained model')
        config=dict(experiment_id='raw-cpu',paths=dict(dataset=str(self.source),lerobot_dataset=str(self.root/'absent-video'),
            smolvla_prepared_base=str(self.root/'prepared'),smolvla_checkpoint_dir=str(self.output)),
            smolvla=dict(dataset_backend='raw',raw_cache_dir=str(self.root/'cache'),init_policy=str(base),offline=True,
                repo_id='m20/raw-cpu',source_fps=50,frame_stride=2,terminal_stop_repeat=8,recovery_terminal_stop_repeat=1,
                steps=10,batch_size=4,num_workers=2,log_freq=10,save_freq=5,seed=1))
        prepared=prepare_smolvla_training_source(config)
        self.assertTrue((prepared/'policy_preprocessor.json').is_file())
        command=smolvla_training_command(config,policy_path=prepared)
        self.assertEqual(command[1:3],['-m','m20pro_vla.training.raw_smolvla'])
        self.assertIn('--dataset.root='+str(self.source),command)
        self.assertFalse((self.root/'absent-video').exists())
        self.assertFalse(list(self.root.rglob('*.mp4')))


if __name__=='__main__': unittest.main()
