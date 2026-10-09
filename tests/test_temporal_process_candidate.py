import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from m20pro_vla.data.temporal_rgb import TemporalRGBSpec
from m20pro_vla.training.temporal_process_candidate import temporal_process_hooks, CONTRACT_FILE
from test_temporal_factory_candidate import factory


class TinyRaw:
    def __init__(self):
        self.settings=dict(source_fps=50,frame_stride=2,terminal_stop_repeat=8,recovery_terminal_stop_repeat=1)
        self.records=[dict(identity='fixture-only',contract={},episode_id=1,task='Approach the red cube.',mode='search',frames=17,artifacts={'fixture':'sha-placeholder'})]
        self.num_frames=17;self.num_episodes=1;self.delta_indices={'action':list(range(50)),'observation.state':[0],'observation.images.front':[0],'observation.images.rear':[0]}
        self.meta=SimpleNamespace(fps=25,camera_keys=['observation.images.front','observation.images.rear'],features={'observation.state':{'shape':(32,)}})
    def __len__(self):return self.num_frames


class TemporalProcessTests(unittest.TestCase):
    def config(self,path):
        return SimpleNamespace(output_dir=Path(path),resume=False,peft=None,policy=SimpleNamespace(type='smolvla',use_peft=False,max_state_dim=32,adapt_to_pi_aloha=False,pretrained_path=None,image_features={'observation.images.front':None,'observation.images.rear':None}))
    def official(self):return SimpleNamespace(make_dataset=lambda cfg:None,make_policy=factory)
    def test_explicit_process_same_spec_counts_and_original_factories_restored(self):
        with tempfile.TemporaryDirectory() as folder, patch('m20pro_vla.training.raw_smolvla.official_raw_dataset_factory',return_value=TinyRaw()):
            official=self.official();original=(official.make_dataset,official.make_policy);cfg=self.config(folder);spec=TemporalRGBSpec((-200,-50,0))
            with temporal_process_hooks(official,{},spec):
                dataset=official.make_dataset(cfg);policy=official.make_policy(cfg=cfg.policy,ds_meta=dataset.meta,rename_map={})
                self.assertEqual((dataset.num_frames,dataset.num_episodes),(17,1));self.assertEqual(policy.model.spec,spec)
                self.assertEqual(json.loads((Path(folder)/CONTRACT_FILE).read_text())['sampling']['temporal_rgb'],spec.to_dict())
            self.assertEqual((official.make_dataset,official.make_policy),original)
    def test_wrong_rate_or_slot_count_rejected_before_hook_install(self):
        official=self.official();original=(official.make_dataset,official.make_policy)
        for spec in (TemporalRGBSpec((-10,0)),TemporalRGBSpec((-400,-100,0),fps=50)):
            with self.assertRaises(ValueError):
                with temporal_process_hooks(official,{},spec):pass
            self.assertEqual((official.make_dataset,official.make_policy),original)
    def test_restore_factories_on_error(self):
        official=self.official();original=(official.make_dataset,official.make_policy)
        with self.assertRaisesRegex(RuntimeError,'intentional'):
            with temporal_process_hooks(official,{},TemporalRGBSpec()):raise RuntimeError('intentional')
        self.assertEqual((official.make_dataset,official.make_policy),original)
    def test_policy_before_dataset_or_wrong_metadata_rejected(self):
        with tempfile.TemporaryDirectory() as folder, patch('m20pro_vla.training.raw_smolvla.official_raw_dataset_factory',return_value=TinyRaw()):
            official=self.official();cfg=self.config(folder)
            with temporal_process_hooks(official,{},TemporalRGBSpec()):
                with self.assertRaises(ValueError):official.make_policy(cfg.policy)
                dataset=official.make_dataset(cfg)
                with self.assertRaises(ValueError):official.make_policy(cfg.policy,object())
                with self.assertRaises(ValueError):official.make_policy(cfg.policy,dataset.meta,rename_map={'front':'other'})
    def test_resume_sampling_source_and_missing_contract_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            official=self.official();cfg=self.config(folder);raw=TinyRaw()
            with patch('m20pro_vla.training.raw_smolvla.official_raw_dataset_factory',return_value=raw):
                with temporal_process_hooks(official,{},TemporalRGBSpec()):official.make_dataset(cfg)
                cfg.resume=True
                with temporal_process_hooks(official,{},TemporalRGBSpec()):self.assertEqual(len(official.make_dataset(cfg)),17)
                with temporal_process_hooks(official,{},TemporalRGBSpec((-200,-50,0))):
                    with self.assertRaises(ValueError):official.make_dataset(cfg)
                raw.records[0]['identity']='different-source'
                with temporal_process_hooks(official,{},TemporalRGBSpec()):
                    with self.assertRaises(ValueError):official.make_dataset(cfg)
                (Path(folder)/CONTRACT_FILE).unlink()
                with temporal_process_hooks(official,{},TemporalRGBSpec()):
                    with self.assertRaises(ValueError):official.make_dataset(cfg)
    def test_fresh_run_contract_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder, patch('m20pro_vla.training.raw_smolvla.official_raw_dataset_factory',return_value=TinyRaw()):
            official=self.official();cfg=self.config(folder)
            with temporal_process_hooks(official,{},TemporalRGBSpec()):official.make_dataset(cfg)
            saved=(Path(folder)/CONTRACT_FILE).read_bytes()
            with temporal_process_hooks(official,{},TemporalRGBSpec()):
                with self.assertRaises(ValueError):official.make_dataset(cfg)
            self.assertEqual(saved,(Path(folder)/CONTRACT_FILE).read_bytes())
    def test_original_32d_nonpeft_required_before_dataset_factory(self):
        with tempfile.TemporaryDirectory() as folder, patch('m20pro_vla.training.raw_smolvla.official_raw_dataset_factory') as called:
            official=self.official();cfg=self.config(folder);cfg.peft={'enabled':True}
            with temporal_process_hooks(official,{},TemporalRGBSpec()):
                with self.assertRaises(ValueError):official.make_dataset(cfg)
            called.assert_not_called()

if __name__=='__main__':unittest.main()
