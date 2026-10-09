import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from m20pro_vla.data.temporal_rgb import TemporalRGBSpec
from m20pro_vla.data.temporal_raw_candidate import TemporalRawDatasetCandidate
from m20pro_vla.training.temporal_factory_candidate import make_temporal_policy_candidate
from test_temporal_policy_candidate import BasePolicyStub


def get_policy_class(name):return BasePolicyStub

def factory(cfg,ds_meta):
    cls=get_policy_class(cfg.type)
    if cfg.pretrained_path:return cls.from_pretrained(pretrained_name_or_path=cfg.pretrained_path,config=cfg)
    return cls(cfg)


class TemporalFactoryCandidateTests(unittest.TestCase):
    def config(self,path=None):return SimpleNamespace(type='smolvla',use_peft=False,pretrained_path=path,
        max_state_dim=32,adapt_to_pi_aloha=False,image_features={'observation.images.front':None,'observation.images.rear':None})
    def dataset(self,spec=None):
        raw=SimpleNamespace(meta=SimpleNamespace(fps=25,camera_keys=['observation.images.front','observation.images.rear'],features={'observation.state':{'shape':(32,)}}),delta_indices={})
        return TemporalRawDatasetCandidate(raw,spec or TemporalRGBSpec())
    def test_scratch_factory_selects_same_spec_without_mutating_original_globals(self):
        original=factory.__globals__['get_policy_class'];spec=TemporalRGBSpec((-200,-50,0))
        policy=make_temporal_policy_candidate(factory,self.config(),self.dataset(spec));self.assertEqual(policy.model.spec,spec);self.assertIs(factory.__globals__['get_policy_class'],original);self.assertIsInstance(factory(self.config(),None),BasePolicyStub)
    def test_invalid_backend_or_unwrapped_dataset_rejected(self):
        cfg=self.config();cfg.use_peft=True
        with self.assertRaises(ValueError):make_temporal_policy_candidate(factory,cfg,self.dataset())
        with self.assertRaises(ValueError):make_temporal_policy_candidate(factory,self.config(),object())
    def test_resume_without_checkpoint_rejected(self):
        with self.assertRaises(ValueError):make_temporal_policy_candidate(factory,self.config(),self.dataset(),require_temporal_contract=True)
    def test_factory_requires_known_policy_class_interface(self):
        with self.assertRaises(ValueError):make_temporal_policy_candidate(lambda cfg,ds_meta:None,self.config(),self.dataset())

if __name__=='__main__':unittest.main()
