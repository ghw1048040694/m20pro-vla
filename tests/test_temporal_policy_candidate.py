import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from m20pro_vla.data.temporal_rgb import TemporalRGBSpec
from m20pro_vla.training.temporal_policy_candidate import temporal_policy_class, CONTRACT_FILE


class BasePolicyStub(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__(); self.config=config
        self.model=nn.Module(); self.model.state_proj=nn.Linear(32,6)
    def prepare_images(self,batch):
        keys=tuple(self.config.image_features)
        return [batch[k]*2-1 for k in keys], [batch[k+'_padding_mask'] for k in keys]
    def get_optim_params(self):return self.parameters()
    def _save_pretrained(self,directory):torch.save(self.state_dict(),Path(directory)/'weights.pt')
    @classmethod
    def from_pretrained(cls,path,config,**kwargs):
        strict=kwargs.pop('strict');policy=cls(config,**kwargs)
        policy.load_state_dict(torch.load(Path(path)/'weights.pt',weights_only=True),strict=strict)
        return policy.eval()


class TemporalPolicyCandidateTests(unittest.TestCase):
    def config(self):return SimpleNamespace(max_state_dim=32,adapt_to_pi_aloha=False,
        image_features={'observation.images.front':None,'observation.images.rear':None})

    def test_preparation_keeps_all_slots_and_new_parameters_reach_inherited_optimizer(self):
        cls=temporal_policy_class(BasePolicyStub);policy=cls(self.config())
        batch={k:torch.zeros((1,3,3,2,2)) for k in policy.config.image_features}
        for k in policy.config.image_features:batch[k+'_is_pad']=torch.zeros((1,3),dtype=torch.bool)
        batch['observation.images.front'][:,0]=1
        images,masks=policy.prepare_images(batch)
        self.assertEqual(len(images),6);self.assertTrue(torch.all(images[0]==1));self.assertTrue(torch.all(images[-2]==-1))
        params=list(policy.get_optim_params());self.assertIn(id(policy.model.identity.age_projection.weight),list(map(id,params)))
        self.assertEqual(len(params),len(set(map(id,params))))

    def test_legacy_transfer_then_temporal_strict_reload(self):
        cls=temporal_policy_class(BasePolicyStub);config=self.config()
        with tempfile.TemporaryDirectory() as folder:
            legacy=Path(folder)/'legacy';legacy.mkdir();original=BasePolicyStub(config);original._save_pretrained(legacy)
            policy=cls.from_pretrained(legacy,config=config)
            self.assertTrue(torch.equal(original.model.state_proj.weight,policy.model.core.state_proj.weight))
            self.assertFalse(policy.training);self.assertFalse(policy.model.training)
            temporal=Path(folder)/'temporal';temporal.mkdir();policy._save_pretrained(temporal)
            restored=cls.from_pretrained(temporal,config=config,require_temporal_contract=True)
            self.assertTrue(all(torch.equal(v,restored.state_dict()[k]) for k,v in policy.state_dict().items()))

    def test_missing_contract_or_changed_sampling_resume_rejected(self):
        cls=temporal_policy_class(BasePolicyStub);config=self.config()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);base=BasePolicyStub(config);base._save_pretrained(root)
            with self.assertRaises(ValueError):cls.from_pretrained(root,config=config,require_temporal_contract=True)
            policy=cls.from_pretrained(root,config=config);policy._save_pretrained(root)
            with self.assertRaises(ValueError):cls.from_pretrained(root,config=config,temporal_spec=TemporalRGBSpec((-200,-50,0)))
            with self.assertRaises(ValueError):cls.from_pretrained(root,config=config,strict=False)

    def test_original_base_class_is_unchanged(self):
        old=BasePolicyStub.prepare_images;cls=temporal_policy_class(BasePolicyStub)
        self.assertIs(BasePolicyStub.prepare_images,old);self.assertNotEqual(cls.prepare_images,old)


if __name__=='__main__':unittest.main()
